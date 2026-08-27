"""Autoregressive SHADOW inference that stays in one CUDA process."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .format import DenseRecord, RVQRecord, ShadowModelFile, TernaryRecord, unpack_rvq, unpack_ternary
from .kernels import (
    PackedRVQWeight, PackedTernaryWeight, TileLangLinear, TorchLinear,
    compile_attention,
)


D = 1536
LAYERS = 10
QUERY_HEADS = 24
KV_HEADS = 2
HEAD_DIM = 64
FFN_DIM = 4224
FINGERPRINT_DIM = 512
VOCAB_SIZE = 131072


def _rms_norm(x, weight, eps: float = 1e-6):
    import torch

    scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (x.float() * scale).to(x.dtype) * weight.to(x.dtype)


def _power_of_two_quantize(x):
    import torch

    maximum = x.abs().amax(-1, keepdim=True).clamp_min(1e-6)
    scale = torch.exp2(torch.ceil(torch.log2(maximum / 127.0)))
    return (x / scale).round().clamp(-127, 127) * scale


class TileLangEngine:
    """Load ``.shdw`` weights and execute the model on one CUDA GPU.

    ``backend="tilelang"`` is the native path. ``backend="torch"`` is a
    deliberately simple reference used to validate graph and kernel parity.
    """

    def __init__(
        self,
        model: str | Path,
        table: str | Path,
        *,
        device: str = "cuda",
        backend: str = "tilelang",
        max_context: int = 2048,
    ):
        import torch

        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            raise RuntimeError("a CUDA-capable PyTorch installation and GPU are required")
        if backend not in ("tilelang", "torch"):
            raise ValueError("backend must be 'tilelang' or 'torch'")
        self.device = torch.device(device)
        # Training and the original runtime quantize activations with BF16's
        # exponent range. FP16 overflows in the structural step.
        self.dtype = torch.bfloat16
        self.max_context = int(max_context)
        if self.max_context < 1:
            raise ValueError("max_context must be positive")
        self.linear = TileLangLinear() if backend == "tilelang" else TorchLinear()
        self.backend = backend
        self.model_file = ShadowModelFile(model)
        self.weights = self._load_weights()
        self.fingerprints = self._load_fingerprints(table)
        self._inv_frequency = 1.0 / (
            10000.0
            ** (torch.arange(0, HEAD_DIM, 2, device=self.device).float() / HEAD_DIM)
        )
        self.reset()

    def close(self) -> None:
        self.model_file.close()

    def reset(self) -> None:
        import torch

        self.position = 0
        if self.backend == "tilelang":
            cache_shape = (LAYERS, KV_HEADS, self.max_context, HEAD_DIM)
            self.k_cache = torch.empty(
                cache_shape, device=self.device, dtype=self.dtype
            )
            self.v_cache = torch.empty(
                cache_shape, device=self.device, dtype=self.dtype
            )
            self._position_cuda = torch.zeros(
                1, device=self.device, dtype=torch.int32
            )
        else:
            self.k_cache = [[] for _ in range(LAYERS)]
            self.v_cache = [[] for _ in range(LAYERS)]
        self.trunk_cache = []

    def _load_weights(self):
        import torch

        weights = {}
        for record in self.model_file:
            if isinstance(record, DenseRecord):
                array = np.array(record.value, copy=True)
                tensor = torch.from_numpy(array).to(
                    device=self.device,
                    dtype=self.dtype if record.name in ("emb.weight", "head.weight") else torch.float32,
                    non_blocking=False,
                )
            elif isinstance(record, RVQRecord):
                if self.backend == "tilelang":
                    codebooks = torch.from_numpy(np.array(record.codebooks, copy=True)).to(self.device)
                    indices = torch.from_numpy(np.array(record.indices, copy=True)).to(self.device)
                    scales = torch.from_numpy(np.array(record.scales, copy=True)).to(self.device)
                    tensor = PackedRVQWeight(
                        codebooks.unsqueeze(0).expand(
                            record.out_features // 64, -1, -1, -1
                        ).contiguous(),
                        indices, scales, record.out_features, record.in_features,
                        record.group_size, record.stages,
                    )
                else:
                    tensor = torch.from_numpy(unpack_rvq(record)).to(self.device, self.dtype)
            elif isinstance(record, TernaryRecord):
                if self.backend == "tilelang":
                    packed = torch.from_numpy(np.array(record.packed, copy=True)).to(self.device)
                    scales = torch.from_numpy(np.array(record.scales, copy=True)).to(self.device)
                    tensor = PackedTernaryWeight(
                        packed, scales, record.out_features, record.in_features
                    )
                else:
                    tensor = torch.from_numpy(unpack_ternary(record)).to(self.device, self.dtype)
            else:  # pragma: no cover - exhaustive type guard
                raise TypeError(type(record))
            weights[record.name] = (
                tensor.contiguous() if isinstance(tensor, torch.Tensor) else tensor
            )
        expected = {"emb.weight", "head.weight", "tb", "nf.w"}
        missing = expected - weights.keys()
        if missing:
            raise ValueError(f"model is missing required tensors: {sorted(missing)}")
        # These concatenations turn related projections into one GEMV launch.
        # Row order matches the slices in _block, so this is numerically the
        # same operation as three Q/K/V and two up/gate calls.
        for layer in range(LAYERS):
            prefix = f"b.{layer}"
            if self.backend == "tilelang":
                weights[f"{prefix}.qkv"] = self._join_rvq(
                    weights[f"{prefix}.q"], weights[f"{prefix}.k"],
                    weights[f"{prefix}.v"],
                )
                weights[f"{prefix}.up_gate"] = self._join_ternary(
                    weights[f"{prefix}.up"], weights[f"{prefix}.gt"]
                )
            else:
                weights[f"{prefix}.qkv"] = torch.cat(
                    (weights[f"{prefix}.q"], weights[f"{prefix}.k"],
                     weights[f"{prefix}.v"]), dim=0,
                ).contiguous()
                weights[f"{prefix}.up_gate"] = torch.cat(
                    (weights[f"{prefix}.up"], weights[f"{prefix}.gt"]), dim=0
                ).contiguous()
            for suffix in ("q", "k", "v", "up", "gt"):
                del weights[f"{prefix}.{suffix}"]
        return weights

    @staticmethod
    def _join_rvq(*weights: PackedRVQWeight) -> PackedRVQWeight:
        import torch

        first = weights[0]
        if any(
            weight.in_features != first.in_features
            or weight.group_size != first.group_size
            or weight.stages != first.stages
            for weight in weights
        ):
            raise ValueError("cannot concatenate incompatible packed RVQ matrices")
        return PackedRVQWeight(
            torch.cat(tuple(weight.codebooks for weight in weights), dim=0).contiguous(),
            torch.cat(tuple(weight.indices for weight in weights), dim=1).contiguous(),
            torch.cat(tuple(weight.scales for weight in weights), dim=0).contiguous(),
            sum(weight.out_features for weight in weights),
            first.in_features, first.group_size, first.stages,
        )

    @staticmethod
    def _join_ternary(*weights: PackedTernaryWeight) -> PackedTernaryWeight:
        import torch

        first = weights[0]
        if any(weight.in_features != first.in_features for weight in weights):
            raise ValueError("cannot concatenate incompatible packed ternary matrices")
        return PackedTernaryWeight(
            torch.cat(tuple(weight.packed for weight in weights), dim=0).contiguous(),
            torch.cat(tuple(weight.scales for weight in weights), dim=0).contiguous(),
            sum(weight.out_features for weight in weights), first.in_features,
        )

    def _load_fingerprints(self, path: str | Path):
        import torch

        packed = np.load(path)
        if packed.ndim != 2 or packed.shape[0] != VOCAB_SIZE:
            raise ValueError(
                f"fingerprint table must have shape ({VOCAB_SIZE}, bytes), got {packed.shape}"
            )
        bits = np.unpackbits(packed, axis=1)[:, :FINGERPRINT_DIM]
        values = bits.astype(np.float32) * 2.0 - 1.0
        return torch.from_numpy(values).to(self.device, self.dtype).contiguous()

    def _projection(self, name: str, x):
        return self.linear(x, self.weights[name])

    def _rope(self, x, position: int):
        import torch

        angle = float(position) * self._inv_frequency
        cosine, sine = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)

    def _block(self, layer: int, x):
        import torch
        import torch.nn.functional as functional

        prefix = f"b.{layer}"
        z = _rms_norm(x, self.weights[f"{prefix}.n1.w"])
        qkv = self._projection(f"{prefix}.qkv", z)
        q_end = QUERY_HEADS * HEAD_DIM
        kv_width = KV_HEADS * HEAD_DIM
        q = qkv[:q_end].reshape(QUERY_HEADS, HEAD_DIM)
        k = qkv[q_end : q_end + kv_width].reshape(KV_HEADS, HEAD_DIM)
        v = qkv[q_end + kv_width :].reshape(KV_HEADS, HEAD_DIM)
        q = _rms_norm(q, self.weights[f"{prefix}.qn.w"])
        k = _rms_norm(k, self.weights[f"{prefix}.kn.w"])
        q, k = self._rope(q, self.position), self._rope(k, self.position)
        q = _power_of_two_quantize(q)
        k = _power_of_two_quantize(k)
        v = _power_of_two_quantize(v)
        alpha = self.weights[f"{prefix}.alpha"].reshape(QUERY_HEADS)
        alpha = (alpha * 4096.0).round() / 4096.0
        if self.backend == "tilelang":
            slot = self.position % self.max_context
            self.k_cache[layer, :, slot] = k
            self.v_cache[layer, :, slot] = v
            attended = compile_attention(
                QUERY_HEADS, KV_HEADS, HEAD_DIM, self.max_context
            )(
                q, self.k_cache[layer], self.v_cache[layer], alpha,
                self._position_cuda,
            ).reshape(D)
        else:
            self.k_cache[layer].append(k)
            self.v_cache[layer].append(v)
            self.k_cache[layer] = self.k_cache[layer][-self.max_context :]
            self.v_cache[layer] = self.v_cache[layer][-self.max_context :]
            keys = torch.stack(self.k_cache[layer], dim=1)
            values = torch.stack(self.v_cache[layer], dim=1)
            repeat = QUERY_HEADS // KV_HEADS
            keys = keys.repeat_interleave(repeat, dim=0)
            values = values.repeat_interleave(repeat, dim=0)
            scores = torch.einsum("hd,htd->ht", q, keys)
            scores = torch.floor(scores * alpha[:, None])
            probability = torch.exp2(
                (scores - scores.amax(-1, keepdim=True)).clamp_min(-15)
            )
            probability /= probability.sum(-1, keepdim=True)
            attended = torch.einsum(
                "ht,htd->hd", probability.to(values.dtype), values
            ).reshape(D)
        gate = torch.sigmoid(self.weights[f"{prefix}.g"]).to(attended.dtype)
        x = x + self._projection(f"{prefix}.o", attended * gate)
        hidden = _rms_norm(x, self.weights[f"{prefix}.n2.w"])
        up_gate = self._projection(f"{prefix}.up_gate", hidden)
        up, gate_projection = up_gate.split(FFN_DIM)
        gated = functional.silu(gate_projection) * up
        return x + self._projection(f"{prefix}.dn", gated)

    def _structural_step(self, current):
        import torch
        import torch.nn.functional as functional

        context = torch.stack(self.trunk_cache, dim=0)
        query = self._projection("step.Wq", current)
        scores = context @ query / math.sqrt(D)
        recall = torch.softmax(scores.float(), dim=-1).to(self.dtype) @ context
        joined = torch.cat((current, recall))
        hidden = functional.silu(self._projection("step.cin", joined))
        output = current + self._projection("step.cout", hidden)
        return _rms_norm(output, self.weights["step.nf.w"])

    def step(self, token_id: int):
        """Consume one token and return float32 next-token logits."""

        if not 0 <= int(token_id) < VOCAB_SIZE:
            raise ValueError(f"token id {token_id} is outside [0, {VOCAB_SIZE})")
        fingerprint = self.fingerprints[int(token_id)]
        if self.backend == "tilelang":
            self._position_cuda.fill_(self.position)
        hidden = self._projection("emb.weight", fingerprint)
        for layer in range(LAYERS):
            hidden = self._block(layer, hidden)
        self.trunk_cache.append(hidden)
        self.trunk_cache = self.trunk_cache[-self.max_context :]
        hidden = self._structural_step(hidden)
        hidden = _rms_norm(hidden, self.weights["nf.w"])
        projected = self._projection("head.weight", hidden).float()
        logits = projected @ (self.fingerprints.float().T / math.sqrt(FINGERPRINT_DIM))
        logits += self.weights["tb"].float()
        self.position += 1
        return logits

    def prefill(self, token_ids: Iterable[int]):
        logits = None
        for token_id in token_ids:
            logits = self.step(int(token_id))
        if logits is None:
            raise ValueError("prefill requires at least one token")
        return logits

    def generate(
        self,
        token_ids: Iterable[int],
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_k: int = 30,
        repetition_penalty: float = 1.15,
        seed: int = 0,
        stop_ids: tuple[int, ...] = (1, 9),
    ) -> list[int]:
        import torch

        prompt = [int(token) for token in token_ids]
        logits = self.prefill(prompt)
        generated: list[int] = []
        generator = torch.Generator(device=self.device).manual_seed(seed)
        for _ in range(int(max_new_tokens)):
            scores = logits.clone()
            if repetition_penalty != 1.0:
                recent = set((prompt + generated)[-128:])
                index = torch.tensor(sorted(recent), device=self.device)
                selected = scores[index]
                scores[index] = torch.where(
                    selected < 0, selected * repetition_penalty, selected / repetition_penalty
                )
            if temperature <= 0:
                token = int(scores.argmax())
            else:
                k = min(max(1, int(top_k)), scores.numel())
                values, indices = torch.topk(scores, k)
                probabilities = torch.softmax(values / temperature, dim=-1)
                token = int(indices[torch.multinomial(probabilities, 1, generator=generator)])
            generated.append(token)
            if token in stop_ids:
                break
            logits = self.step(token)
        return generated
