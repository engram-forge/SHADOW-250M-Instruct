"""Autoregressive SHADOW inference that stays in one CUDA process."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .format import DenseRecord, RVQRecord, ShadowModelFile, TernaryRecord, unpack_rvq, unpack_ternary
from .kernels import (
    PackedRVQWeight, PackedTernaryWeight, TileLangLinear, TorchLinear,
    compile_attention, compile_fingerprint_logits, compile_fingerprint_unpack,
    compile_fingerprint_embedding,
    compile_circular_gather, compile_circular_store, compile_fingerprint_gather,
    compile_fingerprint_unpack_batch,
    compile_prefill_attention, compile_rms_norm,
    compile_power_of_two_quantize, compile_rope, compile_rope_quantize,
    compile_qk_rms_norm,
    compile_structural_softmax,
    compile_token_store,
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
            self._token_cuda = torch.zeros(
                1, device=self.device, dtype=torch.int64
            )
            self._trunk_cache_cuda = torch.empty(
                self.max_context, D, device=self.device, dtype=self.dtype
            )
            self._trunk_cache_cuda.zero_()
            self._decode_graph = None
            self._decode_graph_logits = None
            self._greedy_graph = None
            self._greedy_tokens_cuda = torch.empty(
                self.max_context, device=self.device, dtype=torch.int64
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
                    if record.stages != 2:
                        raise ValueError(
                            f"TileLang packed RVQ requires two stages, got {record.stages} for {record.name}"
                        )
                    codebooks = torch.from_numpy(np.array(record.codebooks, copy=True)).to(self.device)
                    pair_codebooks = (
                        codebooks[0][:, None, :] + codebooks[1][:, :, None]
                    ).reshape(record.group_size, 256)
                    packed_indices = np.array(record.indices, copy=True)
                    low = (packed_indices[:, :, :, :] & 15)
                    high = packed_indices[:, :, :, :] >> 4
                    pair_indices = np.concatenate(
                        (low[0] | (low[1] << 4), high[0] | (high[1] << 4)),
                        axis=2,
                    ).transpose(0, 2, 1)
                    indices = torch.from_numpy(pair_indices).to(self.device)
                    scales = torch.from_numpy(np.array(record.scales, copy=True)).to(self.device)
                    tensor = PackedRVQWeight(
                        pair_codebooks.unsqueeze(0).expand(
                            record.out_features // 64, -1, -1
                        ).contiguous(),
                        indices.contiguous(), scales,
                        record.out_features, record.in_features,
                        record.group_size, record.stages,
                    )
                else:
                    tensor = torch.from_numpy(unpack_rvq(record)).to(self.device, self.dtype)
            elif isinstance(record, TernaryRecord):
                if self.backend == "tilelang":
                    source = np.array(record.packed, copy=True)
                    trits = np.empty(
                        (record.out_features, record.in_features), dtype=np.uint8
                    )
                    remaining = source.astype(np.int16)
                    for component in range(5):
                        trits[:, component::5] = (remaining % 3)[:, :
                            trits[:, component::5].shape[1]
                        ]
                        remaining //= 3
                    packed_2bit = np.zeros(
                        (record.out_features, (record.in_features + 4) // 5),
                        dtype=np.uint16,
                    )
                    for component in range(5):
                        values = trits[:, component::5].astype(np.uint16)
                        packed_2bit[:, :values.shape[1]] |= values << (component * 2)
                    packed = torch.from_numpy(packed_2bit).to(self.device)
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
            weights[f"{prefix}.gate"] = torch.sigmoid(
                weights[f"{prefix}.g"]
            ).to(self.dtype).contiguous()
            weights[f"{prefix}.alpha_q"] = (
                (weights[f"{prefix}.alpha"].reshape(QUERY_HEADS) * 4096.0).round()
                / 4096.0
            ).contiguous()
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
            torch.cat(tuple(weight.indices for weight in weights), dim=0).contiguous(),
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
        if (
            packed.ndim != 2 or packed.shape[0] != VOCAB_SIZE
            or packed.shape[1] * 8 < FINGERPRINT_DIM
        ):
            raise ValueError(
                f"fingerprint table must have shape ({VOCAB_SIZE}, bytes), got {packed.shape}"
            )
        if self.backend == "tilelang":
            byte_width = FINGERPRINT_DIM // 8
            return torch.from_numpy(np.array(packed[:, :byte_width], copy=True)).to(
                self.device
            ).contiguous()
        bits = np.unpackbits(packed, axis=1)[:, :FINGERPRINT_DIM]
        values = bits.astype(np.float32) * 2.0 - 1.0
        return torch.from_numpy(values).to(self.device, self.dtype).contiguous()

    def _fingerprint(self, token_id: int):
        if self.backend == "tilelang":
            return compile_fingerprint_unpack(FINGERPRINT_DIM)(
                self.fingerprints[token_id]
            )
        return self.fingerprints[token_id]

    def _fingerprint_cuda(self):
        return compile_fingerprint_gather(VOCAB_SIZE, FINGERPRINT_DIM)(
            self.fingerprints, self._token_cuda
        )

    def _fingerprint_batch(self, indices, batch_size: int):
        import torch

        if self.backend == "tilelang":
            padded = torch.zeros(batch_size, device=self.device, dtype=torch.int64)
            padded[: indices.numel()] = indices
            return compile_fingerprint_unpack_batch(
                batch_size, VOCAB_SIZE, FINGERPRINT_DIM
            )(self.fingerprints, padded)
        return self.fingerprints[indices]

    def _logits(self, projected):
        if self.backend == "tilelang":
            return compile_fingerprint_logits(VOCAB_SIZE, FINGERPRINT_DIM)(
                projected, self.fingerprints, self.weights["tb"].float()
            )
        return (
            projected.float()
            @ (self.fingerprints.float().T / math.sqrt(FINGERPRINT_DIM))
            + self.weights["tb"].float()
        )

    def _projection(self, name: str, x):
        return self.linear(x, self.weights[name])

    def _norm(self, x, weight):
        if self.backend != "tilelang":
            return _rms_norm(x, weight)
        shape = x.shape
        width = shape[-1]
        rows = x.numel() // width
        output = compile_rms_norm(rows, width)(x.reshape(rows, width), weight)
        return output.reshape(shape)

    def _quantize(self, x):
        if self.backend != "tilelang":
            return _power_of_two_quantize(x)
        shape = x.shape
        width = shape[-1]
        rows = x.numel() // width
        output = compile_power_of_two_quantize(rows, width)(
            x.reshape(rows, width)
        )
        return output.reshape(shape)

    def _batch_projection(self, name: str, x):
        if self.backend == "tilelang":
            return self.linear.batch(x, self.weights[name])
        return self.linear(x, self.weights[name])

    def _rope(self, x, position: int, cosine=None, sine=None):
        import torch

        if cosine is None or sine is None:
            angle = float(position) * self._inv_frequency
            cosine, sine = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
        if self.backend == "tilelang":
            return compile_rope(x.shape[0], x.shape[1])(x, cosine, sine)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)

    def _block(self, layer: int, x, cosine, sine):
        import torch
        import torch.nn.functional as functional

        prefix = f"b.{layer}"
        z = self._norm(x, self.weights[f"{prefix}.n1.w"])
        qkv = self._projection(f"{prefix}.qkv", z)
        q_end = QUERY_HEADS * HEAD_DIM
        kv_width = KV_HEADS * HEAD_DIM
        qk = qkv[:q_end + kv_width].reshape(QUERY_HEADS + KV_HEADS, HEAD_DIM)
        q = qk[:QUERY_HEADS]
        k = qk[QUERY_HEADS:]
        v = qkv[q_end + kv_width :].reshape(KV_HEADS, HEAD_DIM)
        if self.backend == "tilelang":
            qk = compile_qk_rms_norm(QUERY_HEADS, KV_HEADS, HEAD_DIM)(
                qk, self.weights[f"{prefix}.qn.w"], self.weights[f"{prefix}.kn.w"]
            )
            qk = compile_rope_quantize(
                QUERY_HEADS + KV_HEADS, HEAD_DIM
            )(qk, cosine, sine)
            q, k = qk[:QUERY_HEADS], qk[QUERY_HEADS:]
        else:
            q = self._norm(q, self.weights[f"{prefix}.qn.w"])
            k = self._norm(k, self.weights[f"{prefix}.kn.w"])
            q = self._quantize(self._rope(q, self.position, cosine, sine))
            k = self._quantize(self._rope(k, self.position, cosine, sine))
        if self.backend != "tilelang":
            v = self._quantize(v)
        alpha = self.weights[f"{prefix}.alpha_q"]
        if self.backend == "tilelang":
            attended = compile_attention(
                QUERY_HEADS, KV_HEADS, HEAD_DIM, self.max_context
            )(
                q, k, v, self.k_cache[layer], self.v_cache[layer], alpha,
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
        gate = self.weights[f"{prefix}.gate"]
        if self.backend == "tilelang":
            x = self.linear.gated_residual(
                attended, gate, x, self.weights[f"{prefix}.o"]
            )
        else:
            x = x + self._projection(f"{prefix}.o", attended * gate)
        hidden = self._norm(x, self.weights[f"{prefix}.n2.w"])
        if self.backend == "tilelang":
            gated = self.linear.swiglu(
                hidden, self.weights[f"{prefix}.up_gate"]
            )
            return self.linear.residual(gated, x, self.weights[f"{prefix}.dn"])
        up_gate = self._projection(f"{prefix}.up_gate", hidden)
        up, gate_projection = up_gate.split(FFN_DIM)
        gated = functional.silu(gate_projection) * up
        return x + self._projection(f"{prefix}.dn", gated)

    def _prefill_block(self, layer: int, x, start_position: int):
        import torch
        import torch.nn.functional as functional

        prefix = f"b.{layer}"
        tokens = x.shape[0]
        z = self._norm(x, self.weights[f"{prefix}.n1.w"])
        qkv = self._batch_projection(f"{prefix}.qkv", z)
        q_end = QUERY_HEADS * HEAD_DIM
        kv_width = KV_HEADS * HEAD_DIM
        q = qkv[:, :q_end].reshape(tokens, QUERY_HEADS, HEAD_DIM)
        k = qkv[:, q_end : q_end + kv_width].reshape(tokens, KV_HEADS, HEAD_DIM)
        v = qkv[:, q_end + kv_width :].reshape(tokens, KV_HEADS, HEAD_DIM)
        q = self._norm(q, self.weights[f"{prefix}.qn.w"])
        k = self._norm(k, self.weights[f"{prefix}.kn.w"])
        positions = torch.arange(
            start_position, start_position + tokens, device=self.device, dtype=torch.float32
        )
        angle = positions[:, None] * self._inv_frequency[None]
        cosine, sine = angle.cos().to(q.dtype), angle.sin().to(q.dtype)
        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]
        q = torch.stack((q_even * cosine[:, None] - q_odd * sine[:, None],
                         q_even * sine[:, None] + q_odd * cosine[:, None]), -1).flatten(-2)
        k = torch.stack((k_even * cosine[:, None] - k_odd * sine[:, None],
                         k_even * sine[:, None] + k_odd * cosine[:, None]), -1).flatten(-2)
        q, k, v = (_power_of_two_quantize(value) for value in (q, k, v))
        slots = torch.arange(start_position, start_position + tokens, device=self.device)
        slots %= self.max_context
        self.k_cache[layer, :, slots] = k.transpose(0, 1)
        self.v_cache[layer, :, slots] = v.transpose(0, 1)
        alpha = self.weights[f"{prefix}.alpha"].reshape(QUERY_HEADS)
        alpha = (alpha * 4096.0).round() / 4096.0
        attended = compile_prefill_attention(
            tokens, QUERY_HEADS, KV_HEADS, HEAD_DIM
        )(q, k, v, alpha).reshape(tokens, D)
        gate = torch.sigmoid(self.weights[f"{prefix}.g"]).to(attended.dtype)
        x = x + self._batch_projection(f"{prefix}.o", attended * gate)
        hidden = self._norm(x, self.weights[f"{prefix}.n2.w"])
        up_gate = self._batch_projection(f"{prefix}.up_gate", hidden)
        up, gate_projection = up_gate.split(FFN_DIM, dim=-1)
        return x + self._batch_projection(
            f"{prefix}.dn", functional.silu(gate_projection) * up
        )

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
        return self._norm(output, self.weights["step.nf.w"])

    def _structural_step_cuda(self, current):
        import torch
        import torch.nn.functional as functional

        query = self._projection("step.Wq", current)
        scores = self._trunk_cache_cuda @ query / math.sqrt(D)
        probability = compile_structural_softmax(self.max_context)(
            scores, self._position_cuda
        )
        recall = probability @ self._trunk_cache_cuda
        hidden = self.linear.split_silu(
            current, recall, self.weights["step.cin"]
        )
        output = self.linear.rvq_residual(
            hidden, current, self.weights["step.cout"]
        )
        return self._norm(output, self.weights["step.nf.w"])

    def _decode_trunk_cuda(self):
        angle = self._position_cuda.float() * self._inv_frequency
        cosine = angle.cos().to(self.dtype)
        sine = angle.sin().to(self.dtype)
        hidden = compile_fingerprint_embedding(
            VOCAB_SIZE, FINGERPRINT_DIM, D
        )(
            self.fingerprints, self._token_cuda,
            self.weights["emb.weight"],
        )
        for layer in range(LAYERS):
            hidden = self._block(layer, hidden, cosine, sine)
        return hidden

    def _decode_cuda(self):
        hidden = self._decode_trunk_cuda()
        compile_circular_store(self.max_context, D)(
            hidden, self._trunk_cache_cuda, self._position_cuda
        )
        hidden = self._structural_step_cuda(hidden)
        hidden = self._norm(hidden, self.weights["nf.w"])
        projected = self._projection("head.weight", hidden)
        return self._logits(projected)

    def _decode_graph_step(self):
        """Replay a complete dynamic-token model step as one CUDA graph."""

        import torch

        self._ensure_decode_graph()
        self._decode_graph.replay()
        # Graph outputs have stable storage and are overwritten by the next
        # replay. Preserve normal step() value semantics for callers retaining
        # logits from more than one token.
        return self._decode_graph_logits.clone()

    def _ensure_decode_graph(self):
        """Capture the dynamic decode graph without changing logical position."""

        import torch

        if self._decode_graph is None:
            # Compile every lazy TileLang specialization before capture. The
            # warm execution writes the same dynamic cache slot that capture
            # immediately overwrites, so it does not advance model state.
            self._decode_cuda()
            torch.cuda.synchronize(self.device)
            self._decode_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._decode_graph):
                self._decode_graph_logits = self._decode_cuda()
            # Capture records work but does not execute it. Replay immediately
            # so the first graphed token observes populated output storage.

    def _ensure_greedy_graph(self):
        """Capture token selection, state advance, and one complete decode."""

        import torch

        if self._greedy_graph is None:
            self._ensure_decode_graph()
            torch.cuda.synchronize(self.device)
            self._greedy_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._greedy_graph):
                compile_token_store(self.max_context)(
                    self._token_cuda, self._greedy_tokens_cuda,
                    self._position_cuda,
                )
                logits = self._decode_cuda()
                self._token_cuda.copy_(logits.argmax().reshape(1))
                self._position_cuda.add_(1)

    def _generate_greedy_cuda(self, logits, token_count: int) -> list[int]:
        """Generate fixed-length greedy tokens with one final host transfer."""

        import torch

        if token_count <= 0:
            return []
        start_position = self.position
        self._token_cuda.copy_(logits.argmax().reshape(1))
        self._ensure_greedy_graph()
        generated = []
        completed = 0
        while completed < token_count:
            chunk = min(self.max_context, token_count - completed)
            for _ in range(chunk):
                self._greedy_graph.replay()
            slots = torch.arange(
                start_position + completed,
                start_position + completed + chunk,
                device=self.device, dtype=torch.int64,
            ) % self.max_context
            generated.extend(self._greedy_tokens_cuda[slots].cpu().tolist())
            completed += chunk
        self.position += token_count
        return generated

    def _consume(self, token_id: int, *, return_logits: bool):
        if not 0 <= int(token_id) < VOCAB_SIZE:
            raise ValueError(f"token id {token_id} is outside [0, {VOCAB_SIZE})")
        if self.backend == "tilelang":
            self._token_cuda.fill_(int(token_id))
            self._position_cuda.fill_(self.position)
        angle = float(self.position) * self._inv_frequency
        cosine, sine = angle.cos().to(self.dtype), angle.sin().to(self.dtype)
        if self.backend == "tilelang":
            if return_logits:
                logits = self._decode_graph_step()
                self.position += 1
                return logits
            hidden = self._decode_trunk_cuda()
        else:
            hidden = self._projection(
                "emb.weight", self._fingerprint(int(token_id))
            )
            for layer in range(LAYERS):
                hidden = self._block(layer, hidden, cosine, sine)
        if self.backend != "tilelang":
            self.trunk_cache.append(hidden)
            self.trunk_cache = self.trunk_cache[-self.max_context :]
        else:
            compile_circular_store(self.max_context, D)(
                hidden, self._trunk_cache_cuda, self._position_cuda
            )
        self.position += 1
        if not return_logits:
            return None
        hidden = (
            self._structural_step_cuda(hidden)
            if self.backend == "tilelang" else self._structural_step(hidden)
        )
        hidden = self._norm(hidden, self.weights["nf.w"])
        projected = self._projection("head.weight", hidden)
        return self._logits(projected)

    def step(self, token_id: int):
        """Consume one token and return float32 next-token logits."""

        return self._consume(token_id, return_logits=True)

    def prefill(self, token_ids: Iterable[int]):
        tokens = [int(token_id) for token_id in token_ids]
        if not tokens:
            raise ValueError("prefill requires at least one token")
        if (
            self.backend == "tilelang" and self.position == 0
            and 1 < len(tokens) <= self.max_context
        ):
            import torch

            if any(not 0 <= token < VOCAB_SIZE for token in tokens):
                raise ValueError(f"token IDs must be inside [0, {VOCAB_SIZE})")
            token_count = len(tokens)
            batch_size = 1 << (token_count - 1).bit_length()
            index = torch.tensor(tokens, device=self.device)
            fingerprints = self._fingerprint_batch(index, batch_size)
            hidden = self._batch_projection("emb.weight", fingerprints)
            for layer in range(LAYERS):
                hidden = self._prefill_block(layer, hidden, 0)
            hidden = hidden[:token_count]
            self.trunk_cache = list(hidden.unbind(0))
            self._trunk_cache_cuda[:token_count].copy_(hidden)
            self.position = token_count
            self._position_cuda.fill_(self.position)
            final = self._structural_step(hidden[-1])
            final = self._norm(final, self.weights["nf.w"])
            projected = self._projection("head.weight", final)
            return self._logits(projected)
        for token_id in tokens[:-1]:
            self._consume(token_id, return_logits=False)
        return self._consume(tokens[-1], return_logits=True)

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
        if (
            self.backend == "tilelang"
            and temperature <= 0
            and repetition_penalty == 1.0
            and not stop_ids
        ):
            return self._generate_greedy_cuda(logits, int(max_new_tokens))
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
