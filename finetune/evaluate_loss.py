"""Evaluate a trainer checkpoint with pure assistant-token MLE on a JSONL file."""
import argparse
import os
import pathlib
import random
import sys

for key, value in {"SHADOW_D": "1536", "SHADOW_NL": "10", "SHADOW_NH": "24", "SHADOW_NKV": "2",
                   "SHADOW_HD": "64", "SHADOW_FFNH": "4224", "SHADOW_FAST_ATTN": "1",
                   "SHADOW_KV_BITS": "1", "SHADOW_KV_TWO_TIER": "1"}.items(): os.environ.setdefault(key, value)
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "modeling")); sys.path.insert(0, str(ROOT / "shadow_runtime"))
import numpy as np
import torch
import torch.nn.functional as F
import common
from common import requant
from finetune import Packer
from model_250m import Shadow250M

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--data", required=True)
    parser.add_argument("--table", default=str(ROOT / "deployment" / "fp131072.npy")); parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--batches", type=int, default=16); parser.add_argument("--micro-batch", type=int, default=2); parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_forward = common.RVQ.forward
    def ternary_forward(module, value):
        if module.g == 32:
            weight = module.weight; scale = 1.0 / weight.abs().mean(dim=1, keepdim=True).clamp_(min=1e-5)
            quantized = weight + ((weight * scale).round().clamp(-1, 1) / scale - weight).detach()
            return F.linear(value, quantized.to(value.dtype))
        return original_forward(module, value)
    common.RVQ.forward = ternary_forward
    original_encode = common.RVQ.enc
    def encode(module):
        if module.g != 32: original_encode(module)
    common.RVQ.enc = encode
    fp = np.unpackbits(np.load(args.table), axis=1)[:, :512]; cent = torch.tensor(fp.astype(np.float32) * 2 - 1, device=device)
    model = Shadow250M(cent, F.normalize(cent, dim=-1), cent.shape[0]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k,v in checkpoint["model"].items()}); requant(model); model.eval()
    rng = random.Random(args.seed); packer = Packer(args.data, args.ctx, rng, 0.02, recovery_ratio=0.0)
    # Evaluate the full file as a dedicated pool, independent of Packer's fallback split.
    packer.val = packer.ex
    losses = []; tokens = 0
    with torch.no_grad():
        for _ in range(args.batches):
            x, y = packer.pack(args.micro_batch, val=True); x, y = x.to(device), y.to(device)
            hidden, _ = model.trunk(x); projected = model.head(hidden).float().reshape(-1, 512); target = y.reshape(-1); valid = target >= 0
            projected, target = projected[valid], target[valid]; total = 0.0
            for start in range(0, len(target), 8192):
                logits = projected[start:start+8192] @ model.cent_n.T + model.tied_bias
                total += F.cross_entropy(logits, target[start:start+8192], reduction="sum")
            count = int(valid.sum()); losses.append(float(total)); tokens += count
    loss = sum(losses) / tokens
    print(f"checkpoint={args.checkpoint} data={args.data} tokens={tokens} mle_loss={loss:.6f} perplexity={np.exp(loss):.4f}")

if __name__ == "__main__": main()
