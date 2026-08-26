"""Dump deployment-model logits for one or more token IDs."""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

for key, value in {
    "SHADOW_D": "1536", "SHADOW_NL": "10", "SHADOW_NH": "24",
    "SHADOW_NKV": "2", "SHADOW_HD": "64", "SHADOW_FFNH": "4224",
    "SHADOW_FAST_ATTN": "0", "SHADOW_KV_BITS": "1",
    "SHADOW_KV_TWO_TIER": "1",
}.items():
    os.environ.setdefault(key, value)

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "modeling"))
from load_shdw import load_shdw_model


def parse_tokens(text):
    values = [int(value) for value in text.replace(",", " " ).split()]
    if not values:
        raise ValueError("at least one token ID is required")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--tokens", required=True, help="space- or comma-separated token IDs")
    parser.add_argument("--out", help="optional .npy file for full float32 logits [tokens,vocab]")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--last-only", action="store_true", help="emit only the final input position")
    args = parser.parse_args()

    tokens = parse_tokens(args.tokens)
    model, version = load_shdw_model(args.model, args.table, args.device)
    with torch.inference_mode():
        logits = model(torch.tensor([tokens], dtype=torch.long, device=args.device))[0].float().cpu().numpy()
    if args.last_only:
        logits = logits[-1:]
    if args.out:
        output = pathlib.Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, logits.astype(np.float32, copy=False))
    top_k = min(args.top_k, logits.shape[-1])
    summary = []
    for position, row in enumerate(logits):
        indices = np.argpartition(-row, top_k - 1)[:top_k]
        indices = indices[np.argsort(-row[indices], kind="stable")]
        summary.append({
            "position": len(tokens) - 1 if args.last_only else position,
            "input_token": tokens[-1] if args.last_only else tokens[position],
            "top_k": [{"token": int(index), "logit": float(row[index])} for index in indices],
        })
    print(json.dumps({"format": "shadow-shdw-logits-v1", "shdw_version": version,
                      "shape": list(logits.shape), "positions": summary}, indent=2))


if __name__ == "__main__":
    main()
