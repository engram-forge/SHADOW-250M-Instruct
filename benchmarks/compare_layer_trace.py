"""Locate the first Python/native activation divergence for one token sequence."""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

import torch

for key, value in {
    "SHADOW_D": "1536", "SHADOW_NL": "10", "SHADOW_NH": "24",
    "SHADOW_NKV": "2", "SHADOW_HD": "64", "SHADOW_FFNH": "4224",
    "SHADOW_FAST_ATTN": "0", "SHADOW_KV_BITS": "1",
    "SHADOW_KV_TWO_TIER": "1",
}.items():
    os.environ.setdefault(key, value)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "modeling"))
from load_shdw import load_shdw_model


def parse_tokens(text):
    return [int(value) for value in text.replace(",", " " ).split()]


def summary(tensor):
    value = tensor.detach().float().cpu()
    return torch.stack((value[..., 0], value[..., 1], value.square().sum(-1)), -1)[0].tolist()


def python_trace(model, tokens):
    rows = {}
    hooks = [model.inp.register_forward_hook(lambda _m, _i, out: rows.update(emb=summary(out)))]
    for layer, block in enumerate(model.b):
        hooks.append(block.register_forward_hook(
            lambda _m, _i, out, layer=layer: rows.update({f"layer{layer}": summary(out)})))
    hooks.append(model.struct.register_forward_hook(
        lambda _m, _i, out: rows.update(struct=summary(out[0]))))
    try:
        with torch.inference_mode():
            model(torch.tensor([tokens], dtype=torch.long))
    finally:
        for hook in hooks:
            hook.remove()
    return rows


def native_trace(args, tokens):
    environment = os.environ.copy()
    environment["SHADOW_THREADS"] = str(args.threads)
    result = subprocess.run(
        [args.kernel, args.model, args.table, " ".join(map(str, tokens)), "1", "--trace"],
        check=True, capture_output=True, text=True, env=environment,
    )
    pattern = re.compile(r"^TRACE (emb|layer\d+|struct) (-?[0-9.e+]+) (-?[0-9.e+]+) (-?[0-9.e+]+)$", re.I)
    rows = {"emb": [], "struct": []}
    rows.update({f"layer{layer}": [] for layer in range(10)})
    for line in result.stderr.splitlines():
        match = pattern.match(line)
        if match:
            rows[match.group(1)].append([float(match.group(i)) for i in range(2, 5)])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()
    tokens = parse_tokens(args.tokens)
    model, _ = load_shdw_model(args.model, args.table)
    reference = python_trace(model, tokens)
    candidate = native_trace(args, tokens)
    report = []
    for stage in ["emb"] + [f"layer{i}" for i in range(10)] + ["struct"]:
        if len(reference[stage]) != len(candidate[stage]):
            raise RuntimeError(f"trace length mismatch for {stage}")
        for position, (left, right) in enumerate(zip(reference[stage], candidate[stage])):
            absolute = [abs(a - b) for a, b in zip(left, right)]
            report.append({
                "stage": stage, "position": position,
                "max_summary_abs": max(absolute),
                "x0_abs": absolute[0], "x1_abs": absolute[1], "squared_norm_abs": absolute[2],
            })
    stages = ["emb"] + [f"layer{i}" for i in range(10)] + ["struct"]
    report.sort(key=lambda row: (row["position"], stages.index(row["stage"])))
    print(json.dumps({"tokens": len(tokens), "trace": report}, indent=2))


if __name__ == "__main__":
    main()
