"""Compare exact-shiftmax score boundaries for the smallest failing prompt."""
import argparse
import os
import pathlib
import re
import subprocess
import sys

import torch

for key, value in {
    "SHADOW_D": "1536", "SHADOW_NL": "10", "SHADOW_NH": "24",
    "SHADOW_NKV": "2", "SHADOW_HD": "64", "SHADOW_FFNH": "4224",
    "SHADOW_FAST_ATTN": "0", "SHADOW_KV_BITS": "1", "SHADOW_KV_TWO_TIER": "1",
}.items(): os.environ.setdefault(key, value)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "modeling"))
from common import cs, pot, rope
from load_shdw import load_shdw_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True); parser.add_argument("--tokens", required=True)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args(); tokens = [int(x) for x in args.tokens.split()]
    environment = os.environ.copy(); environment["SHADOW_THREADS"] = str(args.threads)
    run = subprocess.run([args.kernel, args.model, args.table, args.tokens, "1", "--trace"],
                         check=True, capture_output=True, text=True, env=environment)
    pattern = re.compile(r"TRACE score (\d+) (\d+) (\d+) (\d+) ([-+0-9.e]+) ([-+0-9.e]+)")
    native = {}
    for line in run.stderr.splitlines():
        match = pattern.fullmatch(line)
        if match: native[tuple(map(int, match.groups()[:4]))] = tuple(map(float, match.groups()[4:]))

    model, _ = load_shdw_model(args.model, args.table); index = torch.tensor([tokens])
    cos, sin = cs(len(tokens), "cpu"); x = model.inp(model.cent[index]); mismatches = []
    layer_max_raw_abs = []
    with torch.inference_mode():
        for layer, block in enumerate(model.b):
            z = block.n1(x); batch, length, _ = z.shape
            q = block.qn(block.q(z).view(batch, length, 24, 64)).transpose(1, 2)
            k = block.kn(block.k(z).view(batch, length, 2, 64)).transpose(1, 2)
            q = pot(rope(q, cos, sin)).to(torch.bfloat16)
            k = pot(rope(k, cos, sin)).to(torch.bfloat16).repeat_interleave(12, 1)
            dot = q @ k.transpose(-1, -2)
            alpha = (block.alpha * 4096).round() / 4096
            raw = alpha * dot; score = raw.floor()
            position = len(tokens) - 1
            max_raw_abs = 0.0
            for head in range(24):
                for key_position in range(len(tokens)):
                    key = (position, layer, head, key_position)
                    native_raw, native_score = native[key]
                    python_raw = float(raw[0, head, position, key_position])
                    python_score = float(score[0, head, position, key_position])
                    max_raw_abs = max(max_raw_abs, abs(python_raw - native_raw))
                    if python_score != native_score:
                        mismatches.append({"layer": layer, "head": head, "key_position": key_position,
                                           "python_raw": python_raw, "native_raw": native_raw,
                                           "python_floor": python_score, "native_floor": native_score})
            layer_max_raw_abs.append(max_raw_abs)
            x = block(x, cos, sin)
    print({"score_mismatches": len(mismatches), "layer_max_raw_abs": layer_max_raw_abs,
           "first": mismatches[:10]})


if __name__ == "__main__": main()
