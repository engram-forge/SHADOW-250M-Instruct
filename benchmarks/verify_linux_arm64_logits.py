"""Compare strict ARM64 NEON logits with scalar kernels and Python .shdw oracle."""
import argparse
import json
import os
import pathlib
import subprocess
import tempfile

import numpy as np


def run_native(args, tokens, output, scalar):
    environment = {**os.environ, "SHADOW_THREADS": str(args.threads), "SHADOW_FAST_LOGITS": "0"}
    if scalar:
        environment.update(SHADOW_TERNARY_REFERENCE="1", SHADOW_LOGITS_REFERENCE="1")
    subprocess.run([args.kernel, args.model, args.table, " ".join(map(str, tokens)), "1",
                    "--dump-logits", str(output)], check=True, capture_output=True, text=True, env=environment)
    return np.load(output)[0]


def run_python(args, tokens, output):
    subprocess.run([args.python, "finetune/dump_shdw_logits.py", "--model", args.model,
                    "--table", args.table, "--tokens", " ".join(map(str, tokens)),
                    "--out", str(output), "--last-only"],
                   check=True, capture_output=True, text=True)
    return np.load(output)[0]


def metrics(left, right):
    difference = right.astype(np.float64) - left.astype(np.float64)
    k = 10
    return {"max_abs": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "argmax_match": bool(np.argmax(left) == np.argmax(right)),
            "top10_overlap": len(set(np.argpartition(left, -k)[-k:]) & set(np.argpartition(right, -k)[-k:])) / k}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True); parser.add_argument("--fixture", required=True)
    parser.add_argument("--threads", type=int, default=4); parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--python", default="python"); parser.add_argument("--out", required=True)
    args = parser.parse_args(); fixture = json.loads(pathlib.Path(args.fixture).read_text())
    rows = []
    with tempfile.TemporaryDirectory(prefix="shadow-linux-parity-") as directory:
        directory = pathlib.Path(directory)
        for case in fixture["cases"][:args.limit]:
            optimized = run_native(args, case["tokens"], directory / "optimized.npy", False)
            scalar = run_native(args, case["tokens"], directory / "scalar.npy", True)
            python = run_python(args, case["tokens"], directory / "python.npy")
            rows.append({"id": case["id"], "scalar_vs_neon": metrics(scalar, optimized),
                         "python_vs_neon": metrics(python, optimized)})
            print(f"verified {case['id']}", flush=True)
    result = {"format": "shadow-linux-arm64-parity-v1", "threads": args.threads, "cases": rows}
    output = pathlib.Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))
    if not all(row["scalar_vs_neon"]["max_abs"] == 0.0 for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
