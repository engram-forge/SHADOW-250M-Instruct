"""Compare strict and fast macOS logits over a tokenized verification fixture."""

import argparse
import json
import os
import pathlib
import subprocess
import tempfile

import numpy as np


def run_logits(args, tokens, output, fast):
    environment = {**os.environ, "SHADOW_THREADS": str(args.threads)}
    environment["SHADOW_FAST_LOGITS"] = "1" if fast else "0"
    subprocess.run(
        [args.kernel, args.model, args.table, " ".join(map(str, tokens)), "1",
         "--dump-logits", str(output)],
        check=True, capture_output=True, text=True, env=environment,
    )
    return np.load(output)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fixture = json.loads(pathlib.Path(args.fixture).read_text())
    max_abs = 0.0
    squared_error = 0.0
    value_count = 0
    argmax_matches = 0
    top10_overlap = 0
    with tempfile.TemporaryDirectory(prefix="shadow-logits-") as directory:
        strict_path = pathlib.Path(directory) / "strict.npy"
        fast_path = pathlib.Path(directory) / "fast.npy"
        for index, case in enumerate(fixture["cases"], 1):
            strict = run_logits(args, case["tokens"], strict_path, False)
            fast = run_logits(args, case["tokens"], fast_path, True)
            difference = fast.astype(np.float64) - strict.astype(np.float64)
            max_abs = max(max_abs, float(np.max(np.abs(difference))))
            squared_error += float(np.dot(difference, difference))
            value_count += difference.size
            argmax_matches += int(np.argmax(strict) == np.argmax(fast))
            strict_top = set(np.argpartition(strict, -10)[-10:])
            fast_top = set(np.argpartition(fast, -10)[-10:])
            top10_overlap += len(strict_top & fast_top)
            if index % 25 == 0 or index == len(fixture["cases"]):
                print(f"verified {index}/{len(fixture['cases'])}", flush=True)

    count = len(fixture["cases"])
    result = {
        "format": "shadow-macos-logits-verification-v1",
        "cases": count,
        "logit_values": value_count,
        "max_abs": max_abs,
        "rmse": (squared_error / value_count) ** 0.5,
        "argmax_agreement": argmax_matches / count,
        "top10_overlap": top10_overlap / (count * 10),
    }
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
