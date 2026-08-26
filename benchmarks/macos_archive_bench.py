"""Benchmark hybrid archive decode with or without persistent Metal resources."""
import argparse
import json
import os
import statistics
import subprocess


def run(args, cache):
    environment = os.environ.copy()
    environment["SHADOW_THREADS"] = str(args.threads)
    environment["SHADOW_METAL_CACHE"] = "1" if cache else "0"
    result = subprocess.run([
        args.kernel, args.model, args.table, args.tokens, str(args.generate),
        "--archive", args.archive, "--archive-backend", "metal",
        "--archive-topk", str(args.archive_top_k), "--bench",
    ], check=True, capture_output=True, text=True, env=environment)
    marker = " | decode "
    line = next(line for line in result.stderr.splitlines() if marker in line)
    return float(line.split(marker, 1)[1].split(" tok/s", 1)[0])


def main():
    parser = argparse.ArgumentParser()
    for name in ("kernel", "model", "table", "archive"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--tokens", default="2")
    parser.add_argument("--generate", type=int, default=33)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--archive-top-k", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--cache", choices=("on", "off"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(); enabled = args.cache == "on"
    for _ in range(args.warmup): run(args, enabled)
    values = [run(args, enabled) for _ in range(args.runs)]
    ordered = sorted(values)
    payload = {"format": "shadow-macos-archive-benchmark-v1", "cache": args.cache,
               "runs": values, "decode_tok_s_median": statistics.median(values),
               "decode_tok_s_p05": ordered[max(0, int(0.05 * len(ordered)) - 1)],
               "decode_tok_s_p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered))) ]}
    with open(args.out, "w") as stream: json.dump(payload, stream, indent=2); stream.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
