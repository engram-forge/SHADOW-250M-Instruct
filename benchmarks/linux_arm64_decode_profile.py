"""Aggregate decode-only runtime stage profiles across repeated runs."""

import argparse
import json
import os
import re
import statistics
import subprocess


PROFILE = re.compile(r"PROFILE (\{.*\})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--tokens", default="2")
    parser.add_argument("--generate", type=int, default=129)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    environment = {**os.environ, "SHADOW_THREADS": str(args.threads),
                   "SHADOW_FAST_LOGITS": "0"}
    command = [args.kernel, args.model, args.table, args.tokens,
               str(args.generate), "--profile"]

    def run():
        result = subprocess.run(command, check=True, capture_output=True,
                                text=True, env=environment)
        match = PROFILE.search(result.stderr)
        if not match:
            raise RuntimeError(f"profile missing from output: {result.stderr}")
        return json.loads(match.group(1))

    for _ in range(args.warmup):
        run()
    rows = [run() for _ in range(args.runs)]
    stages = ["embedding_s", "qkv_s", "attention_s", "output_s",
              "ffn_up_gate_s", "ffn_down_s", "structural_s", "head_s",
              "logits_s"]
    medians = {stage: statistics.median(row[stage] for row in rows)
               for stage in stages}
    measured = statistics.median(row["decode_s"] for row in rows)
    attributed = sum(medians.values())
    payload = {"format": "shadow-linux-arm64-decode-profile-v1",
               "threads": args.threads, "warmup": args.warmup,
               "runs": rows, "decode_s_median": measured,
               "decode_steps_median": statistics.median(
                   row["decode_steps"] for row in rows),
               "stages_s_median": medians,
               "stages_percent": {
                   stage: value / measured * 100.0
                   for stage, value in medians.items()},
               "unattributed_s": max(0.0, measured - attributed)}
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
