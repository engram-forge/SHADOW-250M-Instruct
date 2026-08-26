"""Paired sequential/batch-4 prefill qualification for Linux ARM64."""

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import tempfile


PREFILL = re.compile(r"prefill ([0-9.]+)s")
RSS = re.compile(r"maximum_rss_kib=(\d+)")


def percentile(values, fraction):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize(rows, prompt_tokens):
    seconds = [row["prefill_s"] for row in rows]
    rss = [row["maximum_rss_kib"] for row in rows]
    median = statistics.median(seconds)
    return {
        "prefill_s_median": median,
        "prefill_s_p05": percentile(seconds, 0.05),
        "prefill_s_p95": percentile(seconds, 0.95),
        "prefill_tok_s_median": prompt_tokens / median,
        "spread": (max(seconds) - min(seconds)) / median,
        "maximum_rss_kib_median": statistics.median(rss),
        "maximum_rss_kib_max": max(rss),
    }


def run(args, tokens, batch):
    environment = {
        **os.environ,
        "SHADOW_THREADS": str(args.threads),
        "SHADOW_FAST_LOGITS": "0",
        "SHADOW_BATCH_PREFILL": "1" if batch else "0",
    }
    command = [args.kernel, args.model, args.table, tokens, "1", "--bench"]
    result = subprocess.run(
        ["/usr/bin/time", "-f", "maximum_rss_kib=%M", *command],
        check=True, capture_output=True, text=True, env=environment)
    prefill = PREFILL.search(result.stderr)
    rss = RSS.search(result.stderr)
    if not prefill or not rss:
        raise RuntimeError(f"cannot parse benchmark output: {result.stderr}")
    return {"prefill_s": float(prefill.group(1)),
            "maximum_rss_kib": int(rss.group(1))}


def verify_logits(args, tokens):
    with tempfile.TemporaryDirectory(prefix="shadow-prefill-parity-") as directory:
        outputs = []
        for batch, name in ((False, "sequential"), (True, "batch4")):
            output = pathlib.Path(directory) / f"{name}.npy"
            environment = {**os.environ, "SHADOW_THREADS": str(args.threads),
                           "SHADOW_FAST_LOGITS": "0",
                           "SHADOW_BATCH_PREFILL": "1" if batch else "0"}
            subprocess.run([args.kernel, args.model, args.table, tokens, "1",
                            "--dump-logits", str(output)], check=True,
                           capture_output=True, text=True, env=environment)
            outputs.append(output.read_bytes())
        return outputs[0] == outputs[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4, 16, 64, 256])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--token", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    suites = []
    for length in args.lengths:
        tokens = " ".join([str(args.token)] * length)
        for index in range(args.warmup):
            for batch in ((False, True) if index % 2 == 0 else (True, False)):
                run(args, tokens, batch)
        rows = {"sequential": [], "batch4": []}
        for index in range(args.runs):
            order = ((False, "sequential"), (True, "batch4"))
            if index % 2:
                order = tuple(reversed(order))
            for batch, name in order:
                rows[name].append(run(args, tokens, batch))
        sequential = summarize(rows["sequential"], length)
        batch4 = summarize(rows["batch4"], length)
        parity = verify_logits(args, tokens)
        if not parity:
            raise SystemExit(f"logit parity failed for prompt length {length}")
        suites.append({"prompt_tokens": length, "logits_byte_identical": parity,
                       "sequential": sequential, "batch4": batch4,
                       "gain": (batch4["prefill_tok_s_median"] /
                                sequential["prefill_tok_s_median"] - 1.0),
                       "runs": rows})
        print(f"length={length} gain={suites[-1]['gain']:+.1%} parity=exact",
              flush=True)
    payload = {"format": "shadow-linux-arm64-prefill-matrix-v1",
               "threads": args.threads, "warmup": args.warmup,
               "measured_runs_per_path": args.runs, "suites": suites}
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
