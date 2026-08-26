"""Reproducible Linux ARM64 runtime benchmark with RSS and percentile metrics."""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import statistics
import subprocess
import time

SPEED = re.compile(r"decode ([0-9.]+) tok/s")
PREFILL = re.compile(r"prefill ([0-9.]+)s")
RSS = re.compile(r"maximum_rss_kib=(\d+)")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999))]


def run_once(args):
    environment = {**os.environ, "SHADOW_THREADS": str(args.threads),
                   "SHADOW_FAST_LOGITS": "1" if args.fast_logits else "0"}
    command = [args.kernel, args.model, args.table, args.tokens, str(args.generate), "--bench"]
    started = time.perf_counter()
    result = subprocess.run(
        ["/usr/bin/time", "-f", "maximum_rss_kib=%M", *command],
        capture_output=True, text=True, check=True, env=environment,
    )
    wall = time.perf_counter() - started
    speed = SPEED.search(result.stderr); prefill = PREFILL.search(result.stderr); rss = RSS.search(result.stderr)
    if not speed or not prefill or not rss:
        raise RuntimeError(f"cannot parse benchmark output: {result.stderr}")
    return {"decode_tok_s": float(speed.group(1)), "prefill_s": float(prefill.group(1)),
            "wall_s": wall, "maximum_rss_kib": int(rss.group(1)),
            "output_tokens": len(result.stdout.split())}


def compiler_note(kernel):
    result = subprocess.run(["readelf", "-p", ".comment", kernel], capture_output=True, text=True)
    return " ".join(line.strip() for line in result.stdout.splitlines() if "compiler:" in line or "clang" in line.lower() or "gcc" in line.lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True); parser.add_argument("--tokens", default="2")
    parser.add_argument("--generate", type=int, default=65); parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3); parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--fast-logits", action="store_true"); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for _ in range(args.warmup): run_once(args)
    rows = [run_once(args) for _ in range(args.runs)]
    speeds = [row["decode_tok_s"] for row in rows]; prefills = [row["prefill_s"] for row in rows]
    rss = [row["maximum_rss_kib"] for row in rows]
    release = platform.freedesktop_os_release()
    payload = {
        "format": "shadow-linux-arm64-benchmark-v1", "machine": platform.machine(),
        "platform": platform.platform(), "distribution": release.get("PRETTY_NAME"),
        "wsl": "microsoft" in platform.release().lower(), "cpu_count": os.cpu_count(),
        "compiler_comment": compiler_note(args.kernel),
        "kernel_sha256": sha256(args.kernel), "model_sha256": sha256(args.model),
        "table_sha256": sha256(args.table), "threads": args.threads,
        "fast_logits": args.fast_logits, "prompt_tokens": len(args.tokens.split()),
        "requested_tokens": args.generate, "warmup": args.warmup, "runs": rows,
        "decode_tok_s_median": statistics.median(speeds),
        "decode_tok_s_p05": percentile(speeds, 0.05), "decode_tok_s_p95": percentile(speeds, 0.95),
        "decode_tok_s_spread": (max(speeds) - min(speeds)) / statistics.median(speeds),
        "prefill_s_median": statistics.median(prefills), "prefill_s_p95": percentile(prefills, 0.95),
        "maximum_rss_kib_median": statistics.median(rss), "maximum_rss_kib_max": max(rss),
    }
    path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
