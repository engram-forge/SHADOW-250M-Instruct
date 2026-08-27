"""Qualify an RK3566 runtime artifact before selecting optimized defaults."""

import argparse
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


def read_optional(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return None


def temperature_millic():
    values = []
    for path in pathlib.Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            values.append(int(path.read_text().strip()))
        except (OSError, ValueError):
            pass
    return max(values, default=None)


def run_once(args, tokens, threads, extra_env):
    env = {**os.environ, "SHADOW_THREADS": str(threads), **extra_env}
    command = [args.kernel, args.model, args.table, tokens, str(args.generate),
               "--bench"]
    before = temperature_millic()
    result = subprocess.run(
        ["/usr/bin/time", "-f", "maximum_rss_kib=%M", *command],
        check=True, capture_output=True, text=True, env=env)
    after = temperature_millic()
    speed = SPEED.search(result.stderr); prefill = PREFILL.search(result.stderr)
    rss = RSS.search(result.stderr)
    if not speed or not prefill or not rss:
        raise RuntimeError(f"cannot parse runtime output: {result.stderr}")
    return {"decode_tok_s": float(speed.group(1)),
            "prefill_s": float(prefill.group(1)),
            "maximum_rss_kib": int(rss.group(1)),
            "temperature_before_millic": before,
            "temperature_after_millic": after}


def summarize(rows, prompt_length):
    speeds = [row["decode_tok_s"] for row in rows]
    prefills = [row["prefill_s"] for row in rows]
    return {"decode_tok_s_median": statistics.median(speeds),
            "decode_tok_s_min": min(speeds), "decode_tok_s_max": max(speeds),
            "prefill_tok_s_median": prompt_length / statistics.median(prefills),
            "maximum_rss_kib_max": max(row["maximum_rss_kib"] for row in rows),
            "temperature_millic_max": max(
                (value for row in rows for value in
                 (row["temperature_before_millic"], row["temperature_after_millic"])
                 if value is not None), default=None),
            "runs": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=[32, 128, 512, 1024, 2048])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--generate", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=2); parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--candidate-env", action="append", default=[],
                        help="candidate environment KEY=VALUE; may be repeated")
    args = parser.parse_args()
    capabilities = json.loads(subprocess.check_output(
        [args.kernel, "--capabilities"], text=True))
    candidate_env = dict(item.split("=", 1) for item in args.candidate_env)
    if candidate_env and not (capabilities.get("fp16_arithmetic") and
                              capabilities.get("fp16_fml")):
        raise SystemExit("candidate requested but FP16 arithmetic/FML is unavailable")
    suites = []
    for context in args.contexts:
        tokens = " ".join(["2"] * context)
        for threads in args.threads:
            for _ in range(args.warmup):
                run_once(args, tokens, threads, {})
                if candidate_env:
                    run_once(args, tokens, threads, candidate_env)
            rows = {"control": [], "candidate": []}
            for index in range(args.runs):
                order = [("control", {}), ("candidate", candidate_env)]
                if index % 2: order.reverse()
                for name, env in order:
                    if name == "candidate" and not candidate_env: continue
                    rows[name].append(run_once(args, tokens, threads, env))
            item = {"context": context, "threads": threads,
                    "control": summarize(rows["control"], context)}
            if rows["candidate"]:
                item["candidate"] = summarize(rows["candidate"], context)
                item["decode_gain"] = (item["candidate"]["decode_tok_s_median"] /
                                           item["control"]["decode_tok_s_median"] - 1)
            suites.append(item)
            print(f"context={context} threads={threads}", flush=True)
    payload = {"format": "shadow-rk3566-board-qualification-v1",
               "machine": platform.machine(), "platform": platform.platform(),
               "capabilities": capabilities, "candidate_env": candidate_env,
               "cpuinfo": read_optional("/proc/cpuinfo"),
               "thermal_throttling": read_optional("/sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state"),
               "suites": suites, "timestamp": time.time()}
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
