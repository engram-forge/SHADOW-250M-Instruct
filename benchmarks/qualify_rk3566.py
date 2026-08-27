#!/usr/bin/env python3
"""Build and qualify all SHADOW deployment modes on physical RK3566."""

import argparse
import datetime
import json
import os
import pathlib
import platform
import re
import statistics
import subprocess
import sys

DECODE = re.compile(r"decode ([0-9.]+) tok/s")
PREFILL = re.compile(r"prefill ([0-9.]+)s")
RSS = re.compile(r"maximum_rss_kib=(\d+)")
MODES = {
    "exact": {},
    "compact64": {"SHADOW_DOTPROD_FFN": "compact64"},
    "fp16_qkv": {"SHADOW_FP16_QKV": "1"},
    "combined": {"SHADOW_DOTPROD_FFN": "compact64", "SHADOW_FP16_QKV": "1"},
}


def output(command):
    try:
        return subprocess.run(command, check=False, capture_output=True,
                              text=True).stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def read_optional(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return None


def snapshot():
    thermal = {}
    for path in pathlib.Path("/sys/class/thermal").glob("thermal_zone*"):
        thermal[path.name] = {"type": read_optional(path / "type"),
                              "temp_millic": read_optional(path / "temp")}
    cpufreq = {}
    for path in pathlib.Path("/sys/devices/system/cpu/cpufreq").glob("policy*"):
        cpufreq[path.name] = {name: read_optional(path / name) for name in (
            "scaling_cur_freq", "scaling_min_freq", "scaling_max_freq",
            "scaling_governor", "cpuinfo_max_freq")}
    return {"machine": platform.machine(), "platform": platform.platform(),
            "uname": output(["uname", "-a"]), "lscpu": output(["lscpu"]),
            "cpuinfo": read_optional("/proc/cpuinfo"),
            "meminfo": read_optional("/proc/meminfo"),
            "device_tree_model": read_optional("/proc/device-tree/model"),
            "thermal": thermal, "cpufreq": cpufreq}


def max_temperature():
    values = []
    for path in pathlib.Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            values.append(int(path.read_text()))
        except (OSError, ValueError):
            pass
    return max(values, default=None)


def build(root, artifact, dotprod):
    env = {**os.environ, "SHADOW_ARM_CPU": "cortex-a55",
           "SHADOW_ARM_DOTPROD": "ON" if dotprod else "OFF"}
    subprocess.run([str(root / "native/build_linux_arm64.sh"), str(artifact)],
                   cwd=root, env=env, check=True)


def run_once(args, kernel, tokens, generated, threads, mode):
    env = {**os.environ, "SHADOW_THREADS": str(threads),
           "SHADOW_FAST_LOGITS": "0", **MODES[mode]}
    before = max_temperature()
    result = subprocess.run(["/usr/bin/time", "-f", "maximum_rss_kib=%M",
        str(kernel), str(args.model), str(args.table), tokens, str(generated),
        "--bench"], env=env, check=True, capture_output=True, text=True)
    after = max_temperature()
    prefill, decode, rss = (PREFILL.search(result.stderr),
                            DECODE.search(result.stderr), RSS.search(result.stderr))
    if not prefill or not rss or (generated > 1 and not decode):
        raise RuntimeError(f"cannot parse runtime output: {result.stderr}")
    return {"prefill_s": float(prefill.group(1)),
            "decode_tok_s": float(decode.group(1)) if decode else None,
            "rss_kib": int(rss.group(1)), "temp_before_millic": before,
            "temp_after_millic": after}


def summarize(rows, token_count):
    result = {"prefill_tok_s_median": token_count / statistics.median(
        row["prefill_s"] for row in rows),
        "rss_kib_median": statistics.median(row["rss_kib"] for row in rows),
        "runs": rows}
    speeds = [row["decode_tok_s"] for row in rows if row["decode_tok_s"] is not None]
    if speeds:
        result["decode_tok_s_median"] = statistics.median(speeds)
    return result


def matrix(args, kernels, lengths, generated, checkpoint=None):
    cells = []
    for length in lengths:
        tokens = " ".join([str(args.token)] * length)
        for threads in args.threads:
            rows = {mode: [] for mode in MODES}
            for repetition in range(args.warmup + args.runs):
                order = list(MODES)
                if repetition % 2:
                    order.reverse()
                for mode in order:
                    row = run_once(args, kernels[mode], tokens, generated, threads, mode)
                    if repetition >= args.warmup:
                        rows[mode].append(row)
            cells.append({"tokens": length, "threads": threads,
                "modes": {mode: summarize(values, length)
                          for mode, values in rows.items()}})
            if checkpoint:
                checkpoint(cells)
            print(f"tokens={length} threads={threads} generated={generated}", flush=True)
    return cells


def generation_quality(args, kernels, directory):
    quality = {}
    comparator = args.root / "benchmarks/compare_dotprod_generation.py"
    for mode in ("compact64", "fp16_qkv", "combined"):
        destination = directory / f"quality-{mode}.json"
        command = [sys.executable, str(comparator), "--kernel", str(kernels[mode]),
            "--model", str(args.model), "--table", str(args.table),
            "--fixture", str(args.fixture), "--limit", str(args.quality_cases),
            "--generate", str(args.quality_generate), "--threads",
            str(args.quality_threads), "--out", str(destination)]
        for key, value in MODES[mode].items():
            command.extend(["--candidate-env", f"{key}={value}"])
        subprocess.run(command, cwd=args.root, check=True)
        quality[mode] = json.loads(destination.read_text())["summary"]
    return quality


def report_markdown(payload):
    lines = ["# RK3566 board qualification", "",
        f"Generated: {payload['generated_at']}", "",
        f"Device: {payload['hardware'].get('device_tree_model') or 'unknown'}",
        "", "## Decode (tok/s)", "",
        "| Context | Threads | Exact | Compact64 | FP16 QKV | Combined |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cell in payload["decode"]:
        values = [cell["modes"][mode]["decode_tok_s_median"] for mode in MODES]
        lines.append(f"| {cell['tokens']} | {cell['threads']} | " +
                     " | ".join(f"{value:.2f}" for value in values) + " |")
    lines += ["", "## Prefill (tok/s)", "",
        "| Tokens | Threads | Exact | Compact64 | FP16 QKV | Combined |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cell in payload["prefill"]:
        values = [cell["modes"][mode]["prefill_tok_s_median"] for mode in MODES]
        lines.append(f"| {cell['tokens']} | {cell['threads']} | " +
                     " | ".join(f"{value:.2f}" for value in values) + " |")
    lines += ["", "## Generation quality", "",
        "| Mode | Sequence equality | First argmax | Median prefix | Top-10 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for mode, result in payload["quality"].items():
        lines.append(f"| {mode} | {result['identical_rate']:.2%} | "
                     f"{result['first_argmax_rate']:.2%} | "
                     f"{result['median_prefix']:.1f} | "
                     f"{result['median_top10_overlap']:.2f} |")
    lines += ["", "Raw runs, RSS, temperatures, capabilities, and CPU state are "
              "in qualification.json."]
    return chr(10).join(lines) + chr(10)


def resolve_paths(args):
    args.root = args.root.resolve()
    for name in ("model", "table", "fixture", "output_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, args.root / value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=pathlib.Path,
                        default=pathlib.Path("deployment/shadow250m_instruct.shdw"))
    parser.add_argument("--table", type=pathlib.Path,
                        default=pathlib.Path("deployment/fp131072.npy"))
    parser.add_argument("--fixture", type=pathlib.Path,
                        default=pathlib.Path("benchmarks/pirate_runtime_fixture.json"))
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("benchmarks/board-results"))
    parser.add_argument("--contexts", nargs="+", type=int,
                        default=[32, 128, 512, 1024, 2048])
    parser.add_argument("--prefill-lengths", nargs="+", type=int,
                        default=[4, 16, 64, 256])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--token", type=int, default=2)
    parser.add_argument("--decode-generate", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--quality-cases", type=int, default=472)
    parser.add_argument("--quality-generate", type=int, default=33)
    parser.add_argument("--quality-threads", type=int, default=4)
    parser.add_argument("--allow-non-rk3566", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    resolve_paths(args)
    hardware = snapshot()
    if args.dry_run:
        print(json.dumps({"hardware": hardware, "modes": MODES,
            "decode_cells": len(args.contexts) * len(args.threads),
            "prefill_cells": len(args.prefill_lengths) * len(args.threads),
            "quality_cases_per_candidate": args.quality_cases}, indent=2))
        return
    if platform.machine() not in ("aarch64", "arm64"):
        raise SystemExit("qualification requires native ARM64 Linux")
    model_name = (hardware.get("device_tree_model") or "").lower()
    if not args.allow_non_rk3566 and "rk3566" not in model_name and "zero 3" not in model_name:
        raise SystemExit("not an identified RK3566/Radxa ZERO 3 board")
    feature_text = ((hardware.get("cpuinfo") or "") + hardware["lscpu"]).lower()
    if "asimddp" not in feature_text:
        raise SystemExit("CPU does not report asimddp; refusing DotProd qualification")
    if not pathlib.Path("/usr/bin/time").is_file():
        raise SystemExit("/usr/bin/time is required; install the time package")
    for path in (args.model, args.table, args.fixture):
        if not path.is_file():
            raise SystemExit(f"missing required asset: {path}")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = args.output_dir / stamp
    directory.mkdir(parents=True)
    exact = args.root / "build/rk3566-exact/shadow"
    dotprod = args.root / "build/rk3566-dotprod/shadow"
    build(args.root, exact, False)
    build(args.root, dotprod, True)
    kernels = {mode: exact if mode in ("exact", "fp16_qkv") else dotprod
               for mode in MODES}
    capabilities = {name: json.loads(subprocess.check_output(
        [str(path), "--capabilities"], text=True)) for name, path in
        (("exact", exact), ("dotprod", dotprod))}
    if not (capabilities["exact"].get("fp16_arithmetic") and
            capabilities["exact"].get("fp16_fml")):
        raise SystemExit("runtime does not expose required FP16 arithmetic/FML")
    json_path = directory / "qualification.json"
    markdown_path = directory / "qualification.md"
    payload = {"format": "shadow-rk3566-full-qualification-v1",
        "generated_at": stamp, "status": "running", "hardware": hardware,
        "capabilities": capabilities, "decode": [], "prefill": [], "quality": {}}

    def checkpoint(section, cells):
        payload[section] = cells
        payload["hardware_latest"] = snapshot()
        json_path.write_text(json.dumps(payload, indent=2) + chr(10))

    checkpoint("decode", [])
    payload["decode"] = matrix(
        args, kernels, args.contexts, args.decode_generate,
        lambda cells: checkpoint("decode", cells))
    payload["prefill"] = matrix(
        args, kernels, args.prefill_lengths, 1,
        lambda cells: checkpoint("prefill", cells))
    payload["quality"] = generation_quality(args, kernels, directory)
    payload["hardware_after"] = snapshot()
    payload["status"] = "complete"
    json_path.write_text(json.dumps(payload, indent=2) + chr(10))
    markdown_path.write_text(report_markdown(payload))
    print(f"wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
