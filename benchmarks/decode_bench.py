"""Measure native SHADOW decode throughput on assistant turns in a JSONL dataset.

The kernel's ``--bench`` report times generation internally, so model loading and Python
process startup are excluded from the reported tokens/second.
"""
import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shadow_runtime"))
from retriever import enc

BOS, SOT, EOT = 2, 8, 9
GENERATION = re.compile(
    r"GENERATION:\s+(\d+) tokens in ([0-9.]+)s = ([0-9.]+) ms/token = ([0-9.]+) tok/s \((\d+) threads\)"
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "finetune" / "examples_pirate.jsonl"))
    parser.add_argument("--kernel", default=str(ROOT / "deployment" / "bin" / "windows" / "shadow.exe"))
    parser.add_argument("--model", default=str(ROOT / "deployment" / "shadow250m_instruct.shdw"))
    parser.add_argument("--table", default=str(ROOT / "deployment" / "fp131072.npy"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=50, help="assistant turns to measure; 0 means all")
    parser.add_argument("--max-tokens", type=int, default=256, help="cap paired assistant length")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", help="optional JSON result path")
    return parser.parse_args()


def prompt_ids(messages):
    ids = [BOS]
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        ids += [SOT] + enc(role + "\n") + enc(message["content"]) + [EOT] + enc("\n")
    return ids + [SOT] + enc("model\n")


def examples(path, max_tokens):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            for index, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue
                budget = min(len(enc(message["content"])) + 1, max_tokens)
                if budget:
                    yield line_number, index, prompt_ids(messages[:index]), budget


def main():
    args = arguments()
    paths = [pathlib.Path(value).resolve() for value in (args.kernel, args.model, args.table, args.data)]
    kernel, model, table, data = paths
    for path in paths:
        if not path.exists():
            raise SystemExit(f"not found: {path}")
    if sys.platform.startswith("linux") and kernel.suffix.lower() == ".exe" and not str(kernel).startswith("/mnt/"):
        raise SystemExit(
            "WSL can run Windows executables reliably only from a mounted Windows drive. "
            "Clone or copy the repository under /mnt/c (for example /mnt/c/src/SHADOW-250M-Instruct), "
            "then run this command there."
        )

    selected = list(examples(data, args.max_tokens))
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise SystemExit("the dataset contains no assistant turns")
    env = dict(os.environ, SHADOW_THREADS=str(args.threads))
    runtime_paths = [str(model), str(table)]
    if sys.platform.startswith("linux") and kernel.suffix.lower() == ".exe":
        runtime_paths = [subprocess.check_output(["wslpath", "-w", value], text=True).strip()
                         for value in runtime_paths]
    rows = []
    work = selected[:1] * args.warmup + selected
    for run_index, (line_number, turn_index, ids, budget) in enumerate(work):
        command = [str(kernel), *runtime_paths, " ".join(map(str, ids)), str(budget), "--bench"]
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        output = result.stdout + "\n" + result.stderr
        match = GENERATION.search(output)
        if result.returncode or not match:
            raise SystemExit(f"kernel failed for JSONL line {line_number}:\n{output.strip()}")
        if run_index < args.warmup:
            continue
        tokens, seconds, ms_token, tokens_second, threads = match.groups()
        row = {"line": line_number, "turn": turn_index, "prompt_tokens": len(ids),
               "generated_tokens": int(tokens), "seconds": float(seconds),
               "ms_per_token": float(ms_token), "tokens_per_second": float(tokens_second),
               "threads": int(threads)}
        rows.append(row)
        print(f"[{len(rows):3d}/{len(selected)}] line {line_number}: {row['tokens_per_second']:.1f} tok/s", flush=True)

    total_tokens = sum(row["generated_tokens"] for row in rows)
    total_seconds = sum(row["seconds"] for row in rows)
    summary = {"samples": len(rows), "generated_tokens": total_tokens,
               "weighted_tokens_per_second": total_tokens / total_seconds,
               "median_tokens_per_second": statistics.median(row["tokens_per_second"] for row in rows),
               "threads": args.threads}
    report = {"summary": summary, "runs": rows}
    print(json.dumps(summary, indent=2))
    if args.out:
        output = pathlib.Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
