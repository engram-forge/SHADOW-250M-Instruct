"""Benchmark the PyTorch and TileLang CUDA inference backends.

Model loading, lazy TileLang compilation, CUDA graph capture, and decode prompt
prefill are deliberately outside the measured intervals. CUDA synchronization
and output transfers performed by the normal runtime paths are included.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shadow_tilelang.engine import TileLangEngine


DEFAULT_SIZES = (32, 128, 1024, 2048)
PROMPT_PATTERN = (2, 8, 925, 1234)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=ROOT / "deployment/shadow250m_instruct.shdw"
    )
    parser.add_argument(
        "--table", default=ROOT / "deployment/fp131072.npy"
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
        help="token counts to benchmark",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--max-context", type=int, default=2048,
        help="circular cache capacity",
    )
    parser.add_argument("--out", type=Path, help="optional JSON result path")
    return parser.parse_args()


def synchronize() -> None:
    torch.cuda.synchronize()


def prompt(token_count: int) -> list[int]:
    repeats = (token_count + len(PROMPT_PATTERN) - 1) // len(PROMPT_PATTERN)
    return list((PROMPT_PATTERN * repeats)[:token_count])


def dispose(engine: TileLangEngine) -> None:
    engine.close()
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def summary(seconds: list[float], token_count: int) -> dict:
    median = statistics.median(seconds)
    return {
        "runs_seconds": seconds,
        "median_seconds": median,
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "median_tokens_per_second": token_count / median,
    }


def benchmark_prefill(
    backend: str, token_count: int, repetitions: int, paths: tuple[Path, Path],
    max_context: int,
) -> tuple[dict, int]:
    tokens = prompt(token_count)
    engine = TileLangEngine(*paths, backend=backend, max_context=max_context)
    try:
        # Compile shape specializations and populate allocator/library caches.
        engine.prefill(tokens)
        synchronize()
        seconds = []
        argmax = -1
        for _ in range(repetitions):
            engine.reset()
            synchronize()
            started = time.perf_counter()
            logits = engine.prefill(tokens)
            synchronize()
            seconds.append(time.perf_counter() - started)
            argmax = int(logits.argmax())
        return summary(seconds, token_count), argmax
    finally:
        dispose(engine)


def torch_decode(
    engine: TileLangEngine, logits: torch.Tensor, token_count: int
) -> list[int]:
    generated = []
    for _ in range(token_count):
        token = int(logits.argmax())
        generated.append(token)
        logits = engine.step(token)
    return generated


def tilelang_graph_keys(engine: TileLangEngine, token_count: int) -> list:
    return list(dict.fromkeys(
        engine._decode_graph_key(engine.position + offset)
        for offset in range(token_count)
    ))


def benchmark_decode(
    backend: str, token_count: int, repetitions: int, paths: tuple[Path, Path],
    max_context: int,
) -> tuple[dict, list[int]]:
    seconds = []
    first_output = []
    decode_prompt = list(PROMPT_PATTERN)
    for repetition in range(repetitions):
        engine = TileLangEngine(*paths, backend=backend, max_context=max_context)
        try:
            logits = engine.prefill(decode_prompt)
            if backend == "tilelang":
                for graph_key in tilelang_graph_keys(engine, token_count):
                    engine._ensure_greedy_graph(graph_key)
            synchronize()
            started = time.perf_counter()
            if backend == "tilelang":
                output = engine._generate_greedy_cuda(logits, token_count)
            else:
                output = torch_decode(engine, logits, token_count)
            synchronize()
            seconds.append(time.perf_counter() - started)
            if repetition == 0:
                first_output = output
        finally:
            dispose(engine)
    return summary(seconds, token_count), first_output


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if any(size < 1 or size > args.max_context for size in args.sizes):
        raise SystemExit("sizes must be inside [1, max-context]")
    paths = (Path(args.model).resolve(), Path(args.table).resolve())
    for path in paths:
        if not path.exists():
            raise SystemExit(f"not found: {path}")

    import tilelang

    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "tilelang": tilelang.__version__,
            "commit": git_value("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git_value("status", "--porcelain")),
        },
        "configuration": {
            "sizes": args.sizes,
            "repetitions": args.repetitions,
            "max_context": args.max_context,
            "prefill_prompt_pattern": list(PROMPT_PATTERN),
            "decode_prompt": list(PROMPT_PATTERN),
        },
        "prefill": {},
        "decode": {},
    }
    print(json.dumps(report["environment"], sort_keys=True), flush=True)

    for size in args.sizes:
        row = {}
        argmax = {}
        for backend in ("torch", "tilelang"):
            result, token = benchmark_prefill(
                backend, size, args.repetitions, paths, args.max_context
            )
            row[backend] = result
            argmax[backend] = token
            print(
                f"prefill size={size} backend={backend} "
                f"median={result['median_tokens_per_second']:.3f} tok/s",
                flush=True,
            )
        row["tilelang_speedup"] = (
            row["torch"]["median_seconds"]
            / row["tilelang"]["median_seconds"]
        )
        row["argmax"] = argmax
        row["argmax_match"] = argmax["torch"] == argmax["tilelang"]
        report["prefill"][str(size)] = row

    for size in args.sizes:
        row = {}
        outputs = {}
        for backend in ("torch", "tilelang"):
            result, output = benchmark_decode(
                backend, size, args.repetitions, paths, args.max_context
            )
            row[backend] = result
            outputs[backend] = output
            print(
                f"decode size={size} backend={backend} "
                f"median={result['median_tokens_per_second']:.3f} tok/s",
                flush=True,
            )
        matched_prefix = next(
            (index for index, pair in enumerate(zip(outputs["torch"], outputs["tilelang"]))
             if pair[0] != pair[1]),
            size,
        )
        row["tilelang_speedup"] = (
            row["torch"]["median_seconds"]
            / row["tilelang"]["median_seconds"]
        )
        row["matched_prefix_tokens"] = matched_prefix
        row["tokens_match"] = matched_prefix == size
        if matched_prefix < size:
            row["first_mismatch"] = {
                "index": matched_prefix,
                "torch": outputs["torch"][matched_prefix],
                "tilelang": outputs["tilelang"][matched_prefix],
            }
        report["decode"][str(size)] = row

    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
