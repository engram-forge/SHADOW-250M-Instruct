"""Build deterministic runtime fixtures from the 472-example pirate SFT set.

The dataset is verification input only: user prompts are wrapped with the
published chat template, assistant answers are retained as metadata, and no
training or quality claim is made by this harness.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shadow_runtime import BOS, EOT, SOT
from shadow_runtime.retriever import enc


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_tokens(message):
    return [BOS, SOT] + enc("user\n") + enc(message) + [EOT] + enc("\n") + [SOT] + enc("model\n")


def load_cases(path):
    cases = []
    for index, line in enumerate(pathlib.Path(path).read_text().splitlines()):
        record = json.loads(line)
        messages = record.get("messages", [])
        if len(messages) != 2 or [item.get("role") for item in messages] != ["user", "assistant"]:
            raise ValueError(f"row {index + 1} is not a user/assistant pair")
        user = messages[0]["content"]
        cases.append({
            "id": f"pirate-{index + 1:03d}",
            "source_row": index + 1,
            "prompt_sha256": hashlib.sha256(user.encode()).hexdigest(),
            "assistant_sha256": hashlib.sha256(messages[1]["content"].encode()).hexdigest(),
            "tokens": prompt_tokens(user),
        })
    return cases


def build_fixture(data, model, table, output, limit=None):
    cases = load_cases(data)
    if limit is not None:
        cases = cases[:limit]
    payload = {
        "format": "shadow-pirate-runtime-verification-v1",
        "source_sha256": file_sha256(data),
        "model_sha256": file_sha256(model),
        "table_sha256": file_sha256(table),
        "case_count": len(cases),
        "cases": cases,
    }
    target = pathlib.Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(target), "case_count": len(cases),
                      "max_prompt_tokens": max(map(lambda case: len(case["tokens"]), cases))}, indent=2))


def summarize(reference_path, candidate_path, top_k):
    reference = np.load(reference_path, mmap_mode="r")
    candidate = np.load(candidate_path, mmap_mode="r")
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError(f"logit dump shape mismatch: {reference.shape} != {candidate.shape}")
    maxima = []; means = []; squared = []; agreements = []; overlaps = []
    k = min(top_k, reference.shape[1])
    for left, right in zip(reference, candidate):
        difference = right.astype(np.float64) - left.astype(np.float64)
        absolute = np.abs(difference); maxima.append(float(absolute.max()))
        means.append(float(absolute.mean())); squared.append(float(np.square(difference).mean()))
        agreements.append(int(left.argmax()) == int(right.argmax()))
        lt = np.argpartition(-left, k - 1)[:k]; rt = np.argpartition(-right, k - 1)[:k]
        overlaps.append(len(set(lt.tolist()) & set(rt.tolist())) / k)
    return {
        "cases": reference.shape[0], "vocab": reference.shape[1],
        "max_abs": max(maxima, default=0.0), "mean_abs": float(np.mean(means)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "argmax_agreement": float(np.mean(agreements)),
        f"top_{k}_mean_overlap": float(np.mean(overlaps)),
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--data", required=True); build.add_argument("--model", required=True)
    build.add_argument("--table", required=True); build.add_argument("--out", required=True)
    build.add_argument("--limit", type=int)
    report = sub.add_parser("summarize")
    report.add_argument("reference"); report.add_argument("candidate"); report.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.command == "build": build_fixture(args.data, args.model, args.table, args.out, args.limit)
    else: print(json.dumps(summarize(args.reference, args.candidate, args.top_k), indent=2))


if __name__ == "__main__":
    main()
