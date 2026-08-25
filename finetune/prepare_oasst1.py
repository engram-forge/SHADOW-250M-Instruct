"""Prepare neutral English SFT data from the pinned Apache-2.0 OASST1 export."""
import argparse
import gzip
import hashlib
import json
import pathlib
import random
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "shadow_runtime"))
from finetune import audit_messages, build_ids

SOURCE_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
PIRATE = re.compile(r"\b(arr|aye(?: aye)?|blimey|shiver me timbers|yarr|matey|savvy)\b", re.I)

def label_value(message, name, default=0.0):
    value = message.get("labels", {}).get(name, default)
    return float(value.get("value", default) if isinstance(value, dict) else value)

def acceptable(message, min_quality):
    if message.get("lang") != "en" or message.get("deleted") or message.get("synthetic"):
        return False
    if not message.get("review_result", False): return False
    if message.get("role") == "assistant" and message.get("rank") not in (0, None): return False
    if label_value(message, "quality", 1.0) < min_quality: return False
    for name in ("spam", "lang_mismatch", "not_appropriate", "hate_speech", "sexual_content", "fails_task"):
        if label_value(message, name) > 0.25: return False
    detox = message.get("detoxify") or {}
    if float(detox.get("toxicity", 0.0)) > 0.2 or float(detox.get("sexual_explicit", 0.0)) > 0.2: return False
    text = (message.get("text") or "").strip()
    return 2 <= len(text) <= 12000

def chain(message, by_id):
    path = []
    while message:
        path.append(message); message = by_id.get(message.get("parent_id"))
    path.reverse()
    if not path or path[0].get("role") != "prompter" or path[-1].get("role") != "assistant": return None
    expected = "prompter"
    for item in path:
        if item.get("role") != expected: return None
        expected = "assistant" if expected == "prompter" else "prompter"
    return {"messages": [{"role": "user" if item["role"] == "prompter" else "assistant", "content": item["text"].strip()} for item in path],
            "metadata": {"source": "OpenAssistant/oasst1", "tree_id": path[0]["message_tree_id"]}}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-train", type=int, default=5000); parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--val-percent", type=int, default=10); parser.add_argument("--min-quality", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0); args = parser.parse_args()
    with gzip.open(args.input, "rt", encoding="utf-8") as stream:
        messages = [item for item in map(json.loads, stream) if acceptable(item, args.min_quality)]
    by_id = {item["message_id"]: item for item in messages}
    children = {};
    for item in messages: children.setdefault(item.get("parent_id"), []).append(item["message_id"])
    candidates = []; seen_answers = set()
    for item in messages:
        if item.get("role") != "assistant": continue
        # Keep a path at each accepted assistant turn, not just full-tree leaves.
        example = chain(item, by_id)
        if not example: continue
        if len(build_ids(example["messages"])[0]) > 2048: continue
        audit = audit_messages(example["messages"])
        if audit["repeated_span"] or audit["repeat_4gram_ratio"] > 0.5: continue
        if PIRATE.search(" \n".join(message["content"] for message in example["messages"])): continue
        fingerprint = hashlib.sha256(example["messages"][-1]["content"].strip().lower().encode()).hexdigest()
        if fingerprint in seen_answers: continue
        seen_answers.add(fingerprint); candidates.append(example)
    rng = random.Random(args.seed); rng.shuffle(candidates); train, val = [], []
    for example in candidates:
        tree_id = example["metadata"]["tree_id"]
        bucket = int(hashlib.sha256(tree_id.encode()).hexdigest()[:8], 16) % 100
        target, limit = (val, args.max_val) if bucket < args.val_percent else (train, args.max_train)
        if len(target) < limit: target.append(example)
    output = pathlib.Path(args.out_dir); output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", val)):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {"source": "OpenAssistant/oasst1", "revision": SOURCE_REVISION, "license": "Apache-2.0",
                "input_sha256": hashlib.sha256(pathlib.Path(args.input).read_bytes()).hexdigest(),
                "split": f"tree_id sha256 bucket < {args.val_percent} is validation", "min_quality": args.min_quality,
                "accepted_messages": len(messages), "candidate_paths": len(candidates), "train": len(train), "validation": len(val), "seed": args.seed}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
