"""Compare repetition behavior of one or more deployed SHADOW models.

Example:
    python benchmarks/anti_repetition_eval.py \
        --model base=deployment/shadow250m_instruct.shdw \
        --model tuned=finetune/my_model.shdw --out results/repetition.json
"""
import argparse
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shadow_runtime import Engine, guarded_result

PROMPTS = [
    "Write a short poem about AI.",
    "Explain photosynthesis in two sentences.",
    "Give five distinct tips for staying focused while studying.",
    "Summarize why regular exercise is useful without repeating yourself.",
    "Write a Python function that removes duplicates from a list while preserving order.",
    "What is the capital of France? Answer briefly.",
    "Compare solar and wind energy in one concise paragraph.",
    "Write a 200-word story about a robot learning to paint.",
]

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--table", default=str(ROOT / "deployment" / "fp131072.npy"))
    parser.add_argument("--prompts", help="optional JSON array of prompt strings")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=("greedy", "sampled", "guarded"), default="guarded")
    return parser.parse_args()

def distinct(ids, n):
    grams = [tuple(ids[i:i + n]) for i in range(max(0, len(ids) - n + 1))]
    return 1.0 if not grams else len(set(grams)) / len(grams)

def normalize_prompt(item, index):
    if isinstance(item, str):
        return {"id": f"prompt_{index + 1:03d}", "category": "uncategorized",
                "prompt": item, "allow_repetition": False}
    required = {"id", "category", "prompt"}
    missing = required - item.keys()
    if missing: raise SystemExit(f"prompt {index + 1} missing fields: {sorted(missing)}")
    return {**item, "allow_repetition": bool(item.get("allow_repetition", False))}

def aggregate(rows):
    return {
        "count": len(rows),
        "loop_rate": sum(row["looped"] for row in rows) / len(rows),
        "mean_repeat_4gram_ratio": statistics.mean(row["repeat_4gram_ratio"] for row in rows),
        "mean_distinct_2": statistics.mean(row["distinct_2"] for row in rows),
        "mean_distinct_3": statistics.mean(row["distinct_3"] for row in rows),
        "mean_tokens": statistics.mean(row["tokens"] for row in rows),
        "retry_rate": sum(row["retried"] for row in rows) / len(rows),
    }

def main():
    args = arguments()
    raw_prompts = json.loads(pathlib.Path(args.prompts).read_text(encoding="utf-8")) if args.prompts else PROMPTS
    prompts = [normalize_prompt(item, index) for index, item in enumerate(raw_prompts)]
    results = {"profile": args.profile, "prompts": prompts, "models": {}}
    for spec in args.model:
        if "=" not in spec: raise SystemExit("--model must be NAME=PATH")
        name, path = spec.split("=", 1)
        engine = Engine(path, args.table)
        rows = []
        for index, prompt_record in enumerate(prompts):
            prompt = prompt_record["prompt"]
            kwargs = {"n": args.tokens, "seed": args.seed + index}
            if args.profile == "greedy": kwargs.update(greedy=True, guard=False)
            elif args.profile == "sampled": kwargs.update(greedy=False, guard=False)
            else: kwargs.update(greedy=False, guard=True)
            metadata = engine.chat(prompt, return_metadata=True, **kwargs)
            measured = guarded_result(metadata["text"])
            rows.append({**prompt_record, **metadata, "tokens": len(measured["ids"]),
                         "distinct_2": distinct(measured["ids"], 2),
                         "distinct_3": distinct(measured["ids"], 3)})
        categories = sorted(set(row["category"] for row in rows))
        ordinary = [row for row in rows if not row["allow_repetition"]]
        controls = [row for row in rows if row["allow_repetition"]]
        results["models"][name] = {**aggregate(rows),
            "ordinary_loop_rate": sum(row["looped"] for row in ordinary) / len(ordinary) if ordinary else 0.0,
            "legitimate_repeat_flag_rate": sum(row["looped"] for row in controls) / len(controls) if controls else 0.0,
            "by_category": {category: aggregate([row for row in rows if row["category"] == category])
                            for category in categories}, "samples": rows}
    output = pathlib.Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in result.items() if key != "samples"}
                      for name, result in results["models"].items()}, indent=2))

if __name__ == "__main__":
    main()
