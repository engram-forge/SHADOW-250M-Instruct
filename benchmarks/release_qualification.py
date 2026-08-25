"""Release qualification for deployed SHADOW models.

Runs deterministic instruction/format checks and writes a promotion report plus a blinded
human-review packet. This evaluates deployment files, not trainer checkpoints.
"""
import argparse
import ast
import collections
import json
import pathlib
import random
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shadow_runtime import Engine, guarded_result

PIRATE = re.compile(r"\b(arr|aye(?: aye)?|blimey|shiver me timbers|yarr|matey|savvy)\b", re.I)

def case(case_id, category, prompt, validator, **expected):
    return {"id": case_id, "category": category, "prompt": prompt,
            "validator": validator, "expected": expected}

def cases():
    rows = []
    for i in range(1, 16):
        keys = [f"field_{i}", "status", "items"]
        rows.append(case(f"json_{i:02d}", "json",
            f"Return only valid JSON with exactly these keys: {', '.join(keys)}. Set status to ok and items to an array of {i % 4 + 1} integers.",
            "json", keys=keys, array_key="items", array_length=i % 4 + 1, value_key="status", value="ok"))
    for i in range(1, 16):
        count = i % 5 + 3
        rows.append(case(f"list_{i:02d}", "numbered_list",
            f"Give exactly {count} distinct numbered steps for checking a small application's health. Use 1 through {count}.",
            "numbered_list", count=count))
    python_tasks = [
        ("add(a, b)", "return a + b"), ("is_even(n)", "return n % 2 == 0"),
        ("square(n)", "return n * n"), ("first(items)", "return items[0] if items else None"),
        ("last(items)", "return items[-1] if items else None"),
    ]
    for i in range(15):
        signature, hint = python_tasks[i % len(python_tasks)]
        name = signature.split("(")[0]
        rows.append(case(f"python_{i+1:02d}", "python",
            f"Write only a Python code block defining `{signature}`. Its core behavior is `{hint}`.",
            "python", function=name))
    for i in range(1, 16):
        rows.append(case(f"table_{i:02d}", "markdown_table",
            f"Return a Markdown table with exactly 3 columns and {i % 3 + 2} data rows comparing option A and option B. Include a header.",
            "markdown_table", columns=3, rows=i % 3 + 2))
    for i in range(1, 21):
        sentence_count = i % 3 + 2
        required = f"marker{i}"
        rows.append(case(f"constraint_{i:02d}", "constraints",
            f"Explain one benefit of careful planning in exactly {sentence_count} sentences. Include the exact token {required} once. Do not use the word pirate.",
            "constraints", sentences=sentence_count, required=required, forbidden="pirate"))
    neutral = [
        "Explain why the sky appears blue in two sentences.", "Write a polite appointment reminder.",
        "Summarize the purpose of unit testing.", "Give three tips for storing vegetables.",
        "Describe a quiet morning in a city park.", "Explain what a budget is.",
        "Write a brief professional thank-you note.", "Compare walking and cycling for commuting.",
        "Explain why sleep is important.", "Give a concise definition of an algorithm.",
    ]
    for i in range(20):
        rows.append(case(f"neutral_{i+1:02d}", "neutral_style", neutral[i % len(neutral)], "neutral"))
    assert len(rows) == 100
    return rows

def strip_fence(text):
    match = re.search(r"```(?:python|json|text)?\s*(.*?)```", text, re.S | re.I)
    return match.group(1).strip() if match else text.strip()

def validate(item, text):
    expected = item["expected"]; kind = item["validator"]; details = []
    try:
        if kind == "json":
            value = json.loads(strip_fence(text)); ok = isinstance(value, dict)
            ok &= set(value) == set(expected["keys"])
            ok &= isinstance(value.get(expected["array_key"]), list) and len(value.get(expected["array_key"], [])) == expected["array_length"]
            ok &= value.get(expected["value_key"]) == expected["value"]
        elif kind == "numbered_list":
            numbers = [int(x) for x in re.findall(r"(?m)^\s*(\d+)[.)]\s+", text)]
            ok = numbers == list(range(1, expected["count"] + 1)); details.append(f"numbers={numbers}")
        elif kind == "python":
            tree = ast.parse(strip_fence(text)); names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            ok = expected["function"] in names; details.append(f"functions={names}")
        elif kind == "markdown_table":
            lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|") and line.strip().endswith("|")]
            widths = [len(line.strip("|").split("|")) for line in lines]
            ok = len(lines) == expected["rows"] + 2 and widths and all(width == expected["columns"] for width in widths)
            details.append(f"lines={len(lines)}, widths={widths}")
        elif kind == "constraints":
            sentences = [x for x in re.split(r"(?<=[.!?])(?:[\"']?\s+|$)", text.strip()) if x.strip()]
            ok = len(sentences) == expected["sentences"]
            ok &= text.count(expected["required"]) == 1 and expected["forbidden"].lower() not in text.lower()
            details.append(f"sentences={len(sentences)}")
        else:
            ok = bool(text.strip()) and not PIRATE.search(text)
    except (SyntaxError, ValueError, TypeError, json.JSONDecodeError) as error:
        ok = False; details.append(str(error))
    return bool(ok), "; ".join(details)

def aggregate(rows):
    return {"count": len(rows), "compliance": sum(r["valid"] for r in rows) / len(rows),
            "loop_rate": sum(r["looped"] for r in rows) / len(rows),
            "pirate_leakage": sum(r["pirate"] for r in rows) / len(rows),
            "mean_tokens": statistics.mean(r["tokens"] for r in rows)}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--table", default=str(ROOT / "deployment" / "fp131072.npy")); parser.add_argument("--out", required=True)
    parser.add_argument("--review-out", required=True); parser.add_argument("--tokens", type=int, default=192); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(); suite = cases(); report = {"suite_size": len(suite), "models": {}}; all_rows = {}
    for spec in args.model:
        name, path = spec.split("=", 1); engine = Engine(path, args.table); rows = []
        for index, item in enumerate(suite):
            metadata = engine.chat(item["prompt"], n=args.tokens, seed=args.seed + index, return_metadata=True)
            measured = guarded_result(metadata["text"]); valid, detail = validate(item, metadata["text"])
            rows.append({**item, **metadata, "valid": valid, "validation_detail": detail,
                         "pirate": bool(PIRATE.search(metadata["text"])), "tokens": len(measured["ids"])})
        categories = sorted(set(r["category"] for r in rows)); overall = aggregate(rows)
        report["models"][name] = {**overall, "by_category": {c: aggregate([r for r in rows if r["category"] == c]) for c in categories}, "samples": rows}
        all_rows[name] = rows
    names = list(all_rows); base = report["models"][names[0]]; review_rng = random.Random(17); review = []
    review_ids = [item["id"] for item in suite if item["category"] in ("neutral_style", "constraints")][:25]
    for case_id in review_ids:
        candidates = [{"model_key": name, "text": next(r["text"] for r in rows if r["id"] == case_id)} for name, rows in all_rows.items()]
        review_rng.shuffle(candidates); prompt = next(item["prompt"] for item in suite if item["id"] == case_id)
        review.append({"id": case_id, "prompt": prompt, "candidates": [{"label": chr(65+i), "text": c["text"]} for i,c in enumerate(candidates)],
                       "answer_key": {chr(65+i): c["model_key"] for i,c in enumerate(candidates)},
                       "scores": {"preferred": None, "complete": {}, "relevant": {}, "correct": {}}})
    candidate_reports = {}
    for name in names[1:]:
        candidate = report["models"][name]
        gates = {"loop_rate_zero": candidate["loop_rate"] == 0, "pirate_leakage_below_1pct": candidate["pirate_leakage"] < 0.01,
                 "compliance_not_over_5pt_below_base": candidate["compliance"] >= base["compliance"] - 0.05,
                 "length_between_70_and_120pct": 0.7 * base["mean_tokens"] <= candidate["mean_tokens"] <= 1.2 * base["mean_tokens"]}
        candidate_reports[name] = {"gates": gates, "automatic_pass": all(gates.values())}
    report["promotion"] = candidate_reports
    output = pathlib.Path(args.out); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2) + "\n")
    review_path = pathlib.Path(args.review_out); review_path.parent.mkdir(parents=True, exist_ok=True); review_path.write_text("".join(json.dumps(x) + "\n" for x in review))
    print(json.dumps({"models": {n: {k:v for k,v in r.items() if k not in ("samples","by_category")} for n,r in report["models"].items()}, "promotion": candidate_reports}, indent=2))

if __name__ == "__main__": main()
