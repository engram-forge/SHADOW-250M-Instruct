"""Compare a runtime's greedy tokens with the Modal Linux pirate golden."""
import argparse
import json
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True); parser.add_argument("--golden", required=True)
    parser.add_argument("--kernel", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True); parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    fixture = json.load(open(args.fixture)); golden = json.load(open(args.golden))
    if fixture["model_sha256"] != golden["model_sha256"] or fixture["table_sha256"] != golden["table_sha256"]:
        raise SystemExit("fixture/golden asset hashes differ")
    expected = {item["id"]: item["output"] for item in golden["outputs"]}
    cases = fixture["cases"][:args.limit] if args.limit else fixture["cases"]
    matched = 0; first_failure = None; total_prefix = 0
    for case in cases:
        result = subprocess.run(
            [args.kernel, args.model, args.table, " ".join(map(str, case["tokens"])), str(golden["generate_tokens"])],
            capture_output=True, text=True, check=True, env={**os.environ, "SHADOW_THREADS": str(args.threads)},
        )
        got = [int(value) for value in result.stdout.split()]; want = expected[case["id"]]
        prefix = 0
        for left, right in zip(got, want):
            if left != right: break
            prefix += 1
        total_prefix += prefix
        if got == want: matched += 1
        elif first_failure is None: first_failure = {"id": case["id"], "expected": want, "got": got, "matching_prefix": prefix}
    print(json.dumps({"cases": len(cases), "exact_matches": matched,
                      "exact_match_rate": matched / len(cases) if cases else 0.0,
                      "mean_matching_prefix": total_prefix / len(cases) if cases else 0.0,
                      "first_failure": first_failure}, indent=2))
    raise SystemExit(0 if matched == len(cases) else 1)


if __name__ == "__main__":
    main()
