"""Check a runner against fixtures captured by generate_runtime_golden.py."""
import argparse
import hashlib
import json
import pathlib
import subprocess


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--golden", required=True)
    args = parser.parse_args()
    fixture = json.loads(pathlib.Path(args.golden).read_text())
    if fixture["model_sha256"] != digest(args.model) or fixture["table_sha256"] != digest(args.table):
        raise SystemExit("golden fixture does not match the selected model/table")
    failed = []
    for case in fixture["cases"]:
        result = subprocess.run(
            [args.kernel, args.model, args.table, " ".join(map(str, case["tokens"])), str(case["generate"])],
            capture_output=True, text=True, check=True,
        )
        got = [int(value) for value in result.stdout.split()]
        if got != case["output"]:
            failed.append({"id": case["id"], "expected": case["output"], "got": got})
    if failed:
        raise SystemExit(json.dumps(failed, indent=2))
    print(f"{len(fixture['cases'])} runtime golden cases passed")


if __name__ == "__main__":
    main()
