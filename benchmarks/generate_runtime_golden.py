"""Capture deterministic token fixtures from an existing deployed kernel.

Run this on Linux or Windows with the shipped x86 binary. The resulting small
JSON file can be committed and consumed by every new platform runtime.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

CASES = [
    {"id": "bos", "tokens": [2], "generate": 8},
    {"id": "empty_user_turn", "tokens": [2, 8, 127026, 9, 8], "generate": 8},
    {"id": "short_ids", "tokens": [2, 8, 42, 314, 9, 8, 127020], "generate": 8},
]


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for case in CASES:
        command = [args.kernel, args.model, args.table, " ".join(map(str, case["tokens"])), str(case["generate"])]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        rows.append({**case, "output": [int(value) for value in result.stdout.split()]})
    payload = {
        "format": "shadow-runtime-golden-v1",
        "model_sha256": digest(args.model), "table_sha256": digest(args.table),
        "cases": rows,
    }
    path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
