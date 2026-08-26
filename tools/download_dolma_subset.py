"""Build a bounded, resumable Dolma subset counted with SHADOW model tokens."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shadow_runtime"))
from retriever import enc  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=10_000_000_000)
    parser.add_argument("--output-shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--seed", default="shadow-dolma-v1.7-10b-v1")
    return parser.parse_args()


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main():
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    work.mkdir(exist_ok=True)
    state_path = args.out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "tokens": 0, "documents": 0, "next_url": 0, "output_shard": 0
    }
    urls = [line.strip() for line in args.urls.read_text().splitlines() if line.strip()]
    urls.sort(key=lambda url: hashlib.sha256(f"{args.seed}\0{url}".encode()).digest())
    manifest = args.out / "source_urls.txt"
    manifest.write_text("\n".join(urls) + "\n")

    output = None
    output_tokens = 0
    try:
        for url_index in range(state["next_url"], len(urls)):
            url = urls[url_index]
            source = work / Path(url).name
            print(f"download {url_index + 1}/{len(urls)}: {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "shadow-pretraining-data/1.0"})
            with urllib.request.urlopen(request) as response, source.open("wb") as destination:
                shutil.copyfileobj(response, destination, length=1024 * 1024)
            with gzip.open(source, "rt", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                    except (json.JSONDecodeError, AttributeError):
                        continue
                    count = len(enc(text)) + 1
                    if state["tokens"] + count > args.target_tokens:
                        state["next_url"] = url_index
                        save_state(state_path, state)
                        print(f"target reached: {state['tokens']:,} tokens", flush=True)
                        return
                    if output is None or output_tokens + count > args.output_shard_tokens:
                        if output is not None:
                            output.close()
                        output_path = args.out / f"part-{state['output_shard']:05d}.jsonl.gz"
                        output = gzip.open(output_path, "at", encoding="utf-8")
                        state["output_shard"] += 1
                        output_tokens = 0
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_tokens += count
                    state["tokens"] += count
                    state["documents"] += 1
            source.unlink()
            state["next_url"] = url_index + 1
            save_state(state_path, state)
            print(f"total: {state['tokens']:,} tokens, {state['documents']:,} documents", flush=True)
    finally:
        if output is not None:
            output.close()
        shutil.rmtree(work, ignore_errors=True)
        save_state(state_path, state)


if __name__ == "__main__":
    main()
