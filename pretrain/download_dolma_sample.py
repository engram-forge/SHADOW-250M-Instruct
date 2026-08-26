"""Download the official Dolma v1.6 8B sample without decompressing it."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import shutil
import urllib.request

MANIFEST_URL = (
    "https://huggingface.co/datasets/allenai/dolma/resolve/main/urls/"
    "v1_6-sample.txt"
)


def request(url, method=None, headers=None):
    combined = {"User-Agent": "shadow-dolma-pretrain/1.0", **(headers or {})}
    return urllib.request.Request(url, method=method, headers=combined)


def remote_size(url):
    with urllib.request.urlopen(request(url, method="HEAD")) as response:
        return int(response.headers["Content-Length"])


def download_one(url, output, expected_size):
    final = output / Path(url).name
    partial = final.with_suffix(final.suffix + ".part")
    if final.exists() and final.stat().st_size == expected_size:
        return final, "present"
    if final.exists():
        final.replace(partial)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset == expected_size:
        partial.replace(final)
        return final, "resumed"
    if offset > expected_size:
        partial.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with urllib.request.urlopen(request(url, headers=headers)) as response:
        append = offset and response.status == 206
        if offset and not append:
            offset = 0
        with partial.open("ab" if append else "wb") as stream:
            shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"size mismatch for {final.name}: {partial.stat().st_size} != {expected_size}"
        )
    partial.replace(final)
    return final, "downloaded"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--verify-gzip", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(request(MANIFEST_URL)) as response:
        urls = [line for line in response.read().decode().splitlines() if line]
    if len(urls) != 103:
        raise SystemExit(f"expected 103 sample shards, manifest has {len(urls)}")
    manifest = args.out.parent / "v1_6-sample.urls.txt"
    manifest.write_text("\n".join(urls) + "\n", encoding="utf-8")

    print("reading remote sizes", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        sizes = list(pool.map(remote_size, urls))
    required = sum(sizes)
    free = shutil.disk_usage(args.out).free
    existing = sum(
        min((args.out / Path(url).name).stat().st_size, size)
        for url, size in zip(urls, sizes)
        if (args.out / Path(url).name).exists()
    )
    if free < required - existing + 2 * 1024**3:
        raise SystemExit("insufficient disk: require corpus remainder plus 2 GiB margin")

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, url, args.out, size): url
            for url, size in zip(urls, sizes)
        }
        for future in as_completed(futures):
            path, status = future.result()
            completed += 1
            print(f"[{completed:03d}/{len(urls)}] {status}: {path.name}", flush=True)

    files = sorted(args.out.glob("*.json.gz"))
    actual = sum(path.stat().st_size for path in files)
    if len(files) != len(urls) or actual != required:
        raise SystemExit(f"download incomplete: {len(files)} files, {actual}/{required} bytes")
    if args.verify_gzip:
        print("verifying gzip streams", flush=True)
        def verify(path):
            with gzip.open(path, "rb") as stream:
                while stream.read(8 * 1024 * 1024):
                    pass
            return path.name
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, name in enumerate(pool.map(verify, files), 1):
                print(f"[{index:03d}/{len(files)}] valid: {name}", flush=True)
    metadata = {
        "manifest_url": MANIFEST_URL,
        "files": len(files),
        "compressed_bytes": actual,
        "format": "gzip-compressed JSONL, unchanged from upstream",
    }
    (args.out.parent / "download.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
