"""Generate a deterministic scan-only SHKV fixture without model inference."""
import argparse
import hashlib
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune"))
from build_kv_archive import ALIGNMENT, HEADER, MAGIC, align, sha256


def write_fixture(path, model, table, count, width, seed):
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    positions_offset = align(HEADER.size)
    positions_bytes = count * 8
    keys_offset = align(positions_offset + positions_bytes)
    values_offset = align(keys_offset + count * width)
    tokens_offset = align(values_offset + count * width)
    length = align(tokens_offset + count * 4)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    try:
        with open(temporary, "w+b") as stream:
            stream.truncate(length)
            stream.seek(0)
            stream.write(HEADER.pack(
                MAGIC, 1, HEADER.size, 1, 1, width, 256, count,
                positions_offset, keys_offset, values_offset, tokens_offset,
                sha256(model), sha256(table), b"",
            ))
            positions = np.memmap(stream, dtype="<u8", mode="r+", offset=positions_offset, shape=(count,))
            positions[:] = np.arange(count, dtype=np.uint64); positions.flush(); del positions
            keys = np.memmap(stream, dtype=np.uint8, mode="r+", offset=keys_offset, shape=(count, width))
            rng = np.random.default_rng(seed)
            block = 16_384
            for begin in range(0, count, block):
                end = min(count, begin + block)
                keys[begin:end] = rng.integers(0, 256, size=(end - begin, width), dtype=np.uint8)
            keys.flush(); del keys
            tokens = np.memmap(stream, dtype="<u4", mode="r+", offset=tokens_offset, shape=(count,))
            tokens[:] = np.arange(count, dtype=np.uint32); tokens.flush(); del tokens
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists(): temporary.unlink()
    print({"path": str(target), "tokens": count, "packed_width": width,
           "bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True); parser.add_argument("--tokens", type=int, default=1_000_000)
    parser.add_argument("--width", type=int, default=64); parser.add_argument("--seed", type=int, default=250)
    args = parser.parse_args(); write_fixture(args.out, args.model, args.table, args.tokens, args.width, args.seed)


if __name__ == "__main__": main()
