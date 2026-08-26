"""Build a memory-mappable SHADOW 1-bit cold-KV archive.

The builder replays a tokens.u32 stream through cached inference, preserving the
same per-layer codec state used by training. It writes atomically and never loads
the final archive payload into one contiguous host allocation.
"""
import argparse
import hashlib
import os
import pathlib
import struct
import sys
import tempfile

import numpy as np
import torch

for key, value in {
    "SHADOW_D": "1536", "SHADOW_NL": "10", "SHADOW_NH": "24",
    "SHADOW_NKV": "2", "SHADOW_HD": "64", "SHADOW_FFNH": "4224",
    "SHADOW_FAST_ATTN": "1", "SHADOW_KV_BITS": "1",
    "SHADOW_KV_TWO_TIER": "1",
}.items():
    os.environ.setdefault(key, value)

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "modeling"))
from common import requant
from paged_kv import PagedKVArchive, PagedKVView, ExactChunkCountView
from model_250m import Shadow250M, load_vocab

HEADER = struct.Struct("<8s6I5Q32s32s120s")
MAGIC = b"SHARKV1\0"
ALIGNMENT = 4096


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def align(value, width=ALIGNMENT):
    return (value + width - 1) // width * width


def pad_to(stream, offset):
    missing = offset - stream.tell()
    if missing < 0:
        raise ValueError("archive section overlap")
    if missing:
        stream.write(b"\0" * missing)


def reset_cold(cache):
    old = cache["cold_archive"]
    fresh = PagedKVArchive(old.batch, old.heads, old.packed_width, page_size=old.page_size, device=old.device)
    cache["cold_archive"] = fresh
    cache["cold_k"] = PagedKVView(fresh, "k")
    cache["cold_v"] = PagedKVView(fresh, "v")
    cache["chunk_keys"] = ExactChunkCountView(fresh, cache["cold_chunk"])


def flush_cold(layers, spool_files):
    """Flush complete evicted tokens to per-layer/head files and release their pages."""
    flushed = None
    for layer, cache in enumerate(layers):
        archive = cache["cold_archive"]
        count = len(archive)
        if flushed is None:
            flushed = count
        elif count != flushed:
            raise RuntimeError("layer cold archives diverged")
        if not count:
            continue
        for kind in ("k", "v"):
            payload = archive.materialize(kind)[0].cpu().numpy()  # heads,tokens,width
            for head in range(payload.shape[0]):
                spool_files[(kind, layer, head)].write(payload[head].tobytes(order="C"))
        reset_cold(cache)
    return int(flushed or 0)


def finalize_spooled_archive(path, spool_paths, hot_keys, hot_values, positions, tokens, model_hash, table_hash, page_size):
    layers, heads, hot_count, width = hot_keys.shape
    cold_bytes = spool_paths[("k", 0, 0)].stat().st_size
    if cold_bytes % width:
        raise RuntimeError("misaligned KV spool")
    cold_count = cold_bytes // width
    count = cold_count + hot_count
    if count != len(tokens):
        raise RuntimeError(f"archive payload has {count} tokens but stream has {len(tokens)}")
    positions_offset = align(HEADER.size)
    keys_bytes = layers * heads * count * width
    keys_offset = align(positions_offset + positions.nbytes)
    values_offset = align(keys_offset + keys_bytes)
    tokens_offset = align(values_offset + keys_bytes)
    target = pathlib.Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    try:
        with open(temporary, "wb") as stream:
            stream.write(HEADER.pack(MAGIC, 1, HEADER.size, layers, heads, width, int(page_size), count,
                                     positions_offset, keys_offset, values_offset, tokens_offset, model_hash, table_hash, b""))
            pad_to(stream, positions_offset); stream.write(positions.tobytes(order="C"))
            for kind, offset, hot in (("k", keys_offset, hot_keys), ("v", values_offset, hot_values)):
                pad_to(stream, offset)
                for layer in range(layers):
                    for head in range(heads):
                        with open(spool_paths[(kind, layer, head)], "rb") as source:
                            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                                stream.write(chunk)
                        stream.write(hot[layer, head].tobytes(order="C"))
            pad_to(stream, tokens_offset); stream.write(tokens.astype("<u4", copy=False).tobytes(order="C"))
            pad_to(stream, align(tokens_offset + tokens.nbytes)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists(): temporary.unlink()


def write_archive(path, keys, values, positions, tokens, model_hash, table_hash, page_size=256):
    """Write tensors shaped (layers, heads, tokens, packed_width)."""
    keys = np.asarray(keys, dtype=np.uint8)
    values = np.asarray(values, dtype=np.uint8)
    positions = np.asarray(positions, dtype="<u8")
    tokens = np.asarray(tokens, dtype="<u4")
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("keys and values must share (layers,heads,tokens,width)")
    layers, heads, count, width = keys.shape
    if positions.shape != (count,) or tokens.shape != (count,):
        raise ValueError("position/token count differs from KV payload")
    if len(model_hash) != 32 or len(table_hash) != 32:
        raise ValueError("archive hashes must be SHA-256 digests")
    positions_offset = align(HEADER.size)
    keys_offset = align(positions_offset + positions.nbytes)
    values_offset = align(keys_offset + keys.nbytes)
    tokens_offset = align(values_offset + values.nbytes)
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    try:
        with open(temporary, "wb") as stream:
            stream.write(HEADER.pack(
                MAGIC, 1, HEADER.size, layers, heads, width, int(page_size), count,
                positions_offset, keys_offset, values_offset, tokens_offset,
                model_hash, table_hash, b"",
            ))
            pad_to(stream, positions_offset); stream.write(positions.tobytes(order="C"))
            pad_to(stream, keys_offset); stream.write(keys.tobytes(order="C"))
            pad_to(stream, values_offset); stream.write(values.tobytes(order="C"))
            pad_to(stream, tokens_offset); stream.write(tokens.tobytes(order="C"))
            pad_to(stream, align(tokens_offset + tokens.nbytes))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def build(checkpoint, model_file, table, token_file, output, max_ctx=2048, block=256, page_size=256):
    with open(model_file, "rb") as stream:
        if stream.read(4) != b"SHDW" or struct.unpack("<I", stream.read(4))[0] != 2:
            raise ValueError("cold-KV archives require a .shdw v2 exported with --with-codecs")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cent, cent_n, vocab = load_vocab(table, device)
    model = Shadow250M(cent, cent_n, vocab, use_memory=True).to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data["model"], strict=False)
    model.eval(); requant(model)
    tokens = np.memmap(token_file, np.uint32, "r")
    if not len(tokens):
        raise ValueError("cannot build an archive from an empty token stream")
    initial = min(len(tokens), max_ctx)
    first = torch.from_numpy(np.asarray(tokens[:initial], dtype=np.int64)).view(1, -1).to(device)
    with tempfile.TemporaryDirectory(prefix="shadow-kv-") as spool_dir:
        spool_paths = {}
        for kind in ("k", "v"):
            for layer in range(10):
                for head in range(2):
                    spool_paths[(kind, layer, head)] = pathlib.Path(spool_dir) / f"{kind}-{layer}-{head}.bin"
        spool_files = {key: open(path, "wb") for key, path in spool_paths.items()}
        try:
            with torch.inference_mode():
                _, state = model.prefill_cached(first, max_ctx=max_ctx, use_memory=True, memory_chunk=page_size)
                cursor = initial
                while cursor < len(tokens):
                    stop = min(len(tokens), cursor + block)
                    chunk = torch.from_numpy(np.asarray(tokens[cursor:stop], dtype=np.int64)).view(1, -1).to(device)
                    model.ingest_cached(chunk, state); cursor = stop
                    flush_cold(state["layers"], spool_files)
                    print(f"encoded {cursor}/{len(tokens)} tokens", flush=True)
        finally:
            for stream in spool_files.values(): stream.close()
        hot_keys = np.stack([cache["k"][0].cpu().numpy() for cache in state["layers"]])
        hot_values = np.stack([cache["v"][0].cpu().numpy() for cache in state["layers"]])
        positions = np.arange(len(tokens), dtype=np.uint64)
        finalize_spooled_archive(output, spool_paths, hot_keys, hot_values, positions, tokens,
                                 sha256(model_file), sha256(table), page_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True, help="deployed .shdw v2 whose hash is stored in the archive")
    parser.add_argument("--table", required=True)
    parser.add_argument("--tokens", required=True, help="uint32 tokens.u32")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-ctx", type=int, default=2048)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=256)
    args = parser.parse_args()
    build(args.checkpoint, args.model, args.table, args.tokens, args.out, args.max_ctx, args.block, args.page_size)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
