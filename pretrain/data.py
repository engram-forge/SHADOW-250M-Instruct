"""Deterministic streaming tokenization and packing for compressed Dolma JSONL."""

from collections import deque
import gzip
import hashlib
from itertools import islice
import json
import multiprocessing as mp
from pathlib import Path
import random

EOS = 1
_TOKENIZER = None


class ShadowTokenizer:
    def __init__(self, model, remap):
        import numpy as np
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(model))
        self.new_to_old = np.fromfile(remap, np.uint32).astype(np.int64)
        self.old_to_new = np.full(self.sp.get_piece_size(), -1, np.int64)
        self.old_to_new[self.new_to_old] = np.arange(len(self.new_to_old))
        self.expansion = {}

    def encode(self, text):
        result = []
        for old_id in self.sp.encode(text):
            new_id = self.old_to_new[old_id]
            if new_id >= 0:
                result.append(int(new_id))
                continue
            sequence = self.expansion.get(old_id)
            if sequence is None:
                piece = self.sp.id_to_piece(int(old_id)).replace("▁", " ")
                raw = (
                    self.sp.decode([int(old_id)]).encode("utf-8")
                    if not piece.startswith("<0x")
                    else bytes([int(piece[3:5], 16)])
                )
                sequence = [
                    int(self.old_to_new[self.sp.piece_to_id(f"<0x{byte:02X}>")])
                    for byte in raw
                ]
                self.expansion[old_id] = sequence
            result.extend(sequence)
        return result


def _init_worker(model, remap):
    global _TOKENIZER
    _TOKENIZER = ShadowTokenizer(model, remap)


def _encode_text(text):
    return _TOKENIZER.encode(text)


def split_shards(paths, seed=1337):
    ordered = sorted(
        (Path(path) for path in paths),
        key=lambda path: hashlib.sha256(f"{seed}\0{path.name}".encode()).digest(),
    )
    if len(ordered) < 2:
        raise ValueError("at least two shards are required for train/validation")
    return ordered[1:], ordered[:1]


class DolmaPacker:
    """Ordered multiprocessing tokenizer with serializable pending state."""

    def __init__(self, shards, model, remap, context=2048, workers=8, chunk_docs=256, seed=1337):
        self.shards = [str(Path(path)) for path in shards]
        random.Random(seed).shuffle(self.shards)
        self.model, self.remap = str(model), str(remap)
        self.context, self.workers, self.chunk_docs = context, workers, chunk_docs
        self.shard_index = 0
        self.line_index = 0
        self.tokens = deque()
        self.windows = deque()
        self.documents = self.bad_records = self.consumed_tokens = 0
        self._stream = None
        self._pool = None

    def _ensure_pool(self):
        if self._pool is None and self.workers > 0:
            context = mp.get_context("spawn")
            self._pool = context.Pool(
                self.workers, _init_worker, (self.model, self.remap)
            )

    def _open(self):
        if self.shard_index >= len(self.shards):
            return False
        self._stream = gzip.open(
            self.shards[self.shard_index], "rt", encoding="utf-8", errors="replace"
        )
        for _ in range(self.line_index):
            if not self._stream.readline():
                break
        return True

    def _read_chunk(self):
        texts = []
        while len(texts) < self.chunk_docs and self.shard_index < len(self.shards):
            if self._stream is None and not self._open():
                break
            line = self._stream.readline()
            if not line:
                self._stream.close()
                self._stream = None
                self.shard_index += 1
                self.line_index = 0
                continue
            self.line_index += 1
            try:
                text = json.loads(line).get("text")
                if not isinstance(text, str) or not text:
                    raise ValueError
                texts.append(text)
            except (json.JSONDecodeError, AttributeError, ValueError):
                self.bad_records += 1
        return texts

    def _fill(self):
        texts = self._read_chunk()
        if not texts:
            if len(self.tokens) > self.context:
                self.windows.append(list(islice(self.tokens, self.context + 1)))
                for _ in range(self.context):
                    self.tokens.popleft()
            return False
        self._ensure_pool()
        if self._pool is None:
            tokenizer = ShadowTokenizer(self.model, self.remap)
            encoded = map(tokenizer.encode, texts)
        else:
            encoded = self._pool.imap(_encode_text, texts, chunksize=8)
        for ids in encoded:
            self.documents += 1
            self.tokens.extend(ids)
            self.tokens.append(EOS)
            while len(self.tokens) >= self.context + 1:
                self.windows.append(list(islice(self.tokens, self.context + 1)))
                for _ in range(self.context):
                    self.tokens.popleft()
        return True

    def next_window(self):
        while not self.windows:
            if not self._fill():
                if not self.windows:
                    raise StopIteration
        window = self.windows.popleft()
        self.consumed_tokens += self.context
        return window

    def next_batch(self, batch_size):
        import torch
        rows = [self.next_window() for _ in range(batch_size)]
        tensor = torch.tensor(rows, dtype=torch.long)
        return tensor[:, :-1], tensor[:, 1:]

    def state_dict(self):
        return {
            "shard_index": self.shard_index,
            "line_index": self.line_index,
            "tokens": list(self.tokens),
            "windows": list(self.windows),
            "documents": self.documents,
            "bad_records": self.bad_records,
            "consumed_tokens": self.consumed_tokens,
            "shards": self.shards,
        }

    def load_state_dict(self, state):
        if state["shards"] != self.shards:
            raise ValueError("checkpoint shard order does not match this corpus")
        for key in ("shard_index", "line_index", "documents", "bad_records", "consumed_tokens"):
            setattr(self, key, state[key])
        self.tokens = deque(state["tokens"])
        self.windows = deque(state["windows"])

    def close(self):
        if self._stream is not None:
            self._stream.close()
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
