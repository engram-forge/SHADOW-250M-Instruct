"""Strict, zero-copy reader for the version-1 ``.shdw`` container.

The packed arrays point into an mmap owned by :class:`ShadowModelFile`.  They
are copied only when a record is materialized for CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import mmap
from pathlib import Path
import struct
from typing import Iterator

import numpy as np


MAGIC = b"SHDW"
VERSION = 1


@dataclass(frozen=True)
class DenseRecord:
    name: str
    kind: int
    value: np.ndarray


@dataclass(frozen=True)
class RVQRecord:
    name: str
    kind: int
    out_features: int
    in_features: int
    group_size: int
    stages: int
    codebooks: np.ndarray
    indices: np.ndarray
    scales: np.ndarray


@dataclass(frozen=True)
class TernaryRecord:
    name: str
    kind: int
    out_features: int
    in_features: int
    packed: np.ndarray
    scales: np.ndarray


Record = DenseRecord | RVQRecord | TernaryRecord


class ShadowModelFile:
    """Read and validate every record in a SHADOW deployment file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.records: dict[str, Record] = {}
        try:
            self._parse()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        # NumPy views keep the mapping alive at the OS level after close on
        # Unix. Explicitly drop our record views first for portable behavior.
        self.records.clear()
        if getattr(self, "_mmap", None) is not None:
            self._mmap.close()
            self._mmap = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "ShadowModelFile":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records.values())

    def __getitem__(self, name: str) -> Record:
        return self.records[name]

    def _need(self, offset: int, size: int, what: str) -> None:
        if offset < 0 or size < 0 or offset + size > len(self._mmap):
            raise ValueError(f"truncated .shdw while reading {what}")

    def _u32(self, offset: int, what: str) -> tuple[int, int]:
        self._need(offset, 4, what)
        return struct.unpack_from("<I", self._mmap, offset)[0], offset + 4

    def _array(
        self, offset: int, dtype: np.dtype | str, shape: tuple[int, ...], what: str
    ) -> tuple[np.ndarray, int]:
        dt = np.dtype(dtype)
        count = int(np.prod(shape, dtype=np.int64))
        size = count * dt.itemsize
        self._need(offset, size, what)
        value = np.ndarray(shape, dtype=dt, buffer=self._mmap, offset=offset)
        return value, offset + size

    def _parse(self) -> None:
        self._need(0, 12, "header")
        if self._mmap[:4] != MAGIC:
            raise ValueError(f"{self.path} is not a SHDW model")
        version, count = struct.unpack_from("<II", self._mmap, 4)
        if version != VERSION:
            raise ValueError(f"unsupported SHDW version {version}; expected {VERSION}")
        offset = 12
        for record_index in range(count):
            name_length, offset = self._u32(offset, "record name length")
            self._need(offset, name_length, "record name")
            try:
                name = self._mmap[offset : offset + name_length].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"record {record_index} has a non-UTF-8 name") from exc
            offset += name_length
            if not name or name in self.records:
                raise ValueError(f"invalid or duplicate record name {name!r}")
            kind, offset = self._u32(offset, f"kind for {name}")
            if kind in (0, 5):
                ndim, offset = self._u32(offset, f"rank for {name}")
                if ndim > 8:
                    raise ValueError(f"unreasonable rank {ndim} for {name}")
                self._need(offset, 4 * ndim, f"shape for {name}")
                shape = struct.unpack_from(f"<{ndim}I", self._mmap, offset) if ndim else ()
                offset += 4 * ndim
                value, offset = self._array(
                    offset, "<f4" if kind == 0 else "<f2", tuple(shape), name
                )
                record: Record = DenseRecord(name, kind, value)
            elif kind == 1:
                self._need(offset, 16, f"RVQ dimensions for {name}")
                out_features, in_features, group_size, stages = struct.unpack_from(
                    "<IIII", self._mmap, offset
                )
                offset += 16
                if not group_size or in_features % group_size or not stages:
                    raise ValueError(f"invalid RVQ dimensions for {name}")
                padded_out = (out_features + 63) & ~63
                groups = in_features // group_size
                codebooks, offset = self._array(
                    offset, "<f4", (stages, group_size, 16), f"codebooks for {name}"
                )
                indices, offset = self._array(
                    offset,
                    "u1",
                    (stages, padded_out // 64, groups, 32),
                    f"indices for {name}",
                )
                scales, offset = self._array(
                    offset, "<f4", (padded_out,), f"scales for {name}"
                )
                record = RVQRecord(
                    name, kind, out_features, in_features, group_size, stages,
                    codebooks, indices, scales,
                )
            elif kind == 4:
                self._need(offset, 8, f"ternary dimensions for {name}")
                out_features, in_features = struct.unpack_from("<II", self._mmap, offset)
                offset += 8
                packed, offset = self._array(
                    offset,
                    "u1",
                    (out_features, (in_features + 4) // 5),
                    f"ternary weights for {name}",
                )
                scales, offset = self._array(
                    offset, "<f4", (out_features,), f"ternary scales for {name}"
                )
                record = TernaryRecord(
                    name, kind, out_features, in_features, packed, scales
                )
            else:
                raise ValueError(f"unsupported record kind {kind} for {name}")
            self.records[name] = record
        if offset != len(self._mmap):
            raise ValueError(f"trailing data in .shdw: {len(self._mmap) - offset} bytes")


def unpack_rvq(record: RVQRecord) -> np.ndarray:
    """Expand a packed RVQ matrix to row-major float32."""

    padded_out = record.scales.shape[0]
    groups = record.in_features // record.group_size
    output = np.zeros(
        (padded_out, groups, record.group_size), dtype=np.float32
    )
    codes = np.empty((padded_out, groups), dtype=np.uint8)
    for stage in range(record.stages):
        for chunk in range(padded_out // 64):
            packed = record.indices[stage, chunk]
            start = chunk * 64
            codes[start : start + 32] = (packed & 0x0F).T
            codes[start + 32 : start + 64] = (packed >> 4).T
        codebook = record.codebooks[stage]
        for component in range(record.group_size):
            output[:, :, component] += codebook[component][codes]
    matrix = output.reshape(padded_out, record.in_features)
    matrix *= record.scales[:, None]
    return matrix[: record.out_features]


def unpack_ternary(record: TernaryRecord) -> np.ndarray:
    """Expand five-trits-per-byte weights to row-major float32."""

    trits = np.empty((*record.packed.shape, 5), dtype=np.int8)
    value = record.packed.astype(np.uint16)
    divisor = 1
    for component in range(5):
        trits[..., component] = ((value // divisor) % 3).astype(np.int8) - 1
        divisor *= 3
    matrix = trits.reshape(record.out_features, -1)[:, : record.in_features].astype(
        np.float32
    )
    matrix *= record.scales[:, None]
    return matrix
