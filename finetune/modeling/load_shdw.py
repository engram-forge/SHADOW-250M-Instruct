"""Load the packed deployment ``.shdw`` format into the Python model.

Unlike loading the master ``.pt`` checkpoint, this module reconstructs the
exact quantized matrices stored in the deployment artifact. It is intended as
a correctness oracle for native runtimes and export round-trip checks.
"""
from __future__ import annotations

import dataclasses
import pathlib
import struct
from typing import BinaryIO

import numpy as np
import torch

def _rvq_unpack(codebook, indices, scale, rows, columns, group, stages):
    groups = columns // group
    padded_rows = scale.shape[0]
    chunks = padded_rows // 64
    weight = np.zeros((padded_rows, columns), dtype=np.float32)
    for stage in range(stages):
        for chunk in range(chunks):
            packed = indices[stage, chunk]
            low = (packed & 0x0F).T
            high = (packed >> 4).T
            for codes, row_offset in ((low, chunk * 64), (high, chunk * 64 + 32)):
                for local_row in range(32):
                    selected = codes[local_row]
                    weight[row_offset + local_row] += codebook[stage][:, selected].T.reshape(groups * group)
    return weight[:rows] * scale[:rows, None]


@dataclasses.dataclass(frozen=True)
class ShdwRecord:
    name: str
    kind: int
    value: np.ndarray


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated SHDW file")
    return data


def _unpack_ternary(packed: np.ndarray, rows: int, columns: int, kind: int) -> np.ndarray:
    if kind == 3:
        digits = np.stack(tuple((packed >> (2 * index)) & 3 for index in range(4)), -1)
    elif kind == 4:
        digits = np.stack(tuple((packed // (3 ** index)) % 3 for index in range(5)), -1)
    else:
        raise ValueError(f"not a ternary record kind: {kind}")
    return digits.reshape(rows, -1)[:, :columns].astype(np.float32) - 1.0


def read_shdw(path: str | pathlib.Path) -> tuple[int, dict[str, ShdwRecord]]:
    records: dict[str, ShdwRecord] = {}
    with open(path, "rb") as stream:
        if _read_exact(stream, 4) != b"SHDW":
            raise ValueError("invalid SHDW magic")
        version, count = struct.unpack("<II", _read_exact(stream, 8))
        if version not in (1, 2):
            raise ValueError(f"unsupported SHDW version {version}")
        for _ in range(count):
            name_length, = struct.unpack("<I", _read_exact(stream, 4))
            name = _read_exact(stream, name_length).decode("utf-8")
            kind, = struct.unpack("<I", _read_exact(stream, 4))
            if kind in (0, 5):
                dimensions, = struct.unpack("<I", _read_exact(stream, 4))
                shape = struct.unpack("<" + "I" * dimensions, _read_exact(stream, 4 * dimensions))
                dtype = np.dtype("<f4") if kind == 0 else np.dtype("<f2")
                value = np.frombuffer(_read_exact(stream, int(np.prod(shape)) * dtype.itemsize), dtype).reshape(shape)
                value = value.astype(np.float32, copy=True)
            elif kind == 1:
                rows, columns, group, stages = struct.unpack("<IIII", _read_exact(stream, 16))
                padded_rows = (rows + 63) & ~63
                groups = columns // group
                chunks = padded_rows // 64
                codebook = np.frombuffer(
                    _read_exact(stream, stages * group * 16 * 4), np.dtype("<f4")
                ).reshape(stages, group, 16)
                indices = np.frombuffer(
                    _read_exact(stream, stages * chunks * groups * 32), np.uint8
                ).reshape(stages, chunks, groups, 32)
                scale = np.frombuffer(_read_exact(stream, padded_rows * 4), np.dtype("<f4"))
                value = _rvq_unpack(codebook, indices, scale, rows, columns, group, stages)
            elif kind in (3, 4):
                rows, columns = struct.unpack("<II", _read_exact(stream, 8))
                stride = columns // 4 if kind == 3 else (columns + 4) // 5
                packed = np.frombuffer(_read_exact(stream, rows * stride), np.uint8).reshape(rows, stride)
                scale = np.frombuffer(_read_exact(stream, rows * 4), np.dtype("<f4"))
                value = _unpack_ternary(packed, rows, columns, kind) * scale[:, None]
            else:
                raise ValueError(f"unknown SHDW record kind {kind} for {name}")
            if name in records:
                raise ValueError(f"duplicate SHDW record {name}")
            records[name] = ShdwRecord(name, kind, np.ascontiguousarray(value))
        if stream.read(1):
            raise ValueError("trailing data after SHDW records")
    return version, records


def _model_name(record_name: str) -> str:
    if record_name == "tb":
        return "tied_bias"
    if record_name.startswith("emb."):
        return "inp." + record_name.removeprefix("emb.")
    if record_name.startswith("step."):
        return "struct." + record_name.removeprefix("step.")
    return record_name


def load_shdw_model(model_path: str | pathlib.Path, table_path: str | pathlib.Path, device: str = "cpu"):
    """Return ``(model, version)`` using deployment weights from ``model_path``."""
    # Imports are delayed so callers can set SHADOW_* architecture variables
    # before constructing the fixed published model.
    from common import RVQ
    from model_250m import Shadow250M, load_vocab

    version, records = read_shdw(model_path)
    cent, cent_n, vocab = load_vocab(str(table_path), device)
    model = Shadow250M(cent, cent_n, vocab).to(device)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    for record in records.values():
        name = _model_name(record.name)
        tensor = torch.from_numpy(record.value).to(device)
        if record.kind in (1, 3, 4):
            module = model.get_submodule(name)
            if not isinstance(module, RVQ):
                raise ValueError(f"compressed matrix {record.name} does not map to an RVQ module")
            if tuple(module.weight.shape) != tuple(tensor.shape):
                raise ValueError(f"shape mismatch for {record.name}: {tuple(tensor.shape)}")
            module.weight.data.copy_(tensor)
            module._q = tensor.clone()
        elif name in parameters:
            target = parameters[name]
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(f"shape mismatch for {record.name}: {tuple(tensor.shape)} != {tuple(target.shape)}")
            target.data.copy_(tensor.to(target.dtype))
        elif name in buffers:
            target = buffers[name]
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(f"shape mismatch for {record.name}: {tuple(tensor.shape)} != {tuple(target.shape)}")
            target.data.copy_(tensor.to(target.dtype))
        else:
            raise ValueError(f"SHDW record does not map to the Python model: {record.name}")

    model.eval()
    return model, version
