#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path

import numpy as np


def records(path: Path):
    with path.open("rb") as stream:
        while header := stream.read(16):
            if len(header) != 16:
                raise ValueError("truncated activation header")
            magic, stage, layer, count = struct.unpack("<4I", header)
            if magic != 0x31414653:
                raise ValueError("invalid activation dump")
            values = np.frombuffer(stream.read(count * 4), dtype="<f4").copy()
            if values.size != count:
                raise ValueError("truncated activation values")
            yield stage, layer, values


def quantize(values: np.ndarray, group: int):
    restored = np.empty_like(values)
    for begin in range(0, values.size, group):
        part = values[begin : begin + group]
        scale = np.max(np.abs(part)) / 127.0
        restored[begin : begin + group] = (
            np.clip(np.rint(part / scale), -127, 127) * scale if scale else 0
        )
    return restored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    args = parser.parse_args()
    rows = list(records(args.dump))
    print(f"records={len(rows)}")
    for stage_id, name in ((0, "up_gate_input"), (1, "down_input")):
        selected = [values for stage, _, values in rows if stage == stage_id]
        print(f"stage={name} records={len(selected)}")
        for group in (0, 128, 64):
            errors = []
            for values in selected:
                restored = quantize(values, values.size if group == 0 else group)
                errors.append(np.linalg.norm(restored - values) / np.linalg.norm(values))
            label = "full" if group == 0 else str(group)
            print(f"  group={label} median_relative_rmse={np.median(errors):.8f} p95={np.percentile(errors, 95):.8f}")


if __name__ == "__main__":
    main()
