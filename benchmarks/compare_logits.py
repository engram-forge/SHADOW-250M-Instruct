"""Compare two float logit dumps produced by SHADOW runtimes."""
import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-abs", type=float, help="fail if max absolute error exceeds this value")
    args = parser.parse_args()
    reference = np.load(args.reference).astype(np.float32, copy=False)
    candidate = np.load(args.candidate).astype(np.float32, copy=False)
    if reference.shape != candidate.shape:
        raise SystemExit(f"shape mismatch: {reference.shape} != {candidate.shape}")
    difference = candidate - reference
    flat = np.abs(difference)
    positions = reference.reshape(-1, reference.shape[-1])
    compared = candidate.reshape(-1, candidate.shape[-1])
    k = min(args.top_k, positions.shape[-1])
    reference_top = np.argpartition(-positions, k - 1, axis=-1)[:, :k]
    candidate_top = np.argpartition(-compared, k - 1, axis=-1)[:, :k]
    overlap = [len(set(left.tolist()) & set(right.tolist())) / k for left, right in zip(reference_top, candidate_top)]
    summary = {
        "shape": list(reference.shape),
        "max_abs": float(flat.max(initial=0.0)),
        "mean_abs": float(flat.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))),
        "argmax_agreement": float(np.mean(positions.argmax(-1) == compared.argmax(-1))),
        f"top_{k}_mean_overlap": float(np.mean(overlap)),
    }
    print(json.dumps(summary, indent=2))
    if args.max_abs is not None and summary["max_abs"] > args.max_abs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
