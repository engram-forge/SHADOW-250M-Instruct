"""Command-line entry point compatible with the bundled token-ID runtime."""

import argparse

from .engine import TileLangEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="SHADOW TileLang CUDA engine")
    parser.add_argument("model")
    parser.add_argument("table")
    parser.add_argument("tokens", help="space-separated prompt token IDs")
    parser.add_argument("count", type=int, help="maximum generated token count")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--rep", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("tilelang", "torch"), default="tilelang")
    parser.add_argument("--max-context", type=int, default=2048)
    args = parser.parse_args()
    prompt = [int(token) for token in args.tokens.split()]
    engine = TileLangEngine(
        args.model, args.table, backend=args.backend, max_context=args.max_context
    )
    try:
        output = engine.generate(
            prompt, args.count, temperature=args.temp, top_k=args.topk,
            repetition_penalty=args.rep, seed=args.seed,
        )
        print(" ".join(map(str, output)))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
