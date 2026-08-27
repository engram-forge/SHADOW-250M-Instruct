"""Interactive text chat over the TileLang CUDA engine."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _prompt(message: str) -> list[int]:
    from shadow_runtime.prompt import BOS, EOT, SOT, nl
    from shadow_runtime.retriever import enc

    return [BOS, SOT] + enc("user\n") + enc(message) + [EOT] + nl() + [SOT] + enc("model\n")


def main() -> None:
    from shadow_runtime.retriever import _dec

    from .engine import TileLangEngine

    parser = argparse.ArgumentParser(description="Chat with SHADOW on CUDA through TileLang")
    parser.add_argument("--model", default=ROOT / "deployment/shadow250m_instruct.shdw")
    parser.add_argument("--table", default=ROOT / "deployment/fp131072.npy")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("tilelang", "torch"), default="tilelang")
    args = parser.parse_args()
    engine = TileLangEngine(args.model, args.table, backend=args.backend)
    print("SHADOW TileLang CUDA. Type your message, 'quit' to stop.")
    try:
        while True:
            try:
                message = input("you> ").strip()
            except EOFError:
                break
            if message in ("quit", "exit"):
                break
            if not message:
                continue
            engine.reset()
            output = engine.generate(
                _prompt(message), args.tokens, temperature=args.temperature,
                top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                seed=args.seed,
            )
            visible = output
            for stop in (9, 1):
                if stop in visible:
                    visible = visible[: visible.index(stop)]
            print("shadow>", _dec(visible).strip())
    finally:
        engine.close()


if __name__ == "__main__":
    main()
