"""Start chatting with SHADOW. Picks the right binary for your system automatically.
    python shadow_chat.py
"""
import argparse, sys, pathlib
ap = argparse.ArgumentParser()
ap.add_argument("--greedy", action="store_true")
ap.add_argument("--no-guard", action="store_true")
ap.add_argument("--temperature", type=float, default=0.25)
ap.add_argument("--top-k", type=int, default=30)
ap.add_argument("--repetition-penalty", type=float, default=1.15)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--kv-archive")
ap.add_argument("--archive-backend", choices=("auto", "cpu", "metal"), default="auto")
ap.add_argument("--archive-top-k", type=int, default=32)
a = ap.parse_args()
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shadow_runtime import Engine
from shadow_runtime.platform import bundled_kernel
try: k = bundled_kernel(HERE)
except RuntimeError as error: sys.exit(str(error))
eng = Engine(str(HERE / "deployment" / "shadow250m_instruct.shdw"), str(HERE / "deployment" / "fp131072.npy"),
             kernel=str(k), kv_archive=a.kv_archive, archive_backend=a.archive_backend,
             archive_top_k=a.archive_top_k)
print("SHADOW 250M. Type your message, 'quit' to stop.")
while True:
    try: q = input("you> ").strip()
    except EOFError: break
    if q in ("quit", "exit"): break
    if q: print("shadow>", eng.chat(q, greedy=a.greedy, temp=a.temperature, topk=a.top_k,
                                      rep=a.repetition_penalty, seed=a.seed, guard=not a.no_guard))
