import argparse, sys
from . import Engine
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--table", required=True)
ap.add_argument("--archive"); ap.add_argument("--ask"); ap.add_argument("--chat", action="store_true")
ap.add_argument("--greedy", action="store_true"); ap.add_argument("--no-guard", action="store_true")
ap.add_argument("--temperature", type=float, default=0.25); ap.add_argument("--top-k", type=int, default=30)
ap.add_argument("--repetition-penalty", type=float, default=1.15); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
eng = Engine(a.model, a.table, archive=a.archive)
if a.ask: print(eng.answer(a.ask))
elif a.chat:
    while True:
        try: q = input("you> ")
        except EOFError: break
        if not q.strip(): continue
        print("shadow>", eng.chat(q, greedy=a.greedy, temp=a.temperature, topk=a.top_k,
                                    rep=a.repetition_penalty, seed=a.seed, guard=not a.no_guard))
