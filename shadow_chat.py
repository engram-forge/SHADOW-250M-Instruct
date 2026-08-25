"""Start chatting with SHADOW. Picks the right binary for your system automatically.
    python shadow_chat.py
"""
import argparse, os, sys, platform, pathlib
ap = argparse.ArgumentParser()
ap.add_argument("--greedy", action="store_true")
ap.add_argument("--no-guard", action="store_true")
ap.add_argument("--temperature", type=float, default=0.25)
ap.add_argument("--top-k", type=int, default=30)
ap.add_argument("--repetition-penalty", type=float, default=1.15)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
HERE = pathlib.Path(__file__).resolve().parent
osname = platform.system()
if osname == "Windows": k = HERE / "deployment" / "bin" / "windows" / "shadow.exe"
elif osname == "Linux": k = HERE / "deployment" / "bin" / "linux" / "shadow"
else: sys.exit("macOS build available on request: saikiranbathula1@gmail.com")
if osname != "Windows": os.chmod(k, 0o755)
sys.path.insert(0, str(HERE))
from shadow_runtime import Engine
eng = Engine(str(HERE / "deployment" / "shadow250m_instruct.shdw"), str(HERE / "deployment" / "fp131072.npy"), kernel=str(k))
print("SHADOW 250M. Type your message, 'quit' to stop.")
while True:
    try: q = input("you> ").strip()
    except EOFError: break
    if q in ("quit", "exit"): break
    if q: print("shadow>", eng.chat(q, greedy=a.greedy, temp=a.temperature, topk=a.top_k,
                                      rep=a.repetition_penalty, seed=a.seed, guard=not a.no_guard))
