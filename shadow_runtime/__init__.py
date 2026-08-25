"""SHADOW 250M runtime: chat through the bundled CPU kernel, and question answering over an on-disk archive.

    from shadow_runtime import Engine
    eng = Engine("shadow250m_instruct.shdw", "fp131072.npy", archive="path/to/archive_dir")
    print(eng.answer("your question"))

An archive directory holds tokens.u32 (uint32 token stream). The lexical index is built once on first use
and cached next to it. Chat without an archive:

    eng = Engine("shadow250m_instruct.shdw", "fp131072.npy")
    print(eng.chat("Explain photosynthesis in two sentences."))

CLI:  python -m shadow_runtime --model shadow250m_instruct.shdw --table fp131072.npy \
          --archive path/to/archive --ask "your question"
"""
import os, sys, pathlib, subprocess
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retriever import Inverted, load_archive, enc, _dec
from answer_engine import Engine as _Extractor, ABSTAIN
from prompt import EOT

BOS, SOT, EOS = 2, 8, 1

def repetition_metrics(ids):
    """Return a loop start and recent repeated 4-gram ratio."""
    loop_start = None
    for width in range(min(32, len(ids) // 3), 0, -1):
        for end in range(width * 3, len(ids) + 1):
            block = ids[end - width:end]
            if all(ids[end - (r + 1) * width:end - r * width] == block for r in (1, 2)):
                loop_start = end - width * 3
                break
        if loop_start is not None: break
    recent = ids[-96:]
    grams = [tuple(recent[i:i + 4]) for i in range(max(0, len(recent) - 3))]
    ratio = 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)
    return loop_start, ratio

def guarded_result(text):
    ids = enc(text)
    loop_start, ratio = repetition_metrics(ids)
    return {"text": text, "ids": ids, "loop_start": loop_start, "repeat_4gram_ratio": ratio,
            "looped": loop_start is not None or (len(ids) >= 48 and ratio > 0.35)}

class Engine:
    def __init__(s, model, table, archive=None, kernel=None, threads=None):
        s.model = str(pathlib.Path(model).resolve()); s.table = str(pathlib.Path(table).resolve())
        s.kernel = str(pathlib.Path(kernel).resolve()) if kernel else str((HERE.parent / "deployment" / "bin" / ("windows/shadow.exe" if os.name == "nt" else "linux/shadow")).resolve())
        s.env = dict(os.environ)
        if threads: s.env["SHADOW_THREADS"] = str(threads)
        s.ext = None
        if archive:
            tok, meta, _bank = load_archive(str(archive))
            inv = Inverted(tok)
            s.ext = _Extractor(tok, inv, model_ask=s.chat)
    def _gen(s, ids, n=140, extra=()):
        r = subprocess.run([s.kernel, s.model, s.table, " ".join(map(str, ids)), str(n), *extra],
                           capture_output=True, text=True, env=s.env)
        out = [int(x) for x in r.stdout.split()]
        for stop in (EOT, EOS):
            if stop in out: out = out[:out.index(stop)]
        return _dec(out).strip()
    def chat(s, message, n=160, greedy=False, temp=0.25, topk=30, rep=1.15, seed=0,
             guard=True, retry_on_loop=True, return_metadata=False):
        ids = [BOS, SOT] + enc("user\n") + enc(message) + [EOT] + enc("\n") + [SOT] + enc("model\n")
        extra = () if greedy else ("--temp", str(temp), "--topk", str(topk), "--rep", str(rep), "--seed", str(seed))
        first = guarded_result(s._gen(ids, n, extra))
        chosen = first; retried = False; truncated = False
        if guard and first["looped"] and retry_on_loop:
            retry_extra = ("--temp", "0.35", "--topk", "40", "--rep", "1.25", "--seed", str(seed + 1))
            second = guarded_result(s._gen(ids, n, retry_extra)); retried = True
            chosen = min((first, second), key=lambda item: (item["looped"], item["repeat_4gram_ratio"]))
        if guard and chosen["looped"] and chosen["loop_start"] is not None:
            chosen = dict(chosen)
            chosen["text"] = _dec(np.asarray(chosen["ids"][:chosen["loop_start"]], np.int64)).strip()
            truncated = True
        metadata = {"text": chosen["text"], "looped": chosen["looped"], "retried": retried,
                    "truncated": truncated, "repeat_4gram_ratio": chosen["repeat_4gram_ratio"]}
        return metadata if return_metadata else metadata["text"]
    def answer(s, question):
        if s.ext is None: return s.chat(question)
        a, how = s.ext.answer(question)
        return a
