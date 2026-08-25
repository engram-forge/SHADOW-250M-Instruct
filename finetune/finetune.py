"""Fine-tune SHADOW 250M Instruct on your own chat data, on a single GPU (8 GB is enough).

Data: a .jsonl file, one conversation per line:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Run:
    python finetune.py --data my_data.jsonl --steps 300 --out my_model
    python export_model.py my_model/finetuned.pt my_model.shdw          # 52 MB deploy file
    ./shadow my_model.shdw fp131072.npy --chat                          # your model, on CPU

Defaults are safe for style and domain fine-tunes: low learning rate, loss only on assistant
tokens, quantisation kept in the loop so the exported model behaves like the trained one.
"""
import argparse, json, math, os, sys, time, random, pathlib
for k, v in {"SHADOW_D": "1536", "SHADOW_NL": "10", "SHADOW_NH": "24", "SHADOW_NKV": "2", "SHADOW_HD": "64",
             "SHADOW_FFNH": "4224", "SHADOW_FAST_ATTN": "1", "SHADOW_KV_BITS": "1", "SHADOW_KV_TWO_TIER": "1"}.items():
    os.environ.setdefault(k, v)
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "modeling")); sys.path.insert(0, str(ROOT / "shadow_runtime"))
import numpy as np, torch, torch.nn.functional as F
import common
from common import requant
from model_250m import Shadow250M
from retriever import enc

BOS, EOS, SOT, EOT = 2, 1, 8, 9

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl with {'messages': [...]} per line")
    ap.add_argument("--val-data", help="separate held-out JSONL; disables random validation split")
    ap.add_argument("--init", default=str(HERE / "shadow250m_instruct.pt"))
    ap.add_argument("--table", default=str(ROOT / "deployment" / "fp131072.npy"))
    ap.add_argument("--out", default="finetuned")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ul-alpha", type=float, default=0.2)
    ap.add_argument("--ul-window", type=int, default=64)
    ap.add_argument("--ul-ngram", type=int, default=3)
    ap.add_argument("--recovery-ratio", type=float, default=0.10)
    ap.add_argument("--repeat-policy", choices=("error", "drop", "warn"), default="warn")
    ap.add_argument("--overlength", choices=("error", "truncate"), default="error")
    ap.add_argument("--audit-report")
    return ap.parse_args()

def build_ids(messages):
    ids, msk = [BOS], [0]
    for m in messages:
        role = "user" if m["role"] != "assistant" else "model"
        head = [SOT] + enc(role + "\n"); ids += head; msk += [0] * len(head)
        body = enc(m["content"]) + [EOT] + enc("\n")
        ids += body
        msk += ([1] * (len(body) - 1) + [0]) if role == "model" else [0] * len(body)
    return ids, msk

def repeated_span_start(ids, min_n=3, max_n=32, repeats=3):
    """Return the first adjacent repeated-span start, or None."""
    for n in range(min(max_n, len(ids) // repeats), min_n - 1, -1):
        for end in range(n * repeats, len(ids) + 1):
            block = ids[end - n:end]
            if all(ids[end - (r + 1) * n:end - r * n] == block for r in range(1, repeats)):
                return end - n * repeats
    return None

def repeated_ngram_ratio(ids, n=4):
    grams = [tuple(ids[i:i + n]) for i in range(max(0, len(ids) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)

def audit_messages(messages):
    assistant_ids = [enc(m.get("content", "")) for m in messages if m.get("role") == "assistant"]
    flat = [token for part in assistant_ids for token in part]
    return {
        "empty_assistant": any(not part for part in assistant_ids),
        "repeated_span": repeated_span_start(flat) is not None,
        "repeat_4gram_ratio": repeated_ngram_ratio(flat),
    }

def truncate_complete_turns(messages, ctx):
    """Keep the longest complete prefix ending in an assistant turn."""
    kept = []
    for message in messages:
        candidate = kept + [message]
        if len(build_ids(candidate)[0]) > ctx:
            break
        kept = candidate
    while kept and kept[-1].get("role") != "assistant":
        kept.pop()
    return kept

def recovery_example(ids, msk, rng):
    """Inject a short repeated assistant prefix, supervised only after the perturbation."""
    supervised = [i for i, value in enumerate(msk) if value]
    if len(supervised) < 12:
        return ids, msk
    start = supervised[0]
    width = min(rng.randint(3, 12), max(3, len(supervised) // 3))
    pattern = ids[start:start + width]
    injected = pattern + pattern
    return ids[:start] + injected + ids[start:], msk[:start] + [0] * len(injected) + msk[start:]

class Packer:
    def __init__(s, path, ctx, rng, val_frac, repeat_policy="warn", overlength="error",
                 recovery_ratio=0.10, audit_report=None, val_path=None):
        s.ex = []
        def load(source, is_validation=False):
            loaded = []
            with open(source, encoding="utf-8") as stream:
                for lineno, line in enumerate(stream, 1):
                    line = line.strip()
                    if not line: continue
                    audit["total"] += 1
                    messages = json.loads(line)["messages"]
                    info = audit_messages(messages)
                    pathological = info["repeated_span"] or info["repeat_4gram_ratio"] > 0.5
                    location = f"{source}:{lineno}" if val_path else lineno
                    if info["empty_assistant"]: audit["empty_assistant"].append(location)
                    if pathological:
                        audit["pathological_repeat"].append(location)
                        if repeat_policy == "drop": audit["dropped"] += 1; continue
                        if repeat_policy == "error": raise SystemExit(f"pathological repetition at {location}")
                    ids, msk = build_ids(messages)
                    if len(ids) > ctx:
                        audit["overlength"].append(location)
                        if overlength == "error": raise SystemExit(f"{location} has {len(ids)} tokens, over --ctx {ctx}")
                        messages = truncate_complete_turns(messages, ctx)
                        if not messages: raise SystemExit(f"{location} has no complete assistant turn within --ctx {ctx}")
                        ids, msk = build_ids(messages)
                    loaded.append((np.asarray(ids, np.int64), np.asarray(msk, np.int64)))
                    audit["accepted"] += 1
            return loaded
        audit = {"total": 0, "accepted": 0, "dropped": 0, "overlength": [],
                 "pathological_repeat": [], "empty_assistant": []}
        s.ex = load(path)
        separate_val = load(val_path, True) if val_path else None
        if audit_report:
            report_path = pathlib.Path(audit_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print(f"audit: {audit['total']} total, {len(audit['pathological_repeat'])} repetitive, "
              f"{len(audit['overlength'])} overlength, {audit['dropped']} dropped")
        if len(s.ex) < 2: raise SystemExit("need at least 2 conversations in the data file")
        rng.shuffle(s.ex)
        if separate_val is not None:
            if not separate_val: raise SystemExit("separate validation file is empty")
            s.train, s.val = s.ex, separate_val
        else:
            nval = min(max(1, int(len(s.ex) * val_frac)), max(1, len(s.ex) // 5))
            s.val = s.ex[:nval]; s.train = s.ex[nval:]
        s.ctx = ctx; s.rng = rng
        s.recovery_ratio = recovery_ratio
        print(f"data: {len(s.train)} train / {len(s.val)} val conversations")
    def pack(s, B, val=False):
        pool = s.val if val else s.train
        X = np.zeros((B, s.ctx), np.int64); Y = np.full((B, s.ctx), -100, np.int64)
        for r in range(B):
            pos = 0; misses = 0
            while pos < s.ctx and misses < len(pool):
                ids, m = pool[s.rng.randrange(len(pool))]
                ids, m = ids.tolist(), m.tolist()
                if not val and s.rng.random() < s.recovery_ratio:
                    ids, m = recovery_example(ids, m, s.rng)
                if len(ids) > s.ctx - pos:
                    misses += 1
                    continue
                X[r, pos:pos + len(ids)] = ids
                ids_arr = np.asarray(ids, np.int64); mask_arr = np.asarray(m, np.int64)
                tgt = np.full(len(ids), -100, np.int64)
                tgt[:-1] = np.where(mask_arr[1:] == 1, ids_arr[1:], -100)
                Y[r, pos:pos + len(ids)] = tgt; pos += len(ids)
                misses = 0
                if pos > s.ctx * 0.9: break
            if pos == 0: raise RuntimeError("no complete example fits in the training context")
        return torch.tensor(X), torch.tensor(Y)

def unlikelihood_pairs(x, y, window=64, ngram=3):
    """Positions and tokens that would complete a locally repeated n-gram."""
    pairs = []
    for row in range(x.shape[0]):
        ids = x[row].tolist(); gold = y[row].tolist()
        for pos in range(ngram - 1, len(ids)):
            if gold[pos] < 0: continue
            prefix = tuple(ids[pos - ngram + 2:pos + 1])
            begin = max(ngram - 2, pos - window)
            negatives = set()
            for old in range(begin, pos):
                if tuple(ids[old - ngram + 2:old + 1]) == prefix:
                    negatives.add(ids[old + 1])
            negatives.discard(gold[pos])
            pairs.extend((row * x.shape[1] + pos, token) for token in negatives)
    return pairs

def main():
    a = get_args(); rng = random.Random(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    _of = common.RVQ.forward
    def _tern(s, x):
        if s.g == 32:
            w = s.weight; sc = 1.0 / w.abs().mean(dim=1, keepdim=True).clamp_(min=1e-5)
            return F.linear(x, (w + ((w * sc).round().clamp(-1, 1) / sc - w).detach()).to(x.dtype))
        return _of(s, x)
    common.RVQ.forward = _tern
    _oenc = common.RVQ.enc
    def _enc2(s):
        if s.g == 32: return
        _oenc(s)
    common.RVQ.enc = _enc2
    fp = np.unpackbits(np.load(a.table), axis=1)[:, :512]
    cent = torch.tensor(fp.astype(np.float32) * 2 - 1, device=dev); cent_n = F.normalize(cent, dim=-1)
    model = Shadow250M(cent, cent_n, cent.shape[0]).to(dev)
    ck = torch.load(a.init, map_location=dev, weights_only=False)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()}
    model.load_state_dict(sd); requant(model)
    for md in model.modules():
        if isinstance(md, common.KVCodec1): md.eval()
    print(f"loaded {a.init} on {dev}")
    if not 0 <= a.recovery_ratio <= 1: raise SystemExit("--recovery-ratio must be in [0, 1]")
    if a.ul_alpha < 0 or a.ul_window < 1 or a.ul_ngram < 2: raise SystemExit("invalid unlikelihood settings")
    data = Packer(a.data, a.ctx, rng, a.val_frac, a.repeat_policy, a.overlength,
                  a.recovery_ratio, a.audit_report, a.val_data)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    def loss_of(x, y, include_ul=True):
        h, _ = model.trunk(x); ph = model.head(h).float().reshape(-1, 512)
        yf = y.reshape(-1); v = yf >= 0; ph = ph[v]; yf = yf[v]
        ce = 0.0
        for i in range(0, ph.shape[0], 8192):
            lg = ph[i:i + 8192] @ model.cent_n.T + model.tied_bias
            ce = ce + F.cross_entropy(lg, yf[i:i + 8192], reduction="sum")
        mle = ce / max(1, int(v.sum()))
        if not include_ul or a.ul_alpha == 0: return mle
        pairs = unlikelihood_pairs(x, y, a.ul_window, a.ul_ngram)
        if not pairs: return mle
        positions = torch.tensor([p for p, _ in pairs], device=x.device)
        negatives = torch.tensor([c for _, c in pairs], device=x.device)
        all_ph = model.head(h).float().reshape(-1, 512)
        ul_sum = 0.0
        for i in range(0, len(pairs), 1024):
            pos_chunk = positions[i:i + 1024]; neg_chunk = negatives[i:i + 1024]
            logits = all_ph[pos_chunk] @ model.cent_n.T + model.tied_bias
            probs = logits.softmax(-1).gather(1, neg_chunk[:, None]).squeeze(1)
            ul_sum = ul_sum - torch.log1p(-probs.clamp(max=1 - 1e-6)).sum()
        ul = ul_sum / len(pairs)
        return mle + a.ul_alpha * ul
    @torch.no_grad()
    def val():
        model.eval(); tot = 0.0
        for _ in range(4):
            x, y = data.pack(a.micro_batch, val=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                tot += float(loss_of(x.to(dev), y.to(dev), include_ul=False))
        model.train()
        for md in model.modules():
            if isinstance(md, common.KVCodec1): md.eval()
        return tot / 4
    v0 = val(); print(f"step 0  val loss {v0:.4f}")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * (0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            x, y = data.pack(a.micro_batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                (loss_of(x.to(dev), y.to(dev)) / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); requant(model)
        if step % a.log_every == 0:
            el = time.time() - t0
            print(f"step {step:>4}  lr {lr:.2e}  {el/step:.1f}s/step  eta {(a.steps-step)*el/step/60:.0f}min", flush=True)
    v1 = val()
    torch.save({"model": model.state_dict()}, out / "finetuned.pt")
    print(f"done  val loss {v0:.4f} -> {v1:.4f}  saved {out/'finetuned.pt'}")

if __name__ == "__main__":
    main()
