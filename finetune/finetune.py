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
import arm_qat
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
    ap.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--val-batches",type=int,default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ul-alpha", type=float, default=0.2)
    ap.add_argument("--ul-window", type=int, default=64)
    ap.add_argument("--ul-ngram", type=int, default=3)
    ap.add_argument("--recovery-ratio", type=float, default=0.10)
    ap.add_argument("--repeat-policy", choices=("error", "drop", "warn"), default="warn")
    ap.add_argument("--overlength", choices=("error", "truncate"), default="error")
    ap.add_argument("--audit-report")
    ap.add_argument("--ffn-act-qat", action=argparse.BooleanOptionalAction, default=True,
                    help="fake-quantize both FFN activation boundaries to per-token INT8")
    ap.add_argument("--ffn-act-warmup-steps",type=int,default=0,
                    help="linearly introduce activation QAT over this many optimizer steps")
    ap.add_argument("--ffn-weight-dtype",choices=("auto","ternary","int4_row"),default="auto",
                    help="inherit the checkpoint alphabet or explicitly select it")
    ap.add_argument("--kv-format",choices=("auto","1bit","2bit","int4"),default="auto",
                    help="inherit checkpoint KV QAT or select the A55 cache format")
    ap.add_argument("--kv-hot-tokens",type=int,default=None,
                    help="exact recent KV tokens; defaults to checkpoint value")
    ap.add_argument("--mtp-loss-weight",type=float,default=None,
                    help="auxiliary loss per future head; defaults to checkpoint value")
    ap.add_argument("--mtp-loss-warmup-steps",type=int,default=0,
                    help="linearly introduce MTP loss during this fine-tune")
    ap.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16",
                    help="CUDA compute dtype; parameters and AdamW state remain FP32")
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
    """Replace an assistant prefix with a repeated span, then supervise a later continuation."""
    supervised = [i for i, value in enumerate(msk) if value]
    if len(supervised) < 24:
        return ids, msk
    answer_start, answer_end = supervised[0], supervised[-1] + 1
    width = min(rng.randint(3, 12), max(3, len(supervised) // 6))
    max_start = answer_start + max(0, len(supervised) // 3 - width)
    start = rng.randint(answer_start, max_start)
    continuation = min(answer_end, start + width + rng.randint(4, max(4, width * 2)))
    if continuation >= answer_end:
        return ids, msk
    pattern = ids[start:start + width]
    repeats = rng.randint(2, 4)
    corrupted = ids[:start] + pattern * repeats + ids[continuation:]
    # The entire context through the corruption has zero loss. Gold supervision starts on a
    # genuinely later continuation, never on another copy of the repeated span.
    mask = [0] * (start + width * repeats) + msk[continuation:]
    return corrupted, mask

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
    if a.device=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA is unavailable")
    dev=torch.device("cuda" if a.device=="cuda" or
                     (a.device=="auto" and torch.cuda.is_available()) else "cpu")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.ffn_act_warmup_steps<0: raise SystemExit("--ffn-act-warmup-steps must be nonnegative")
    if a.mtp_loss_warmup_steps<0: raise SystemExit("--mtp-loss-warmup-steps must be nonnegative")
    if a.val_batches<1: raise SystemExit("--val-batches must be positive")
    ck = torch.load(a.init, map_location=dev, weights_only=False)
    mtp_horizon=int(ck.get("cfg",{}).get("mtp_horizon",1))
    if mtp_horizon not in (1,2): raise SystemExit(f"invalid checkpoint MTP horizon {mtp_horizon}")
    mtp_loss_weight=(float(ck.get("cfg",{}).get("mtp_loss_weight",0.3))
                     if a.mtp_loss_weight is None else a.mtp_loss_weight)
    if mtp_loss_weight<0: raise SystemExit("--mtp-loss-weight must be nonnegative")
    checkpoint_weight=ck.get("cfg",{}).get("ffn_weight_dtype","ternary")
    weight_dtype=checkpoint_weight if a.ffn_weight_dtype=="auto" else a.ffn_weight_dtype
    if a.ffn_weight_dtype!="auto" and "ffn_weight_dtype" in ck.get("cfg",{}) and weight_dtype!=checkpoint_weight:
        raise SystemExit(f"checkpoint FFN dtype is {checkpoint_weight}, requested {weight_dtype}")
    checkpoint_kv=ck.get("cfg",{}).get("kv_format","1bit")
    kv_format=checkpoint_kv if a.kv_format=="auto" else a.kv_format
    kv_hot_tokens=int(ck.get("cfg",{}).get("kv_hot_tokens",128)
                      if a.kv_hot_tokens is None else a.kv_hot_tokens)
    if kv_hot_tokens<0: raise SystemExit("--kv-hot-tokens must be nonnegative")
    initial_strength=arm_qat.activation_qat_strength(0,a.ffn_act_warmup_steps,a.ffn_act_qat)
    qat_cfg=arm_qat.configure(a.ffn_act_qat,weight_dtype,initial_strength)
    common.set_kv_format(kv_format,kv_hot_tokens)
    amp_dtype,amp_enabled,scaler=arm_qat.autocast_and_scaler(dev,a.amp_dtype)
    fp = np.unpackbits(np.load(a.table), axis=1)[:, :512]
    cent = torch.tensor(fp.astype(np.float32) * 2 - 1, device=dev); cent_n = F.normalize(cent, dim=-1)
    model=Shadow250M(cent,cent_n,cent.shape[0],mtp_horizon=mtp_horizon).to(dev)
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
    def loss_of(x,y,mtp_weight,include_ul=True,return_metrics=False):
        h,_=model.trunk(x)
        metrics=model.language_model_metrics(h,y,mtp_weight,chunk=8192,
                                             conditioning_ids=x[:,1:])
        mle=metrics["loss"]
        if not include_ul or a.ul_alpha == 0:
            return (mle,metrics) if return_metrics else mle
        pairs = unlikelihood_pairs(x, y, a.ul_window, a.ul_ngram)
        if not pairs: return (mle,metrics) if return_metrics else mle
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
        total=mle+a.ul_alpha*ul
        return (total,metrics) if return_metrics else total
    @torch.no_grad()
    def val():
        training_strength=common.FFN_ACT_QAT_STRENGTH
        common.set_ffn_activation_qat_strength(1.0 if a.ffn_act_qat else 0.0)
        model.eval(); tot = 0.0
        try:
            for _ in range(a.val_batches):
                x, y = data.pack(a.micro_batch, val=True)
                with torch.autocast(dev.type,dtype=amp_dtype,enabled=amp_enabled):
                    tot += float(loss_of(x.to(dev),y.to(dev),mtp_loss_weight,include_ul=False))
        finally:
            model.train()
            for md in model.modules():
                if isinstance(md, common.KVCodec1): md.eval()
            common.set_ffn_activation_qat_strength(training_strength)
        return tot / max(1,a.val_batches)
    v0 = val(); print(f"step 0  val loss {v0:.4f}")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        common.set_ffn_activation_qat_strength(arm_qat.activation_qat_strength(
            step,a.ffn_act_warmup_steps,a.ffn_act_qat))
        lr = a.lr * min(1.0, step / a.warmup) * (0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        current_mtp_weight=(mtp_loss_weight if a.mtp_loss_warmup_steps<=0 else
                            mtp_loss_weight*min(1.0,step/a.mtp_loss_warmup_steps))
        latest_metrics=None
        metric_sums={key:torch.zeros((),device=dev) for key in
                     ("base_loss_sum","base_correct","mtp_loss_sum","mtp_correct")}
        base_metric_tokens=mtp_metric_tokens=0
        for _ in range(a.accum):
            x, y = data.pack(a.micro_batch)
            with torch.autocast(dev.type,dtype=amp_dtype,enabled=amp_enabled):
                total,latest_metrics=loss_of(x.to(dev),y.to(dev),current_mtp_weight,
                                             return_metrics=True)
                loss=total/a.accum
            scaler.scale(loss).backward()
            with torch.no_grad():
                metric_sums["base_loss_sum"]+=latest_metrics["base_loss"].detach()*latest_metrics["base_tokens"]
                metric_sums["base_correct"]+=latest_metrics["base_accuracy"].detach()*latest_metrics["base_tokens"]
                metric_sums["mtp_loss_sum"]+=latest_metrics["mtp_loss"].detach()*latest_metrics["mtp_tokens"]
                metric_sums["mtp_correct"]+=latest_metrics["mtp_accuracy"].detach()*latest_metrics["mtp_tokens"]
                base_metric_tokens+=latest_metrics["base_tokens"]
                mtp_metric_tokens+=latest_metrics["mtp_tokens"]
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); requant(model)
        if step % a.log_every == 0:
            el = time.time() - t0
            values=torch.stack(tuple(metric_sums.values())).cpu().tolist()
            print(json.dumps({"step":step,"lr":lr,"seconds_per_step":el/step,
                "eta_minutes":(a.steps-step)*el/step/60,
                "base_loss":values[0]/max(1,base_metric_tokens),
                "base_accuracy":values[1]/max(1,base_metric_tokens),
                "mtp_loss":values[2]/max(1,mtp_metric_tokens),
                "mtp_accuracy":values[3]/max(1,mtp_metric_tokens),
                "mtp_loss_weight":current_mtp_weight}),flush=True)
    v1 = val()
    common.set_ffn_activation_qat_strength(1.0 if a.ffn_act_qat else 0.0)
    torch.save({"model": model.state_dict(),"cfg":{
        "V":cent.shape[0],**qat_cfg,"training_amp_dtype":a.amp_dtype,
        "kv_format":kv_format,"kv_hot_tokens":kv_hot_tokens,
        "kv_key_scale":"per_token_head_symmetric_fp16" if kv_format=="int4" else None,
        "kv_value_scale":"group32_asymmetric_fp16" if kv_format=="int4" else None,
        "parameter_dtype":"float32","ffn_act_warmup_steps":a.ffn_act_warmup_steps,
        "architecture_version":2,
        "mtp_variant":"a55_k2_conditioned_residual_mlp" if mtp_horizon==2 else "none",
        "mtp_horizon":mtp_horizon,"mtp_loss_weight":mtp_loss_weight,
        "mtp_loss_warmup_steps":a.mtp_loss_warmup_steps}},out/"finetuned.pt")
    print(f"done  val loss {v0:.4f} -> {v1:.4f}  saved {out/'finetuned.pt'}")

if __name__ == "__main__":
    main()
