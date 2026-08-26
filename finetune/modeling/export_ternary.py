import os, sys, struct
os.environ.setdefault("SHADOW_D", "1536"); os.environ.setdefault("SHADOW_NL", "10")
os.environ.setdefault("SHADOW_NH", "24"); os.environ.setdefault("SHADOW_NKV", "2")
os.environ.setdefault("SHADOW_HD", "64"); os.environ.setdefault("SHADOW_FFNH", "4224")
os.environ.setdefault("SHADOW_KV_TWO_TIER", "1")
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import RVQ, requant
from weight_formats import int4_row_pack,ternary_pack
from model_250m import Shadow250M
from export_rvq import rvq_pack

CK, OUT = sys.argv[1], sys.argv[2]; COMPACT = "--compact" in sys.argv      
ck = torch.load(CK, map_location="cpu", weights_only=False)
V, FPD = ck.get("cfg", {}).get("V", 131072), 512
MTP_HORIZON=int(ck.get("cfg",{}).get("mtp_horizon",1))
model=Shadow250M(torch.zeros(V,FPD),torch.zeros(V,FPD),V,mtp_horizon=MTP_HORIZON)
WEIGHT_DTYPE=ck.get("cfg",{}).get("ffn_weight_dtype","ternary")
common.set_ffn_qat(WEIGHT_DTYPE,ck.get("cfg",{}).get("ffn_act_qat",False),1.0)
model.load_state_dict(ck["model"], strict=False); model.eval(); requant(model)

class Wrap(torch.nn.Module):                                      
    def __init__(s, m):
        super().__init__(); s.emb=m.inp; s.b=m.b; s.step=m.struct; s.nf=m.nf; s.head=m.head; s.mtp=m.mtp; s.tb=m.tied_bias
wrap = Wrap(model)
recs = []; ntern = nrvq = 0; tern_bytes = rvq_bytes = dense_bytes = 0
rvq_ids = set()
for name, mod in wrap.named_modules():
    if isinstance(mod, RVQ):
        rvq_ids.add(id(mod))
        if mod.g == 32:                                           
            if WEIGHT_DTYPE=="ternary": packed,rs,deq=ternary_pack(mod.weight,COMPACT); kind=4 if COMPACT else 3
            else: packed,rs,deq=int4_row_pack(mod.weight); kind=6
            recs.append((name,kind,(mod.o,mod.i,packed,rs))); ntern += 1; tern_bytes += packed.nbytes + rs.nbytes
        else:
            cbT, idx, scale = rvq_pack(mod)
            recs.append((name, 1, (mod.o, mod.i, mod.g, mod.st, cbT, idx, scale))); nrvq += 1
            rvq_bytes += cbT.nbytes + idx.nbytes + scale.nbytes
for name, p in wrap.named_parameters():
    owner = name.rsplit(".", 1)[0]
    mod = dict(wrap.named_modules()).get(owner)
    if mod is not None and id(mod) in rvq_ids: continue           
    a = p.detach().float().numpy()
    if COMPACT and a.ndim >= 1 and a.size >= 4096: recs.append((name, 5, a.astype(np.float16))); dense_bytes += a.size * 2
    else: recs.append((name, 0, a)); dense_bytes += a.nbytes
for name, b in wrap.named_buffers():
    if any(x in name for x in ("cent", "cb", "initialized", "sign", "mu", "ctv", "low", "high", "updates", "inv")): continue
    a = b.detach().float().numpy(); recs.append((name, 0, a)); dense_bytes += a.nbytes

with open(OUT, "wb") as f:
    f.write(b"SHDW"); f.write(struct.pack("<II", 1, len(recs)))
    for name, kind, pay in recs:
        nb = name.encode(); f.write(struct.pack("<I", len(nb))); f.write(nb); f.write(struct.pack("<I", kind))
        if kind == 0:
            a = np.ascontiguousarray(pay, np.float32); f.write(struct.pack("<I", a.ndim)); f.write(struct.pack("<" + "I" * a.ndim, *a.shape)); f.write(a.tobytes())
        elif kind == 1:
            o, i, g, st, cbT, idx, scale = pay
            f.write(struct.pack("<IIII", o, i, g, st)); f.write(np.ascontiguousarray(cbT, np.float32).tobytes())
            f.write(np.ascontiguousarray(idx, np.uint8).tobytes()); f.write(np.ascontiguousarray(scale, np.float32).tobytes())
        elif kind == 5:
            a = np.ascontiguousarray(pay, np.float16); f.write(struct.pack("<I", a.ndim)); f.write(struct.pack("<" + "I" * a.ndim, *a.shape)); f.write(a.tobytes())
        else:
            o, i, packed, rs = pay
            f.write(struct.pack("<II", o, i)); f.write(np.ascontiguousarray(packed, np.uint8).tobytes()); f.write(rs.tobytes())
size = os.path.getsize(OUT)
print(f"loaded checkpoint ({ck.get('step','finetuned')})")
print(f"{WEIGHT_DTYPE} FFN modules {ntern}: {tern_bytes/1e6:.2f} MB | RVQ 1-bit modules {nrvq}: {rvq_bytes/1e6:.2f} MB | dense fp32: {dense_bytes/1e6:.2f} MB")
print(f".shdw = {size/1e6:.2f} MB  (+ 8.39 MB fp131072 table = {(size+8388736)/1e6:.2f} MB deploy)")

mod=model.b[0].up; wq=common.ffn_weight(mod.weight).detach().float()
_,_,deq=(ternary_pack(mod.weight,COMPACT) if WEIGHT_DTYPE=="ternary" else int4_row_pack(mod.weight))
print(f"{WEIGHT_DTYPE} round-trip max|err| vs trainer forward:",float((torch.tensor(deq)-wq).abs().max()))
