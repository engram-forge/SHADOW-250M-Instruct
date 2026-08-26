"""Capture real RoPE-transformed Q/K and V tensors from a trainer checkpoint."""

import argparse
import os
from pathlib import Path
import sys

for key,value in {"SHADOW_D":"1536","SHADOW_NL":"10","SHADOW_NH":"24",
                  "SHADOW_NKV":"2","SHADOW_HD":"64","SHADOW_FFNH":"4224",
                  "SHADOW_FAST_ATTN":"1","SHADOW_KV_BITS":"1",
                  "SHADOW_KV_TWO_TIER":"1"}.items(): os.environ.setdefault(key,value)

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"finetune"/"modeling"),str(ROOT/"shadow_runtime")]

import numpy as np
import torch
import torch.nn.functional as F

import common
from model_250m import Shadow250M
from retriever import enc


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--prompt",action="append",required=True)
    parser.add_argument("--layer",type=int,default=0)
    parser.add_argument("--max-tokens",type=int,default=256)
    parser.add_argument("--table",type=Path,default=ROOT/"deployment/fp131072.npy")
    parser.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    return parser.parse_args()


@torch.no_grad()
def main():
    args=arguments()
    if args.max_tokens<1: raise SystemExit("--max-tokens must be positive")
    if args.device=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
    device=torch.device("cuda" if args.device=="cuda" or
                        (args.device=="auto" and torch.cuda.is_available()) else "cpu")
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    cfg=checkpoint.get("cfg",{})
    packed=np.unpackbits(np.load(args.table),axis=1)[:,:512]
    cent=torch.tensor(packed.astype(np.float32)*2-1,device=device)
    common.set_ffn_qat(cfg.get("ffn_weight_dtype","ternary"),
                       cfg.get("ffn_act_qat",False),1.0)
    common.set_kv_format(cfg.get("kv_format","1bit"),cfg.get("kv_hot_tokens",128))
    model=Shadow250M(cent,F.normalize(cent,dim=-1),len(cent),
                     mtp_horizon=cfg.get("mtp_horizon",1)).to(device)
    model.load_state_dict(checkpoint["model"],strict=True); common.requant(model); model.eval()
    if not 0<=args.layer<len(model.b): raise SystemExit("--layer outside model")
    captured=[]
    for prompt in args.prompt:
        ids=enc(prompt)[-args.max_tokens:]
        if not ids: raise SystemExit("a prompt encoded to zero tokens")
        tokens=torch.tensor([ids],dtype=torch.long,device=device)
        cos,sin=common.cs(tokens.shape[1],device)
        hidden=model.inp(model.cent[tokens])
        for index,block in enumerate(model.b):
            if index==args.layer:
                q,k,v=block._qkv(block.n1(hidden),cos,sin)
                # Standardize GQA to one row per batch/KV head. Query-head groups
                # become additional query positions without duplicating K/V.
                batch,query_heads,length,width=q.shape
                groups=query_heads//common.NKV
                q=q.reshape(batch,common.NKV,groups,length,width).reshape(
                    batch*common.NKV,groups*length,width)
                k=k.reshape(batch*common.NKV,length,width)
                v=v.reshape(batch*common.NKV,length,width)
                captured.append((q.cpu(),k.cpu(),v.cpu()))
                break
            hidden=block(hidden,cos,sin)
    min_tokens=min(row[1].shape[1] for row in captured)
    groups=common.NH//common.NKV
    queries=torch.cat([row[0][:,-groups*min_tokens:] for row in captured],0).numpy().astype(np.float32)
    keys=torch.cat([row[1][:,-min_tokens:] for row in captured],0).numpy().astype(np.float32)
    values=torch.cat([row[2][:,-min_tokens:] for row in captured],0).numpy().astype(np.float32)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,queries=queries,keys=keys,values=values,
                        layer=np.int32(args.layer))
    print(f"wrote {args.out}: Q{queries.shape} K{keys.shape} V{values.shape} from layer {args.layer}")


if __name__=="__main__": main()
