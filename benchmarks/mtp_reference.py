"""Measure greedy K=2 MTP acceptance using exact sequential PyTorch verification."""

import argparse
import json
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
from mtp_reference import AcceptanceMetrics,reference_k2_step
from retriever import enc


def arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--table",default=str(ROOT/"deployment"/"fp131072.npy"))
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt",action="append",help="plain text prefix; repeatable")
    source.add_argument("--prompts",help="JSON array of strings or records with a prompt field")
    parser.add_argument("--cycles",type=int,default=16,help="two-token comparison cycles per prompt")
    parser.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    parser.add_argument("--out",help="optional detailed JSON output")
    return parser.parse_args()


def load_prompts(args):
    if args.prompt: return args.prompt
    raw=json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    return [item if isinstance(item,str) else item["prompt"] for item in raw]


def load_model(args,device):
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    cfg=checkpoint.get("cfg",{})
    if int(cfg.get("mtp_horizon",1))!=2:
        raise SystemExit("checkpoint is not a K=2 MTP model")
    common.set_ffn_qat(cfg.get("ffn_weight_dtype","ternary"),
                       cfg.get("ffn_act_qat",False),1.0)
    packed=np.unpackbits(np.load(args.table),axis=1)[:,:512]
    cent=torch.tensor(packed.astype(np.float32)*2-1,device=device)
    model=Shadow250M(cent,F.normalize(cent,dim=-1),len(cent),mtp_horizon=2).to(device)
    model.load_state_dict({name:(value.float() if value.is_floating_point() else value)
                           for name,value in checkpoint["model"].items()})
    common.requant(model); model.eval()
    return model


def main():
    args=arguments()
    if args.cycles<1: raise SystemExit("--cycles must be positive")
    if args.device=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
    device=torch.device("cuda" if args.device=="cuda" or
                        (args.device=="auto" and torch.cuda.is_available()) else "cpu")
    model=load_model(args,device); total=AcceptanceMetrics(); rows=[]
    for prompt in load_prompts(args):
        context=torch.tensor([enc(prompt)],dtype=torch.long,device=device)
        if context.shape[1]<1: raise SystemExit("a prompt encoded to zero tokens")
        generated=[]; local=AcceptanceMetrics()
        for _ in range(args.cycles):
            step=reference_k2_step(model,context); local.update(step); total.update(step)
            first=int(step.reference_first[0]); second=int(step.reference_second[0])
            generated.extend((first,second))
            context=torch.cat((context,step.reference_first[:,None],
                               step.reference_second[:,None]),1)
        rows.append({"prompt":prompt,"generated_token_ids":generated,**local.as_dict()})
    result={"checkpoint":str(Path(args.checkpoint).resolve()),"device":str(device),
            "k":2,"verification":"two_ordinary_sequential_greedy_forwards",
            **total.as_dict(),"prompts":rows}
    if args.out:
        output=Path(args.out); output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:value for key,value in result.items() if key!="prompts"},indent=2))


if __name__=="__main__": main()
