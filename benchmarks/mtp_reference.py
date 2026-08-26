"""Measure greedy K=2 MTP acceptance using exact sequential PyTorch verification."""

import argparse
import gzip
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
    source.add_argument("--jsonl",help="held-out JSONL(.gz); reads text or first user message")
    parser.add_argument("--limit",type=int,default=32)
    parser.add_argument("--cycles",type=int,default=16,help="two-token comparison cycles per prompt")
    parser.add_argument("--max-context",type=int,default=2048)
    parser.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    parser.add_argument("--out",help="optional detailed JSON output")
    return parser.parse_args()


def load_prompts(args):
    if args.prompt: return args.prompt
    if args.jsonl:
        opener=gzip.open if str(args.jsonl).endswith(".gz") else open
        raw=[]
        with opener(args.jsonl,"rt",encoding="utf-8") as stream:
            for index,line in enumerate(stream):
                if len(raw)>=args.limit: break
                record=json.loads(line); prompt=record.get("text")
                if prompt is None:
                    prompt=next((item.get("content","") for item in record.get("messages",[])
                                 if item.get("role")!="assistant"),"")
                if prompt: raw.append({"prompt":prompt,"id":f"jsonl_{index+1:05d}",
                                       "category":"held_out"})
        return raw
    raw=json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    return [({"prompt":item,"id":f"prompt_{index+1:03d}","category":"uncategorized"}
             if isinstance(item,str) else item) for index,item in enumerate(raw)]


def load_model(args,device):
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    cfg=checkpoint.get("cfg",{})
    if int(cfg.get("mtp_horizon",1))!=2:
        raise SystemExit("checkpoint is not a K=2 MTP model")
    common.set_ffn_qat(cfg.get("ffn_weight_dtype","ternary"),
                       cfg.get("ffn_act_qat",False),1.0)
    common.set_kv_format(cfg.get("kv_format","1bit"),cfg.get("kv_hot_tokens",128))
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
    if args.max_context<3: raise SystemExit("--max-context must be at least three")
    if args.device=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
    device=torch.device("cuda" if args.device=="cuda" or
                        (args.device=="auto" and torch.cuda.is_available()) else "cpu")
    model=load_model(args,device); total=AcceptanceMetrics(); rows=[]
    raw_prompts=load_prompts(args)
    if args.prompt:
        raw_prompts=[{"prompt":text,"id":f"prompt_{index+1:03d}",
                      "category":"uncategorized"} for index,text in enumerate(raw_prompts)]
    for prompt_record in raw_prompts:
        prompt=prompt_record["prompt"]
        ids=[2,8]+enc("user\n")+enc(prompt)+[9]+enc("\n")+[8]+enc("model\n")
        ids=ids[-(args.max_context-2):]
        context=torch.tensor([ids],dtype=torch.long,device=device)
        if context.shape[1]<1: raise SystemExit("a prompt encoded to zero tokens")
        generated=[]; local=AcceptanceMetrics()
        for _ in range(args.cycles):
            if context.shape[1]+2>args.max_context: break
            step=reference_k2_step(model,context)
            first=int(step.reference_first[0]); second=int(step.reference_second[0])
            eligible=step.reference_first.ne(1)&step.reference_first.ne(9)
            local.update(step,eligible); total.update(step,eligible)
            generated.append(first)
            context=torch.cat((context,step.reference_first[:,None]),1)
            if first in (1,9): break
            generated.append(second)
            context=torch.cat((context,step.reference_second[:,None]),1)
            if second in (1,9): break
        rows.append({**prompt_record,"generated_token_ids":generated,**local.as_dict()})
    categories=sorted(set(row.get("category","uncategorized") for row in rows))
    by_category={}
    for category in categories:
        subset=[row for row in rows if row.get("category","uncategorized")==category]
        sequences=sum(row["sequences"] for row in subset); first=sum(row["first_accepted"] for row in subset)
        eligible=sum(row["second_eligible"] for row in subset); second=sum(row["second_accepted"] for row in subset)
        by_category[category]={"sequences":sequences,"first_acceptance_rate":first/max(1,sequences),
                               "second_acceptance_rate":second/max(1,eligible),
                               "pair_acceptance_rate":second/max(1,sequences)}
    result={"checkpoint":str(Path(args.checkpoint).resolve()),"device":str(device),
            "k":2,"verification":"two_ordinary_sequential_greedy_forwards",
            "prompt_template":"shadow_chat_v1","stop_token_ids":[1,9],
            "max_context":args.max_context,
            **total.as_dict(),"by_category":by_category,"prompts":rows}
    if args.out:
        output=Path(args.out); output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:value for key,value in result.items() if key!="prompts"},indent=2))


if __name__=="__main__": main()
