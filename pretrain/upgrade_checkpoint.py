"""Create an explicit K=2 model-only warm start from a legacy K=1 checkpoint."""

import argparse
import hashlib
import os
from pathlib import Path
import sys

os.environ.setdefault("SHADOW_D","1536")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"finetune"/"modeling"))

import torch
from common import D
from model_250m import MTPModule


def digest(path):
    result=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): result.update(block)
    return result.hexdigest()


def upgrade(source,output,loss_weight=0.3,loss_warmup_tokens=100_000_000,seed=1337):
    checkpoint=torch.load(source,map_location="cpu",weights_only=False)
    cfg=dict(checkpoint.get("cfg",{})); old_horizon=int(cfg.get("mtp_horizon",1))
    if old_horizon!=1 or any(name.startswith("mtp.") for name in checkpoint["model"]):
        raise ValueError("source must be a legacy K=1 checkpoint without MTP tensors")
    torch.manual_seed(seed); mtp=MTPModule(D)
    state=dict(checkpoint["model"]); state.update({f"mtp.{name}":value
        for name,value in mtp.state_dict().items()})
    cfg.update(architecture_version=2,mtp_variant="a55_k2_conditioned_residual_mlp",
               mtp_horizon=2,mtp_loss_weight=float(loss_weight),
               mtp_loss_warmup_tokens=int(loss_warmup_tokens))
    payload={"version":2,"checkpoint_type":"model_only_warm_start",
             "cfg":cfg,"model":state,"provenance":{
                 "operation":"add_a55_k2_mtp","source":str(source.resolve()),
                 "source_sha256":digest(source),"mtp_seed":int(seed),
                 "optimizer_state_reused":False,"data_cursor_reused":False}}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+".tmp"); torch.save(payload,temporary); temporary.replace(output)
    return payload


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mtp-loss-weight",type=float,default=0.3)
    parser.add_argument("--mtp-loss-warmup-tokens",type=int,default=100_000_000)
    parser.add_argument("--seed",type=int,default=1337); args=parser.parse_args()
    if args.mtp_loss_weight<0 or args.mtp_loss_warmup_tokens<0: raise SystemExit("MTP loss settings must be nonnegative")
    try: payload=upgrade(args.input,args.output,args.mtp_loss_weight,args.mtp_loss_warmup_tokens,args.seed)
    except ValueError as error: raise SystemExit(str(error)) from error
    print(f"wrote K=2 model-only warm start {args.output}")
    print(f"source sha256 {payload['provenance']['source_sha256']}")
    print("start a new run with --init-checkpoint; do not use --resume")


if __name__=="__main__": main()
