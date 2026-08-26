#!/usr/bin/env python3
import argparse, json, os, pathlib, subprocess, tempfile
import numpy as np

def run(args, tokens, group, output):
    env={**os.environ,"SHADOW_THREADS":str(args.threads),"SHADOW_FAST_LOGITS":"0"}
    if group: env["SHADOW_DOTPROD_FFN"]=str(group)
    result=subprocess.run([args.kernel,args.model,args.table," ".join(map(str,tokens)),str(args.generate),"--dump-logits",str(output)],env=env,check=True,capture_output=True,text=True)
    generated=[int(v) for v in result.stdout.split()]
    return generated,np.load(output)

def main():
    p=argparse.ArgumentParser();p.add_argument("--kernel",required=True);p.add_argument("--model",required=True);p.add_argument("--table",required=True);p.add_argument("--fixture",required=True);p.add_argument("--group",type=int,required=True);p.add_argument("--limit",type=int,default=25);p.add_argument("--generate",type=int,default=33);p.add_argument("--threads",type=int,default=1);p.add_argument("--out",required=True);a=p.parse_args()
    cases=json.loads(pathlib.Path(a.fixture).read_text())["cases"][:a.limit];rows=[]
    with tempfile.TemporaryDirectory(prefix="shadow-dotprod-") as d:
      d=pathlib.Path(d)
      for case in cases:
        bt,bl=run(a,case["tokens"],0,d/"base.npy");ct,cl=run(a,case["tokens"],a.group,d/"candidate.npy")
        prefix=next((i for i,(x,y) in enumerate(zip(bt,ct)) if x!=y),min(len(bt),len(ct)))
        first=cl[0].astype(np.float64)-bl[0].astype(np.float64);k=10
        rows.append({"id":case["id"],"tokens_equal":bt==ct,"prefix":prefix,"first_rmse":float(np.sqrt(np.mean(first*first))),"first_max_abs":float(np.max(np.abs(first))),"first_argmax_match":bool(np.argmax(bl[0])==np.argmax(cl[0])),"first_top10_overlap":len(set(np.argpartition(bl[0],-k)[-k:])&set(np.argpartition(cl[0],-k)[-k:]))/k})
        print(case["id"],rows[-1]["tokens_equal"],prefix,flush=True)
    summary={"cases":len(rows),"group":a.group,"identical_rate":sum(r["tokens_equal"] for r in rows)/len(rows),"first_argmax_rate":sum(r["first_argmax_match"] for r in rows)/len(rows),"median_prefix":float(np.median([r["prefix"] for r in rows])),"median_first_rmse":float(np.median([r["first_rmse"] for r in rows])),"median_top10_overlap":float(np.median([r["first_top10_overlap"] for r in rows]))}
    pathlib.Path(a.out).write_text(json.dumps({"summary":summary,"rows":rows},indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
