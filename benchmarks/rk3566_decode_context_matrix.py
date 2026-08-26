#!/usr/bin/env python3
"""Matched exact versus Compact64 decode-context matrix."""
import argparse, json, os, pathlib, re, statistics, subprocess

PREFILL=re.compile(r"prefill ([0-9.]+)s"); DECODE=re.compile(r"decode ([0-9.]+) tok/s"); RSS=re.compile(r"maximum_rss_kib=(\d+)")

def run(a,tokens,threads,mode):
    env={**os.environ,"SHADOW_THREADS":str(threads),"SHADOW_FAST_LOGITS":"0"}
    if mode=="compact64":env["SHADOW_DOTPROD_FFN"]="compact64"
    cmd=["/usr/bin/time","-f","maximum_rss_kib=%M",a.kernel,a.model,a.table,tokens,str(a.generate),"--bench"]
    p=subprocess.run(cmd,env=env,check=True,capture_output=True,text=True); pre=PREFILL.search(p.stderr);dec=DECODE.search(p.stderr);rss=RSS.search(p.stderr)
    if not pre or not dec or not rss:raise RuntimeError(p.stderr)
    return {"prefill_s":float(pre.group(1)),"decode_tok_s":float(dec.group(1)),"rss_kib":int(rss.group(1))}

def main():
    p=argparse.ArgumentParser();p.add_argument("--kernel",required=True);p.add_argument("--model",required=True);p.add_argument("--table",required=True);p.add_argument("--lengths",nargs="+",type=int,default=[32,128,512,1024,2048]);p.add_argument("--threads",nargs="+",type=int,default=[1,2,4]);p.add_argument("--generate",type=int,default=17);p.add_argument("--runs",type=int,default=3);p.add_argument("--out",required=True);a=p.parse_args();cells=[]
    for length in a.lengths:
      tokens=" ".join(["2"]*length)
      for threads in a.threads:
        rows={"exact":[],"compact64":[]}
        for repetition in range(a.runs):
          order=("exact","compact64") if repetition%2==0 else ("compact64","exact")
          for mode in order:rows[mode].append(run(a,tokens,threads,mode))
        summary={m:{"decode_tok_s_median":statistics.median(x["decode_tok_s"] for x in xs),"prefill_s_median":statistics.median(x["prefill_s"] for x in xs),"rss_kib_median":statistics.median(x["rss_kib"] for x in xs),"runs":xs} for m,xs in rows.items()}
        summary["compact64"]["decode_gain"]=summary["compact64"]["decode_tok_s_median"]/summary["exact"]["decode_tok_s_median"]-1
        cells.append({"context_tokens":length,"threads":threads,"summary":summary});print(f"context={length} threads={threads} exact={summary['exact']['decode_tok_s_median']:.2f} compact64={summary['compact64']['decode_tok_s_median']:.2f} gain={summary['compact64']['decode_gain']*100:.1f}%",flush=True)
    pathlib.Path(a.out).write_text(json.dumps({"format":"shadow-rk3566-decode-context-matrix-v1","generated_tokens":a.generate,"measured_runs":a.runs,"cells":cells},indent=2)+"\n")
if __name__=="__main__":main()
