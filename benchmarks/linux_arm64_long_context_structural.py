"""Compare structural recurrence scheduling at long context lengths."""
import argparse, json, os, pathlib, re, statistics, subprocess, tempfile

PREFILL = re.compile(r"prefill ([0-9.]+)s")
DECODE = re.compile(r"decode ([0-9.]+) tok/s")

def run(args, tokens, mode, dump=None):
    env = {**os.environ, "SHADOW_THREADS": str(args.threads), "SHADOW_FAST_LOGITS": "0",
           "SHADOW_PARALLEL_SCORE": "1" if mode in ("score", "both") else "0",
           "SHADOW_PARALLEL_RECALL": "1" if mode in ("recall", "both") else "0"}
    command = [args.kernel, args.model, args.table, tokens, str(args.generate), "--bench"]
    if dump: command += ["--dump-logits", str(dump)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    prefill, decode = PREFILL.search(result.stderr), DECODE.search(result.stderr)
    if not prefill or not decode: raise RuntimeError(result.stderr)
    return {"prefill_s": float(prefill.group(1)), "decode_tok_s": float(decode.group(1))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True); parser.add_argument("--lengths", nargs="+", type=int, default=[32,128,512,1024,2048])
    parser.add_argument("--threads", type=int, default=4); parser.add_argument("--generate", type=int, default=17)
    parser.add_argument("--runs", type=int, default=3); parser.add_argument("--out", required=True)
    args = parser.parse_args(); modes = ("baseline", "score", "recall", "both"); suites = []
    for length in args.lengths:
        tokens = " ".join(["2"] * length); rows = {mode: [] for mode in modes}
        for repetition in range(args.runs):
            order = modes if repetition % 2 == 0 else tuple(reversed(modes))
            for mode in order: rows[mode].append(run(args, tokens, mode))
        with tempfile.TemporaryDirectory(prefix="shadow-long-context-") as temp:
            reference = pathlib.Path(temp) / "baseline.npy"; run(args, tokens, "baseline", reference)
            reference_bytes = reference.read_bytes(); parity = {}
            for mode in modes[1:]:
                candidate = pathlib.Path(temp) / (mode + ".npy"); run(args, tokens, mode, candidate)
                parity[mode] = candidate.read_bytes() == reference_bytes
                if not parity[mode]: raise SystemExit(f"parity failed: {length} {mode}")
        summary = {mode: {"prefill_s_median": statistics.median(x["prefill_s"] for x in rows[mode]),
                          "decode_tok_s_median": statistics.median(x["decode_tok_s"] for x in rows[mode]),
                          "runs": rows[mode]} for mode in modes}
        suites.append({"context_tokens": length, "summary": summary, "logits_byte_identical": parity})
        print("length=" + str(length) + " " + " ".join(f"{m}={summary[m]['decode_tok_s_median']:.2f}" for m in modes), flush=True)
    pathlib.Path(args.out).write_text(json.dumps({"format":"shadow-linux-arm64-long-context-structural-v1",
        "threads":args.threads,"generated_tokens":args.generate,"measured_runs":args.runs,"suites":suites}, indent=2) + "\n")

if __name__ == "__main__": main()
