"""Build a deterministic OASST1 + Pirate training mixture."""
import argparse
import json
import pathlib
import random

def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--oasst", required=True); parser.add_argument("--pirate", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--pirate-share", type=float, default=0.15); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(); oasst, pirate = read(args.oasst), read(args.pirate)
    if not 0 < args.pirate_share < 1: raise SystemExit("--pirate-share must be in (0, 1)")
    rng = random.Random(args.seed); rng.shuffle(oasst); rng.shuffle(pirate)
    pirate_count = round(len(oasst) * args.pirate_share / (1 - args.pirate_share))
    selected = [dict(item, metadata={**item.get("metadata", {}), "mixture_source": "oasst1"}) for item in oasst]
    for index in range(pirate_count):
        item = pirate[index % len(pirate)]
        selected.append(dict(item, metadata={**item.get("metadata", {}), "mixture_source": "pirate"}))
    rng.shuffle(selected); output = pathlib.Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected), encoding="utf-8")
    print(f"wrote {len(selected)} rows: {len(oasst)} OASST1 + {pirate_count} Pirate ({pirate_count/len(selected):.1%})")

if __name__ == "__main__": main()
