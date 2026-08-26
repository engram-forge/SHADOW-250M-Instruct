# SHADOW quantization tutorial

This tutorial traces the quantization code in `finetune/modeling/` from
trainable tensors to deployed bytes. It is divided by mechanism so that the
three meanings of “one bit” do not get mixed together.

## Reading order

1. [System map and quantization-aware training](01-system-map.md)
2. [Residual vector quantization (RVQ)](02-rvq.md)
3. [Ternary FFN weights](03-ternary.md)
4. [One-bit and two-bit KV codecs](04-kv-codecs.md)
5. [Hot cache, cold archive, and retrieval](05-cache-and-retrieval.md)
6. [`.shdw` export and storage accounting](06-export-format.md)

## The terminology trap

| Phrase | Object | Actual meaning |
|---|---|---|
| RVQ “1-bit” | projection weights | two 4-bit indices per group of 8 weights, or $2\times4/8=1$ index bit/weight |
| ternary / “2-bit” | FFN weights | values $\{-s,0,+s\}$, first held in 2-bit symbols and finally packed at 1.6 bits/weight |
| 1-bit KV | cached K and V activations | one calibrated binary decision per transformed scalar |
| 2-bit KV | cached K and V activations | one of four levels per transformed scalar, plus a vector scale |

RVQ is vector codebook quantization. Ternary weights are scalar quantization. KV
quantization compresses dynamic activations rather than model parameters.

## Source map

| Topic | Primary implementation |
|---|---|
| STE, RVQ, KV codecs, attention | `finetune/modeling/common.py` |
| model wiring and tensor flow | `finetune/modeling/model_250m.py` |
| RVQ byte layout | `finetune/modeling/export_rvq.py` |
| ternary conversion and records | `finetune/modeling/export_ternary.py` |
| base-3 and FP16 repacking | `finetune/modeling/repack_shdw.py` |
| paged cold KV storage | `finetune/modeling/paged_kv.py` |
| training-time overrides | `finetune/finetune.py` |

