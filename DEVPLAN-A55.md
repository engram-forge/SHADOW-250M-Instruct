# Cortex-A55 DotProd model/QAT development plan

## Goal and constraints

Maximize batch-one autoregressive decode throughput on RK3566 Cortex-A55 while preserving the
quality benefit of the current SwiGLU architecture. The optimized path targets AArch64 AdvSIMD
with the optional DotProd feature (`asimddp`, compiler target `armv8.2-a+dotprod`). Runtime feature
detection is mandatory: ARMv8.2-A by itself must not be treated as proof that `SDOT` is available.
The existing ARMv8-A widening kernel remains the fallback.

This repository has prebuilt runtime binaries but no native engine source. It can define and test
the QAT/export contract and standalone kernels; integrating dispatch and threading into the shipped
engine requires that source. Prefill and decode are separate workloads. Decode GEMV is the first
target; prefill needs a tiled GEMM kernel and should not reuse decode conclusions blindly.

## Corrections to the initial performance hypothesis

`SDOT` performs four signed byte products per INT32 lane, but it is not a universal single-cycle
replacement for every four products, and actual latency/throughput are implementation-specific.
More importantly, one generated token streams all dense FFN weights. The current ten-layer SwiGLU
contains 194,641,920 ternary FFN weights: direct INT8 trits require about 185.6 MiB of execution
weight reads per token, while signed nibbles require about 92.8 MiB. Once arithmetic is reduced by
`SDOT`, memory bandwidth is likely to become the limiting resource. A 2.5--3.3x FFN speedup or
40--55 token/s is therefore a benchmark hypothesis, not an architectural guarantee.

## Model and training decisions

1. Keep `D=1536`, ten layers, 24 query heads, two KV heads, head dimension 64, and SwiGLU width
   4224 for the first controlled experiment. A two-matrix FFN must be widened for a fair quality
   comparison and would lose much of its traffic advantage.
2. Keep FP32 parameters and optimizer state. Use BF16 autocast by default; use FP16 autocast only
   with dynamic loss scaling. Converting FP16-only stored parameters to FP32 cannot restore lost
   precision.
3. Keep row-scaled ternary `up`, `gate`, and `down` weights. Quantize the post-RMSNorm activation
   once and share it between `up` and `gate`; quantize the post-SwiGLU activation before `down`.
4. Use symmetric per-token INT8 activation QAT with power-of-two scales. This is compatible with
   signed `SDOT`, cheap scaling, and exact INT32 accumulation.
5. Preserve base-3 release storage. At model load, repack each FFN matrix into the selected A55
   execution layout. Keep the base-3 data only if memory permits or release it after repacking.
6. Do not add sparsity unless it has a fixed structure consumed by a measured kernel. Random zeros
   reduce neither dense `SDOT` instructions nor memory reads.
7. Do not assume ternary is the best new-pretraining weight alphabet. Ternary and symmetric INT4
   consume the same signed-nibble execution bandwidth and the same `SDOT` count after unpacking.
   INT4 may buy materially better quality at equal decode speed/RAM, while ternary retains the
   smaller base-3 release file. Train both from identical seeds before choosing.

## Numerical contract

For each output row,

$$
s_r=\max(\operatorname{mean}_k|W_{r,k}|,10^{-5}),\qquad
T_{r,k}=\operatorname{clip}(\operatorname{round}(W_{r,k}/s_r),-1,1).
$$

For each token vector,

$$
\Delta_x=2^{\lceil\log_2(\max_k|x_k|/127)\rceil},\qquad
q_k=\operatorname{clip}(\operatorname{round}(x_k/\Delta_x),-127,127).
$$

The A55 kernel computes

$$
A_r=\sum_k T_{r,k}q_k\quad\text{in INT32},\qquad
y_r=s_r\Delta_x A_r.
$$

No INT16 partial sums are needed on the DotProd path. The largest FFN dimension, 4224, has the
safe bound $4224\times127=536{,}448$, far inside signed INT32. FP32 is retained for row-scale
application, SiLU, the gate product, normalization reductions, and residual addition initially.
FP16 elementwise/residual storage is a later measured ablation, not a prerequisite for `SDOT`.

## A55 decode layout

The A53 input-major layout is wrong for `SDOT`. Use every `SDOT` lane as a different output row.
Replicate four consecutive activations across the four 32-bit lanes and store one row-major
$4\times4$ weight tile. The A55 layout is:

```text
[output tile of 4 rows][input block of 4][4 rows x 4 signed weights]
```

One `SDOT` consumes the $4\times4$ weight tile and replicated activation, producing four output
partial sums directly in its four INT32 lanes. Four accumulator vectors are interleaved over
successive input blocks to hide dependency latency, then added before one vector store. There is no
horizontal reduction. Test four versus eight output rows to balance activation reuse, dependency
chains, register pressure, and prefetch distance. `up` and `gate` use the same quantized activation
and can be stored as adjacent tile streams; fuse their traversal only if the larger working stream
does not increase cache misses. Threads partition output tiles, not the input dimension.

## Execution-format candidates

| Format | FFN RAM/read per token | Decode work | Role |
|---|---:|---|---|
| Direct signed INT8 trits | 185.6 MiB | load + `SDOT` | Compute-minimal reference |
| Signed/biased nibbles | 92.8 MiB | unpack 2/byte + `SDOT` | Primary candidate |
| Interleaved 2-bit codes | 46.4 MiB | unpack 4/byte + `SDOT` | Continue only if counters win |
| Two bitplanes | 46.4 MiB | mask expansion + `SDOT` | Re-evaluate, not presumed winner |

Nibble codes use biased values `0,1,2` for `-1,0,+1`; low and high nibbles are unzipped into
signed bytes immediately before `SDOT`. Base-3 is a distribution format only: division/modulo-3
must never appear in the token hot path.

For fresh pretraining, add row-scaled signed INT4 as a separate weight-alphabet candidate. It uses
codes $[-7,7]$ (or `[-8,7]` only if the asymmetric endpoint proves useful), one FP32 scale per
output row, and the same nibble-unpack/`SDOT` kernel. Per-group scales are not the initial design:
they improve fidelity but require multiple scaled partial sums per row and complicate the hot path.
Compare ternary versus row-INT4 using loss/quality, packed RAM, actual kernel time, and release size.

## Whole-engine design

- Dispatch with Linux `getauxval(AT_HWCAP) & HWCAP_ASIMDDP`; retain the A53/portable path.
- Compile DotProd code in a separate translation unit using `-march=armv8.2-a+dotprod
  -mtune=cortex-a55`; do not make the entire binary illegal on older CPUs.
- Pin a fixed worker pool to the four A55 cores. Partition output rows, reuse workers across layers,
  and use one barrier after `up/gate` plus one after `down`. Compare one, two, and four threads: the
  memory controller may saturate before all cores do.
- Fuse RMSNorm output quantization where useful: compute FP32 sum-of-squares and maximum, normalize,
  derive one power-of-two scale, and emit INT8 once. Do not quantize `up` and `gate` separately.
- Compute `up` and `gate` INT32 projections, apply row/activation scales, then vectorize SiLU and
  multiply. Quantize that 4224-element result once for `down`.
- Keep one-bit KV storage initially. Profile attention score/value kernels separately; DotProd does
  not automatically accelerate the current binary KV representation.
- Profile RVQ Q/K/V/O and `StructStep`. Only then compare their current representation against
  per-group INT4/INT8 activation QAT. FFN optimization can shift the bottleneck to these modules.
- Use FP16 arithmetic only behind `fphp`/`asimdhp` runtime checks and only after a quality and
  throughput comparison. It is not required for the integer FFN path.

## Multi-token prediction and future speculative decode

Fresh pretraining uses `K=2`: the normal head predicts offset one and one token-conditioned
residual MLP predicts offset two. The module is `RMSNorm -> D/2 bottleneck -> SiLU -> D` and
shares the model embedding, base fingerprint head, vocabulary projection, and tied bias. At
`D=1536` it adds about 2.36M quantized matrix weights plus norm parameters. Fine-tuning preserves
the module, and export packs its two matrices using the selected FFN ternary/INT4 format.
This preserves the key DeepSeek-style token conditioning and parameter sharing, but it is not the
exact DeepSeek MTP block: the embedded target omits the extra Transformer block to control A55
draft cost and cache pressure. Export metadata records that distinction explicitly.

These heads are proposal generators, not an automatic throughput multiplier. A future native
engine must select candidates, verify multiple causal positions with the main model, accept the
longest valid prefix, and commit or roll back KV state. Measure acceptance rate and verification
cost on-device before making token/s claims. The current bundled binary does not implement this
protocol and is marked incompatible with horizon greater than one.

## Experimental TurboQuant KV baseline

TurboQuant is retained only as a precision/storage reference, not as the default A55 cache.
The linked 0xsero/turboquant repository is GPL-3.0 and is not vendored into release code;
finetune/modeling/turboquant_kv.py is an independent experiment based on the published
algorithm. Keys use a two-bit spherical Lloyd-Max stage plus an independent one-bit QJL
residual-sign sketch. Values use conventional asymmetric four-bit quantization in groups of 32.
This is two aligned bitstreams, not a packed cross-byte three-bit integer format.

At head dimension 64 the per-token, per-head payload is exact:

| Component | Bytes |
|---|---:|
| Key two-bit indices | 16 |
| Key one-bit QJL signs | 8 |
| Key FP16 norm and residual norm | 4 |
| Value four-bit codes | 32 |
| Value two FP16 scales and minima | 8 |
| **Total K+V** | **68** |

The current one-bit K+V payload is 16 bytes/token/head, so this experiment is 4.25 times larger,
though it remains 3.76 times smaller than FP16 K+V. Static random matrices and codebooks are not
charged per token.

For Cortex-A55, the costly operation is not three-bit extraction. The two-bit and one-bit streams
are cheap to unpack, but the exact QR rotation and Gaussian QJL projection each require dense
64-by-64 floating-point work. Historical attention then combines a Polar contribution and a QJL
correction. SDOT does not directly eliminate these FP32 transforms or the softmax-weighted value
accumulation. A randomized FWHT can reduce rotation complexity, but QJL still adds a second score
path and must be evaluated against an ordinary INT4 cache.

Real checkpoint tensors can be captured with benchmarks/capture_qkv.py and then passed to
benchmarks/turboquant_kv.py through its --capture option. The comparison reports reconstruction,
score correlation, top-eight overlap, attention-output error, payload bytes, and separate codec
timings for one-bit, two-bit, A55 INT4, and experimental TurboQuant. Non-A55 timing remains
diagnostic only.

The experiment is complete enough to answer the precision question with
benchmarks/turboquant_kv.py. Promotion requires captures of real Q/K/V tensors, attention-output
quality gates, and native RK3566 measurements. Unless those gates show a clear end-to-end win,
the production direction remains an exact recent FP16/INT8 tier, A55-aligned INT4 warm K/V with
QAT, and the existing one-bit cold retrieval tier.

The A55-aligned INT4 warm cache uses 74 bytes/token/head at width 64: 32 key-code bytes plus one
FP16 key scale, and 32 value-code bytes plus two FP16 scales and minima for group size 32. It is
slightly larger than the 68-byte TurboQuant experiment, but avoids dense QR/QJL transforms and
maps key scoring to nibble expansion plus signed INT8 SDOT. Its advantage must therefore be judged
by native latency and energy, not storage alone.

## Phases and gates

### Phase 1: QAT contract

- Centralize ternary and per-token INT8 power-of-two fake quantizers.
- Apply identical FFN boundaries in pretraining, fine-tuning, evaluation, and export metadata.
- Log activation saturation, zero fraction, scale range, NMSE, ternary zero fraction, and loss.
- Compare ternary-only, activation-QAT from token zero, and gradual activation-QAT warm-in from the
  same initialization/data order.
- Add a row-scaled symmetric INT4-weight/A8-activation run at the same initialization, model shape,
  optimizer, and token order. Its execution format is the same nibble layout as ternary.

Gate: CPU tests pass, gradients remain finite, and deployment integer equations match QAT values.

### Phase 2: standalone A55 decode kernels

- Implement direct INT8-trit `SDOT` with four-row and eight-row output tiles.
- Implement nibble unpack + `SDOT`, then 2-bit interleaved codes if justified.
- Read actual `up`, `gate`, and `down` tensors from `.shdw`; validate every INT32 output against
  scalar code before timing.
- Inspect generated assembly for `sdot`, spills, redundant moves, scalar tails, and failed inlining.

Gate: exact INT32 results and real RK3566 measurements using pinned cores and fixed clocks.

### Phase 3: fused FFN and multicore

- Repack adjacent `up/gate` tile streams and benchmark fused versus separate traversal.
- Add row scaling, vector SiLU/gate multiply, down-input quantization, and down projection.
- Benchmark one/two/four threads, warm and cold cache, and thermal steady state.
- Record cycles, instructions, IPC, L1/L2/last-level misses, memory bandwidth, and microseconds.

Gate: select execution format by end-to-end FFN time, not GEMV arithmetic alone.

### Phase 4: engine integration

- Add runtime `asimddp` dispatch and load-time base-3 repacking.
- Compare logits against QAT evaluation and run model-quality gates.
- Measure decode token/s at context lengths 1, 128, 512, and 2048; measure prefill separately.
- Re-profile the whole token loop and optimize the new largest consumers.

Gate: higher median steady-state decode throughput without unsupported instructions or unacceptable
quality regression.

### Phase 5: model-structure search

Only after the optimized kernel establishes milliseconds per layer, compare `FFN=4224/4096/3840`,
eight/tens/twelve-layer allocations at a fixed time budget, alternating SwiGLU/ReLU-squared blocks,
and removal/redesign of `StructStep`. Compare at fixed training tokens, storage, measured latency,
and quality. Include ternary versus row-INT4 weights at fixed nibble execution layout. Do not
optimize parameter count or release-file size in isolation.
