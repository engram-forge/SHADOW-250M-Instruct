# RK3566 / Cortex-A55 optimization plan

The deployment target is now the Radxa ZERO 3W running Ubuntu ARM64. Its RK3566
CPU has four Cortex-A55 cores. WSL2 on the Surface Laptop remains the development
environment; its timings are not Radxa performance claims. Confirm the production
image with `lscpu` and `/proc/cpuinfo` before enabling optional ISA paths. In
particular, require the Linux `asimddp` feature before running DotProd binaries.

## Build baselines

The default Linux ARM64 build targets `-mcpu=cortex-a55` without DotProd. Build
the optional candidate separately:

```bash
native/build_linux_arm64.sh
SHADOW_ARM_DOTPROD=ON native/build_linux_arm64.sh build/linux-arm64-dotprod/shadow
```

Do not use `-march=native`. Keep the non-DotProd binary as a compatibility and
exactness baseline. A DotProd binary must never be distributed to a CPU whose
feature flags have not been checked.

## Revised priority order

1. Rebuild and benchmark the existing exact signed-nibble/FP32 engine for A55.
   This isolates core, clock, cache, and compiler effects from algorithm changes.
2. Implement an isolated `sdot` ternary GEMV benchmark with dynamically quantized
   INT8 activations. Test per-tensor, group-128, then group-64 scales. Include
   quantization and dequantization time, signed-INT8 expanded weight RSS, and a
   compact-nibble-to-`sdot` candidate.
3. Gate approximate kernels with generation evaluation, not operator RMSE alone:
   first-step argmax/top-10 overlap, first-step logit RMSE, matching token prefix,
   complete sequence equality, and task score. The previous per-tensor FFN trial
   showed that about 0.4% operator relative RMSE can still alter trajectories.
4. If quality passes, integrate FFN `up/gt/dn` first, then projection matrices.
   Preserve FP32 embedding, normalization, attention scores/softmax, recurrence,
   and logits until independently justified.
5. Revisit FP16 only through a measured A55 kernel. FP16 support does not imply a
   2x end-to-end gain, and model weights are ternary rather than FP16.
6. Run prefill lengths 4/16/64/256 and decode contexts
   32/128/512/1024/2048 with 1/2/4 threads on the physical Radxa. Record clocks,
   temperature, throttling, RSS, TTFT, and tokens/s.

## Performance expectations

Do not treat projected 1.8--2.2x gains as established results. DotProd only helps
when both operands use an integer formulation; it does not accelerate the current
FP32-activation nibble kernel automatically. End-to-end gain is bounded by the
accelerated profile share and by quantization, unpacking, memory traffic, and
quality-preserving group overhead. Physical-board measurements are authoritative.

The 1-bit archive path remains valid: its ARM64 CPU implementation already uses
NEON XOR, byte population count, and pairwise accumulation. It should be profiled
again on A55 but does not require DotProd.

## Initial DotProd operator gate

The ternary INT8 activation microbenchmark now contains a genuine `sdot` kernel,
confirmed by disassembly. Timing includes dynamic quantization, integer GEMV,
per-group FP32 conversion, and output accumulation. The isolated layout expands
weights to signed INT8 and uses 6.188 MiB per tested matrix, twice the nibble layout.

| Shape | Group | FP32 | DotProd total | Speedup | Relative RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| up/gt | full | 534.5 us | 102.6 us | 5.21x | 0.393% |
| up/gt | 128 | 534.5 us | 94.4 us | 5.67x | 0.388% |
| up/gt | 64 | 534.5 us | 97.4 us | 5.49x | 0.391% |
| down | full | 530.0 us | 108.8 us | 4.87x | 0.396% |
| down | 128 | 530.0 us | 102.6 us | 5.17x | 0.411% |
| down | 64 | 530.0 us | 107.2 us | 4.94x | 0.408% |

These are Surface ARM64 WSL development measurements, not RK3566 measurements.
The candidate clears the operator-speed gate, but runtime integration remains
blocked on real-activation error and full generation quality. Group-128 is the
first candidate; group-64 continues only if real activation outliers justify it.

### Real FFN activation audit

An exact diagnostic run captured 140 FFN inputs across seven evaluated tokens and
all ten transformer layers. Capture does not alter computation. Activation-only
quantize/dequantize error was:

| Stage | Group | Median relative RMSE | p95 |
| --- | ---: | ---: | ---: |
| up/gate input | full | 1.333% | 2.610% |
| up/gate input | 128 | 0.806% | 1.249% |
| up/gate input | 64 | 0.685% | 0.979% |
| down input | full | 4.445% | 7.949% |
| down input | 128 | 1.668% | 2.070% |
| down input | 64 | 1.294% | 1.527% |

Real FFN activations have materially stronger outliers than the synthetic uniform
operator input. This rejects per-tensor quantization and justifies continuing
group-64 as the quality-first runtime experiment. Group-128 remains the speed
control. Neither is accepted until sequence-level generation evaluation passes.

### Guarded runtime generation gate

The DotProd build now exposes `SHADOW_DOTPROD_FFN=64|128`; exact FP32 remains the
default. A four-thread lifetime defect found during qualification was fixed by
using per-call quantization buffers that remain alive and read-only throughout
worker dispatch. The earlier main-thread `thread_local` buffers were unsafe for
worker access.

On the first ten fixture prompts, group-128 with 17 generated tokens achieved
80% complete token-sequence equality, 90% first-token argmax agreement, median
matching prefix 17, median first-step logit RMSE 0.165, and median top-10 overlap
1.0. This is promising approximate behavior but fails strict parity, so the path
remains opt-in and cannot replace the exact default. Full fixture and task-score
evaluation are required before it can be described as quality-qualified.

### Baseline decision

Group-64 is the baseline for all future approximate DotProd FFN experiments. On
the same 25-case, 33-token comparison it retained 36% complete sequences and a
median 19-token matching prefix, versus 28% and 15 tokens for group-128. Both
had 96% first-token agreement and full median top-10 overlap. Group-128 had the
lower median first-step logit RMSE (0.179 versus 0.201), but group-64 preserved
the autoregressive trajectory better, which is the more relevant quality signal.

This does not make group-64 the production default: exact FP32 remains default.
Group-128 is retained only as a speed control. New compact-weight kernels,
calibration, selective-layer quantization, and physical-board measurements must
compare against group-64 first.

### Full 472-case generation result

Both candidates were evaluated with four threads and 33 generated tokens on all
472 fixture prompts. Group-64 wins every differentiating sequence-quality metric.

| Metric | Group-64 | Group-128 |
| --- | ---: | ---: |
| Complete sequence equality | 42.58% | 37.29% |
| First-token argmax agreement | 96.82% | 95.55% |
| Median matching prefix | 24/33 | 20.5/33 |
| Median first-step logit RMSE | 0.1598 | 0.1712 |
| Median first-step top-10 overlap | 100% | 100% |

This confirms group-64 as the approximate DotProd quality baseline. It still does
not satisfy strict parity and remains opt-in. The next required measurement is a
matched exact/group-64/group-128 throughput and RSS matrix; earlier standalone
smoke measurements are not sufficient for a release claim.

### A55 compact bit-plane reconsideration

The previous exact two-bitplane kernel was 32--38% slower than signed nibbles on
the development host when compiled for A53. Cortex-A55 may change instruction
scheduling and memory behavior, so one isolated retest is justified, but neither
the 52 MiB model estimate nor `bandwidth / model size` predicts realized tokens/s.
Decode also reads RVQ and dense weights and performs representation decode, FP32
accumulation, attention, recurrence, logits, synchronization, and repeated cache
traffic. There is no demonstrated mechanism that loads only "active" ternary
weights: dense ternary GEMV consumes every weight unless the model gains trained
structured sparsity or a separate conditional-computation design.

Retest only the existing exact bit-plane operator at real up/gate, down, paired,
and batch-4 shapes using the A/B/C rule in `AGENTS.md`:

1. A: exact signed-nibble baseline, group quantization disabled.
2. B: exact bit-plane candidate, group quantization disabled. Require exact output
   and at least 1.5x isolated improvement before any runtime integration.
3. C: only if B passes, design a compact bit-plane-to-DotProd group-64 kernel and
   compare its total speed, RSS, and 472-case quality with expanded-INT8 group-64.

If B again loses, close the exact bit-plane route. A compact approximate kernel is
still independently valuable because it could remove the current doubled INT8
weight working set, but it must be judged as a new decode algorithm, not as an
automatic consequence of A55 memory bandwidth.

### Matched throughput/RSS and exact bit-plane decision

With the same binary, prompt, 65-token decode, four threads, three warmups, and 20
runs, the WSL development medians were:

| Path | Decode tok/s | Gain vs exact | Median RSS | 472-case equality |
| --- | ---: | ---: | ---: | ---: |
| Exact FP32/nibble | 114.27 | baseline | 176.0 MiB | 100% by definition |
| DotProd group-64 | 142.15 | +24.4% | 357.6 MiB | 42.58% |
| DotProd group-128 | 135.96 | +19.0% | 357.6 MiB | 37.29% |

Group-64 wins both approximate quality and median throughput. Its memory cost is
material: signed-INT8 FFN expansion adds about 181.6 MiB RSS. Group-128 also had
far higher throughput spread (50.8% versus 21.8% group-64 and 17.0% exact). These
are WSL development results, not RK3566 claims.

The existing exact bit-plane operator was rebuilt for Cortex-A55 with DotProd
disabled. It remained bit-identical but measured 870.9 us versus 572.3 us for
signed nibbles: 34.3% slower. It fails the 1.5x gate decisively, reproducing the
earlier A53 conclusion. The exact bit-plane runtime route is closed. Future work
may still test a fundamentally different compact bit-plane-to-DotProd group-64
algorithm because its purpose is to recover memory from the approximate INT8 path,
not to replace the exact nibble kernel.

### Compact bit-plane-to-DotProd operator result

An isolated group-64 kernel packs four ternary values into one byte, expands a
16-row by 4-column tile in NEON registers, and immediately consumes it with
`sdot`. Its output is identical to the expanded-INT8 group-64 operator.

| Shape | Expanded INT8 | Compact 2-bit | Compact / FP32 | Weight memory |
| --- | ---: | ---: | ---: | ---: |
| up/gate | 97.2 us | 190.3 us | 2.62x | 1.547 vs 6.188 MiB |
| down | 115.0 us | 191.3 us | 2.64x | 1.547 vs 6.188 MiB |

The compact layout removes 75% of the expanded weight memory but is 1.66--1.96x
slower because shifts, masks, subtracts, and zips sit in the inner loop. It does
not retain most of expanded DotProd throughput. One guarded whole-runtime test is
still justified to measure the actual RSS/speed tradeoff; reject it if memory does
not approach the exact runtime or end-to-end decode loses nearly all group-64 gain.

### Compact64 guarded runtime result

`SHADOW_DOTPROD_FFN=compact64` stores only the packed 2-bit FFN layout and expands
each 16-row by 4-column tile immediately before `sdot`. Its generated tokens and
all dumped logits were bit-identical to expanded group-64 in the smoke parity run,
so it inherits group-64's measured approximation boundary rather than adding a
second numerical change.

The same-binary matched rerun measured:

| Path | Decode median | Gain vs exact | Median RSS | Spread |
| --- | ---: | ---: | ---: | ---: |
| Exact | 110.53 tok/s | baseline | 171.9 MiB | 7.0% |
| Expanded group-64 | 148.01 tok/s | +33.9% | 357.6 MiB | 18.9% |
| Compact group-64 | 164.42 tok/s | +48.8% | 218.4 MiB | 7.3% |

Although compact unpack is slower in the isolated cache-friendly operator, the
whole runtime is faster on this WSL host, consistent with the much smaller weight
working set reducing memory/cache pressure. Compact64 saves about 139 MiB versus
expanded group-64 and retains only about 46.5 MiB overhead versus exact. This is
development evidence, not an RK3566 claim; physical-board context/thread matrices
remain mandatory. Compact64 becomes the preferred approximate candidate, while
expanded group-64 remains its numerical and diagnostic reference.
### Adaptive thread scheduling result

An exact scheduling-only candidate used one thread below 256 rows, two threads
below 1024 rows, and the configured maximum otherwise. Logits were byte-identical,
but its isolated 20-run A/B gate measured 90.70 tok/s versus 101.87 tok/s control
(11.0% slower), with 68.5% versus 22.7% spread. The extra pool and runtime switch
were fully removed. The existing full-pool dispatch remains preferred.

Linux affinity is deferred to physical RK3566 testing. WSL virtual CPU numbering,
host scheduling, and Snapdragon topology do not represent the four homogeneous A55
cores, and earlier WSL affinity tests increased variance. On-board qualification
should compare unpinned versus CPUs 0-3 pinned, then individual 1/2/4-core masks,
without combining affinity with another unqualified optimization.
### Manual attention value NEON result

An exact token-order-preserving NEON FMA loop for the 64-element attention value
accumulation produced byte-identical logits but regressed decode throughput by
4.8% at context 32, 3.1% at 512, and 2.3% at 2048. The compiler already handles
the simple scalar inner loop effectively; explicit load/FMA/store did not help.
The runtime switch and kernel were removed. The next attention experiment must
change data layout or fuse score/value traversal rather than restating this loop
with intrinsics.
### Head-major KV cache result

The exact KV cache layout changed from token-major `[token][kv-head][64]` to
head-major `[kv-head][token][64]`. Attention scans one KV head across every token,
so the new layout removes the unused other-head row between consecutive accesses.
Token order and arithmetic are unchanged; context-512 logits were byte-identical
against the pre-change binary. Alternating five-run exact measurements showed:

| Context | Token-major | Head-major | Gain |
| ---: | ---: | ---: | ---: |
| 32 | 117.53 tok/s | 118.89 tok/s | +1.2% |
| 512 | 67.09 tok/s | 70.56 tok/s | +5.2% |
| 2048 | 35.16 tok/s | 39.88 tok/s | +13.4% |

The gain grows with context as expected for a locality improvement. Retain the
layout as an exact optimization and rerun it with Compact64 as the C integration
control. Ring-buffer replacement remains separate future work for contexts that
advance beyond the 2048-entry cap.
### Shared multi-query key scoring result

For each of two KV heads, 12 query heads scan the same key cache. The exact shared
path loads each 64-float key row once and updates 12 independent FP32 accumulators,
preserving every head's dot-product reduction and token order. Context-512 logits
were byte-identical. Alternating exact measurements showed -1.6% at context 32,
+2.8% at 512, +14.9% at 1024, and +6.8% at 2048. The implementation therefore
activates automatically only at 1024 or more cached tokens; the existing per-head
path remains below that threshold. Cold archive attention retains the established
path because its per-head shortlist may differ.

### FP16 KV cache result

An opt-in experiment stored the already-BF16-rounded K/V cache in FP16 and
converted it to FP32 for attention. Context-32 and context-1024 logits were
byte-identical in the parity fixtures, but conversion overhead outweighed the
smaller cache at every measured long context on the WSL development host:

| Context | FP32 decode | FP16 decode | Change | FP32 RSS | FP16 RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 8.65 tok/s | 8.39 tok/s | -3.0% | 179.8 MiB | 174.7 MiB |
| 1024 | 4.59 tok/s | 4.49 tok/s | -2.2% | 194.4 MiB | 185.6 MiB |
| 2048 | 2.47 tok/s | 2.20 tok/s | -10.8% | 225.4 MiB | 207.1 MiB |

At context 2048, prefill also regressed 14.2%. The maximum memory saving was
about 18.3 MiB, insufficient to justify slower inference and extra cache paths.
The experiment was removed; FP32 KV storage remains the baseline. These are WSL
development results and should not be interpreted as RK3566 measurements.

### Mixed-FP16 attention result

A second experiment used FP16 Q/K/V multiplication with FP32 dot-product and
value accumulation; score scaling, flooring, softmax, normalization, and output
remained FP32. It required the optional `asimdfhm`/FP16FML extension and retained
an FP32 fallback. Isolated corrected operator tests on the Surface WSL host found
shared-MQA Q/K 1.23--1.29x faster and a channel-blocked FP16 V layout 1.86--1.99x
faster than its same-layout FP32 control. FP16 V in the original token-major
layout remained slower.

The whole-runtime gate did not preserve those gains. After removing per-head
allocation, vectorizing probability conversion, and restoring shared-MQA scans,
context-1024 exact decode measured 67.45 tok/s versus 77.95 tok/s for FP32
(-13.5%). Median RSS fell only from 194.8 MiB to 185.6 MiB. Short-prompt logits
and generated tokens were identical, but the performance gate failed before a
full quality matrix was justified. Batch-4 prefill was intentionally disabled in
the experimental path and therefore its slower TTFT was not used as acceptance
evidence. All runtime and microbenchmark code was removed.

This result does not rule out a board-specific assembly kernel, but another retry
must first explain the operator-to-runtime gap and demonstrate an end-to-end win
on physical RK3566 hardware. Do not enable FP16 attention merely because Cortex-A55
supports FP16FML.

### Fair batch-4 FP16 prefill result

The prefill question was retested without the earlier scheduling mismatch. Both
control and candidate used the same batch-4 scheduler and FP32 KV layout; only
Q/K and probability/V multiplication changed to FP16FML, with FP32 accumulation
and FP32 softmax. Final prompt logits were byte-identical for every tested length.
Three-run WSL medians were:

| Threads | Length | FP32 | Mixed FP16 | Change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 59.28 tok/s | 58.50 tok/s | -1.3% |
| 1 | 16 | 59.46 tok/s | 57.37 tok/s | -3.5% |
| 1 | 64 | 58.86 tok/s | 56.43 tok/s | -4.1% |
| 1 | 256 | 56.49 tok/s | 55.51 tok/s | -1.7% |
| 2 | 4 | 100.44 tok/s | 78.06 tok/s | -22.3% |
| 2 | 16 | 98.17 tok/s | 79.04 tok/s | -19.5% |
| 2 | 64 | 96.36 tok/s | 87.36 tok/s | -9.3% |
| 2 | 256 | 89.01 tok/s | 97.72 tok/s | +9.8% |
| 4 | 4 | 161.15 tok/s | 147.86 tok/s | -8.2% |
| 4 | 16 | 178.17 tok/s | 162.46 tok/s | -8.8% |
| 4 | 64 | 181.97 tok/s | 166.89 tok/s | -8.3% |
| 4 | 256 | 139.71 tok/s | 150.57 tok/s | +7.8% |

RSS was effectively unchanged because cache representation was deliberately held
constant. The candidate failed the no-regression rule and the 15% gain gate at
lengths 64/256. Compact64 integration was not run because weight format cannot
remove this isolated attention regression. The FP16 prefill code and benchmark
switch were removed; FP32 remains the accepted prefill path.

### FP16 range audit

Temporary aggregate instrumentation measured 44 stratified cases from the
472-case fixture: 32 evenly distributed cases plus the 16 longest prompts,
deduplicated to 44 prompts and 3,475 input tokens (maximum length 144).

| Stage | Values | Range | Overflow | Zero underflow | Relative RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q cache-ready | 53.4M | -8.875 to 7.562 | 0 | 0 | 0 |
| K cache-ready | 4.45M | -37.5 to 35 | 0 | 0 | 0 |
| V cache-ready | 4.45M | -121 to 115 | 0 | 0 | 0 |
| Q/K/V projected | 62.3M | -240.1 to 252.6 | 0 | 0 | ~2.08e-4 |
| FFN input | 53.4M | -5.179 to 6.93 | 0 | 2 | ~2.07e-4 |
| FFN product | 146.8M | -295.5 to 262 | 0 | 119 | ~2.07e-4 |
| Residual after attention | 53.4M | -329.6 to 2381 | 0 | 0 | ~2.21e-4 |
| Residual after FFN | 53.4M | -670.9 to 2446 | 0 | 1 | ~2.05e-4 |
| Logits | 455.5M | -75.62 to 57.58 | 0 | 0 | ~2.09e-4 |

Cache-ready Q/K/V are power-of-two quantized and BF16-rounded; every observed
value was exactly representable in FP16. Residuals retained about 26.8x range
headroom below FP16 maximum finite value 65,504. Rare zero-underflows affected
only extremely small FFN/residual values and do not prove quality safety.

Decision: FP16 storage is range-safe for measured Q/K/V and is a reasonable
future A55 storage baseline. Keep RMS statistics, attention reductions, softmax,
residual accumulation, structural recurrence, and logits FP32 initially. FFN and
residual FP16 storage remain approximate candidates requiring full 472-case
generation validation. Range safety does not override the rejected WSL speed
results; persistent FP16 storage should next be measured on physical RK3566.

### Coherent opt-in FP16 QKV island

`SHADOW_FP16_QKV=1` converts Q once, K/V once at insertion, and directly consumes
stored FP16 operands with FP16FML and FP32 accumulation. Softmax stays FP32 and
probabilities convert once for P/V FML. Sequential decode and batch-4 prefill
share the representation; Compact64 is compatible. Cold archives retain FP32.

| Context | Exact FP32 | Exact FP16 | Gain |
| ---: | ---: | ---: | ---: |
| 32 | 127.87 tok/s | 131.25 tok/s | +2.6% |
| 128 | 119.61 tok/s | 122.50 tok/s | +2.4% |
| 512 | 100.42 tok/s | 108.54 tok/s | +8.1% |
| 1024 | 68.29 tok/s | 84.65 tok/s | +24.0% |
| 2048 | 52.18 tok/s | 59.00 tok/s | +13.1% |

At context 1024, exact prefill improved 25.6% and median RSS fell about 9.2 MiB.
Compact64 improved 16.3% decode and 10.1% prefill at context 1024; at context 512
decode improved 14.7%. These are three-run WSL development results.

The 16-case/17-token screen produced 15/16 complete-sequence equality, 16/16
first-token agreement, 98.53% stepwise argmax agreement, and median logit RMSE
5.30e-6. One divergent trajectory reached RMSE 2.33.

The full 472-case/17-token evaluation produced 451/472 (95.55%) identical
sequences and 471/472 (99.79%) first-token argmax agreement. Median identical
prefix was all 17 tokens, median first-step RMSE was zero, and median first-step
top-10 overlap was 1.0. The raw result is in
`rk3566_fp16_qkv_generation.json`. This is strong approximate-quality evidence,
but any divergence means the mode remains opt-in. The physical RK3566 performance
matrix remains required before considering it as a default.

### Rejected FP16 RVQ projection inputs

An isolated `SHADOW_FP16_PROJECTION_INPUT` experiment converted the FP32 RMSNorm
output once and reused it across Q/K/V, while mirroring RVQ codebooks as FP16 and
using FP16FML with FP32 lookup/output accumulation. It was removed after the
matched three-run, four-thread WSL screen: decode regressed 9.6% at context 32,
7.3% at context 512, and 7.8% at context 1024. Prefill also regressed about 1--2%,
and the generated trajectory diverged (full-logit RMSE 7.64 in the screen).

RVQ codebooks are small and heavily reused, so FP16 codebook loads do not remove
a material bandwidth bottleneck. Conversion, scalar FP16 broadcasts, and FP16FML
lookup construction cost more than the existing four-way FP32 kernel. Do not
retry persistent FP16 hidden/projection inputs without a materially different RVQ
layout or physical-board evidence. The accepted FP16 QKV cache island is
unaffected.

### Shared long-context FP16 MQA key scan

The FP16 QKV path now loads each MQA key once and evaluates all 12 query heads
sharing that KV head with independent FP32 accumulators. It activates only from
1536 cached tokens; shorter contexts retain the lower-register-pressure per-head
kernel. Logits and generated tokens were byte-identical to the prior FP16 QKV
path in the context-512 check.

At context 2048 and four threads, seven interleaved WSL runs improved median
decode from 57.37 to 60.71 tok/s (+5.8%); the shared scan won six of seven paired
runs. Initial context-512 and context-1024 screens showed no benefit, which is why
this is thresholded. These are Surface WSL development results; confirm the
threshold and gain on physical RK3566.

### Compact64 fused batch-4 prefill

Compact64 prefill now quantizes four normalized token states independently,
unpacks each packed ternary weight tile once, and reuses it across four `sdot`
streams. FP32 group scaling and output accumulation preserve the qualified
group-64 numerical definition. The fused path produced byte-identical logits and
tokens versus four sequential Compact64 calls.

Three-run WSL medians against exact batch-4 prefill were uniformly positive:

| Threads | 4 tokens | 16 tokens | 64 tokens | 256 tokens |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +31.5% | +28.2% | +30.5% | +29.8% |
| 2 | +22.6% | +31.5% | +38.7% | +26.9% |
| 4 | +25.2% | +28.6% | +22.1% | +26.6% |

At four threads the resulting Compact64 throughput was 236.04, 253.30, 250.28,
and 235.14 prompt tok/s respectively. Raw measurements are in
`rk3566_compact64_batch4_prefill_wsl.json`. These are Surface WSL development
results; repeat the matrix on the physical RK3566 before publishing board claims.

### Compact64 fused up/gate decode

The paired up/gate path now dynamically quantizes their shared FFN input once and
dispatches both Compact64 matrices together. Each matrix retains its own packed
weights, FP32 group scaling, and output, so logits and generated tokens were
byte-identical to two separate Compact64 matvec calls.

At context 128 with four threads and 65 generated tokens, seven interleaved WSL
runs improved median decode from 167.63 to 175.24 tok/s (+4.5%). The fused path
won all seven pairs. This gain is additional to Compact64 itself and comes from
removing duplicate activation quantization and one worker dispatch. Confirm on
physical RK3566 before treating it as a board result.

### Vectorized Compact64 activation quantization

Group-64 peak detection and float-to-INT8 rounding now use ARM NEON in DotProd
builds. The conversion uses round-to-nearest-even, matching `nearbyint`; clamping,
group scales, and all downstream arithmetic are unchanged. Generated tokens and
dumped logits were byte-identical to the scalar quantizer.

Seven-run, four-thread WSL medians improved context-128 decode from 173.04 to
176.62 tok/s (+2.1%). Length-256 batch-4 prefill time fell from 1.1358 to 1.1131
seconds (+2.0% throughput). This is a small numerical-equivalent gain layered on
the fused Compact64 kernels; confirm it on physical RK3566.
