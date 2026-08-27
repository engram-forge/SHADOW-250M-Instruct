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
