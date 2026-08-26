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
