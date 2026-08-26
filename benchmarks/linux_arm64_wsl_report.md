# Linux ARM64 WSL pre-board qualification

## Environment

- Host: ARM64 Windows laptop through WSL2, Ubuntu 24.04.2, 8 Qualcomm cores.
- Runtime: GCC 13.3, `-mcpu=cortex-a53`, strict logits unless noted.
- Model/table and runtime hashes are stored in each JSON result.
- These are development-machine measurements, not Orange Pi H618 claims.

## Correctness

- Native scalar versus strict NEON logits were bitwise identical for pirate
  cases 001, 002, and 003.
- Python `.shdw` versus strict NEON retained top-10 overlap of 0.9, 0.9, and
  1.0. Argmax matched cases 001 and 003 but not case 002. This is consistent
  with the Mac branch's documented long-prompt sensitivity around discontinuous
  deployment rounding, so Python semantic parity is not claimed.
- Fast versus strict logits over the first 10 fixture cases had max absolute
  error 7.63e-5, RMSE 6.90e-6, 100% argmax agreement, and 100% top-10 overlap.
- Native archive ordering, gathering, mmap bounds checks, and portable SHA-256
  validation pass the C++ test suite. All Python tests pass.

## Performance

Each result uses 3 warmups, 20 measured runs, a 1-token prompt, and 65 requested
tokens. RSS is process maximum resident set size.

| Threads | Mode | Median tok/s | p05 | p95 | Spread | Median RSS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | strict | 21.17 | 19.73 | 23.99 | 22.6% | 171.9 MiB |
| 2 | strict | 38.89 | 33.17 | 39.90 | 19.2% | 171.9 MiB |
| 4 | strict | 44.91 | 32.07 | 52.98 | 50.4% | 171.9 MiB |
| 8 | strict | 50.21 | 27.64 | 75.13 | 96.3% | 171.9 MiB |
| 2 | fast | 24.47 | 18.25 | 32.75 | 63.3% | 171.8 MiB |

These initial results used GCC 13.3. Rebuilding the identical Cortex-A53-fenced
source with Clang 18 produced bitwise-identical scalar/NEON output and improved
the 10-run strict medians to 27.49, 49.19, 87.23, and 118.61 tok/s at 1, 2, 4,
and 8 threads respectively. The 8-thread result is 2.36x the GCC median. Clang
18 is therefore the preferred compiler.

Fast logits under Clang reduced isolated profile time, but the 3-warmup/20-run
257-token suite had an 81.14 tok/s median and 101% spread. It remains opt-in
until WSL scheduling can be controlled and the full 472-case numerical suite
passes. Compact base-3 decoding and GCC LTO were measured regressions and were
rejected. A native-ISA diagnostic build was also rejected because it is invalid
for H618 and remained far below the target.

The current reproducible peak is 118.61 tok/s strict at 8 threads. A 400 tok/s
goal requires another 3.37x improvement and cannot be claimed from the current
implementation or projected to H618. Profiles identify ternary FFN, RVQ, and
vocabulary logits as the remaining dominant costs.

## Extreme optimization phases

Potential gains are directional and non-additive: Q/K/V dispatch fusion 5–20%,
persistent/vectorized RVQ 10–35%, tiled compact ternary 20–60%, fused greedy
logits 5–20%, affinity 0–15%, activation/attention fusion 5–20%, and ring
buffers negligible at short context but material near 2,048 tokens.

- Priority 1, first Q/K/V fusion attempt: rejected. It preserved bitwise parity
  but replaced the optimized 8-row RVQ body with scalar row gathering and
  regressed an isolated 257-token run to 42.51 tok/s. Dispatch fusion must retain
  the tiled RVQ body and is deferred until the vector-friendly representation.
- Priority 2a, persistent RVQ lookup scratch: accepted. Replacing approximately
  43 per-token heap allocations with reusable thread-local storage preserved
  bitwise scalar/NEON parity on pirate cases 001–003. The 10-run strict
  8-thread median improved from 118.61 to 146.25 tok/s (+23.3%); an isolated
  257-token run reached 152.89 tok/s.
- Priority 2b, RVQ `TBL` gather: pending an operator microbenchmark. ARMv8-A has
  byte table lookup but no FP32 gather, so the candidate must reconstruct four
  byte planes without changing accumulation order and beat the current Clang
  scalar-indexed 8-row loop before integration.

The operator benchmark subsequently rejected both alternatives while retaining
bitwise equality: the current 8-row loop measured 117.12 us, a 16-row scalar
unroll measured 193.40 us (-39.4%), and four-byte-plane `TBL` reconstruction
measured 167.22 us (-30.0%). Baseline NEON's lack of FP32 gather makes the
table-reconstruction cost larger than the saved scalar indexed loads.

- Priority 3, compact ternary `TBL`: rejected. A bitwise-equal operator
  benchmark measured 564.92 us for the current expanded 16-row kernel and
  2084.91 us for compact 8-row `TBL` decoding (-72.9%). Selecting from a
  256-entry base-3 table requires four 64-byte NEON table operations per digit,
  which overwhelms the reduced weight traffic.

- Priority 5, Linux affinity: rejected on WSL. Explicit guest CPU pinning
  reduced the 129-token strict median from 141.29 to 130.33 tok/s and left
  spread above 70%. WSL's virtual CPU scheduler remains the controlling source
  of variance; affinity must be reconsidered on the physical H618.
- Priority 7, segmented ring buffers: rejected. Replacing contiguous KV/trunk
  vectors with `std::deque` preserved parity but an isolated 257-token strict
  run fell to 95.92 tok/s because segmented storage degraded attention and
  recurrence locality. A future 2,048-token implementation must use fixed
  contiguous storage with a logical head index.
- Alternative approximate activation route: rejected. A vector polynomial
  exponential preserved argmax/top-10 on two initial fixtures with max logit
  error below 2.9e-5, but activation-only decode collapsed to 14.61 tok/s.
  Vector division and additional arithmetic caused severe whole-runtime
  contention, so the experimental path was removed.
- ThinLTO: rejected. Clang 18 and LLD 18 produced a Cortex-A53-safe binary,
  but a paired four-thread run fell from 88.27 tok/s to 74.40 tok/s. LLD is a
  build-time linker only and is not required on the Orange Pi.
- Clang PGO: rejected for the default artifact. The profile-trained binary was
  bitwise identical to strict NEON on pirate cases 001--003 and a short
  four-thread run initially improved from 88.27 to 103.76 tok/s. A longer paired
  129-token, eight-thread run reversed the result: standard Clang measured
  165.68 tok/s versus 145.55 tok/s for PGO. A Surface/WSL-trained profile is too
  workload- and host-specific to ship for Cortex-A53; PGO should only be
  reconsidered by training and measuring directly on the physical H618.

### Priority-order implementation audit

- Strict logits, bias, and greedy argmax are already fused where it matters:
  bias application and deterministic lowest-index argmax share one parallel
  pass. The fingerprint dot-product pass still materializes logits because the
  same runtime supports tracing, `.npy` dumps, repetition penalties, top-k, and
  temperature sampling. A greedy-only no-output API would need a separate
  measured fast path rather than changing these semantics.
- Phase-level profiling is already implemented for dense, RVQ, ternary, logits,
  attention, and unclassified work, including call counts for matrix kernels. A
  representative four-thread strict run attributed about 0.298 s to ternary,
  0.134 s to RVQ, 0.109 s to logits, and 0.011 s to attention.
- Exact SiLU and gate multiplication are already fused in place through
  `silu_multiply_inplace`; the two ternary `up`/`gt` matrices also share one
  parallel dispatch through `matvec_pair_into`. No approximate activation is
  enabled.
- Q/K/V remains the next scheduling candidate, but it must share one worker
  dispatch while retaining the current persistent lookup scratch and optimized
  eight-row RVQ decoder for every matrix. The earlier scalar-row fusion result
  is not a valid basis for this implementation.
- Q/K/V RVQ shared dispatch was subsequently implemented with independent
  persistent lookup regions and the existing eight-row decoder. It preserved
  bitwise scalar/NEON parity on pirate cases 001--003, but a paired four-thread,
  129-token suite measured 102.46 tok/s for three normal dispatches and 99.31
  tok/s for the shared dispatch (-3.1%), with comparable spread. Q has 1,536
  output rows while K and V have only 128 each, so concatenating their row
  spaces creates worker imbalance; two saved barriers do not repay that cost.
  The runtime therefore remains on the normal optimized RVQ calls.
- Expanded ternary column tiling: rejected at the operator gate. All 64-column,
  128-column, and 128-column-plus-prefetch variants were bitwise identical to
  the current 16-row expanded kernel. Across repeated 100-iteration runs,
  64-column tiling was consistently slower, while 128-column and prefetch
  results varied between small wins and losses and did not reliably beat the
  roughly 0.52--0.57 ms baseline. The existing layout already streams each
  16-row block sequentially, so tile boundaries add loop cost without reducing
  its working set. No runtime layout change was made.
- Explicit attention value NEON: rejected. Four 128-bit FMA lanes preserved
  bitwise parity on pirate cases 001--003, but a four-thread 129-token run fell
  from the paired 102.46 tok/s reference to 97.74 tok/s. Clang already
  vectorizes the simple 64-element scalar loop effectively; explicit
  load/FMA/store intrinsics increased traffic and were reverted.
- Full signed-int8 ternary expansion: rejected after a promising operator
  result. The byte-expanded kernel was bitwise identical and reduced the
  operator median from about 566 us to 507 us (roughly 10%), but expanding all
  ternary matrices changed a paired four-thread runtime only from 98.35 to
  98.05 tok/s while median RSS rose from 176,050 KiB to 366,294 KiB. The larger
  working set erased the cheaper decode and is a poor H618 tradeoff. The runtime
  option was removed; the microbenchmark remains as evidence for a possible
  future selective expansion experiment.
- Selective signed-int8 expansion was also rejected, this time replacing rather
  than retaining the selected nibble buffers. In a paired four-thread suite,
  the nibble baseline was 94.38 tok/s at 176,000 KiB RSS; `dn`-only expansion
  was 85.65 tok/s at 207,630 KiB, and `up`/`gt`-only expansion was 93.22 tok/s
  at 239,380 KiB. `dn` preserved bitwise parity on pirate cases 001--003. The
  `up`/`gt` mode failed the performance gate before further qualification. Even
  selective byte expansion increases streamed weight traffic enough to erase
  the isolated decode advantage on this host, so the runtime mechanism was
  removed.
- Paired `up`/`gt` plus exact SiLU fusion: rejected. Computing the same exact
  activation as each worker completed its 16-row tile preserved bitwise parity
  on pirate cases 001--003 and avoided the full gate-buffer readback, but a
  paired four-thread suite fell from 96.51 to 85.74 tok/s (-11.2%). Moving
  scalar `std::exp` calls into the worker phase lengthened the synchronization-
  critical ternary dispatch and disrupted its tight kernel; the small memory
  saving did not compensate. The original paired matvec followed by the exact
  in-place activation remains active.
- Packed-nibble mask-select ternary: rejected at the operator gate. The corrected
  kernel preserved bitwise equality by selecting `-x`, zero, or `+x` for each
  lane, but repeated runs measured about 1,251 us versus about 580 us for the
  current conversion/FMA kernel (roughly 2.16x slower). ARMv8-A NEON has no
  direct byte-mask-to-FP32 select, so widening to 32-bit masks, comparisons, and
  nested bit selects cost much more than integer-to-float conversion and FMA.
  No runtime change was made.
- Worker-pool non-owning task thunk: rejected. Replacing the per-dispatch
  `std::function` with a synchronous stack callable pointer and function thunk
  preserved bitwise parity on pirate cases 001--003, but an adjacent 20-run,
  four-thread suite measured 102.64 tok/s for `std::function` versus 96.15 tok/s
  for the thunk. The thunk also had higher spread (22.0% versus 11.4%). Clang's
  small-callable `std::function` path is not the dominant dispatch cost; atomic
  publication, worker wakeup, and completion spinning remain. The original
  worker pool was retained.
- K/V-only RVQ shared dispatch: rejected. The implementation retained separate
  persistent lookup tables and the existing eight-row decoder, and it preserved
  bitwise parity on pirate cases 001--003. An adjacent 20-run four-thread suite
  measured 101.51 tok/s for separate K and V dispatches versus 99.66 tok/s for
  the paired dispatch (-1.8%). The pair was more stable but saving one barrier
  per layer did not compensate for the larger worker body and loss of separate
  scheduling. Separate optimized K and V calls remain active.
- Worker synchronization microbenchmark: current spinning retained. A persistent
  four-thread pool compared generation-counter spin, C++20 `atomic::wait`, and a
  64-yield hybrid over K/V-, Q-, ternary-, and logits-like tasks. Across five
  runs, spin was about 2.1 us for K/V versus 57--60 us for wait/hybrid, about
  17.5 us for Q versus 71--75 us, about 92 us for ternary versus 145--160 us,
  and roughly 89--103 us for logits versus 140--159 us. WSL's futex-backed wake
  latency overwhelms these short dispatches. Waiting may reduce power, but it is
  not a throughput optimization on this host. `shadow_worker_microbench` is kept
  for direct reruns on the physical H618 before making a board-specific choice.
- Four-token ternary prefill operator: accepted as a foundation. The batch
  kernel loads and decodes each 16-row weight vector once, then updates four
  independent accumulator sets while preserving each token's input-column
  order. It was bitwise identical to four independent GEMVs. Across five runs,
  four calls measured about 2.46--2.65 ms while batch-4 measured about
  1.08--1.14 ms, a 2.25--2.33x throughput improvement. Runtime integration is
  intentionally separate: prompt execution must become layer-major with causal
  attention updates, while the structural trunk recurrence still executes in
  token order. Decode and sampling semantics are unchanged.
- Four-token RVQ row decoder: accepted at the operator gate. Reusing one packed
  index traversal across four independent lookup tables preserved bitwise
  equality. Across five runs, four decodes measured about 476--493 us while the
  batch-4 decoder measured about 340--357 us, a 37--44% improvement (roughly
  1.4x throughput). Full runtime RVQ batching must additionally construct the
  four lookup tables while sharing the codebook traversal.
- Paired `up`/`gt` batch-4 operator: accepted. Reusing both matrices across four
  token states measured about 2.05--2.09 ms versus 4.56--5.08 ms for the
  corresponding independent work, a 2.18--2.47x throughput improvement. This
  completes the high-cost operator gates required before implementing the exact
  layer-major causal prefill scheduler.

## Remaining hardware gates

- Run the same suites on the owner's Orange Pi H618 with 1, 2, and 4 threads.
- Record board OS/glibc, temperature, throttling, power mode, and sustained RSS.
- Run the full 472-case strict/fast fixture unattended before enabling any
  approximate path.
