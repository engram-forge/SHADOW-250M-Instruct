# Apple Silicon numerical parity and performance plan

## Current measured state

- Target: Apple Silicon, macOS 14+, with CPU decode and Metal cold-KV scanning.
- Phase 0 M4 Max decode baseline: **11.1487 tok/s** with 10 worker threads.
- Phase 2 exit result: **102.6801 tok/s** (median of three independent suites,
  each with 3 warmups, 20 measured runs, and 32 decode steps), a **9.21x** gain
  over Phase 0. Individual suite medians were 102.6801, 99.1367, and 106.3277
  tok/s; retain their p05/p95 because scheduler noise is material.
- The historical Linux x86 runtime is advertised at approximately **400 tok/s**.
- A Python deployment oracle now loads the exact compressed `.shdw`; it does not
  use the different training-time FFN weights in the master `.pt`.
- Matching the deployment rounding points reduced native versus Python `.shdw`
  error for token IDs `2 8` to max 3.53e-5 and RMSE 6.71e-6, with identical
  argmax and top-10. Longer prompts amplify tiny BLAS/kernel differences through
  the discontinuous exact-shiftmax `floor`; pirate-001 has max 0.661, RMSE 0.147,
  identical argmax, and 90% top-10 overlap.
- The historical Linux binary differs from both the Python `.shdw` oracle and the
  macOS runtime. It remains a compatibility reference, not the numerical oracle.

The original 400–900 tok/s figures are targets, not verified properties. All
optimization decisions below are gated by reproducible measurements.

## Verification contract

### Datasets and fixtures

Use all 472 `user -> assistant` pairs in
`/Users/quan/workspace/SHADOW-250M/finetune/examples_pirate.jsonl`. Only the user
message is inference input; the assistant text is retained by hash for provenance
and is not treated as a runtime correctness target. Each user message is wrapped
with the published chat template and tokenized once into
`benchmarks/pirate_runtime_fixture.json`.

Maintain three oracles:

1. Python `.shdw` oracle: semantic reference from the exact deployed weights.
2. Native scalar kernels: bitwise optimization oracle for every changed operator.
3. Historical Linux x86 runner on Modal: compatibility record only.
4. macOS optimized runner: full logits and greedy token IDs.

The fixture records dataset/model/table SHA-256 values. Reject results if any hash
differs. Do not commit the source dataset; the deterministic token fixture contains
no assistant text.

### Parity gates

Run the 472 prompts in ascending source-row order. A macOS build reaches parity only
when all of these pass:

- Native scalar versus optimized kernels: bitwise-equal full logits and greedy
  tokens for the benchmark sequence and representative pirate prompts.
- Python versus macOS first-step argmax: 100%.
- Python versus macOS top-10 set overlap is reported; exact equality is expected
  for short prompts but not asserted across `floor` boundaries on long prompts.
- Python versus macOS logits are reported with max absolute error and RMSE. The
  strict <=0.01/<=0.002 gate applies to minimal operator reproductions; long-prompt
  acceptance is semantic (argmax/tokens) plus zero native optimization regression.
- Repeated native runs with identical thread count and seed are bitwise deterministic.

If output-logit gates pass, skip per-layer comparison. Per-layer dumps are a temporary
diagnostic activated only for the first failing prompt and removed or disabled from
release builds after parity is restored.

### Failure localization

When a gate fails, reduce to the first prompt, position, and token where divergence
appears, then compare in this order:

1. fingerprint input projection;
2. each block's normalized input and Q/K/V projections;
3. RoPE, PoT rounding, shiftmax scores, and attention output;
4. output projection and ternary SwiGLU branches;
5. structural recurrent step, final norm, fingerprint projection, and tied bias.

Compare tensor shape, first 16 elements, L2 norm, max absolute error, and SHA-256 of
raw float32 bytes. Stop at the first divergent tensor. Match deployment semantics
exactly: BF16 conversion points, `floor`, base-2 exponential clamp, PoT rounding, and
causal/cache indexing must not be replaced with mathematically similar operations.

## Optimization roadmap

### Phase 0 — establish trustworthy measurement

- Add one benchmark command that records chip, macOS version, runner/model hashes,
  thread count, prompt length, generated tokens, median/p95 prefill and decode latency,
  RSS, and backend capabilities as JSON.
- Warm up three runs, then measure at least 20 runs. Report decode only when at least
  32 actual post-prefill steps execute.
- Benchmark prompt lengths 1, 16, 128, 512, and 2,048; decode lengths 32 and 128;
  thread counts 1, 2, 4, performance-core count, and all cores.
- Record 11.25 tok/s only as the current baseline. Do not use one-token timing.

Exit: stable measurements vary by less than 5% across three consecutive suites.

### Phase 1 — remove orchestration overhead

- Replace per-matvec `std::thread` construction with a persistent spin/sleep worker
  pool created once per model. Pin or QoS-classify performance workers only after
  comparing scheduler behavior; keep `SHADOW_THREADS` authoritative.
- Allocate all activation, Q/K/V, attention, FFN, lookup, and logit scratch buffers
  once. Eliminate vectors and string construction from token decode.
- Parse `.shdw` through read-only `mmap`; keep record payloads as spans into mapped
  pages rather than copying 52 MB into heap vectors.
- Cache tensor handles by layer and operator, removing per-token hash-map lookups.

Measured result: **11.4521 tok/s**, +2.72% versus Phase 0. Profiling showed
ternary FFN consumed about 94% of runtime, so the original 2x orchestration-only
gate was infeasible. Phase 1 was closed with identical 32-step greedy output and
the bottleneck evidence, then kernel work began.

### Phase 2 — optimized compressed matrix kernels

- Ternary base-3 FFN: process packed bytes in cache-sized output tiles, decode with
  precomputed 256-entry digit masks, use NEON table lookup and multiple float32
  accumulators, and fuse row scale application. Avoid materializing weights.
- RVQ projections: precompute codebook dot tables per input group, vectorize packed
  nibble extraction, and accumulate 32/64 output rows per tile. Keep Q/K/V/O matrices
  in their exported packed layout.
- Tune tile sizes independently for 1536x1536, 4224x1536, and 1536x4224 using an
  operator microbenchmark. Preserve a scalar reference behind a test-only switch.
- Parallelize by contiguous output tiles to avoid false sharing; assign enough rows per
  worker that synchronization cost stays below 2% of kernel time.

Exit: compressed matvec kernels consume below 70% of decode time and total speed is at
least 100 tok/s on the M4 Max, with all parity gates unchanged.

Phase 2 completed at **102.6801 tok/s**. Accepted changes are a
branch-free base-3 kernel, 8-row output tiling, 8-row RVQ tiling, and 8-row
fingerprint-logit tiling, plus tile-major base-3 storage and NEON base-3 digit
extraction for eight rows at once. All accepted kernels are bitwise equal to scalar
references over 33 full-vocabulary logit rows. Rejected experiments include a
float lookup table (-6.6%), 16-row ternary/logit tiles, RVQ scratch reuse, and a
packed-byte contribution table (max logit error 0.0221, RMSE 0.002043). The latest
Phase 2 exit also passed zero-diff final-logit checks for pirate cases 001, 002,
and 100, 15 Python tests, native tests, and `git diff --check`.

### Phase 3 — fuse transformer decode operations

- Fuse RMSNorm into Q/K/V and FFN input projections without changing float/BF16
  rounding order required by parity.
- Fuse RoPE, PoT quantization, and KV cache append. Store hot K/V in the layout consumed
  by grouped attention and remove per-step head replication.
- Implement GQA directly as 2 KV heads serving 12 query heads each. Vectorize dot
  products and value accumulation over head width 64.
- Fuse `silu(gate) * up`, ternary down projection input preparation, residual addition,
  and applicable scale operations.
- Specialize the structural step for single-token decode and its 2,048-token window.

Exit: at least 250 tok/s on M4 Max, parity suite fully passing, RSS measured and
documented.

Phase 3 completed at **259.0197 tok/s**, using the median of three independent
3-warmup/20-run suites (258.9336, 260.4130, and 259.0197 tok/s). Accepted work
includes fused up/gate base-3 projection, C++20 barrier worker rendezvous, NEON
fingerprint logits, NEON RVQ codebook lookup, 16-row base-3 ILP, tile-aligned
worker partitions, NEON greedy argmax, and parallel tied-bias application. The
final build remained bitwise equal to scalar kernels for 33 full-vocabulary logit
rows and pirate cases 001, 002, and 100. Fifteen Python tests, native tests, and
`git diff --check` passed. Rejected Phase 3 experiments include RVQ contiguous
row addressing, 32-row base-3 tiles, 16-row worker alignment, and parallel SwiGLU
because their gains were within noise or regressed performance.

### Phase 4 — fingerprint logits and scheduling

- Keep the 8 MiB packed fingerprint table and compute all 131,072 scores using NEON
  sign-select/add-subtract tiles. Evaluate transposed/block layouts produced at load
  time only if their memory cost and preprocessing time are justified.
- Fuse tied bias and top-k selection so generation does not require a second full-vocab
  pass. Full logits remain available under `--dump-logits` for verification.
- Overlap independent work only where dependencies allow; do not send serial decode to
  Metal merely to claim GPU use. CPU decode remains preferred unless measured otherwise.
- Re-evaluate performance-core-only versus all-core scheduling after kernels become
  bandwidth-bound.

Exit: pursue 400 tok/s. If measured speed remains below target, publish the achieved
median/p95 and profiler breakdown rather than weakening correctness gates.

Phase 4 closed at **276.6903 tok/s**, the median of three suite medians
(276.5095, 276.6903, and 278.5436 tok/s). This is **24.82x** the Phase 0
baseline, but below the 400 tok/s target. The final profile was approximately
52% ternary, 21% RVQ+dense, 17% fingerprint logits, and 10% attention/other.
Accepted work fused greedy argmax into the parallel tied-bias pass and expanded
NEON fingerprint logits to 16 output rows. Scalar/optimized final logits were
bitwise equal for pirate cases 001, 002, and 100; all tests passed. More aggressive
bias/argmax fusion and a 32-row logits experiment were rejected after no gain or
a failed build; neither remains in the final path.

### Phase 5 — Metal cold-KV production path

- Precompile `archive.metal` when the optional Xcode Metal Toolchain is installed; keep
  embedded-source compilation and CPU fallback. Cache the Metal device, pipeline, queue,
  mapped file, and buffers for the model lifetime.
- Submit all layer/head Hamming scans through bounded candidate buffers; avoid 20 serial
  command-buffer waits. Batch compatible scans and merge deterministic `(distance,index)`
  pairs on CPU.
- Validate CPU/Metal shortlist equality for empty, partial-page, tied-distance, 1M, and
  10M archives. Measure 100M only when the private archive is present.
- Determine the CPU/Metal break-even size empirically and store it in benchmark results;
  the current 64 MiB threshold is provisional.

Exit: CPU/Metal retrieval results are exact, hybrid decode passes the same token gates,
and 1M/10M latency plus memory use are published.

Phase 5 completed for the available assets. Persistent `mmap`, shared Metal buffer,
query buffer, and output buffer caching improved forced-Metal hybrid decode on the
32-token fixture from **54.4056 to 75.5151 tok/s** (+38.8%) in 3-warmup/20-run
suites; CPU and Metal full logits were bitwise equal. A synthetic scan-only fixture
(one layer/head, 64-byte keys, deterministic seed 250) measured exact CPU/Metal
top-8 parity at both scales: 1M warm scans were **18.0767 ms CPU / 3.0630 ms
Metal**, and 10M warm scans were **160.646 ms CPU / 7.3622 ms Metal**. `auto`
selected Metal for both. A 9k partial-page/tied-distance fixture also matched
exactly; CPU was 0.281 ms while Metal cold start was 37.678 ms, supporting the
conservative small-archive CPU path. No real 100M archive was available, so no
100M claim is made. Synthetic 1M/10M files remain build artifacts, not releases.

### Phase 6 — Apple Silicon decode kernels

Phase 6 reaches the 400 tok/s target on an M4 Max with 10 performance-core
workers. The macOS Python launcher enables a 64 KiB byte-contribution table for fingerprint
logits. In the 472-case pirate fixture it retained **100% argmax agreement** and
**100% top-10 overlap** against strict logits across 61,865,984 compared values;
RMSE was **6.91e-6** and maximum absolute error was **1.03e-4**. Set
`SHADOW_FAST_LOGITS=0` when bitwise strict logits are required.

The 3-warmup/20-run, 65-token benchmark measured:

| Mode | Median | p05 | p95 | Numerical contract |
| --- | ---: | ---: | ---: | --- |
| macOS fast logits | **401.216 tok/s** | 395.885 | 405.015 | 472/472 argmax and top-10 match |
| strict logits | 354.068 tok/s | 348.095 | 357.999 | bitwise-equal to scalar reference |

Strict optimized and scalar output was byte-for-byte identical over 33 complete
logit rows and pirate cases 001, 002, and 100. Accepted changes include a persistent
generation-counter worker pool, Apple performance QoS, native FP16 widening, load-time
signed-nibble expansion of base-3 weights, 16-row NEON ternary kernels, and 8-row
fingerprint contribution-table evaluation. The signed-nibble cache costs one half-byte
per ternary weight in addition to the on-disk base-3 representation; it is generated
at load time and does not alter `.shdw`. Fast logits also keep a dedicated 8 MiB
8-row-blocked view of the fingerprint table for contiguous reads. Explicit two-way loop unrolling was faster
than wider unrolling on the tested M4 Max.

Reproduce the benchmark with:

    uv run python benchmarks/macos_runtime_bench.py \
      --kernel deployment/bin/macos/shadow \
      --model deployment/shadow250m_instruct.shdw \
      --table deployment/fp131072.npy --tokens 2 --generate 65 \
      --threads 10 --warmup 3 --runs 20 --fast-logits \
      --out benchmarks/macos_phase6_fast_logits.json

The corresponding strict command omits `--fast-logits`. The full pirate comparison
is recorded in `benchmarks/macos_phase6_logits.json` and can be regenerated with
`benchmarks/verify_macos_logits.py`.

## Modal Linux reference workflow

The logged-in Modal profile is `qwen-wenquan`. Modal CLI 1.5.4 is available through
`uvx` without modifying the project lock file.

Build the deterministic fixture:

    uv run python benchmarks/pirate_runtime_verification.py build \
      --data /Users/quan/workspace/SHADOW-250M/finetune/examples_pirate.jsonl \
      --model deployment/shadow250m_instruct.shdw \
      --table deployment/fp131072.npy \
      --out benchmarks/pirate_runtime_fixture.json

Generate Linux x86 greedy goldens remotely:

    uvx --from 'modal>=1.5' modal run benchmarks/modal_linux_golden.py \
      --fixture benchmarks/pirate_runtime_fixture.json \
      --out benchmarks/pirate_linux_golden.json --generate-tokens 8

The Modal image contains only the historical Linux runner, model, table, and benchmark
fixtures. It uses four vCPUs and no GPU. A two-case smoke test must pass operationally
before launching all 472 prompts.

## Release checklist

- All 472 pirate prompts have Python/native semantic results recorded; historical
  Linux differences are reported separately and are not silently treated as truth.
- Python/native logit gates pass; no release-only diagnostic tracing remains enabled.
- 15+ Python tests, native tests, archive CPU/Metal parity, and `git diff --check` pass.
- ARM64 binary reports macOS 14 minimum and correct `--capabilities`.
- Binary and optional metallib hashes, sizes, build command, benchmark JSON, and model
  hashes are recorded.
- README quotes only measured performance and calls unverified goals targets.
