# Native ARM64 runtimes

The shared C++20 core runs the existing compressed `.shdw` model on Apple
Silicon and Linux ARM64. Radxa ZERO 3W uses the Linux CPU path: Cortex-A55 NEON
for inference and exact CPU Hamming search for cold-KV
archives. Metal remains a macOS-only archive accelerator.

## Radxa ZERO 3W / RK3566 Linux ARM64

The primary workflow uses the owner's current ARM64 WSL2 environment—Ubuntu
24.04 (Noble), running natively as `aarch64`—followed by deployment and final
qualification on the owner's Radxa ZERO 3W. Install CMake, a C++ compiler,
and Ninja if desired, then build in WSL with:

    native/build_linux_arm64.sh

The script prefers `clang++-18`, whose code generation is substantially faster
for the current compressed NEON kernels, and falls back to the system C++
compiler. Set `CXX` explicitly to override this selection.

The default compiles with `-mcpu=cortex-a55`. Do not add `-march=native`. An
optional `SHADOW_ARM_DOTPROD=ON` build adds `-march=armv8.2-a+simd+dotprod`; use
it only after the production image reports `asimddp`. DotProd benefits integer
activation kernels, not the current FP32 activation kernel automatically.
The build script prints the newest referenced glibc symbol. A binary built in
this Ubuntu 24.04 WSL environment must run on a Radxa image providing that
glibc version or newer. If the board image is older, rebuild natively on the
board or in a matching ARM64 userspace; instruction compatibility and userspace
ABI compatibility are separate requirements.

Run and inspect the selected backends:

    build/linux-arm64/shadow --capabilities
    SHADOW_THREADS=4 build/linux-arm64/shadow \
      deployment/shadow250m_instruct.shdw deployment/fp131072.npy "2" 32 --bench

Linux supports `--archive-backend auto` and `cpu`; both select the exact CPU
scanner. Requesting `metal` fails explicitly. A QEMU user-mode smoke test must
use a sysroot matching the binary's build userspace, for example Ubuntu 24.04:

    qemu-aarch64 -cpu cortex-a55 -L /path/to/noble-arm64-sysroot build/linux-arm64/shadow --capabilities

Static disassembly checks for newer instructions are useful safeguards, but
the owner's Radxa ZERO 3W is the compatibility and performance authority. Record
median/p95 decode speed, RSS, temperature, and throttling with 1, 2, and 4
threads before selecting a default.

Pre-board parity and performance qualification on the current WSL host uses:

    uv run python benchmarks/verify_linux_arm64_logits.py \
      --kernel build/linux-arm64/shadow \
      --model deployment/shadow250m_instruct.shdw \
      --table deployment/fp131072.npy \
      --fixture benchmarks/pirate_runtime_fixture.json \
      --threads 4 --limit 3 --python .venv/bin/python \
      --out benchmarks/linux_arm64_parity.json

    uv run python benchmarks/linux_arm64_runtime_bench.py \
      --kernel build/linux-arm64/shadow \
      --model deployment/shadow250m_instruct.shdw \
      --table deployment/fp131072.npy --tokens 2 --generate 65 \
      --threads 2 --warmup 3 --runs 20 \
      --out benchmarks/linux_arm64_threads2_strict.json

Use `benchmarks/verify_macos_logits.py` with this Linux kernel to compare strict
and fast logits. Pass `--limit 25` for a representative interactive run; omit
the limit for the full 472-case unattended qualification. WSL measurements are
development records and must not be presented as RK3566 performance.
Current WSL findings and raw-result links are summarized in
[`benchmarks/linux_arm64_wsl_report.md`](../benchmarks/linux_arm64_wsl_report.md).

### Exact batch-4 prefill foundation

`Tensor::matvec_batch4_into` reuses each decoded base-3 ternary weight vector
across four prompt token states while retaining each token's input-column
accumulation order. The operator benchmark is bitwise equal to four independent
calls and measures approximately 2.25--2.33x higher ternary throughput on the
current WSL ARM64 host.

The primitive is not used by single-token decode. Exact prompt batching also
needs batch-4 RVQ and paired `up`/`gt` primitives plus a layer-major scheduler.
Within each layer, attention positions must still enter the KV cache in order so
one prompt token cannot observe a future token. Sampling behavior is unchanged.
The RVQ operator benchmark also validates batch-4 packed-index reuse: it is
bitwise equal and approximately 1.4x faster than four independent row decodes.

## Apple Silicon

The native runner targets Apple Silicon on macOS 14 or newer. It reads the
existing compressed `.shdw` model directly, executes decode on ARM64/NEON, and
uses Metal for large exact-Hamming cold-KV scans. Its positional CLI remains
compatible with `shadow_runtime.Engine`.

Build without additional tools:

    native/build_macos.sh

If the optional Xcode Metal Toolchain component is installed, the build also
emits `shadow.metallib`. Otherwise the runner compiles its embedded shader on
first use and continues normally.

Or use CMake 3.24 or newer:

    cmake -S native -B build/macos -DCMAKE_BUILD_TYPE=Release
    cmake --build build/macos --parallel

Inspect the selected backends:

    build/macos/shadow --capabilities

The design follows the useful deployment boundaries also used by Ollama: one
runner process, explicit backend discovery, conservative accelerator fallback,
and content-addressed model assets. No Ollama source code is included.

## Cold-KV archive

Export a v2 model containing calibrated codec records, then build an archive:

    python finetune/export_model.py checkpoint.pt model-v2.shdw --with-codecs
    python finetune/build_kv_archive.py \
      --checkpoint checkpoint.pt --model model-v2.shdw \
      --table deployment/fp131072.npy --tokens archive/tokens.u32 \
      --out archive.shkv

Scan one layer/head with a 64-byte query encoded as 128 hex characters:

    deployment/bin/macos/shadow --scan archive.shkv QUERY_HEX 0 0 16 --backend auto

Run decode with cold-KV retrieval integrated into every transformer layer:

    deployment/bin/macos/shadow model-v2.shdw deployment/fp131072.npy "2 8" 32 \
      --archive archive.shkv --archive-backend auto --archive-topk 32

`auto` uses CPU for archives below 64 MiB and Metal for larger files when Metal
is available. `cpu` and `metal` force a backend for validation or benchmarking.

## Correctness boundary

The repository does not contain the historical x86 kernel source. Capture its
deterministic output fixtures on Linux or Windows with
`benchmarks/generate_runtime_golden.py`, then validate this runner with
`benchmarks/check_runtime_golden.py`. Until those fixtures are supplied and
pass, the Apple runner is an independently reconstructed implementation and
must not be described as token-identical to the historical binary.

The deployment artifact itself can be used as a Python logit oracle without
loading the master checkpoint:

    python finetune/dump_shdw_logits.py \
      --model deployment/shadow250m_instruct.shdw \
      --table deployment/fp131072.npy --tokens "2 8" \
      --out reference-logits.npy --top-k 10 --last-only

Compare that array with another instrumented runtime using:

    python benchmarks/compare_logits.py reference-logits.npy candidate-logits.npy

The macOS runner can emit its prediction-step logits directly:

    deployment/bin/macos/shadow model.shdw table.npy "2 8" 1 \
      --dump-logits native-logits.npy

On an M4 Max, the current reconstructed compressed path measures approximately
11 tok/s for a one-token prompt with 10 worker threads. This is a
correctness-first baseline, not the 750–800 tok/s projection in the original
proposal. The 9,000-key synthetic scan measured 0.18 ms on CPU and 84.6 ms on
first-use Metal (including shader compilation), so small archives deliberately
stay on CPU.
