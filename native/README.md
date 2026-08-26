# Apple Silicon runtime

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
