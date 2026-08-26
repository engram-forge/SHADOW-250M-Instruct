# Repository Guidelines

## Project Structure & Module Organization

`shadow_chat.py` is the simplest local entry point. Core Python inference and archive-retrieval code lives in `shadow_runtime/`, with `__main__.py` providing the module CLI. Prebuilt model weights, vocabulary data, and Linux/Windows executables are under `deployment/`. Fine-tuning, export code, model definitions, and example JSONL data live in `finetune/`. Reproducible evaluation scripts and published results are in `benchmarks/`; tokenizer artifacts are in `tokenizer/`. Keep generated checkpoints and private archive data out of version control unless they are intentional release assets.

## Build, Test, and Development Commands

- `python shadow_chat.py` starts an interactive chat using the bundled platform binary.
- `python -m shadow_runtime --model deployment/shadow250m_instruct.shdw --table deployment/fp131072.npy --chat` runs the Python module CLI.
- `python benchmarks/run.py --tiers 1M` evaluates a local archive in `data/archives/1m/`; benchmark archives are not distributed.
- `cd finetune && python finetune.py --data examples_pirate.jsonl --steps 150 --out pirate_model` fine-tunes on a CUDA-capable system.
- `cd finetune && python export_model.py pirate_model/finetuned.pt pirate.shdw` exports a CPU runtime model and performs its round-trip check.

Install the documented runtime dependencies (`torch`, `numpy`, and `sentencepiece`) in an isolated environment. No package manifest or build system is currently provided.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants. Use `pathlib.Path` for filesystem paths and keep CLI behavior in small entry-point modules. Preserve established model-file suffixes such as `.shdw`, `.pt`, `.npy`, and `.u32`. There is no configured formatter or linter; keep imports readable and changes focused.

## Testing Guidelines

There is no dedicated unit-test suite or coverage threshold. For runtime changes, perform a short chat or single-question smoke test. For retrieval changes, run the smallest available benchmark tier and compare generated results with `benchmarks/results.json`. Fine-tuning/export changes should complete the export round-trip check. Document unavailable GPU hardware or private benchmark archives in the pull request.

### Native Runtime Optimization Workflow

- **Iteration:** Propose → isolate one variable → implement behind a control →
  verify parity → measure performance/RSS → commit or fully revert. Do not combine
  unqualified changes. State the bottleneck, expected gain, correctness boundary,
  memory cost, and Cortex-A55 ISA requirement before implementation.
- **Three-way isolation:** Evaluate every new optimization with group quantization
  disabled first: (A) untouched exact FP32 baseline versus (B) exact candidate.
  This attributes speed and output changes to the optimization alone. Only after B
  passes should it be combined with DotProd group-64 as (C). Report B/A as the
  exact optimization gain, C/B as the group-64 integration gain, and C/A as the
  total deployable gain. Never compare only C/A or credit group-64's approximate
  speedup to an unrelated optimization. Exact FP32 remains the correctness and
  production baseline; group-64 is the approximate DotProd integration baseline.
- **Correctness:** Run native and Python tests. Exact changes require strict
  scalar/NEON and logit or seeded-sampling parity. Approximate paths require the
  472-case sequence evaluation: first-token agreement, top-10 overlap, first-step
  logit RMSE, matching-prefix length, complete-sequence equality, and task score.
- **Measurement:** Use identical inputs, threads, sampling, warmups, and runs.
  Alternate baseline/candidate order; report raw runs, median, p05/p95, spread,
  RSS, and the affected profile share. WSL is development evidence, never an
  RK3566 performance claim.
- **Prefill gate:** Test lengths 4/16/64/256 with 1/2/4 threads, including batch
  tails. Report prompt tok/s, TTFT, median/p05/p95, RSS, final-logit parity, and a
  decode regression control. Use 3 warmups plus 20 runs for 4/16 and 10 runs for
  64/256.
- **Decode gate:** Test contexts 32/128/512/1024/2048 with 1/2/4 threads and at
  least 16 generated tokens. Report each context separately. Retain a guarded
  context-specific path only if it wins consistently after branch/code-size cost.
- **ISA and release:** Never use `-march=native`. Scan the exact artifact against
  Cortex-A55 and DotProd artifacts against ARMv8.2-A+DotProd. Require `asimddp`
  on the production image. Rerun the full matrix on physical RK3566 hardware.
- **Decision:** Commit one cohesive passing change with evidence. On correctness,
  stability, memory, or performance failure, remove runtime code and temporary
  switches; retain only useful benchmark or rejection documentation.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects, sometimes scoped with a prefix, for example `runtimes: portable builds`. Follow that pattern and keep each commit cohesive. Pull requests should explain the motivation, list validation commands and results, and identify changed model or binary assets with their sizes. Link relevant issues; include screenshots only for visible documentation or terminal-output changes.
