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

Treat prefill and decode as separate performance products. Prefill reports prompt
tokens per second and time-to-first-token; decode reports generated tokens per
second after prefill. Never combine them into one throughput number. Keep WSL
measurements labeled as development evidence rather than Radxa ZERO 3W claims.

Develop each optimization incrementally:

1. Propose the expected bottleneck, intended gain, correctness boundary, memory
   cost, and Cortex-A55 compatibility constraints. Treat DotProd as an optional
   path that requires an `asimddp` feature check on the production image.
2. Break the proposal into the smallest independently measurable change. Change
   one kernel, layout, scheduling decision, or compiler option at a time.
3. Try the candidate behind a temporary same-binary control when practical.
   Alternate control-first and candidate-first runs to reduce ordering and WSL
   scheduler bias. Do not combine unqualified optimizations.
4. Verify correctness before accepting performance results. Require existing
   native and Python tests, strict scalar/NEON parity where applicable, and
   byte-identical logits or seeded sampling for transformations intended to be
   exact. Scan baseline artifacts against the Cortex-A55 boundary and DotProd
   artifacts against ARMv8.2-A+DotProd; never use `-march=native`. Approximate
   integer paths additionally require sequence-level generation evaluation.
5. Verify stable performance with warmups, repeated raw measurements, medians,
   spread, RSS, and identical inputs, threads, and sampling settings. Profile the
   affected stage to confirm that any gain comes from the intended component.
6. Commit a cohesive accepted improvement with its benchmark evidence. If the
   candidate fails correctness, memory, stability, or performance gates, fully
   revert its runtime code and temporary switches, document the rejection, and
   commit only useful harness or report changes.

Prefill qualification must cover prompt lengths 4, 16, 64, and 256 with 1, 2,
and 4 threads. Compare sequential and batched paths, including prompts not
divisible by the batch width. Record prefill median/p05/p95, prompt tok/s, TTFT,
spread, RSS, final-logit parity, seeded-sampling parity, and a decode regression
control. Use at least 3 warmups and 20 measured runs for 4/16-token prompts and
at least 3 warmups and 10 measured runs for 64/256-token prompts when making a
stable claim.

Decode qualification must cover context lengths 32, 128, 512, 1024, and 2048.
Report decode throughput separately at each length because attention, KV cache,
and structural recurrence costs grow with context. Use at least 16 generated
tokens after prefill, alternating candidate/control order, exact final-logit
parity at every length, and stable repeated measurements. A candidate may be
retained as a guarded context-dependent path only when it wins consistently
above a documented threshold and repays its branch, code-size, and maintenance
cost. Always rerun the 1/2/4-thread matrix on physical RK3566 hardware before
making release-performance claims.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects, sometimes scoped with a prefix, for example `runtimes: portable builds`. Follow that pattern and keep each commit cohesive. Pull requests should explain the motivation, list validation commands and results, and identify changed model or binary assets with their sizes. Link relevant issues; include screenshots only for visible documentation or terminal-output changes.
