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

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects, sometimes scoped with a prefix, for example `runtimes: portable builds`. Follow that pattern and keep each commit cohesive. Pull requests should explain the motivation, list validation commands and results, and identify changed model or binary assets with their sizes. Link relevant issues; include screenshots only for visible documentation or terminal-output changes.
