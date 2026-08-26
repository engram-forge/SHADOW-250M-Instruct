# Pretraining SHADOW 250M on the compressed Dolma 8B sample

This is a best-effort reproduction of the published SHADOW 250M architecture, not a
bit-for-bit reproduction of its original pretraining recipe. The original optimizer,
initialization, sample order, and learning-rate schedule were not published.

The pipeline keeps Dolma in its upstream gzip-compressed JSONL representation. It does not
create an expanded token cache. Worker processes decompress and tokenize records while the
trainer packs a deterministic stream of 2,048-token causal-LM windows.

## 1. Download and verify the sample

The following command is resumable. Complete files are skipped, partial files use HTTP range
requests, and files are atomically renamed after their upstream sizes match.

```bash
uv run python pretrain/download_dolma_sample.py \
  --out /home/dlisuser/quanwen/SHADOW-250M-Instruct/data/dolma/sample-8b \
  --workers 6 --verify-gzip
```

Expected result: 103 `.json.gz` files totaling 16,430,279,784 compressed bytes. Corpus files,
run outputs, and checkpoints are ignored by Git.

## 2. Inspect and benchmark real input

`scan` counts SHADOW-remapped tokens directly from gzip. A complete scan may take time and
writes only a small JSON report. Use `--scan-max-docs` for a quick pipeline check.

```bash
uv run python -m pretrain.train scan \
  --data /home/dlisuser/quanwen/SHADOW-250M-Instruct/data/dolma/sample-8b \
  --out pretrain_runs/dolma-8b

uv run python -m pretrain.train benchmark \
  --data /home/dlisuser/quanwen/SHADOW-250M-Instruct/data/dolma/sample-8b \
  --out pretrain_runs/dolma-8b --benchmark-steps 50
```

The benchmark warms up for three steps and records input, GPU-compute, end-to-end throughput,
peak VRAM, and an 8B-token ETA in `benchmark.json`.

## 3. Train or resume

```bash
uv run python -m pretrain.train train \
  --data /home/dlisuser/quanwen/SHADOW-250M-Instruct/data/dolma/sample-8b \
  --out pretrain_runs/dolma-8b

uv run python -m pretrain.train train \
  --data /home/dlisuser/quanwen/SHADOW-250M-Instruct/data/dolma/sample-8b \
  --out pretrain_runs/dolma-8b \
  --resume pretrain_runs/dolma-8b/checkpoints/tokens-000500000000.pt
```

Choose the FFN alphabet when starting a new run. It is part of the strict resume contract:

```bash
uv run python -m pretrain.train train --data DATA --out RUN \
  --ffn-weight-dtype ternary --ffn-act-warmup-tokens 100000000

uv run python -m pretrain.train train --data DATA --out RUN_INT4 \
  --ffn-weight-dtype int4_row --ffn-act-warmup-tokens 100000000
```

Fresh pretraining defaults to a prediction horizon of two: the ordinary next-token head plus one
token-conditioned residual MLP for offset two. It uses RMSNorm and a `D -> D/2 -> D` path while
sharing the input embedding, base fingerprint head, vocabulary projection, and tied bias. The
auxiliary loss has weight `0.3`. Both settings are part of the strict resume contract:

```bash
uv run python -m pretrain.train train --data DATA --out RUN \
  --mtp-horizon 2 --mtp-loss-weight 0.3
```

The MTP heads train candidate proposals only. Actual speculative decode still requires causal
verification, acceptance/rejection, and KV-cache commit/rollback support in the native engine.
This is an A55-oriented derivative of DeepSeek's token-conditioned MTP idea, not its exact MTP
block: the full Transformer block is deliberately replaced by the smaller residual bottleneck MLP.

Measure exact greedy proposal acceptance before native-engine work. This slow reference recomputes
the base model sequentially for both verification positions and writes optional per-prompt details:

```bash
python benchmarks/mtp_reference.py \
  --checkpoint pretrain_runs/dolma-8b/checkpoints/final.pt \
  --prompt "The quick brown fox" --cycles 16 \
  --device cpu --out pretrain_runs/dolma-8b/mtp-acceptance.json
```

`first_acceptance_rate` should be 1.0 because the first proposal is the ordinary base greedy token.
`second_acceptance_rate` is the useful MTP measurement: it compares the conditioned proposal with
the second ordinary sequential greedy token. Oracle tokens are appended after each comparison so a
rejected proposal cannot contaminate later evaluation contexts.

Defaults are context 2,048, micro-batch 12, accumulation 16, BF16 compute, AdamW, a
`3e-4` peak learning rate, 1% warmup, cosine decay, validation every 100M tokens, and an
atomic checkpoint every 500M tokens. The checkpoint includes model, optimizer, loss-scaler, and
all RNG states, the corpus identity, pending packed tokens, and the exact gzip source cursor.
Parameters and AdamW state remain FP32. BF16 is the default compute type; FP16 uses dynamic loss
scaling. Per-token INT8 FFN activation QAT is introduced linearly during the configured warm-in
and validation always uses full deployment strength.

The default `--max-tokens 8000000000` follows the upstream sample label. Run a complete scan
and override it with the measured SHADOW-token total if exactly one corpus pass is required.
The training command is intentionally not launched automatically.

## 4. Run with nohup

First run the same preflight checks used by the background launcher:

~~~bash
script/start_pretrain_nohup.sh --preflight
~~~

Start the default 8B-token run:

~~~bash
script/start_pretrain_nohup.sh
~~~

The launcher prints the PID and timestamped log path. It refuses to start another job while
the recorded PID is alive. Monitor with `tail -f LOG_PATH`, inspect the GPU with
`nvidia-smi`, and request a normal process shutdown with `kill PID`. The latest completed
atomic checkpoint remains resumable.

Resume explicitly from a checkpoint:

~~~bash
script/start_pretrain_nohup.sh \
  --resume pretrain_runs/dolma-8b/checkpoints/final.pt
~~~

The scripts use the already validated environment at
`/home/dlisuser/quanwen/SHADOW-250M-Instruct/.venv/bin/python`, avoiding a duplicate Torch
installation. Override it with `SHADOW_PYTHON_BIN`. Other supported overrides are
`SHADOW_WORKERS`, `SHADOW_MICRO_BATCH`, `SHADOW_ACCUM`, `SHADOW_MAX_TOKENS`,
`SHADOW_DIAGNOSTICS_EVERY`, `SHADOW_DATA_DIR`, `SHADOW_RUN_DIR`, `SHADOW_AMP_DTYPE`,
`SHADOW_FFN_WEIGHT_DTYPE`, and `SHADOW_FFN_ACT_WARMUP_TOKENS`. Additional
`SHADOW_MTP_HORIZON`, `SHADOW_MTP_LOSS_WEIGHT`,
arguments, such as `--lr 2e-4`, are forwarded to the Python trainer.

## 5. Stability diagnostics

Starting with the next process launch or checkpoint resume, the trainer records the pre-clip
global gradient norm on every normal `metrics.jsonl` line. Every 10 updates by default, one
compact record is appended to `diagnostics.jsonl`. It contains:

- normalized global, median-layer, p10-layer, and worst-layer gradient participation;
- weighted and worst-module NMSE for RVQ and the selected ternary/INT4 QAT weights;
- the worst tensor/module names and diagnostic measurement time.

It does not dump every tensor, so the diagnostic log remains small. Set
`SHADOW_DIAGNOSTICS_EVERY=0` to disable or choose a different interval. Diagnostics are loaded
when a training process starts or resumes.
