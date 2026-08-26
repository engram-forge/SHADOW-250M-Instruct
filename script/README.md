# Pretraining scripts

These wrappers run the compressed Dolma 8B pretraining job from the
`feat/a55-dotprod-qat` worktree. They verify the exact 103-shard corpus, its compressed
byte count, the Python environment, and the NVIDIA GPU before training starts.

## Start

Run the non-mutating preflight first:

~~~bash
cd /home/dlisuser/quanwen/SHADOW-250M-A55
script/start_pretrain_nohup.sh --preflight
~~~

Start the job under `nohup`:

~~~bash
script/start_pretrain_nohup.sh
~~~

The launcher prints the child PID and timestamped log. It writes the current PID to
`pretrain_runs/dolma-8b/train.pid` and refuses to launch a duplicate live process.

## Monitor

Follow the timestamped console log printed by the launcher:

~~~bash
tail -f pretrain_runs/dolma-8b/logs/train-YYYYMMDDTHHMMSSZ.log
~~~

The normal metrics stream contains loss, learning rate, throughput, pre-clip gradient norm,
and clipping coefficient:

~~~bash
tail -f pretrain_runs/dolma-8b/metrics.jsonl
~~~

Every 10 optimizer updates, the default diagnostics stream records normalized gradient
participation and QAT reconstruction NMSE without dumping every tensor:

~~~bash
tail -f pretrain_runs/dolma-8b/diagnostics.jsonl
~~~

Each diagnostic record contains global, median-layer, p10-layer, and worst-layer normalized
participation; weighted and worst-module NMSE for RVQ and the selected ternary/INT4 weights; worst module names;
and measurement time. Change the cadence at launch with, for example:

~~~bash
SHADOW_DIAGNOSTICS_EVERY=50 script/start_pretrain_nohup.sh
~~~

Use `SHADOW_DIAGNOSTICS_EVERY=0` to disable it.

## Stop and resume

Request a normal termination using the recorded PID:

~~~bash
kill -- "-$(cat pretrain_runs/dolma-8b/train.pid)"
~~~

The negative PID targets the detached process group, including tokenizer workers. The trainer
currently checkpoints every 500M tokens. A termination between checkpoints does
not create an emergency checkpoint; resume starts from the latest completed atomic checkpoint.
If no checkpoint exists yet, a restart begins from token zero.

Resume explicitly:

~~~bash
script/start_pretrain_nohup.sh \
  --resume pretrain_runs/dolma-8b/checkpoints/final.pt
~~~

## Overrides

The foreground wrapper is `script/pretrain_dolma8b.sh`. The nohup launcher forwards all
arguments to it. Common environment overrides are:

| Variable | Default |
|---|---:|
| `SHADOW_DIAGNOSTICS_EVERY` | `10` updates |
| `SHADOW_WORKERS` | `12` |
| `SHADOW_MICRO_BATCH` | `12` |
| `SHADOW_ACCUM` | `16` |
| `SHADOW_MAX_TOKENS` | `8000000000` |
| `SHADOW_DEVICE` | `cuda` |
| `SHADOW_AMP_DTYPE` | `bf16` |
| `SHADOW_FFN_WEIGHT_DTYPE` | `ternary` |
| `SHADOW_FFN_ACT_WARMUP_TOKENS` | `100000000` |
| `SHADOW_MTP_HORIZON` | `2` (base plus one MTP proposal) |
| `SHADOW_MTP_LOSS_WEIGHT` | `0.3` |
| `SHADOW_MTP_LOSS_WARMUP_TOKENS` | `100000000` |
| `SHADOW_DATA_DIR` | compressed Dolma sample path |
| `SHADOW_RUN_DIR` | `pretrain_runs/dolma-8b` |
| `SHADOW_PYTHON_BIN` | existing project virtualenv |

Additional trainer arguments are accepted directly, such as `--lr 2e-4`.
