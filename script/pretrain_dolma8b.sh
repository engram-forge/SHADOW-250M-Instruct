#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${SHADOW_DATA_DIR:-/home/dlisuser/quanwen/SHADOW-250M-Instruct/data/dolma/sample-8b}"
RUN_DIR="${SHADOW_RUN_DIR:-${ROOT}/pretrain_runs/dolma-8b}"
PYTHON_BIN="${SHADOW_PYTHON_BIN:-/home/dlisuser/quanwen/SHADOW-250M-Instruct/.venv/bin/python}"
UV_BIN="${SHADOW_UV_BIN:-$(command -v uv || true)}"
EXPECTED_SHARDS=103
EXPECTED_BYTES=16430279784

if [[ -x "${PYTHON_BIN}" ]]; then
  runner=("${PYTHON_BIN}")
elif [[ -n "${UV_BIN}" && -x "${UV_BIN}" ]]; then
  runner=("${UV_BIN}" run python)
else
  echo "error: no Python environment found; set SHADOW_PYTHON_BIN or SHADOW_UV_BIN" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "error: Dolma directory does not exist: ${DATA_DIR}" >&2
  exit 1
fi

read -r shard_count compressed_bytes < <(
  find "${DATA_DIR}" -maxdepth 1 -type f -name '*.json.gz' -printf '%s\n' |
    awk '{bytes += $1; files += 1} END {print files + 0, bytes + 0}'
)
if [[ "${shard_count}" -ne "${EXPECTED_SHARDS}" ]]; then
  echo "error: expected ${EXPECTED_SHARDS} Dolma shards, found ${shard_count}" >&2
  exit 1
fi
if [[ "${compressed_bytes}" -ne "${EXPECTED_BYTES}" ]]; then
  echo "error: expected ${EXPECTED_BYTES} compressed bytes, found ${compressed_bytes}" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "error: no accessible NVIDIA GPU" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"

if [[ "${1:-}" == "--preflight" ]]; then
  echo "preflight OK"
  echo "  repository: ${ROOT}"
  echo "  data:       ${DATA_DIR} (${shard_count} compressed shards)"
  echo "  run:        ${RUN_DIR}"
  echo "  runner:     ${runner[*]}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  exit 0
fi

has_resume=0
for argument in "$@"; do
  if [[ "${argument}" == "--resume" || "${argument}" == --resume=* ]]; then
    has_resume=1
  fi
done
if [[ -e "${RUN_DIR}/checkpoints/final.pt" && "${has_resume}" -eq 0 ]]; then
  echo "error: ${RUN_DIR}/checkpoints/final.pt exists" >&2
  echo "pass --resume CHECKPOINT explicitly to continue, or choose a new SHADOW_RUN_DIR" >&2
  exit 1
fi

cd "${ROOT}"
exec "${runner[@]}" -m pretrain.train train \
  --data "${DATA_DIR}" \
  --out "${RUN_DIR}" \
  --workers "${SHADOW_WORKERS:-12}" \
  --chunk-docs "${SHADOW_CHUNK_DOCS:-2048}" \
  --micro-batch "${SHADOW_MICRO_BATCH:-12}" \
  --accum "${SHADOW_ACCUM:-16}" \
  --max-tokens "${SHADOW_MAX_TOKENS:-8000000000}" \
  --diagnostics-every "${SHADOW_DIAGNOSTICS_EVERY:-10}" \
  --device "${SHADOW_DEVICE:-cuda}" \
  --amp-dtype "${SHADOW_AMP_DTYPE:-bf16}" \
  --ffn-weight-dtype "${SHADOW_FFN_WEIGHT_DTYPE:-ternary}" \
  --ffn-act-warmup-tokens "${SHADOW_FFN_ACT_WARMUP_TOKENS:-100000000}" \
  --mtp-horizon "${SHADOW_MTP_HORIZON:-2}" \
  --mtp-loss-weight "${SHADOW_MTP_LOSS_WEIGHT:-0.3}" \
  --mtp-loss-warmup-tokens "${SHADOW_MTP_LOSS_WARMUP_TOKENS:-100000000}" \
  "$@"
