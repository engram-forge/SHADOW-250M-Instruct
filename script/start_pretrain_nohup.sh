#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${SHADOW_RUN_DIR:-${ROOT}/pretrain_runs/dolma-8b}"
PID_FILE="${RUN_DIR}/train.pid"
LOG_DIR="${RUN_DIR}/logs"

if [[ "${1:-}" == "--preflight" ]]; then
  exec "${ROOT}/script/pretrain_dolma8b.sh" --preflight
fi

mkdir -p "${LOG_DIR}"
if [[ -s "${PID_FILE}" ]]; then
  existing_pid="$(<"${PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "error: pretraining already appears to be running as PID ${existing_pid}" >&2
    echo "log directory: ${LOG_DIR}" >&2
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${LOG_DIR}/train-${timestamp}.log"
status_file="${LOG_DIR}/train-${timestamp}.status"

nohup setsid bash -c '
  status_file=$1
  shift
  trap "printf "%s\n" 143 >"$status_file"" TERM
  "$@"
  status=$?
  printf "%s\n" "$status" >"$status_file"
  exit "$status"
' bash "${status_file}" "${ROOT}/script/pretrain_dolma8b.sh" "$@" >"${log_file}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" >"${PID_FILE}"
sleep 1

if ! kill -0 "${pid}" 2>/dev/null; then
  echo "error: pretraining exited during startup; inspect ${log_file}" >&2
  tail -n 30 "${log_file}" >&2 || true
  exit 1
fi

echo "pretraining started"
echo "  PID: ${pid}"
echo "  log: ${log_file}"
echo "  status: ${status_file} (written when the process exits)"
echo "monitor with: tail -f '${log_file}'"
echo "stop with:    kill -- '-${pid}'"
