#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/liuyutian/tcrppo_v2"
PYTHON="/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python"
GPU_ID="${1:-2}"
TRACE_TAG="${2:-trace25_maxstep_curriculum}"
LOG="${ROOT}/logs/test56_maxstep_curriculum_${TRACE_TAG}_controller.log"
PID_FILE="${ROOT}/run_state/test56_maxstep_curriculum_${TRACE_TAG}.controller.pid"

mkdir -p "${ROOT}/logs" "${ROOT}/run_state"

if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(tr -d '[:space:]' < "${PID_FILE}")"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
        echo "controller already running pid=${old_pid}"
        exit 0
    fi
fi

cd "${ROOT}"
nohup setsid "${PYTHON}" -u "${ROOT}/scripts/run_test56_maxstep_curriculum.py" \
    --gpu "${GPU_ID}" \
    --trace-tag "${TRACE_TAG}" \
    --enable-plateau \
    > "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"

echo "launched test56 max_steps curriculum controller"
echo "pid=$(cat "${PID_FILE}")"
echo "gpu=${GPU_ID}"
echo "log=${LOG}"
