#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/liuyutian/tcrppo_v2"
TRAIN_PYTHON="/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python"
TFOLD_PYTHON="/home/liuyutian/server/miniconda3/envs/tfold/bin/python"
CONFIG_PATH="${CONFIG_PATH:-${ROOT}/configs/test51c.yaml}"
ACTION="${1:-status}"
TRACE_TAG="${2:-trace4}"
GPU_ID="${3:-3}"

RUN_NAME="test51c_amp_server_rerun_detached_${TRACE_TAG}"
LOG_DIR="${ROOT}/logs"
STATE_DIR="${ROOT}/run_state"
SOCKET_PATH="${SOCKET_PATH:-/tmp/tfold_server_${TRACE_TAG}.sock}"
W_AFFINITY="${W_AFFINITY:-1.0}"
W_NATURALNESS="${W_NATURALNESS:-0.5}"
W_DIVERSITY="${W_DIVERSITY:-0.2}"
REWARD_MODE="${REWARD_MODE:-v2_no_decoy}"
ENTROPY_COEF="${ENTROPY_COEF:-0.05}"
CONFIG_RELATIVE="${CONFIG_PATH#${ROOT}/}"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}_train.log"
SERVER_LOG="${LOG_DIR}/test51c_tfold_amp_server_${TRACE_TAG}.log"
COMPLETION_LOG="${LOG_DIR}/test51c_tfold_completion_${TRACE_TAG}.log"
CONTROL_LOG="${LOG_DIR}/test51c_amp_stack_${TRACE_TAG}.launch.log"
TRAIN_PID_FILE="${STATE_DIR}/${RUN_NAME}.trainer.pid"
SERVER_PID_FILE="${STATE_DIR}/${RUN_NAME}.server.pid"
LOCK_FILE="${STATE_DIR}/${RUN_NAME}.control.lock"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"
exec 9>"${LOCK_FILE}"
flock -n 9 || {
    echo "Another launcher instance is already operating on ${RUN_NAME}."
    exit 1
}

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_control() {
    printf '%s %s\n' "$(timestamp)" "$*" | tee -a "${CONTROL_LOG}"
}

pid_alive() {
    local pid="${1:-}"
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

read_pid() {
    local pid_file="$1"
    if [[ -f "${pid_file}" ]]; then
        tr -d '[:space:]' < "${pid_file}"
    fi
}

remove_stale_pidfile() {
    local pid_file="$1"
    local pid
    pid="$(read_pid "${pid_file}")"
    if [[ -n "${pid}" ]] && ! pid_alive "${pid}"; then
        rm -f "${pid_file}"
    fi
}

cleanup_stale_socket() {
    remove_stale_pidfile "${SERVER_PID_FILE}"
    local server_pid
    server_pid="$(read_pid "${SERVER_PID_FILE}")"
    if [[ -S "${SOCKET_PATH}" ]] && [[ -z "${server_pid}" ]]; then
        log_control "Removing stale socket ${SOCKET_PATH}"
        rm -f "${SOCKET_PATH}" "${SOCKET_PATH}.pid"
    fi
}

assert_preflight() {
    if ! rg -n '^checkpoint_interval:\s*20000\b' "${CONFIG_PATH}" >/dev/null; then
        echo "Refusing to launch: ${CONFIG_PATH} does not contain checkpoint_interval: 20000"
        exit 1
    fi
    if [[ ! -x "${TRAIN_PYTHON}" ]]; then
        echo "Missing trainer python: ${TRAIN_PYTHON}"
        exit 1
    fi
    if [[ ! -x "${TFOLD_PYTHON}" ]]; then
        echo "Missing tFold python: ${TFOLD_PYTHON}"
        exit 1
    fi
}

wait_for_exit() {
    local pid="$1"
    local label="$2"
    local timeout_s="${3:-60}"
    local waited=0
    while pid_alive "${pid}"; do
        if (( waited >= timeout_s )); then
            log_control "${label} pid=${pid} did not exit after ${timeout_s}s; sending SIGKILL"
            kill -KILL "${pid}" 2>/dev/null || true
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
}

ping_server() {
    if [[ ! -S "${SOCKET_PATH}" ]]; then
        return 1
    fi

    "${TRAIN_PYTHON}" - "${SOCKET_PATH}" <<'PY'
import json
import socket
import struct
import sys

sock_path = sys.argv[1]
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(5)
sock.connect(sock_path)
payload = json.dumps({"cmd": "ping"}).encode("utf-8")
sock.sendall(struct.pack(">I", len(payload)) + payload)
header = sock.recv(4)
if len(header) != 4:
    raise SystemExit(2)
msg_len = struct.unpack(">I", header)[0]
buf = b""
while len(buf) < msg_len:
    chunk = sock.recv(msg_len - len(buf))
    if not chunk:
        raise SystemExit(3)
    buf += chunk
resp = json.loads(buf.decode("utf-8"))
raise SystemExit(0 if resp.get("status") == "pong" else 4)
PY
}

wait_for_server_ready() {
    local server_pid="$1"
    local timeout_s="${2:-900}"
    local waited=0

    while (( waited < timeout_s )); do
        if ! pid_alive "${server_pid}"; then
            log_control "Server pid=${server_pid} died during startup"
            tail -n 80 "${SERVER_LOG}" || true
            return 1
        fi

        if grep -q "READY" "${SERVER_LOG}" 2>/dev/null && ping_server >/dev/null 2>&1; then
            log_control "Server READY on ${SOCKET_PATH} after ${waited}s"
            return 0
        fi

        sleep 1
        waited=$((waited + 1))
    done

    log_control "Server did not become READY within ${timeout_s}s"
    tail -n 80 "${SERVER_LOG}" || true
    return 1
}

start_server() {
    cleanup_stale_socket

    local existing_pid
    existing_pid="$(read_pid "${SERVER_PID_FILE}")"
    if [[ -n "${existing_pid}" ]] && pid_alive "${existing_pid}"; then
        log_control "Server already running pid=${existing_pid}"
        wait_for_server_ready "${existing_pid}"
        return 0
    fi

    : > "${SERVER_LOG}"
    : > "${COMPLETION_LOG}"

    log_control "Starting tFold AMP server on GPU ${GPU_ID}; socket=${SOCKET_PATH}"
    (
        cd "${ROOT}"
        export CUDA_VISIBLE_DEVICES="${GPU_ID}"
        exec 9>&-
        nohup setsid "${TFOLD_PYTHON}" "${ROOT}/scripts/tfold_feature_server.py" \
            --socket "${SOCKET_PATH}" \
            --gpu 0 \
            --use-amp-wrapper \
            --chunk-size 64 \
            --completion-log "${COMPLETION_LOG}" \
            < /dev/null >> "${SERVER_LOG}" 2>&1 &
        echo $! > "${SERVER_PID_FILE}"
    )

    local server_pid
    server_pid="$(read_pid "${SERVER_PID_FILE}")"
    if [[ -z "${server_pid}" ]]; then
        log_control "Failed to capture server pid"
        return 1
    fi

    log_control "Server launched pid=${server_pid}"
    wait_for_server_ready "${server_pid}"
}

start_trainer() {
    remove_stale_pidfile "${TRAIN_PID_FILE}"

    local existing_pid
    existing_pid="$(read_pid "${TRAIN_PID_FILE}")"
    if [[ -n "${existing_pid}" ]] && pid_alive "${existing_pid}"; then
        log_control "Trainer already running pid=${existing_pid}"
        return 0
    fi

    : > "${TRAIN_LOG}"

    log_control "Starting trainer run_name=${RUN_NAME} on GPU ${GPU_ID} reward_mode=${REWARD_MODE} entropy=${ENTROPY_COEF} config=${CONFIG_RELATIVE}"
    (
        cd "${ROOT}"
        export CUDA_VISIBLE_DEVICES="${GPU_ID}"
        exec 9>&-
        nohup setsid "${TRAIN_PYTHON}" -u tcrppo_v2/ppo_trainer.py \
            --config "${CONFIG_RELATIVE}" \
            --run_name "${RUN_NAME}" \
            --seed 42 \
            --reward_mode "${REWARD_MODE}" \
            --affinity_scorer tfold \
            --tfold_server_socket "${SOCKET_PATH}" \
            --encoder esm2 \
            --total_timesteps 2000000 \
            --n_envs 8 \
            --learning_rate 3e-4 \
            --entropy_coef "${ENTROPY_COEF}" \
            --hidden_dim 512 \
            --max_steps 8 \
            --ban_stop \
            --terminal_reward_only \
            --n_contrast_decoys 0 \
            --w_affinity "${W_AFFINITY}" \
            --w_naturalness "${W_NATURALNESS}" \
            --w_diversity "${W_DIVERSITY}" \
            --curriculum_l0 0.5 \
            --curriculum_l1 0.0 \
            --curriculum_l2 0.5 \
            --train_targets data/tfold_excellent_peptides.txt \
            --tfold_cache_path data/tfold_feature_cache.db \
            < /dev/null >> "${TRAIN_LOG}" 2>&1 &
        echo $! > "${TRAIN_PID_FILE}"
    )

    local trainer_pid
    trainer_pid="$(read_pid "${TRAIN_PID_FILE}")"
    if [[ -z "${trainer_pid}" ]]; then
        log_control "Failed to capture trainer pid"
        return 1
    fi
    log_control "Trainer launched pid=${trainer_pid}"
}

stop_stack() {
    remove_stale_pidfile "${TRAIN_PID_FILE}"
    remove_stale_pidfile "${SERVER_PID_FILE}"

    local trainer_pid
    trainer_pid="$(read_pid "${TRAIN_PID_FILE}")"
    if [[ -n "${trainer_pid}" ]] && pid_alive "${trainer_pid}"; then
        log_control "Stopping trainer pid=${trainer_pid}"
        kill -TERM "${trainer_pid}" 2>/dev/null || true
        wait_for_exit "${trainer_pid}" "trainer" 60
    fi
    rm -f "${TRAIN_PID_FILE}"

    local server_pid
    server_pid="$(read_pid "${SERVER_PID_FILE}")"
    if [[ -n "${server_pid}" ]] && pid_alive "${server_pid}"; then
        log_control "Stopping server pid=${server_pid}"
        kill -TERM "${server_pid}" 2>/dev/null || true
        wait_for_exit "${server_pid}" "server" 60
    fi
    rm -f "${SERVER_PID_FILE}"
    rm -f "${SOCKET_PATH}" "${SOCKET_PATH}.pid"
}

print_status() {
    remove_stale_pidfile "${TRAIN_PID_FILE}"
    remove_stale_pidfile "${SERVER_PID_FILE}"

    local trainer_pid server_pid
    trainer_pid="$(read_pid "${TRAIN_PID_FILE}")"
    server_pid="$(read_pid "${SERVER_PID_FILE}")"

    echo "run_name=${RUN_NAME}"
    echo "gpu=${GPU_ID}"
    echo "socket=${SOCKET_PATH}"
    local weights_line
    weights_line=""
    if [[ -f "${TRAIN_LOG}" ]]; then
        weights_line="$(rg -m1 'weights: aff=' "${TRAIN_LOG}" || true)"
    fi
    if [[ -n "${weights_line}" ]]; then
        echo "weights=${weights_line#*weights: }"
    else
        echo "weights=affinity:${W_AFFINITY},naturalness:${W_NATURALNESS},diversity:${W_DIVERSITY}"
    fi
    echo "trainer_pid=${trainer_pid:-none}"
    echo "trainer_alive=$([[ -n "${trainer_pid}" ]] && pid_alive "${trainer_pid}" && echo 1 || echo 0)"
    echo "server_pid=${server_pid:-none}"
    echo "server_alive=$([[ -n "${server_pid}" ]] && pid_alive "${server_pid}" && echo 1 || echo 0)"
    echo "socket_present=$([[ -S "${SOCKET_PATH}" ]] && echo 1 || echo 0)"
    echo "train_log=${TRAIN_LOG}"
    echo "server_log=${SERVER_LOG}"
    echo "completion_log=${COMPLETION_LOG}"

    if [[ -f "${TRAIN_LOG}" ]]; then
        local last_step
        last_step="$(rg -o 'Step [0-9,]+' "${TRAIN_LOG}" | tail -n 1 || true)"
        if [[ -n "${last_step}" ]]; then
            echo "last_train_step=${last_step#Step }"
        fi
    fi

    if [[ -f "${COMPLETION_LOG}" ]] && [[ -s "${COMPLETION_LOG}" ]]; then
        awk '
            match($0, /elapsed_s=([0-9.]+)/, a) {sum += a[1]; n += 1}
            match($0, /fallback=([0-9]+)/, b) {fb += b[1]}
            END {
                if (n > 0) {
                    printf "recent_avg_elapsed_s=%.3f\n", sum / n
                    printf "recent_xs_per_sample=%.3f\n", sum / n
                    printf "recent_fallback_count=%d\n", fb
                    printf "completion_samples=%d\n", n
                }
            }
        ' < <(tail -n 50 "${COMPLETION_LOG}")
    fi
}

start_stack() {
    assert_preflight
    start_server
    start_trainer
    print_status
}

case "${ACTION}" in
    start)
        start_stack
        ;;
    stop)
        stop_stack
        print_status
        ;;
    restart)
        stop_stack
        start_stack
        ;;
    status)
        print_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status} [trace_tag] [gpu_id]"
        exit 1
        ;;
esac
