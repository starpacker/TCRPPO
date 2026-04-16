#!/bin/bash
# Start the tFold feature extraction server in the background.
# This loads the 735M-param tFold model (~5 min) and then accepts
# feature extraction requests via Unix domain socket.
#
# Usage:
#   bash scripts/start_tfold_server.sh [GPU_ID]
#
# The server PID is written to /tmp/tfold_server.sock.pid
# To stop: kill $(cat /tmp/tfold_server.sock.pid)

GPU_ID="${1:-3}"  # Default to GPU 3
SOCKET_PATH="/tmp/tfold_server.sock"
TFOLD_PYTHON="/home/liuyutian/server/miniconda3/envs/tfold/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="/tmp/tfold_server.log"

# Check if already running
if [ -f "${SOCKET_PATH}.pid" ]; then
    PID=$(cat "${SOCKET_PATH}.pid")
    if kill -0 "$PID" 2>/dev/null; then
        echo "tFold server already running (PID=$PID)"
        exit 0
    fi
    rm -f "${SOCKET_PATH}.pid" "${SOCKET_PATH}"
fi

echo "Starting tFold feature server on GPU ${GPU_ID}..."
echo "  Socket: ${SOCKET_PATH}"
echo "  Log: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup ${TFOLD_PYTHON} "${SCRIPT_DIR}/tfold_feature_server.py" \
    --socket "${SOCKET_PATH}" \
    --gpu 0 \
    > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "  PID: ${SERVER_PID}"
echo ""
echo "Waiting for server to be ready (this takes ~5 min for model loading)..."
echo "Monitor with: tail -f ${LOG_FILE}"
echo ""

# Wait for READY signal
for i in $(seq 1 360); do  # 6 min max
    if grep -q "READY" "${LOG_FILE}" 2>/dev/null; then
        echo "Server is READY! (took ~${i}s)"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died. Check ${LOG_FILE}"
        tail -20 "${LOG_FILE}"
        exit 1
    fi
    sleep 1
done

echo "WARNING: Server did not print READY within 6 min. Check ${LOG_FILE}"
echo "It may still be loading the model."
