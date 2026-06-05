#!/usr/bin/env bash
set -euo pipefail

# Trace31: trace11-style target delta with a 2-edit horizon.
# Usage:
#   scripts/launch_trace31_delta_max2.sh start  trace31_delta_max2 3
#   scripts/launch_trace31_delta_max2.sh status trace31_delta_max2 3
#   scripts/launch_trace31_delta_max2.sh stop   trace31_delta_max2 3

ROOT="/share/liuyutian/tcrppo_v2"
ACTION="${1:-start}"
TRACE_TAG="${2:-trace31_delta_max2}"
GPU_ID="${3:-3}"

export CONFIG_PATH="${CONFIG_PATH:-${ROOT}/configs/test51c.yaml}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-test63_maxsteps_sweep}"
export LOG_PREFIX="${LOG_PREFIX:-test63_maxsteps_sweep}"
export SOCKET_PATH="${SOCKET_PATH:-/tmp/tfold_server_${TRACE_TAG}.sock}"

export REWARD_MODE="${REWARD_MODE:-v2_no_decoy_delta}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
export N_ENVS="${N_ENVS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.02}"
export HIDDEN_DIM="${HIDDEN_DIM:-512}"

export MAX_STEPS="${MAX_STEPS:-2}"
export BAN_STOP="${BAN_STOP:-1}"
export SUB_ONLY="${SUB_ONLY:-0}"
export TERMINAL_REWARD_ONLY="${TERMINAL_REWARD_ONLY:-1}"

export CURRICULUM_L0="${CURRICULUM_L0:-0.5}"
export CURRICULUM_L1="${CURRICULUM_L1:-0.0}"
export CURRICULUM_L2="${CURRICULUM_L2:-0.5}"

export W_AFFINITY="${W_AFFINITY:-1.0}"
export W_DECOY="${W_DECOY:-0.0}"
export W_NATURALNESS="${W_NATURALNESS:-0.05}"
export W_DIVERSITY="${W_DIVERSITY:-0.02}"
export N_CONTRAST_DECOYS="${N_CONTRAST_DECOYS:-0}"

export TRAIN_TARGETS="${TRAIN_TARGETS:-data/tfold_excellent_peptides.txt}"
export TFOLD_CACHE_PATH="${TFOLD_CACHE_PATH:-data/tfold_feature_cache.db}"

exec "${ROOT}/scripts/manage_test51c_amp_stack.sh" "${ACTION}" "${TRACE_TAG}" "${GPU_ID}"
