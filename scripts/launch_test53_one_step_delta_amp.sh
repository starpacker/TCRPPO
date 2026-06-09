#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/launch_test53_one_step_delta_amp.sh start  trace19_one_step_delta_amp 3
#   scripts/launch_test53_one_step_delta_amp.sh status trace19_one_step_delta_amp 3
#   scripts/launch_test53_one_step_delta_amp.sh stop   trace19_one_step_delta_amp 3
#
# This is a thin wrapper around the shared tFold-server + PPO manager. It keeps
# the test53 defaults together without duplicating the server lifecycle logic.

ROOT="/share/liuyutian/tcrppo_v2"
ACTION="${1:-start}"
TRACE_TAG="${2:-trace19_one_step_delta_amp}"
GPU_ID="${3:-3}"

export CONFIG_PATH="${CONFIG_PATH:-${ROOT}/configs/test53_one_step_delta_amp.yaml}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-test53_one_step_delta_amp}"
export LOG_PREFIX="${LOG_PREFIX:-test53_one_step_delta_amp}"
export SOCKET_PATH="${SOCKET_PATH:-/tmp/tfold_server_${TRACE_TAG}.sock}"

export REWARD_MODE="${REWARD_MODE:-tfold_delta_amplified}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
export N_ENVS="${N_ENVS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.05}"
export HIDDEN_DIM="${HIDDEN_DIM:-512}"

export MAX_STEPS="${MAX_STEPS:-1}"
export BAN_STOP="${BAN_STOP:-1}"
export SUB_ONLY="${SUB_ONLY:-1}"
export TERMINAL_REWARD_ONLY="${TERMINAL_REWARD_ONLY:-1}"

export CURRICULUM_L0="${CURRICULUM_L0:-0.0}"
export CURRICULUM_L1="${CURRICULUM_L1:-0.0}"
export CURRICULUM_L2="${CURRICULUM_L2:-1.0}"

export W_AFFINITY="${W_AFFINITY:-1.0}"
export W_NATURALNESS="${W_NATURALNESS:-0.5}"
export W_DIVERSITY="${W_DIVERSITY:-0.0}"

export TRAIN_TARGETS="${TRAIN_TARGETS:-data/tfold_easy3_peptides.txt}"
export TFOLD_CACHE_PATH="${TFOLD_CACHE_PATH:-data/tfold_feature_cache.db}"

exec "${ROOT}/scripts/manage_test51c_amp_stack.sh" "${ACTION}" "${TRACE_TAG}" "${GPU_ID}"
