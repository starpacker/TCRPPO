#!/bin/bash
# Sequential FP32 evaluation to avoid server crashes
# Evaluates one trace at a time with dedicated tFold server

set -e

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
TFOLD_PYTHON=/home/liuyutian/server/miniconda3/envs/tfold/bin/python
PROJECT_ROOT=/share/liuyutian/tcrppo_v2
cd $PROJECT_ROOT

OUTPUT_DIR="results/fp32_eval_sequential_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Use separate FP32 cache for each GPU to avoid conflicts
FP32_CACHE_DIR="data/fp32_eval_cache"
mkdir -p $FP32_CACHE_DIR

echo "=========================================="
echo "Sequential FP32 Evaluation"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to evaluate one trace
evaluate_trace() {
    local trace_name=$1
    local checkpoint=$2
    local config=$3
    local gpu=$4
    local cache_path="$FP32_CACHE_DIR/tfold_cache_${trace_name}_fp32.db"
    local socket="/tmp/tfold_server_fp32_${trace_name}.sock"

    echo "=========================================="
    echo "Evaluating $trace_name on GPU $gpu"
    echo "=========================================="
    echo "Checkpoint: $checkpoint"
    echo "Config: $config"
    echo "Cache: $cache_path"
    echo ""

    # Clean up old socket
    rm -f "$socket"

    # Start tFold FP32 server
    echo "Starting FP32 tFold server..."
    nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
        --socket "$socket" \
        --gpu $gpu \
        --chunk-size 64 \
        > logs/tfold_fp32_${trace_name}.log 2>&1 &
    PID_TFOLD=$!

    echo "Server started: PID=$PID_TFOLD"
    echo "Waiting 15s for server to initialize..."
    sleep 15

    # Run evaluation
    echo "Running evaluation..."
    CUDA_VISIBLE_DEVICES=$gpu TFOLD_SERVER_SOCKET="$socket" TFOLD_CACHE_PATH="$cache_path" \
        $PYTHON tcrppo_v2/test_tcrs.py \
        --checkpoint "$checkpoint" \
        --config "$config" \
        --n_tcrs 20 \
        --n_decoys 0 \
        --scorers tfold \
        --output_dir "$OUTPUT_DIR/$trace_name" \
        > "$OUTPUT_DIR/${trace_name}.log" 2>&1

    # Stop server
    echo "Stopping tFold server..."
    kill $PID_TFOLD 2>/dev/null || true
    wait $PID_TFOLD 2>/dev/null || true

    echo "$trace_name evaluation complete!"
    echo ""
}

# Evaluate each trace sequentially
evaluate_trace "trace61" \
    "output/trace61_dynamic_pool/checkpoints/latest.pt" \
    "configs/trace61_dynamic_pool.yaml" \
    0

evaluate_trace "trace72" \
    "output/trace72_delta_from_trace70/checkpoints/latest.pt" \
    "configs/trace72_adaptive_gate_m0p8.yaml" \
    1

evaluate_trace "trace73" \
    "output/trace73_curriculum_exploration/checkpoints/latest.pt" \
    "configs/trace73_curriculum_exploration.yaml" \
    2

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo ""

# Aggregate results
$PYTHON scripts/aggregate_fp32_eval_results.py --results-dir $OUTPUT_DIR

echo "Results saved to: $OUTPUT_DIR"
