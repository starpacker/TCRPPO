#!/bin/bash
# FP32 evaluation with FRESH cache (no BF16 contamination)
# Creates separate FP32-only cache to avoid mixing with AMP results

set -e

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
PROJECT_ROOT=/share/liuyutian/tcrppo_v2
cd $PROJECT_ROOT

OUTPUT_DIR="results/fp32_eval_fresh_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Create fresh FP32-only cache databases
FP32_CACHE_DIR="data/fp32_eval_cache"
mkdir -p $FP32_CACHE_DIR

CACHE_GPU0="$FP32_CACHE_DIR/tfold_cache_gpu0_fp32.db"
CACHE_GPU1="$FP32_CACHE_DIR/tfold_cache_gpu1_fp32.db"
CACHE_GPU2="$FP32_CACHE_DIR/tfold_cache_gpu2_fp32.db"

echo "=========================================="
echo "FP32 Evaluation with FRESH Cache"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "FP32 caches:"
echo "  GPU 0: $CACHE_GPU0"
echo "  GPU 1: $CACHE_GPU1"
echo "  GPU 2: $CACHE_GPU2"
echo ""

# Remove old FP32 caches if they exist (force fresh extraction)
rm -f "$CACHE_GPU0" "$CACHE_GPU1" "$CACHE_GPU2"
echo "Removed old FP32 caches - will extract all features fresh"
echo ""

TFOLD_PYTHON=/home/liuyutian/server/miniconda3/envs/tfold/bin/python

# Start FP32 tFold servers (NO --use-amp-wrapper flag!)
echo "Starting FP32 tFold servers (no AMP)..."

# GPU 0 server
nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_fp32_fresh_gpu0.sock \
    --gpu 0 \
    --chunk-size 64 \
    > logs/tfold_fp32_fresh_gpu0.log 2>&1 &
PID_TFOLD0=$!

# GPU 1 server
nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_fp32_fresh_gpu1.sock \
    --gpu 1 \
    --chunk-size 64 \
    > logs/tfold_fp32_fresh_gpu1.log 2>&1 &
PID_TFOLD1=$!

# GPU 2 server
nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_fp32_fresh_gpu2.sock \
    --gpu 2 \
    --chunk-size 64 \
    > logs/tfold_fp32_fresh_gpu2.log 2>&1 &
PID_TFOLD2=$!

echo "FP32 servers started: GPU0=$PID_TFOLD0, GPU1=$PID_TFOLD1, GPU2=$PID_TFOLD2"
echo "Waiting 15s for servers to initialize..."
sleep 15

echo ""
echo "Launching evaluations with fresh FP32 caches..."

# trace61 on GPU 0 with fresh cache
CUDA_VISIBLE_DEVICES=0 TFOLD_SERVER_SOCKET=/tmp/tfold_server_fp32_fresh_gpu0.sock TFOLD_CACHE_PATH="$CACHE_GPU0" $PYTHON tcrppo_v2/test_tcrs.py \
    --checkpoint output/trace61_dynamic_pool/checkpoints/latest.pt \
    --config configs/trace61_dynamic_pool.yaml \
    --n_tcrs 20 \
    --n_decoys 0 \
    --scorers tfold \
    --output_dir $OUTPUT_DIR/trace61 \
    > $OUTPUT_DIR/trace61.log 2>&1 &
PID1=$!

# trace72 on GPU 1 with fresh cache
CUDA_VISIBLE_DEVICES=1 TFOLD_SERVER_SOCKET=/tmp/tfold_server_fp32_fresh_gpu1.sock TFOLD_CACHE_PATH="$CACHE_GPU1" $PYTHON tcrppo_v2/test_tcrs.py \
    --checkpoint output/trace72_delta_from_trace70/checkpoints/latest.pt \
    --config configs/trace72_adaptive_gate_m0p8.yaml \
    --n_tcrs 20 \
    --n_decoys 0 \
    --scorers tfold \
    --output_dir $OUTPUT_DIR/trace72 \
    > $OUTPUT_DIR/trace72.log 2>&1 &
PID2=$!

# trace73 on GPU 2 with fresh cache
CUDA_VISIBLE_DEVICES=2 TFOLD_SERVER_SOCKET=/tmp/tfold_server_fp32_fresh_gpu2.sock TFOLD_CACHE_PATH="$CACHE_GPU2" $PYTHON tcrppo_v2/test_tcrs.py \
    --checkpoint output/trace73_curriculum_exploration/checkpoints/latest.pt \
    --config configs/trace73_curriculum_exploration.yaml \
    --n_tcrs 20 \
    --n_decoys 0 \
    --scorers tfold \
    --output_dir $OUTPUT_DIR/trace73 \
    > $OUTPUT_DIR/trace73.log 2>&1 &
PID3=$!

echo "Evaluations launched: trace61=$PID1, trace72=$PID2, trace73=$PID3"
echo ""
echo "Monitor progress:"
echo "  tail -f $OUTPUT_DIR/trace61.log"
echo "  tail -f $OUTPUT_DIR/trace72.log"
echo "  tail -f $OUTPUT_DIR/trace73.log"
echo ""
echo "All features will be extracted fresh with FP32 (no cache hits from BF16)"
echo ""

# Wait for all evaluations
wait $PID1 && echo "trace61 completed"
wait $PID2 && echo "trace72 completed"
wait $PID3 && echo "trace73 completed"

# Stop tFold servers
kill $PID_TFOLD0 $PID_TFOLD1 $PID_TFOLD2 2>/dev/null || true

echo ""
echo "=========================================="
echo "All FP32 evaluations complete!"
echo "=========================================="
echo ""

# Aggregate results
$PYTHON scripts/aggregate_fp32_eval_results.py --results-dir $OUTPUT_DIR

echo "Results saved to: $OUTPUT_DIR"
echo "FP32 caches saved for future use:"
echo "  $CACHE_GPU0"
echo "  $CACHE_GPU1"
echo "  $CACHE_GPU2"
