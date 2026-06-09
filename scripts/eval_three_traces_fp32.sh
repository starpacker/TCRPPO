#!/bin/bash
# Simplified FP32 evaluation using existing test_tcrs.py
# Evaluates trace61, trace72, trace73 with FP32 tFold

set -e

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
PROJECT_ROOT=/share/liuyutian/tcrppo_v2
cd $PROJECT_ROOT

OUTPUT_DIR="results/fp32_eval_simple_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "FP32 Evaluation of Three Traces"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Trace configurations
declare -A TRACES
TRACES[trace61]="output/trace61_dynamic_pool/checkpoints/latest.pt|configs/trace61_dynamic_pool.yaml"
TRACES[trace72]="output/trace72_delta_from_trace70/checkpoints/latest.pt|configs/trace72_delta_from_trace70.yaml"
TRACES[trace73]="output/trace73_curriculum_exploration/checkpoints/latest.pt|configs/trace73_curriculum_exploration.yaml"

# Start tFold FP32 servers on GPUs 0, 1, 2
echo "Starting FP32 tFold servers..."

TFOLD_PYTHON=/home/liuyutian/server/miniconda3/envs/tfold/bin/python

# GPU 0 server
nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_fp32_eval_gpu0.sock \
    --gpu 0 \
    --chunk-size 64 \
    > logs/tfold_fp32_eval_gpu0.log 2>&1 &
PID_TFOLD0=$!

# GPU 1 server
nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_fp32_eval_gpu1.sock \
    --gpu 1 \
    --chunk-size 64 \
    > logs/tfold_fp32_eval_gpu1.log 2>&1 &
PID_TFOLD1=$!

# GPU 2 server
nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_fp32_eval_gpu2.sock \
    --gpu 2 \
    --chunk-size 64 \
    > logs/tfold_fp32_eval_gpu2.log 2>&1 &
PID_TFOLD2=$!

echo "tFold FP32 servers started: GPU0=$PID_TFOLD0, GPU1=$PID_TFOLD1, GPU2=$PID_TFOLD2"
echo "Waiting 10s for servers to initialize..."
sleep 10

# Run evaluations in parallel
echo ""
echo "Launching evaluations..."

# trace61 on GPU 0
CUDA_VISIBLE_DEVICES=0 TFOLD_SERVER_SOCKET=/tmp/tfold_server_fp32_eval_gpu0.sock $PYTHON tcrppo_v2/test_tcrs.py \
    --checkpoint output/trace61_dynamic_pool/checkpoints/latest.pt \
    --config configs/trace61_dynamic_pool.yaml \
    --n_tcrs 20 \
    --n_decoys 0 \
    --scorers tfold \
    --output_dir $OUTPUT_DIR/trace61 \
    > $OUTPUT_DIR/trace61.log 2>&1 &
PID1=$!

# trace72 on GPU 1
CUDA_VISIBLE_DEVICES=1 TFOLD_SERVER_SOCKET=/tmp/tfold_server_fp32_eval_gpu1.sock $PYTHON tcrppo_v2/test_tcrs.py \
    --checkpoint output/trace72_delta_from_trace70/checkpoints/latest.pt \
    --config configs/trace72_delta_from_trace70.yaml \
    --n_tcrs 20 \
    --n_decoys 0 \
    --scorers tfold \
    --output_dir $OUTPUT_DIR/trace72 \
    > $OUTPUT_DIR/trace72.log 2>&1 &
PID2=$!

# trace73 on GPU 2
CUDA_VISIBLE_DEVICES=2 TFOLD_SERVER_SOCKET=/tmp/tfold_server_fp32_eval_gpu2.sock $PYTHON tcrppo_v2/test_tcrs.py \
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

# Wait for all evaluations
wait $PID1 && echo "trace61 completed"
wait $PID2 && echo "trace72 completed"
wait $PID3 && echo "trace73 completed"

# Stop tFold servers
kill $PID_TFOLD0 $PID_TFOLD1 $PID_TFOLD2 2>/dev/null || true

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo ""

# Aggregate results
$PYTHON scripts/aggregate_fp32_eval_results.py --results-dir $OUTPUT_DIR

echo "Results saved to: $OUTPUT_DIR"
