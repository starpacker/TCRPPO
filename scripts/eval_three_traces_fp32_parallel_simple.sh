#!/bin/bash
# 简化并行FP32评估：每个peptide生成5个TCR，只评估目标peptide，三个traces并行

set -e

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
TFOLD_PYTHON=/home/liuyutian/server/miniconda3/envs/tfold/bin/python
PROJECT_ROOT=/share/liuyutian/tcrppo_v2
cd $PROJECT_ROOT

OUTPUT_DIR="results/fp32_eval_parallel_simple_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

FP32_CACHE_DIR="data/fp32_eval_cache_simple"
mkdir -p $FP32_CACHE_DIR

echo "=========================================="
echo "并行FP32评估（简化版）"
echo "=========================================="
echo "每个peptide 5个TCR，20个peptides = 100 TCRs/trace"
echo "三个traces并行运行在GPU 0,1,2"
echo ""

# 为每个trace使用独立的cache和socket
CACHE_GPU0="$FP32_CACHE_DIR/tfold_cache_trace61_fp32.db"
CACHE_GPU1="$FP32_CACHE_DIR/tfold_cache_trace72_fp32.db"
CACHE_GPU2="$FP32_CACHE_DIR/tfold_cache_trace73_fp32.db"

SOCKET_GPU0="/tmp/tfold_server_fp32_simple_gpu0.sock"
SOCKET_GPU1="/tmp/tfold_server_fp32_simple_gpu1.sock"
SOCKET_GPU2="/tmp/tfold_server_fp32_simple_gpu2.sock"

# 清理旧的sockets
rm -f "$SOCKET_GPU0" "$SOCKET_GPU1" "$SOCKET_GPU2"

echo "启动三个FP32 tFold servers..."

# GPU 0 server for trace61
nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
    --socket "$SOCKET_GPU0" \
    --gpu 0 \
    --chunk-size 64 \
    > logs/tfold_fp32_simple_gpu0.log 2>&1 &
PID_TFOLD0=$!

# GPU 1 server for trace72
nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
    --socket "$SOCKET_GPU1" \
    --gpu 1 \
    --chunk-size 64 \
    > logs/tfold_fp32_simple_gpu1.log 2>&1 &
PID_TFOLD1=$!

# GPU 2 server for trace73
nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
    --socket "$SOCKET_GPU2" \
    --gpu 2 \
    --chunk-size 64 \
    > logs/tfold_fp32_simple_gpu2.log 2>&1 &
PID_TFOLD2=$!

echo "Servers started: GPU0=$PID_TFOLD0, GPU1=$PID_TFOLD1, GPU2=$PID_TFOLD2"
echo "等待15秒让servers初始化..."
sleep 15

echo ""
echo "启动三个并行评估..."

# trace61 on GPU 0
CUDA_VISIBLE_DEVICES=0 TFOLD_SERVER_SOCKET="$SOCKET_GPU0" TFOLD_CACHE_PATH="$CACHE_GPU0" \
    $PYTHON tcrppo_v2/test_tcrs.py \
    --checkpoint output/trace61_dynamic_pool/checkpoints/latest.pt \
    --config configs/trace61_dynamic_pool.yaml \
    --n_tcrs 5 \
    --n_decoys 0 \
    --scorers tfold \
    --output_dir "$OUTPUT_DIR/trace61" \
    > "$OUTPUT_DIR/trace61.log" 2>&1 &
PID1=$!

# trace72 on GPU 1
CUDA_VISIBLE_DEVICES=1 TFOLD_SERVER_SOCKET="$SOCKET_GPU1" TFOLD_CACHE_PATH="$CACHE_GPU1" \
    $PYTHON tcrppo_v2/test_tcrs.py \
    --checkpoint output/trace72_delta_from_trace70/checkpoints/latest.pt \
    --config configs/trace72_adaptive_gate_m0p8.yaml \
    --n_tcrs 5 \
    --n_decoys 0 \
    --scorers tfold \
    --output_dir "$OUTPUT_DIR/trace72" \
    > "$OUTPUT_DIR/trace72.log" 2>&1 &
PID2=$!

# trace73 on GPU 2
CUDA_VISIBLE_DEVICES=2 TFOLD_SERVER_SOCKET="$SOCKET_GPU2" TFOLD_CACHE_PATH="$CACHE_GPU2" \
    $PYTHON tcrppo_v2/test_tcrs.py \
    --checkpoint output/trace73_curriculum_exploration/checkpoints/latest.pt \
    --config configs/trace73_curriculum_exploration.yaml \
    --n_tcrs 5 \
    --n_decoys 0 \
    --scorers tfold \
    --output_dir "$OUTPUT_DIR/trace73" \
    > "$OUTPUT_DIR/trace73.log" 2>&1 &
PID3=$!

echo "评估进程启动:"
echo "  trace61 (GPU 0): PID=$PID1"
echo "  trace72 (GPU 1): PID=$PID2"
echo "  trace73 (GPU 2): PID=$PID3"
echo ""
echo "监控进度:"
echo "  tail -f $OUTPUT_DIR/trace61.log"
echo "  tail -f $OUTPUT_DIR/trace72.log"
echo "  tail -f $OUTPUT_DIR/trace73.log"
echo ""

# 等待所有评估完成
wait $PID1 && echo "[$(date)] trace61 完成"
wait $PID2 && echo "[$(date)] trace72 完成"
wait $PID3 && echo "[$(date)] trace73 完成"

# 停止servers
echo ""
echo "停止tFold servers..."
kill $PID_TFOLD0 $PID_TFOLD1 $PID_TFOLD2 2>/dev/null || true

echo ""
echo "=========================================="
echo "所有评估完成！"
echo "=========================================="
echo ""

# 聚合结果
$PYTHON scripts/aggregate_fp32_eval_results.py --results-dir "$OUTPUT_DIR"

echo "结果保存到: $OUTPUT_DIR"
