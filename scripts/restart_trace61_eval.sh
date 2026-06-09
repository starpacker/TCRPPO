#!/bin/bash
# 重启trace61的FP32评估（使用GPU 3）

set -e

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
TFOLD_PYTHON=/home/liuyutian/server/miniconda3/envs/tfold/bin/python
PROJECT_ROOT=/share/liuyutian/tcrppo_v2
cd $PROJECT_ROOT

# 使用原有的输出目录
OUTPUT_DIR="results/fp32_eval_fresh_20260603_122247"

# 使用独立的cache和socket
CACHE_GPU3="data/fp32_fresh_cache/trace61_restart_fp32.db"
SOCKET_GPU3="/tmp/tfold_fp32_fresh_gpu3.sock"

# 清理旧socket
rm -f "$SOCKET_GPU3"

echo "=========================================="
echo "重启trace61 FP32评估（GPU 3）"
echo "=========================================="
echo "Cache: $CACHE_GPU3"
echo "Socket: $SOCKET_GPU3"
echo ""

# 启动GPU 3上的FP32 tFold server
echo "启动FP32 tFold server on GPU 3..."
nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
    --socket "$SOCKET_GPU3" \
    --gpu 3 \
    --chunk-size 64 \
    > logs/tfold_fp32_fresh_gpu3.log 2>&1 &
PID_TFOLD3=$!

echo "Server started: GPU3=$PID_TFOLD3"
echo "等待20秒让server初始化..."
sleep 20

echo ""
echo "启动trace61评估..."

# 重新运行trace61评估（覆盖原有log）
CUDA_VISIBLE_DEVICES=3 TFOLD_SERVER_SOCKET="$SOCKET_GPU3" TFOLD_CACHE_PATH="$CACHE_GPU3" \
    $PYTHON tcrppo_v2/test_tcrs.py \
    --checkpoint output/trace61_dynamic_pool/checkpoints/latest.pt \
    --config configs/trace61_dynamic_pool.yaml \
    --n_tcrs 5 \
    --n_decoys 0 \
    --scorers tfold \
    --output_dir "$OUTPUT_DIR/trace61" \
    > "$OUTPUT_DIR/trace61_restart.log" 2>&1 &
PID_EVAL=$!

echo "trace61评估进程启动: PID=$PID_EVAL"
echo "监控日志: tail -f $OUTPUT_DIR/trace61_restart.log"
echo "监控server: tail -f logs/tfold_fp32_fresh_gpu3.log"
echo ""
echo "预计完成时间: 约8小时后"
echo "=========================================="
