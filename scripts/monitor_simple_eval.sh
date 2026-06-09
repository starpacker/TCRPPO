#!/bin/bash
# 监控简化FP32评估进度

OUTPUT_DIR="results/fp32_eval_parallel_simple_20260603_114658"

echo "=========================================="
echo "FP32评估进度（简化版）"
echo "=========================================="
echo "配置: 每个peptide 5个TCR × 20个peptides = 100 TCRs/trace"
echo ""

# 检查进程
echo "运行状态:"
if ps aux | grep -q "[t]est_tcrs.py.*trace61"; then
    echo "  ✓ trace61 运行中"
else
    echo "  ✗ trace61 未运行"
fi

if ps aux | grep -q "[t]est_tcrs.py.*trace72"; then
    echo "  ✓ trace72 运行中"
else
    echo "  ✗ trace72 未运行"
fi

if ps aux | grep -q "[t]est_tcrs.py.*trace73"; then
    echo "  ✓ trace73 运行中"
else
    echo "  ✗ trace73 未运行"
fi

echo ""
echo "评估进度:"

for trace in trace61 trace72 trace73; do
    log_file="$OUTPUT_DIR/${trace}.log"
    if [ -f "$log_file" ]; then
        # 统计已评估的TCR数量（通过tFoldScore记录）
        n_scores=$(grep -c "tFoldScore" "$log_file" || echo 0)
        last_update=$(stat -c %y "$log_file" | cut -d. -f1)

        # 提取最新的peptide
        current_peptide=$(grep "peptide=" "$log_file" | tail -1 | sed 's/.*peptide=\([A-Z]*\).*/\1/')

        # 统计cache命中和新提取
        cache_hits=$(grep -c "source=cache_hit" "$log_file" || echo 0)
        extracts=$(grep -c "source=extract_ok" "$log_file" || echo 0)

        echo "  $trace:"
        echo "    评估数: $n_scores"
        echo "    Cache命中: $cache_hits, 新提取: $extracts"
        echo "    当前peptide: $current_peptide"
        echo "    最后更新: $last_update"
    else
        echo "  $trace: 日志文件不存在"
    fi
    echo ""
done

echo "=========================================="
