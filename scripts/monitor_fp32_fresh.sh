#!/bin/bash
# 监控全新FP32评估进度

OUTPUT_DIR="results/fp32_eval_fresh_20260603_122247"

echo "=========================================="
echo "FP32评估进度（全新cache，无BF16污染）"
echo "=========================================="
echo "启动时间: 12:22"
echo "配置: 每个peptide 5个TCR × 20个peptides"
echo ""

# 检查进程
echo "运行状态:"
for trace in trace61 trace72 trace73; do
    if ps aux | grep -q "[t]est_tcrs.py.*${trace}"; then
        echo "  ✓ $trace 运行中"
    else
        echo "  ✗ $trace 未运行"
    fi
done

echo ""
echo "评估进度:"

for trace in trace61 trace72 trace73; do
    log_file="$OUTPUT_DIR/${trace}.log"
    if [ -f "$log_file" ]; then
        # 统计FP32提取数量（source=extract_ok）
        n_extracts=$(grep -c "source=extract_ok" "$log_file" || echo 0)

        # 统计cache命中（应该很少，因为是全新cache）
        n_cache_hits=$(grep -c "source=cache_hit" "$log_file" || echo 0)

        # 当前cache大小
        cache_size=$(grep "total cache size:" "$log_file" | tail -1 | awk -F'total cache size: ' '{print $2}' | awk -F')' '{print $1}')

        # 当前peptide
        current_peptide=$(grep "peptide=" "$log_file" | tail -1 | sed 's/.*peptide=\([A-Z]*\).*/\1/')

        # 最后更新时间
        last_update=$(stat -c %y "$log_file" | cut -d. -f1)

        echo "  $trace:"
        echo "    FP32提取: $n_extracts"
        echo "    Cache命中: $n_cache_hits (应该为0或很少)"
        echo "    Cache大小: $cache_size"
        echo "    当前peptide: $current_peptide"
        echo "    最后更新: $last_update"
    else
        echo "  $trace: 日志不存在"
    fi
    echo ""
done

# 预估完成时间
echo "预估:"
echo "  每个TCR ~19秒 FP32提取"
echo "  每个peptide 5个TCR = ~95秒"
echo "  20个peptides = ~32分钟/trace"
echo "  预计完成: ~12:54"
echo ""
echo "=========================================="
