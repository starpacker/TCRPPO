#!/bin/bash
# 监控trace61和trace73的FP32评估进度

echo "=========================================="
echo "FP32评估进度监控"
echo "=========================================="
date
echo ""

# 检查进程状态
echo "运行状态:"
if ps aux | grep -q "[t]est_tcrs.py.*trace61"; then
    echo "  ✓ trace61 运行中 (GPU 3)"
else
    echo "  ✗ trace61 未运行"
fi

if ps aux | grep -q "[t]est_tcrs.py.*trace73"; then
    echo "  ✓ trace73 运行中 (GPU 2)"
else
    echo "  ✗ trace73 未运行"
fi

echo ""
echo "=========================================="

# 目标peptides列表
TARGETS=(GILGFVFTL ELAGIGILTV GLCTLVAML RAKFKQLL NLVPMVATV CINGVCWTV TPRVTGGGAM IPSINVHHY KLGGALQAK LLWNGPMAV FLASKIGRLV RLRAEAQVK AVFDRKSDAK ATDALMTGY IMNDMPIYM YLQPRTFLL SLFNTVATLY RLRPGGKKK KRWIILGLNK LLLDRLNQL)

# trace61进度
echo "trace61 (GPU 3, 重启):"
LOG61="results/fp32_eval_fresh_20260603_122247/trace61_restart.log"
if [ -f "$LOG61" ]; then
    n_success=0
    for peptide in "${TARGETS[@]}"; do
        if grep -q "source=extract_ok.*peptide=$peptide " "$LOG61" 2>/dev/null; then
            ((n_success++))
        fi
    done

    n_extracts=$(grep -c "source=extract_ok" "$LOG61" 2>/dev/null || echo 0)
    last_update=$(stat -c %y "$LOG61" 2>/dev/null | cut -d. -f1 || echo "N/A")

    echo "  已完成peptides: $n_success/20"
    echo "  总提取数: $n_extracts"
    echo "  最后更新: $last_update"
else
    echo "  日志不存在"
fi

echo ""

# trace73进度
echo "trace73 (GPU 2, 原始):"
LOG73="results/fp32_eval_fresh_20260603_122247/trace73.log"
if [ -f "$LOG73" ]; then
    n_success=0
    for peptide in "${TARGETS[@]}"; do
        if grep -q "source=extract_ok.*peptide=$peptide " "$LOG73" 2>/dev/null; then
            ((n_success++))
        fi
    done

    n_extracts=$(grep -c "source=extract_ok" "$LOG73" 2>/dev/null || echo 0)
    last_update=$(stat -c %y "$LOG73" 2>/dev/null | cut -d. -f1 || echo "N/A")

    echo "  已完成peptides: $n_success/20"
    echo "  总提取数: $n_extracts"
    echo "  最后更新: $last_update"
else
    echo "  日志不存在"
fi

echo ""
echo "=========================================="
echo "提示: 运行 'python scripts/analyze_target_only_results.py' 查看详细结果"
echo "=========================================="
