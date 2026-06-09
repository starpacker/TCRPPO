#!/bin/bash
# Monitor IL training progress

echo "=========================================="
echo "IL Training Progress Monitor"
echo "=========================================="
echo ""

echo "Running processes:"
ps aux | grep pretrain_il | grep -v grep | awk '{print "  PID:", $2, "| CPU:", $3"%", "| Mem:", $4"%", "| Time:", $10}'
echo ""

echo "----------------------------------------"
echo "Test Script Progress (il_test_features.log):"
echo "----------------------------------------"
tail -20 logs/il_test_features.log
echo ""

echo "----------------------------------------"
echo "Full Experiments Progress (il_experiments_full.log):"
echo "----------------------------------------"
tail -20 logs/il_experiments_full.log
echo ""

echo "----------------------------------------"
echo "Output directories:"
echo "----------------------------------------"
ls -lh output/il_test_* output/il_exp* 2>/dev/null | grep "^d" || echo "  (directories being created...)"
echo ""

echo "To monitor continuously, run:"
echo "  watch -n 10 bash scripts/monitor_il_progress.sh"
