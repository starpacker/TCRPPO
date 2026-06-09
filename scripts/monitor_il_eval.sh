#!/bin/bash
# Monitor IL evaluation progress

echo "=========================================="
echo "IL Checkpoint Evaluation Monitor"
echo "=========================================="
echo ""

echo "Checkpoint being evaluated:"
echo "  /share/liuyutian/tcrppo_v2/output/il_exp4_scratch_early_stop/checkpoints/best.pt"
echo ""

echo "----------------------------------------"
echo "Latest log output:"
echo "----------------------------------------"
tail -20 logs/il_exp4_best_eval.log
echo ""

echo "----------------------------------------"
echo "Generated files:"
echo "----------------------------------------"
ls -lh results/il_exp4_best_eval/ 2>/dev/null || echo "  (directory being created...)"
echo ""

echo "----------------------------------------"
echo "Process status:"
echo "----------------------------------------"
ps aux | grep eval_checkpoint_decoy_reward_tfold | grep -v grep | head -1 || echo "  Process completed or not found"
echo ""

echo "To monitor continuously:"
echo "  watch -n 10 bash scripts/monitor_il_eval.sh"
echo ""
echo "To view full log:"
echo "  tail -f logs/il_exp4_best_eval.log"
