#!/bin/bash
# trace84: Push to Mean A = 0.0
#
# Strategy:
#   1. Resume from trace73 checkpoint (proven base, step 710K)
#   2. Conservative gate schedule: -2.0 → -1.5 → -1.0 → -0.5 → 0.0
#   3. Gradual decoy unlock: D → D+A → D+A+B → D+A+B+C
#   4. Curriculum: L0/L1 → balanced → mostly L2
#   5. 2M steps total
#
# Expected outcome:
#   - Mean A = 0.0 at ~1.5M steps (gate reaches 0.0)
#   - Maintain specificity via decoy penalty
#   - No catastrophic forgetting (conservative schedule)

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

# Launch training
echo ""
echo "Launching trace84: Push to Mean A = 0.0"
echo "  Resume from: trace73 checkpoint (step 710K)"
echo "  Gate schedule: -2.0 → 0.0 over 1.5M steps"
echo "  Decoy unlock: D → D+A → D+A+B → D+A+B+C"
echo "  Curriculum: L0/L1 → balanced → L2"
echo "  Total steps: 2M"
echo ""

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/trace84_push_to_zero.yaml \
    --resume_from output/trace73_curriculum_exploration/checkpoints/latest.pt \
    --resume_reset_optimizer \
    --run_name trace84_push_to_zero \
    --seed 42 \
    > logs/trace84_push_to_zero_train.log 2>&1 &

PID=$!
echo "trace84 launched on GPU 1 (PID: $PID)"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/trace84_push_to_zero_train.log"
echo "  grep 'Mean A' logs/trace84_push_to_zero_train.log | tail -20"
echo ""
echo "Expected milestones:"
echo "  100K steps: Mean A ~ -1.5 (gate = -1.5)"
echo "  500K steps: Mean A ~ -1.0 (gate = -1.0)"
echo "  1M steps: Mean A ~ -0.5 (gate = -0.5)"
echo "  1.5M steps: Mean A ~ 0.0 (gate = 0.0) ← TARGET!"
echo ""
echo "Kill command: kill $PID"
