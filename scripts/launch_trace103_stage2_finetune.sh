#!/bin/bash
# trace103 Stage 2: Delta reward finetune from naturalness pretrain
#
# Strategy: Resume from stage1 (10k nat-pretrained checkpoint), use delta reward.
# Delta reward solves the "absolute affinity often negative" problem.
#
# Expected outcome:
#   - Week 1: Maintain nat ~0.8, affinity delta improves, 20% pass target gate
#   - Week 2: 40% pass gate, naturalness preserved
#   - Week 3: 50%+ both satisfied (good nat + pass gate)

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

# Resume from stage1's 10k checkpoint
RESUME_CKPT="output/trace103_stage1_nat_pretrain/checkpoints/milestone_10000.pt"

if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: Stage1 checkpoint not found: $RESUME_CKPT"
    echo "Please complete stage1 first: ./scripts/launch_trace103_stage1_pretrain.sh"
    echo ""
    echo "Available checkpoints:"
    ls -lh output/trace103_stage1_nat_pretrain/checkpoints/ 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

echo "==================================================================="
echo "trace103 Stage 2: Delta Reward Finetune"
echo "==================================================================="
echo "Resume from: $RESUME_CKPT"
echo "Key features:"
echo "  - use_delta_reward: true (reward improvement, not absolute)"
echo "  - target_decoy_gate_logit: 0.0"
echo "  - target_pass_bonus: 3.0 (strong incentive)"
echo "  - w_naturalness: 8.0 (preserve pretrain gains)"
echo "  - learning_rate: 5e-5 (low to avoid forgetting)"
echo ""
echo "Why delta reward:"
echo "  Absolute: aff=-2, nat=-0.3 → reward = -2 + 8*(-0.3) = -4.4 ❌"
echo "  Delta: Δaff=+3, nat=-0.3 → reward = +3 + 8*(-0.3) = +0.6 ✅"
echo ""
echo "Target metrics:"
echo "  - Naturalness: maintain 0.8+ from stage1"
echo "  - Affinity > gate (0.0): 50%+"
echo "  - Both satisfied: 40%+"
echo "==================================================================="
echo ""

# Launch tFold server first
echo "Starting tFold server..."
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON -u scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_trace103.sock \
    --gpu 0 \
    > logs/trace103_stage2_delta_finetune_tfold_server.log 2>&1 &

TFOLD_PID=$!
echo $TFOLD_PID > logs/trace103_stage2_delta_finetune_tfold.pid
echo "tFold server launched, PID: $TFOLD_PID"

# Wait for tFold to be ready
echo "Waiting for tFold server to initialize..."
sleep 30

# Check if tFold is ready
if ! grep -q "READY" logs/trace103_stage2_delta_finetune_tfold_server.log; then
    echo "WARNING: tFold server may not be ready yet. Check log:"
    tail -20 logs/trace103_stage2_delta_finetune_tfold_server.log
    echo ""
    echo "Proceeding anyway... training will wait for tFold."
    echo ""
fi

# Launch training
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/trace103_stage2_delta_finetune.yaml \
    --run_name trace103_stage2_delta_finetune \
    --resume_from "$RESUME_CKPT" \
    > logs/trace103_stage2_delta_finetune_train.log 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > logs/trace103_stage2_delta_finetune_train.pid
echo "Training launched on GPU 3, PID: $TRAIN_PID"
echo ""

echo "Monitor commands:"
echo "  tail -f logs/trace103_stage2_delta_finetune_train.log"
echo "  python3 scripts/monitor_trace103.py"
echo ""
echo "Stop commands:"
echo "  kill $TRAIN_PID $TFOLD_PID"
echo ""
echo "Hypothesis: Delta reward + naturalness pretrain will achieve"
echo "both high naturalness and good affinity without trade-off."
