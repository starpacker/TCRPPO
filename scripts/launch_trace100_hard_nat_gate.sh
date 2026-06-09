#!/bin/bash
# trace100: Hard naturalness gate from trace99 checkpoint
#
# Hypothesis: trace99 can achieve 74% affinity>0 but only 1.5% naturalness.
# By enabling hard naturalness gate (reject if nat_penalty < -0.3), we force
# the model to maintain naturalness while preserving its affinity capability.
#
# Expected outcome:
#   - Week 1: Nat pass rate 1.5% -> 30%
#   - Week 2: Nat pass rate -> 50%, Affinity>0 maintain >50%
#   - Week 3: Both conditions met rate -> 30%

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

# Resume from trace99's latest checkpoint
RESUME_CKPT="output/trace99_finetune_nat5_from_trace61/checkpoints/latest.pt"

if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: Checkpoint not found: $RESUME_CKPT"
    echo "Available checkpoints:"
    ls -lh output/trace99_finetune_nat5_from_trace61/checkpoints/
    exit 1
fi

echo "==================================================================="
echo "trace100: Hard Naturalness Gate + Resume from trace99"
echo "==================================================================="
echo "Resume from: $RESUME_CKPT"
echo "Key changes:"
echo "  - naturalness_gate_affinity: true (HARD GATE ENABLED)"
echo "  - naturalness_gate_threshold: -0.3 (reject if worse)"
echo "  - w_naturalness: 10.0 (strong penalty)"
echo "  - learning_rate: 5e-5 (prevent forgetting)"
echo "  - entropy_coef: 0.02 (more exploration)"
echo ""
echo "Target metrics:"
echo "  - Naturalness pass rate: 1.5% -> 50%"
echo "  - Affinity > 0 rate: maintain 70%+"
echo "  - Both satisfied: 30%+"
echo "==================================================================="
echo ""

# Launch training
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/trace100_hard_nat_gate.yaml \
    --run_name trace100_hard_nat_gate \
    --resume_checkpoint "$RESUME_CKPT" \
    > logs/trace100_hard_nat_gate_train.log 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > logs/trace100_hard_nat_gate_train.pid
echo "Training launched on GPU 2, PID: $TRAIN_PID"
echo ""

# Launch tFold server
sleep 5
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u tcrppo_v2/tfold_server.py \
    --socket /tmp/tfold_server_trace100.sock \
    --cache data/tfold_cache_trace100.db \
    > logs/trace100_hard_nat_gate_tfold_server.log 2>&1 &

TFOLD_PID=$!
echo $TFOLD_PID > logs/trace100_hard_nat_gate_tfold.pid
echo "tFold server launched, PID: $TFOLD_PID"
echo ""

echo "Monitor commands:"
echo "  tail -f logs/trace100_hard_nat_gate_train.log"
echo "  watch -n 10 'tail -50 logs/trace100_hard_nat_gate_train.log | grep \"Step\"'"
echo ""
echo "Stop commands:"
echo "  kill $TRAIN_PID $TFOLD_PID"
echo ""
echo "Hypothesis: Hard nat gate will force model to explore natural sequences"
echo "while preserving the affinity capability learned in trace99."
