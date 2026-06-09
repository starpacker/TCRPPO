#!/bin/bash
# Launch ultimate experiments - targeting mean affinity 0.0!

set -e

echo "=========================================="
echo "TCRPPO v2 Ultimate Experiments"
echo "Goal: Mean Affinity 0.0+"
echo "=========================================="
echo ""

# Check if we should run fresh start or SFT+RL or both
if [ "$1" == "fresh" ] || [ "$1" == "both" ]; then
    echo "[1/2] Launching trace91_ultimate_fresh_start..."
    echo "  - True fresh start (or resume from trace72)"
    echo "  - 3M timesteps"
    echo "  - Naturalness improved (threshold=1.0, weight=0.15)"
    echo ""

    nohup python tcrppo_v2/train.py \
        --config configs/trace91_ultimate_fresh_start.yaml \
        > logs/trace91_ultimate_fresh_start_train.log 2>&1 &

    PID1=$!
    echo "  Started with PID: $PID1"
    echo "  Log: logs/trace91_ultimate_fresh_start_train.log"
    echo ""
fi

if [ "$1" == "sft" ] || [ "$1" == "both" ]; then
    echo "[2/2] Launching trace92_ultimate_sft_rl..."
    echo "  - Start from SFT checkpoint (or trace72 if no SFT)"
    echo "  - 2M timesteps"
    echo "  - Aggressive gate schedule: -1.5 → +0.5"
    echo ""

    nohup python tcrppo_v2/train.py \
        --config configs/trace92_ultimate_sft_rl.yaml \
        > logs/trace92_ultimate_sft_rl_train.log 2>&1 &

    PID2=$!
    echo "  Started with PID: $PID2"
    echo "  Log: logs/trace92_ultimate_sft_rl_train.log"
    echo ""
fi

if [ "$1" != "fresh" ] && [ "$1" != "sft" ] && [ "$1" != "both" ]; then
    echo "Usage: $0 [fresh|sft|both]"
    echo ""
    echo "Options:"
    echo "  fresh - Run trace91 (fresh start)"
    echo "  sft   - Run trace92 (SFT + RL)"
    echo "  both  - Run both experiments"
    exit 1
fi

echo "=========================================="
echo "Experiments launched!"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/trace91_ultimate_fresh_start_train.log"
echo "  tail -f logs/trace92_ultimate_sft_rl_train.log"
echo ""
echo "Check affinity:"
echo "  grep 'Mean A' logs/trace91_ultimate_fresh_start_train.log | tail -20"
echo "  grep 'Mean A' logs/trace92_ultimate_sft_rl_train.log | tail -20"
echo ""
echo "Target: Mean Affinity 0.0+ 🎯"
echo "=========================================="
