#!/bin/bash
# Launch Test 6: Pure v2 Architecture (A1+A2+A10 only)
# - A1: Indel action space (SUB/INS/DEL/STOP)
# - A2: ESM-2 state encoding
# - A10: Raw ERGO reward (no z-norm, no penalties)
# - NO L0 curriculum (random TCRdb init)
# - NO decoy/naturalness/diversity penalties

CUDA_VISIBLE_DEVICES=5 nohup /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python -u \
    tcrppo_v2/ppo_trainer.py \
    --config configs/test6_pure_v2.yaml \
    --run_name test6_pure_v2_arch \
    --reward_mode v1_ergo_only \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --seed 42 \
    > output/test6_pure_v2_arch_train.log 2>&1 &

echo "Test 6 (Pure v2 Arch: A1+A2+A10) launched on GPU 5, PID: $!"
echo "Log: output/test6_pure_v2_arch_train.log"
echo ""
echo "This experiment tests ONLY the v2 architecture improvements:"
echo "  - Indel action space (A1)"
echo "  - ESM-2 state encoding (A2)"
echo "  - Raw ERGO terminal reward (A10)"
echo "WITHOUT any curriculum (L0/L1) or penalties (decoy/nat/div)"
