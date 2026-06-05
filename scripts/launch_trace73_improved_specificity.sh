#!/bin/bash
# trace73: 从 trace72 继续，强化 specificity
# 关键改进:
# 1. 增加 w_decoy: 0.3 → 0.6
# 2. 严格筛选 online pool: max_decoy_violation 0.5, min_affinity -1.0
# 3. 使用 L0+L1 curriculum (高质量初始化)
# 4. 目标: Advantage 稳定到 0.0 以上

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --run_name trace73_improved_specificity \
    --seed 43 \
    --resume_from output/trace72_delta_from_trace70/checkpoints/latest.pt \
    --total_timesteps 1500000 \
    --n_envs 8 \
    --learning_rate 0.00015 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --terminal_reward_only \
    --affinity_scorer tfold \
    --reward_mode v2_simple_target_gated_decoy \
    --w_affinity 1.0 \
    --w_decoy 0.6 \
    --w_naturalness 0.05 \
    --w_diversity 0.02 \
    --entropy_coef 0.012 \
    --train_targets data/tfold_excellent_peptides.txt \
    --curriculum_schedule '[{"until": 1200000, "L0": 0.6, "L1": 0.4, "L2": 0.0}, {"until": null, "L0": 0.5, "L1": 0.3, "L2": 0.2}]' \
    --online_tcr_pool_enabled \
    --online_tcr_pool_start_step 914176 \
    --online_tcr_pool_warmup_steps 100000 \
    --online_tcr_pool_max_ratio 0.7 \
    --online_tcr_pool_max_per_target 512 \
    --online_tcr_pool_min_affinity -1.0 \
    --online_tcr_pool_max_decoy_violation 0.5 \
    --online_tcr_pool_min_hamming 2 \
    --online_tcr_pool_use_dynamic_bands \
    --tfold_use_cache \
    --tfold_cache_read_only false \
    --checkpoint_interval 50000 \
    > logs/trace73_improved_specificity_train.log 2>&1 &

echo "trace73_improved_specificity launched on GPU 0"
echo "Monitor: tail -f logs/trace73_improved_specificity_train.log"
echo ""
echo "Key improvements:"
echo "  - w_decoy: 0.3 → 0.6 (强化 specificity)"
echo "  - online_pool_max_decoy_violation: 999 → 0.5 (严格筛选)"
echo "  - online_pool_min_affinity: -10 → -1.0 (只接收高亲和力 TCR)"
echo "  - curriculum: 60% L0 + 40% L1 (高质量初始化)"
echo "  - online_pool_max_ratio: 0.5 → 0.7 (优先使用发现的好 TCR)"
echo ""
echo "Expected outcome:"
echo "  - InitA 提升到 -1.5 以上"
echo "  - DecViol 下降到 1.0 以下"
echo "  - Final Advantage 达到 0.0 以上"
