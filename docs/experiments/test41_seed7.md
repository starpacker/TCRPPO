# test41_seed7: Reproduce test41 with seed=7

**Date**: 2026-05-05
**Status**: planned
**GPU**: 6
**Priority**: P0

## Hypothesis

Third seed validation for test41. Combined with seed=42 (0.6243) and seed=123, we can compute mean ± std across 3 seeds to assess robustness.

## Configuration

Exact same as test41 but with seed=7. Two-phase training:

### Phase 1: ERGO warm-start (1M steps)
Uses test33 checkpoint at 1M (same as test41 did).

### Phase 2: Contrastive fine-tuning (1M steps)
```bash
CUDA_VISIBLE_DEVICES=6 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test41_seed7 \
    --seed 7 \
    --resume_from output/test33_twophase_strong_contrastive/checkpoints/milestone_1000000.pt \
    --resume_change_reward_mode contrastive_ergo \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 16 \
    --contrastive_agg mean \
    --affinity_scorer ergo \
    --encoder esm2 \
    --total_timesteps 1000000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/mcpas_12_targets.txt \
    --checkpoint_freq 50000
```

## Expected Outcome

- **Mean AUROC**: 0.57-0.65 (close to test41's 0.6243)
- **3-seed statistics**: mean ± std across seeds 42, 123, 7
- **If std < 0.03**: Approach is highly reproducible
- **If std > 0.08**: High seed sensitivity (like v1_ergo_only)

---

**Created**: 2026-05-05
