# test41_seed123: Reproduce test41 with seed=123

**Date**: 2026-05-05
**Status**: planned
**GPU**: 5
**Priority**: P0

## Hypothesis

test41 achieved 0.6243 AUROC with seed=42. This experiment validates reproducibility with seed=123.
If the result is close (0.58-0.65), test41's approach is robust.

## Configuration

Exact same as test41 but with seed=123. Two-phase training:

### Phase 1: ERGO warm-start (1M steps)
Uses test33 checkpoint at 1M (same as test41 did).

### Phase 2: Contrastive fine-tuning (1M steps)
```bash
CUDA_VISIBLE_DEVICES=5 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test41_seed123 \
    --seed 123 \
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
- **If close**: test41 approach is seed-stable
- **If poor (<0.55)**: test41 result may have been lucky

---

**Created**: 2026-05-05
