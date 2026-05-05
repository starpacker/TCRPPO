# test47: 32 Decoys + 45 Filtered Peptides (from test41)

**Date**: 2026-05-05
**Status**: planned
**GPU**: 4
**Priority**: P0

## Hypothesis

Extending test41's proven two-phase approach with:
1. **More decoys** (32 vs 16) → stronger specificity signal
2. **More peptides** (45 vs 12) → better generalization

Should push AUROC from 0.6243 to **0.64-0.67**.

## Configuration

**Phase 1**: Resume from test41 final checkpoint (already has 2M steps of training)

**Phase 2**: Contrastive fine-tuning with enhanced settings

```bash
CUDA_VISIBLE_DEVICES=4 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test47_32decoys_45peptides \
    --seed 42 \
    --resume_from output/test41_from_test33_1m_16decoys/checkpoints/final.pt \
    --resume_change_reward_mode contrastive_ergo \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 32 \
    --contrastive_agg mean \
    --affinity_scorer ergo \
    --encoder esm2 \
    --total_timesteps 1000000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/45_filtered_peptides.txt \
    --checkpoint_freq 50000
```

## Key Differences from test41

| Parameter | test41 | test47 | Rationale |
|-----------|--------|--------|-----------|
| n_contrast_decoys | 16 | **32** | Stronger specificity signal |
| train_targets | 12 McPAS | **45 filtered** | Better generalization, more diverse training |
| Resume from | test33@1M | **test41 final** | Start from best checkpoint (0.6243) |
| Total new steps | 1M | 1M | Same fine-tuning duration |

## Expected Outcome

- **Mean AUROC**: 0.64-0.67 (improvement over test41's 0.6243)
- **Training time**: ~12-16 hours (n_envs=8, ERGO is fast)
- **If successful**: Proves that more decoys + more peptides improve specificity
- **If failed**: test41's performance may be near the ceiling for ERGO-based training

## Risk Analysis

### Risk 1: 45 peptides too diverse → slower convergence
**Mitigation**: Resume from test41 (already converged), only fine-tune for 1M steps

### Risk 2: 32 decoys too many → noisy gradient
**Mitigation**: Use mean aggregation (smooth), learning_rate=1e-4 (conservative)

### Risk 3: Overfitting to 45 peptides
**Mitigation**: Evaluate on standard 12 McPAS targets (same as test41)

## Success Criteria

1. Mean AUROC ≥ 0.64 on 12 McPAS targets
2. At least 6/12 targets with AUROC > 0.65 (vs test41's 5/12)
3. No regression on test41's strong targets (IVTDFSVIK, YLQPRTFLL, LLWNGPMAV)

---

**Experiment design**: Claude Code  
**Created**: 2026-05-05
