# test51c_decoy: Late-Stage Decoy Penalty Training

**Date**: 2026-05-11  
**Status**: planned  
**GPU**: 3  
**Priority**: P1

## Hypothesis

Adding decoy penalty (w=0.4) after 100K warm-start improves specificity without destabilizing training.

## Configuration

```bash
CUDA_VISIBLE_DEVICES=3 python tcrppo_v2/ppo_trainer.py \
    --config configs/test51c_decoy.yaml \
    --run_name test51c_decoy \
    --seed 42 \
    --reward_mode v2_full \
    --affinity_scorer tfold \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --terminal_reward_only \
    --n_contrast_decoys 4 \
    --w_affinity 1.0 \
    --w_decoy 0.4 \
    --w_naturalness 0.5 \
    --w_diversity 0.2 \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.0 \
    --curriculum_l2 0.5 \
    --train_targets data/tfold_excellent_peptides.txt \
    --tfold_cache_path data/tfold_feature_cache.db \
    --resume_from output/test51c_no_decoy_long_ep/checkpoints/milestone_100000.pt
```

## Key Differences from test51c

| Parameter | test51c | test51c_decoy | Change |
|-----------|---------|---------------|--------|
| reward_mode | v2_no_decoy | v2_full | Enable decoy penalty |
| w_decoy | 0.0 | 0.4 | Conservative decoy weight |
| n_contrast_decoys | 0 | 4 | Sample 4 decoys per step |
| decoy_K | 0 | 32 | LogSumExp over 32 decoys |
| Resume from | - | milestone_100000.pt | Warm-start from 100K |
| GPU | 1 | 3 | Parallel training |

**All other settings identical to test51c**

## Expected Outcome

### Specificity
- **AUROC**: > test51c (target: +0.10)
- **Target score**: Similar to test51c
- **Decoy score**: Lower than test51c (better discrimination)

### Training stability
- **Reward**: May dip initially, then recover
- **No collapse**: Conservative w=0.4 should prevent instability
- **Convergence**: Should stabilize within 200K steps

### If successful
- Proves late-stage decoy helps specificity
- Can increase w_decoy to 0.8 in future experiments

### If failed
- Decoy penalty may be too strong even at 0.4
- May need to reduce to 0.2 or use gradual ramp-up

## Dependencies

- Code changes: None (decoy scorer already exists)
- Requires: test51c reaches 100K steps (~28 hours)
- Blocks: None

## Monitoring

### Key metrics
1. **AUROC**: Track per-target specificity
2. **Reward components**: affinity, decoy, naturalness, diversity
3. **Decoy penalty**: Should be negative (penalizing cross-reactivity)
4. **Training stability**: Watch for reward collapse

### Checkpoints
- Every 100K steps
- Run decoy eval at 200K, 500K, 1M, 2M

### Alerts
- If reward drops by >50% after resume
- If AUROC < test51c (decoy penalty hurting quality)
- If training crashes

## Why w_decoy=0.4?

**Conservative choice** (user decision):
- Lower risk of training instability
- Allows gradual adaptation to decoy signal
- Can increase later if training is stable

**Comparison**:
- w=0.8: Standard v2_full setting (may be too strong for late-stage)
- w=0.4: Conservative (recommended for first attempt)
- w=0.2: Very conservative (fallback if 0.4 fails)

## Launch

```bash
bash scripts/launch_test51c_decoy.sh
```

Script will wait for `milestone_100000.pt` to exist, then launch automatically.

## Analysis Plan

### At 200K steps (100K after resume)
- Compare reward curves: test51c vs test51c_decoy
- Check if decoy penalty is working (negative values)
- Verify no training collapse

### At 500K steps
- Run full decoy eval on both experiments
- Compare AUROC: test51c_decoy should be higher
- Analyze per-target improvements

### At 2M steps (final)
- Full comparison: test51c vs test51c_decoy vs test51c_amp
- Generate TCRs and run comprehensive eval
- Document findings in results/
