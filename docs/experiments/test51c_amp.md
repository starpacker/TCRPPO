# test51c_amp: AMP-Accelerated tFold Training

**Date**: 2026-05-11  
**Status**: planned  
**GPU**: 2  
**Priority**: P1

## Hypothesis

AMP-accelerated tFold scorer provides 4× faster training without quality regression compared to subprocess tFold.

## Configuration

```bash
CUDA_VISIBLE_DEVICES=2 python tcrppo_v2/ppo_trainer.py \
    --config configs/test51c_amp.yaml \
    --run_name test51c_amp \
    --seed 42 \
    --reward_mode v2_no_decoy \
    --affinity_scorer tfold_amp \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --terminal_reward_only \
    --n_contrast_decoys 0 \
    --w_affinity 1.0 \
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

| Parameter | test51c | test51c_amp | Change |
|-----------|---------|-------------|--------|
| affinity_scorer | tfold (subprocess) | tfold_amp (AMP) | 3.97× faster inference |
| Resume from | - | milestone_100000.pt | Warm-start from 100K |
| GPU | 1 | 2 | Parallel training |

**All other settings identical to test51c**

## Expected Outcome

### Speed
- **Rollout time**: 103s (vs 410s for test51c)
- **Training speed**: 4× faster
- **Steps/min**: ~200 (vs ~50 for test51c)

### Quality
- **Mean reward**: Similar to test51c (no regression)
- **AUROC**: Similar to test51c (AMP should not affect quality)
- **Diversity**: Similar to test51c

### If successful
- Proves AMP can accelerate training without quality loss
- Can be used for all future experiments

### If failed
- AMP may introduce numerical precision issues
- May need to tune AMP settings or fall back to subprocess

## Dependencies

- Code changes: `affinity_tfold_amp.py` (completed)
- Requires: test51c reaches 100K steps (~28 hours)
- Blocks: None

## Monitoring

### Key metrics
1. **Training speed**: steps/min (expect ~200 vs ~50)
2. **Reward curve**: should match test51c trajectory
3. **AMP stats**: n_amp_calls, n_subprocess_fallback
4. **Cache growth**: should continue from test51c cache

### Checkpoints
- Every 100K steps
- Compare with test51c at same step counts

### Alerts
- If reward diverges from test51c by >20%
- If AMP fallback rate > 10%
- If training crashes or OOM errors

## Launch

```bash
bash scripts/launch_test51c_amp.sh
```

Script will wait for `milestone_100000.pt` to exist, then launch automatically.
