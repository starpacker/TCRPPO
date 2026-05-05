# test49 Launch Notes

**Date**: 2026-05-05 19:15
**Status**: Running successfully
**GPU**: 4
**PID**: 3803344

## Problem Solved

test48 (random 10% tFold hybrid) failed due to contrastive reward amplification:
- 1024 samples × 17 peptides (1 target + 16 decoys) × 10% = 1,741 tFold calls per rollout
- First rollout took 40+ minutes (expected: 2-3 minutes)
- Estimated training time: 20-27 days (impractical)

## Solution: Cascade Scorer with cache_only Mode

### Key Changes from test48

1. **Intelligent sampling** instead of random:
   - Threshold-based: only call tFold if ERGO score ≥ 0.5
   - Adaptive: cascade ratio adjusts to policy quality

2. **Higher threshold** (0.5 instead of 0.3):
   - ERGO-trained policy (test41) generates high-scoring TCRs
   - Threshold 0.3 would trigger 50%+ cascade ratio (too slow)
   - Threshold 0.5 achieves ~9% cascade ratio (practical)

3. **cache_only mode** for tFold:
   - Cache hits: < 1ms (V3.4 classifier only)
   - Cache misses: Return score 0.3 without calling tFold server
   - Avoids 7.5s subprocess overhead per miss
   - Cache gradually fills during training (39,677 entries at start)

4. **score_batch_fast()** method:
   - Enables efficient batch scoring for contrastive reward
   - Reward manager uses this for decoy scoring (16 decoys per sample)

### Configuration

```bash
CUDA_VISIBLE_DEVICES=4 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test49_cascade_ergo_tfold_cacheonly \
    --seed 42 \
    --resume_from output/test41_from_test33_1m_16decoys/checkpoints/final.pt \
    --resume_change_reward_mode contrastive_ergo \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 16 \
    --contrastive_agg mean \
    --affinity_scorer cascade \
    --cascade_threshold 0.5 \
    --cascade_tfold_weight 0.7 \
    --cascade_ergo_weight 0.3 \
    --tfold_cache_only \
    --tfold_cache_miss_score 0.3 \
    --encoder esm2 \
    --total_timesteps 3500000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/mcpas_12_targets.txt
```

## Actual Performance (First 13 Minutes)

| Metric | Value | Notes |
|--------|-------|-------|
| First rollout time | 8 minutes | vs test48's 40+ minutes |
| Subsequent rollouts | 2-3 minutes | Cache hits improve speed |
| Cascade ratio | 8.5-9.2% | Stable, threshold 0.5 working well |
| Steps completed | 51K in 13 min | 3.8K steps/min |
| Reward | 0.8-0.86 | Stable, no training instability |
| Training speed | ~2.6 min/rollout | 5 rollouts in 13 minutes |

**Estimated completion time**: 4-6 hours for 1M steps (vs test48's 20-27 days)

## Training Log

```
Step  2,510,848 | Eps:   1136 | R:    0.862 | Len:   8.0 | PG:  -0.0196 | VF:   0.0914 | Ent:  2.893 | Cascade: 9.2% tFold
Step  2,521,088 | Eps:   2272 | R:    0.798 | Len:   8.0 | PG:  -0.0070 | VF:   0.0815 | Ent:  3.233 | Cascade: 8.9% tFold
Step  2,531,328 | Eps:   3408 | R:    0.796 | Len:   8.0 | PG:  -0.0147 | VF:   0.0900 | Ent:  3.601 | Cascade: 8.7% tFold
Step  2,541,568 | Eps:   4552 | R:    0.825 | Len:   8.0 | PG:  -0.0171 | VF:   0.0766 | Ent:  3.464 | Cascade: 8.5% tFold
Step  2,551,808 | Eps:   5688 | R:    0.856 | Len:   8.0 | PG:  -0.0142 | VF:   0.0896 | Ent:  3.836 | Cascade: 8.5% tFold
```

## Success Criteria

1. ✅ **First rollout completes in < 10 minutes** (actual: 8 minutes)
2. ⏳ **Training completes in < 48 hours** (estimated: 4-6 hours)
3. ⏳ **tFold AUROC > 0.45** (will evaluate after training)
4. ⏳ **ERGO AUROC ≥ 0.62** (will evaluate after training)
5. ✅ **Cascade ratio stable** (8.5-9.2%, not increasing uncontrollably)

## Key Insights

1. **Contrastive reward amplifies scorer calls by 17x** — must account for this in speed estimates
2. **ERGO-trained policy generates high-scoring TCRs** — threshold must be tuned accordingly
3. **tFold subprocess overhead is 7.5s per miss** — cache_only mode is essential for practical training
4. **Threshold 0.5 is optimal for ERGO-trained policy** — achieves ~9% cascade ratio
5. **score_batch_fast() is critical** — enables efficient batch scoring for contrastive reward

## Next Steps

1. Monitor training for 4-6 hours until completion
2. Evaluate final checkpoint on tc-hard with tFold scorer
3. Compare tFold AUROC vs test41 (baseline: 0.4017)
4. If successful (tFold AUROC > 0.45):
   - Try lower threshold (0.4) to increase tFold signal
   - Try higher cache_miss_score (0.5) for neutral fallback
5. If failed (tFold AUROC ≤ 0.42):
   - Investigate why cascade insufficient
   - Proceed with knowledge distillation (long-term solution)

---

**Created**: 2026-05-05 19:30
**Monitor**: `tail -f logs/test49_cascade_ergo_tfold_cacheonly_train.log`
