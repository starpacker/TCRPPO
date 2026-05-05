# test49: Cascade Scorer (ERGO pre-filter + tFold verify)

**Date**: 2026-05-05
**Status**: running
**GPU**: 4
**Priority**: P0
**Started**: 2026-05-05 19:15
**PID**: 3803344

## Hypothesis

Using ERGO as a fast pre-filter and tFold for verification of high-scoring candidates will:
1. Achieve adaptive speed (fast early, accurate late)
2. Improve tFold AUROC without sacrificing training time
3. Focus tFold computation on promising TCRs only

**Key insight**: Cascade is smarter than random hybrid (test48) — it adapts to training phase.

## Problem Being Solved

**test48 failure**: Random 10% tFold sampling was too slow with contrastive reward:
- 1024 samples × 17 peptides (1 target + 16 decoys) × 10% = ~1,741 tFold calls per rollout
- First rollout took 40+ minutes (expected: 2-3 minutes)
- Estimated training time: 20-27 days (impractical)

**Root cause**: Contrastive reward amplifies scorer calls by 17x, making random sampling wasteful.

**Solution**: Cascade scoring uses ERGO score as a gate — only call tFold if ERGO score > threshold.

## Configuration

**Revised**: threshold raised to 0.5 (from 0.3), cache_only enabled, cache_miss_score=0.3.

Resume from test41 final checkpoint, continue training with cascade scorer.

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

## Key Parameters

- `--affinity_scorer cascade`: Use CascadeScorer
- `--cascade_threshold 0.5`: Call tFold if ERGO score ≥ 0.5 (raised from 0.3)
- `--cascade_tfold_weight 0.7`: tFold weight in combination
- `--cascade_ergo_weight 0.3`: ERGO weight in combination
- `--tfold_cache_only`: Skip tFold server for cache misses (fast mode)
- `--tfold_cache_miss_score 0.3`: Score for cache misses (pessimistic)
- Resume from test41: Start from a strong ERGO-trained policy

**Key insight**: ERGO-trained policy generates high-scoring TCRs. Threshold 0.3 was too low (50%+ cascade ratio). Threshold 0.5 achieves ~9% cascade ratio, which is practical.

## Expected Speed

**Revised estimates based on actual first rollout**:

**Cascade ratio evolution**:
- Early training (resumed from test41): ~9% (ERGO-trained policy scores high)
- Mid training: ~10-15% (as policy explores)
- Late training: ~15-20% (exploitation phase)

**Per-rollout time**:
- First rollout: 8 minutes (actual, vs test48's 40+ minutes)
- Subsequent rollouts: 2-3 minutes (cache hits improve)
- Average: ~2.6 minutes/rollout

**Total training time**: 4-6 hours for 1M steps (vs test48's 20-27 days)

**Actual progress (first 13 minutes)**:
- 5 rollouts completed
- 51K steps (3.8K steps/min)
- Cascade ratio: 8.5-9.2% (stable)
- Reward: 0.8-0.86 (stable)

## Expected Outcome

### Primary metrics
- **ERGO AUROC**: 0.63-0.65 (maintained from test41's 0.6243)
- **tFold AUROC**: 0.45-0.50 (improvement over test41's 0.4017)
- **Gap reduction**: From -0.22 to -0.15 or better

### Secondary metrics
- **Cascade ratio**: Should increase over training (adaptive behavior)
- **First rollout time**: < 10 minutes (vs test48's 40+ minutes)
- **Training stability**: Should remain stable

### Success criteria
1. **First rollout completes in < 10 minutes** (vs test48's 40+ minutes)
2. **Training completes in < 48 hours** (vs test48's estimated 20-27 days)
3. **tFold AUROC > 0.45**: Proves cascade strategy works
4. **ERGO AUROC ≥ 0.62**: No regression
5. **Cascade ratio increases over time**: Confirms adaptive behavior

## Key Differences from Previous Experiments

### vs test48 (random hybrid)
- **Changed**: Intelligent sampling (ERGO threshold) instead of random 10%
- **Expected**: 10-20x faster, adaptive speed

### vs test41 (pure ERGO)
- **Added**: tFold verification for high-scoring TCRs
- **Expected**: Better tFold AUROC, slightly slower training

### vs test44 (pure tFold)
- **Changed**: ERGO pre-filter reduces tFold calls by 50-95%
- **Expected**: 10-50x faster than pure tFold

## Risks and Mitigations

### Risk 1: Threshold too low (too many tFold calls)
**Symptom**: First rollout > 10 minutes, cascade ratio > 50% early
**Mitigation**: Increase threshold to 0.4 or 0.5 (test49b)

### Risk 2: Threshold too high (too few tFold calls)
**Symptom**: tFold AUROC stays at 0.40, cascade ratio < 5%
**Mitigation**: Decrease threshold to 0.2 (test49c)

### Risk 3: Weighted combination suboptimal
**Symptom**: ERGO AUROC regresses below 0.60
**Mitigation**: Adjust weights (e.g., 0.5/0.5 instead of 0.7/0.3)

### Risk 4: tFold cache hit rate too low
**Symptom**: Training much slower than expected
**Mitigation**: Pre-warm cache for 12 McPAS peptides

## Follow-up Experiments

If successful:
- **test49b**: Try threshold 0.2 (more tFold calls)
- **test49c**: Try threshold 0.4 (fewer tFold calls)
- **test50**: Compare with knowledge distillation (long-term solution)

If failed:
- **test49d**: Try target-only hybrid (tFold on target, ERGO on decoys)
- **Investigate**: Analyze which TCRs are above/below threshold
- **Proceed**: With knowledge distillation (Solution 2, 5-7 days)

## Implementation Details

**CascadeScorer class** (`tcrppo_v2/scorers/cascade_scorer.py`):
- Stage 1: Score with ERGO (fast)
- Stage 2: If ERGO score ≥ threshold, score with tFold (accurate)
- Combine: weighted_score = tfold_weight × tFold + ergo_weight × ERGO
- Tracks statistics: primary_only_calls, cascade_calls, cascade_ratio
- **NEW**: `score_batch_fast()` method for efficient contrastive reward scoring

**tFold cache_only mode**:
- Cache hits: < 1ms (V3.4 classifier only, 1.57M params)
- Cache misses: Return `cache_miss_score` (0.3) without calling tFold server
- Avoids 7.5s subprocess overhead per miss
- Cache gradually fills during training (39,677 entries at start)

**Integration**:
- Added to `ppo_trainer.py` as `affinity_model == "cascade"`
- CLI args: `--cascade_threshold`, `--cascade_tfold_weight`, `--cascade_ergo_weight`, `--tfold_cache_only`, `--tfold_cache_miss_score`
- Logging: Prints "Cascade: X.X% tFold" in training loop

---

**Created**: 2026-05-05
**Next**: Launch experiment and monitor for 1-2 days
