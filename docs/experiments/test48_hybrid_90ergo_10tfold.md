# test48: Hybrid Scorer (90% ERGO + 10% tFold)

**Date**: 2026-05-05
**Status**: ❌ ABORTED (first rollout >40min, impractical)
**GPU**: 2
**Priority**: P0
**PID**: 2478937 (killed)

## Hypothesis

Mixing ERGO (fast, 90%) and tFold (accurate, 10%) during training will:
1. Prevent overfitting to ERGO's sequence biases
2. Generate TCRs that score well on both ERGO and tFold
3. Achieve better overall specificity than pure ERGO training

**Key insight**: tFold's 10% signal should be enough to "correct" ERGO's blind spots without slowing training too much.

## Problem Being Solved

**Current situation**:
- test41 (pure ERGO): ERGO AUROC 0.6243, tFold AUROC 0.4017 (-0.22 gap)
- test44 (pure tFold): Too slow (22 days training time, unacceptable)

**Root cause**: RL model optimized for ERGO generates TCRs that lack structural plausibility (tFold's criterion).

**Solution**: Mix both scorers during training to balance speed and accuracy.

## Configuration

Resume from test41 final checkpoint, continue training with hybrid scorer.

```bash
CUDA_VISIBLE_DEVICES=2 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test48_hybrid_90ergo_10tfold \
    --seed 42 \
    --resume_from output/test41_from_test33_1m_16decoys/checkpoints/final.pt \
    --resume_change_reward_mode contrastive_ergo \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 16 \
    --contrastive_agg mean \
    --affinity_scorer hybrid \
    --hybrid_tfold_ratio 0.1 \
    --encoder esm2 \
    --total_timesteps 1000000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/mcpas_12_targets.txt
```

## Key Parameters

- `--affinity_scorer hybrid`: Use HybridScorer
- `--hybrid_tfold_ratio 0.1`: 10% tFold, 90% ERGO
- Resume from test41: Start from a strong ERGO-trained policy

## Expected Speed

**Per-sample scoring time**:
- 90% ERGO: 0.9 × 10ms = 9ms
- 10% tFold: 0.1 × 1000ms = 100ms
- **Average: 109ms/sample** (vs 10ms pure ERGO, 1000ms pure tFold)

**Training time estimate**:
- 1M steps / (8 envs × 128 steps/rollout) = 977 rollouts
- 977 rollouts × 1024 samples/rollout × 109ms/sample = ~109K seconds = **30 hours**
- vs test41: 12-16 hours (2-3x slower)
- vs test44: 22 days (17x faster)

## Results

**Training aborted**: 2026-05-05 16:35 (after 39 minutes, still on first rollout)

**Reason**: Unacceptable speed due to contrastive reward amplification

**Actual performance**:
- First rollout time: >40 minutes (expected: 2-3 minutes)
- Estimated total training time: 100-200 hours (4-8 days)
- Root cause: 1024 samples × 17 peptides (1 target + 16 decoys) × 10% tFold = ~1,741 tFold calls per rollout

**Key findings**:
1. Hybrid scorer works correctly (cache growing, no errors)
2. Contrastive reward amplifies scorer calls by 17x (decoys)
3. Even 10% tFold ratio is too slow with contrastive reward
4. Cache hit rate improvement won't be enough to make this practical

**Hypothesis rejected**: 10% tFold signal is insufficient to balance speed and accuracy when combined with contrastive reward.

**Next steps**:
1. Try **cascade scorer** (Solution 3): ERGO pre-filter → tFold verify only high-scoring samples
2. Try **active learning** (Solution 5): tFold only on ERGO-uncertain samples
3. Try **hybrid on target only**: Apply 10% tFold to target peptide, pure ERGO for decoys (requires code change)
4. Proceed with **knowledge distillation** (Solution 2): Long-term best solution

## Key Differences from Previous Experiments

### vs test41 (pure ERGO)
- **Added**: 10% tFold scoring during training
- **Expected**: Better tFold AUROC, slightly slower training

### vs test44 (pure tFold)
- **Changed**: 90% ERGO instead of 100% tFold
- **Expected**: 17x faster training, slightly lower tFold AUROC

### vs test47 (32 decoys + 45 peptides)
- **Different approach**: Hybrid scorer vs more decoys
- **Complementary**: test47 tests "more contrastive signal", test48 tests "better scorer"

## Risks and Mitigations

### Risk 1: 10% tFold signal too weak
**Symptom**: tFold AUROC stays at 0.40 (no improvement)
**Mitigation**: Try 20% tFold ratio (test48b)

### Risk 2: Training 2-3x slower than expected
**Symptom**: >50 hours training time
**Mitigation**: Reduce to 5% tFold ratio or abort

### Risk 3: tFold cache hit rate too low
**Symptom**: Actual speed much slower than 109ms/sample
**Mitigation**: Pre-warm cache for 12 McPAS peptides

### ⚠️ CRITICAL ISSUE DISCOVERED (2026-05-05 16:23)

**Problem**: First rollout taking 30+ minutes (expected: 2-3 minutes)

**Root cause**: Contrastive reward amplifies scorer calls by 17x:
- Each sample requires scoring against 1 target + 16 decoys = 17 peptides
- 1024 samples/rollout × 17 peptides × 10% tFold = ~1,741 tFold calls
- At ~1s per cache miss, first rollout takes 30+ minutes
- Subsequent rollouts will be faster (higher cache hit rate) but still slow

**Impact**: Training time estimate revised from 30h to **100-200h** (4-8 days)

**Mitigation options**:
1. **Abort and reduce tFold ratio to 2-5%** (test48b) — faster but weaker signal
2. **Apply hybrid only to target peptide, not decoys** — requires code change
3. **Switch to cascade scorer** (Solution 3) — ERGO pre-filter, tFold verify only high-scoring
4. **Switch to active learning** (Solution 5) — tFold only on ERGO-uncertain samples

**Decision**: Let test48 complete first rollout to get actual timing data, then decide whether to:
- Continue (if cache hit rate improves dramatically)
- Abort and try alternative approach

## Follow-up Experiments

If successful:
- **test49_hybrid_80ergo_20tfold**: Increase tFold ratio to 20%
- **test50_hybrid_cascade**: Combine hybrid with cascade (ERGO pre-filter)

If failed:
- **test48b_hybrid_95ergo_5tfold**: Reduce tFold ratio to 5% (faster)
- **Investigate**: Why 10% tFold signal insufficient?

## Implementation Details

**HybridScorer class** (`tcrppo_v2/scorers/hybrid_scorer.py`):
- Randomly selects scorer per sample based on `secondary_ratio`
- Tracks statistics: primary_calls, secondary_calls, actual ratio
- Supports both `score()` and `score_batch()` methods

**Integration**:
- Added to `ppo_trainer.py` as `affinity_model == "hybrid"`
- CLI arg: `--hybrid_tfold_ratio` (default 0.1)
- Logging: Prints "Hybrid: X.X% tFold" in training loop

---

**Created**: 2026-05-05
**Next**: Launch experiment and monitor for 30 hours
