# test48 Post-Mortem: Why Hybrid Scorer Failed

**Date**: 2026-05-05  
**Experiment**: test48_hybrid_90ergo_10tfold  
**Status**: Aborted after 39 minutes (first rollout incomplete)  
**Outcome**: Approach deemed impractical

---

## What We Tried

**Hypothesis**: Mixing ERGO (90%, fast) and tFold (10%, accurate) during training would:
1. Prevent overfitting to ERGO's sequence biases
2. Generate TCRs that score well on both ERGO and tFold
3. Achieve better overall specificity than pure ERGO training

**Implementation**: `HybridScorer` that randomly selects ERGO or tFold per sample with 10% tFold probability.

**Expected speed**: 109ms/sample average (0.9×10ms + 0.1×1000ms)

**Expected training time**: ~30 hours for 1M steps

---

## What Actually Happened

**First rollout time**: >40 minutes (expected: 2-3 minutes)

**Root cause**: Contrastive reward amplifies scorer calls by 17x:

```
Contrastive reward requires scoring:
- 1 target peptide
- 16 decoy peptides
= 17 peptides per sample

Per rollout:
- 8 envs × 128 steps = 1,024 samples
- 1,024 samples × 17 peptides = 17,408 scorer calls
- 10% tFold ratio = ~1,741 tFold calls
- At ~1s per cache miss = ~30-40 minutes per rollout

Total training estimate:
- 1M steps / (8 envs × 128 steps) = 977 rollouts
- 977 rollouts × 30-40 min = 29,310-39,080 minutes
- = 488-651 hours = 20-27 days
```

**Actual vs Expected**:
- Expected: 30 hours
- Actual: 20-27 days (16-22x slower)

---

## Why the Estimate Was Wrong

**Original calculation assumed**:
- 109ms/sample average speed
- Applied to target peptide only

**Reality**:
- Contrastive reward scores 17 peptides per sample
- 109ms × 17 = 1,853ms per sample (not 109ms)
- First rollout has low cache hit rate (most tFold calls are cache misses)
- Even with improving cache hit rate, average speed would be 500-1000ms/sample

---

## Key Lessons

### 1. Contrastive Reward Amplifies Scorer Costs

Any scorer optimization must account for the 17x amplification from decoy scoring. A "10% tFold" strategy is actually:
- 10% × 17 peptides = 1.7 tFold calls per sample
- vs 0.1 tFold calls per sample for non-contrastive reward

### 2. Cache Hit Rate Doesn't Save You

Even if cache hit rate improves to 90% (optimistic):
- 10% cache miss × 1,741 calls × 1s = 174s per rollout
- Still 2.9 minutes per rollout
- Total training: 977 rollouts × 2.9 min = 47 hours

This is better than 20 days but still 3x slower than pure ERGO (16 hours).

### 3. Hybrid Approach Needs Smarter Sampling

Random 10% sampling is wasteful. Better strategies:
- **Cascade**: ERGO pre-filter → tFold verify only high-scoring samples
- **Active learning**: tFold only on ERGO-uncertain samples
- **Target-only hybrid**: Apply tFold to target peptide, ERGO for decoys

---

## Alternative Approaches

### Option 1: Cascade Scorer (RECOMMENDED)

```python
class CascadeScorer:
    def score(self, tcr, peptide):
        ergo_score = ergo.score(tcr, peptide)
        
        # Only use tFold for high-scoring candidates
        if ergo_score > 0.3:  # Threshold
            tfold_score = tfold.score(tcr, peptide)
            return 0.7 * tfold_score + 0.3 * ergo_score
        else:
            return ergo_score
```

**Advantages**:
- Adaptive: fast early (most TCRs score low), accurate late (more high-scoring TCRs)
- Focused: tFold only verifies promising TCRs
- Efficient: avoids wasting tFold on bad TCRs

**Expected speed**:
- Early training: ~20ms/sample (95% below threshold)
- Late training: ~200ms/sample (50% above threshold)
- Average: ~100ms/sample
- Total training: 1-2 days

### Option 2: Active Learning (ALTERNATIVE)

```python
class ActiveLearningScorer:
    def score(self, tcr, peptide):
        ergo_score, ergo_conf = ergo.score_with_uncertainty(tcr, peptide)
        uncertainty = 1.0 - ergo_conf
        
        # Use tFold only for high-uncertainty samples
        if uncertainty > 0.2 and tfold_budget_remaining():
            return tfold.score(tcr, peptide)
        else:
            return ergo_score
```

**Advantages**:
- Targets ERGO's blind spots
- Budget-controlled (strict tFold call limit)
- Efficient: only 5% tFold calls

**Disadvantages**:
- Depends on ERGO uncertainty (we know from test46 this is unreliable)
- May miss "ERGO confident but wrong" samples

### Option 3: Target-Only Hybrid (REQUIRES CODE CHANGE)

Apply hybrid scoring only to target peptide, use pure ERGO for decoys:

```python
# In reward_manager.py, contrastive_ergo mode:
target_score = hybrid_scorer.score(tcr, target_peptide)  # 10% tFold
decoy_scores = [ergo_scorer.score(tcr, decoy) for decoy in decoys]  # Pure ERGO
```

**Advantages**:
- Reduces tFold calls by 16x (only target, not decoys)
- 1,024 samples × 10% = ~102 tFold calls per rollout (vs 1,741)
- Expected training time: ~30 hours (as originally estimated)

**Disadvantages**:
- Inconsistent: target uses tFold, decoys use ERGO
- May not improve tFold AUROC (decoys still scored with ERGO)

### Option 4: Knowledge Distillation (LONG-TERM BEST)

Train a fast student model to mimic tFold:

**Advantages**:
- Best of both worlds: ~10ms speed + tFold-like accuracy
- One-time cost: dataset generation only needed once
- Reusable: can train multiple RL models with same dataset

**Timeline**: 5-7 days implementation + 1 day data generation

---

## Recommendation

**Immediate (today)**:
1. Implement **Cascade Scorer** (Solution 3 from REWARD_MODEL_SOLUTION.md)
2. Launch test49_cascade on GPU 2
3. Expected training time: 1-2 days

**Parallel (this week)**:
1. Start **Knowledge Distillation** preparation
2. Generate 450K training samples (5 GPUs parallel, 1 day)
3. Train distilled model (2-4 hours)
4. Launch test50_distilled next week

**Fallback**:
- If cascade fails, try **Target-Only Hybrid** (requires code change)
- If that fails, proceed with pure ERGO and accept the ERGO-tFold gap

---

## Conclusion

The hybrid scorer approach is **theoretically sound** but **practically infeasible** when combined with contrastive reward. The 17x amplification from decoy scoring makes even a 10% tFold ratio too slow.

**Key insight**: Any solution must either:
1. Reduce the number of tFold calls (cascade, active learning)
2. Make tFold faster (knowledge distillation)
3. Apply tFold selectively (target-only, uncertainty-guided)

Random sampling at 10% is the worst of all worlds: too slow to be practical, too sparse to be effective.

---

**Next action**: Implement cascade scorer and launch test49.
