# TCRPPO v2 Situation Analysis & Action Plan

**Date:** 2026-05-20  
**Analyst:** Claude Code  
**Status:** URGENT - Need evaluation data to guide next steps

---

## Executive Summary

You have **8 active experiments** running (trace11, 22-28) consuming significant GPU resources, but **no recent AUROC evaluations** to determine which approaches actually work. Training rewards (R) and raw affinity logits (A) are not sufficient to measure specificity.

**Critical Gap:** All experiments show A < 0 (negative binding logits), meaning none have achieved strong binding yet. Your best training reward is trace11 @ 500K steps with R≈2.3, A≈-5.7, but we don't know if this translates to good AUROC.

**Historical Context:** Your best result ever was test41 (0.6243 AUROC) using two-phase training: ERGO warm-start → contrastive fine-tuning with 16 decoys. Current pure tFold experiments (trace11+) have not been evaluated against this baseline.

---

## Current Active Experiments

| Trace | Name | Steps | GPU | R (Last100) | A (Last100) | Key Feature | Status |
|-------|------|-------|-----|-------------|-------------|-------------|--------|
| 11 | trace11_delta | 496K | 0 | 2.30 | -5.71 | Baseline: max_steps=8, delta reward | Main line |
| 22 | trace22_max4 | 77K | 0 | 1.43 | -6.69 | max_steps=4 (shorter horizon) | **Promising** |
| 23 | trace23_stop | 36K | 0 | 1.11 | -6.90 | STOP allowed, min_steps=2 | Learning |
| 24 | trace24_subonly | 38K | 0 | 0.73 | -7.32 | SUB only (no INS/DEL) | Weakest |
| 25 | trace25_curriculum | Early | 2 | TBD | TBD | max_steps curriculum 1→2→4→8 | Just started |
| 26 | trace26_active_clip | Early | 7 | TBD | TBD | Active clipping (train on best prefix) | Just started |
| 27 | trace27_target_guard | Early | 1 | TBD | TBD | Target-guarded decoy penalty | Just started |
| 28 | trace28_abs_spec | Early | 0 | TBD | TBD | Absolute specificity curriculum | Just started |

**Resource usage:** 8 experiments × 8 envs × 2 GPUs (trainer + tFold server) = ~16 GPU-equivalents

---

## Key Problems Identified

### 1. No AUROC Evaluation
- **Problem:** Training rewards (R) and affinity logits (A) don't measure specificity
- **Evidence:** All experiments show A < 0, but we don't know if they discriminate targets from decoys
- **Impact:** Cannot determine which approach actually works
- **Action:** Running ERGO evaluation on trace11 @ 400K and trace22 @ 60K now

### 2. Reward Signal Confusion
Different `reward_mode` settings make cross-experiment comparison meaningless:
- `v2_no_decoy_delta`: Optimizes final-initial affinity delta (trace11, 22, 23, 24)
- `tfold_stepwise`: Per-step delta rewards (trace17, 18)
- `v2_delta_minus_decoy`: Target delta minus decoy delta (trace20, 21)
- `v2_absolute_specificity`: Absolute affinity with decoy penalty (trace28)

**Recommendation:** Standardize on one reward mode after evaluation proves which works.

### 3. No Positive Binding Achieved
- **Problem:** All experiments show A < 0 (negative logits), far from A > 0 threshold
- **Best so far:** trace11 @ 500K with A ≈ -5.7 (still 5.7 logits below binding)
- **Possible causes:**
  - Delta rewards are too small (typical delta: 1-3 logits)
  - Initial TCRs are too far from binding (curriculum issue)
  - tFold scoring is miscalibrated for RL-generated sequences
- **Action:** Evaluate actual AUROC to see if relative ranking is correct despite low absolute scores

### 4. Peptide Selection Verified Correct
- **Status:** ✅ GOOD - Using 20 tFold-excellent peptides (AUC ≥ 0.8)
- **File:** `data/tfold_excellent_peptides.txt` matches PEPTIDE_SCORER_MAPPING.md
- **No action needed**

### 5. Experiment Proliferation
- **Problem:** 118 output directories, 8 active experiments, but most lack evaluation
- **Impact:** Wasting GPU time on unproven approaches
- **Action:** Consolidate to 2-3 best approaches after evaluation

---

## Evaluation Plan (IN PROGRESS)

### Priority 1: Evaluate Existing Checkpoints

**Running now:**
- ✅ trace11 @ 400K steps (ERGO scorer) - PID 866166
- ⏳ trace22 @ 60K steps (ERGO scorer) - Queued

**Metrics to measure:**
- Per-target AUROC (specificity: can it discriminate targets from decoys?)
- Mean target affinity score
- Mean decoy affinity score
- Diversity (unique sequences / total sequences)
- Naturalness (ESM perplexity z-score)

**Expected completion:** ~2-3 hours for trace11 (50 TCRs × 12 targets × 50 decoys)

### Priority 2: Compare Against Historical Best

**Baseline:** test41 (0.6243 AUROC) - Two-phase training
- Phase 1: ERGO warm-start to 1M steps
- Phase 2: Contrastive fine-tuning with 16 decoys for 1M steps

**Question:** Can pure tFold delta training (trace11) beat 0.62 AUROC?

---

## Recommended Action Plan

### Phase 1: Evaluate & Decide (CURRENT)

**Actions:**
1. ✅ Launch trace11 @ 400K evaluation (ERGO) - Running
2. ⏳ Launch trace22 @ 60K evaluation (ERGO) - Queued
3. ⏳ Analyze results and compare to test41 baseline
4. ⏳ Decide: Continue pure tFold OR switch to two-phase strategy

**Decision criteria:**
- If trace11 AUROC > 0.55: Continue to 1M steps, evaluate again
- If trace11 AUROC < 0.50: Switch to two-phase (tFold warm-start → contrastive)
- If trace22 AUROC > trace11: Adopt max_steps=4 as new standard

### Phase 2: Consolidate Experiments (AFTER EVALUATION)

**Stop low-value experiments:**
- trace24 (sub_only) - Lowest reward (R=0.73), no clear advantage
- trace25 (curriculum) - Overly complex, trace22 already shows max_steps=4 works
- trace27/28 (just started) - Wait for trace22/26 evaluation first

**Keep running:**
- trace11 - Main baseline, let finish to 1M steps
- trace22 - Most promising (fast learning, stable), extend to 200K
- trace26 (active clipping) - Novel idea, give 50K steps to prove itself

**Resource savings:** Stop 3 experiments = free up ~6 GPU-equivalents

### Phase 3: Launch Next Generation (AFTER DECISION)

**Option A: If pure tFold works (AUROC > 0.55)**
- Continue trace11 to 1M steps
- Extend trace22 to 200K steps
- Launch two-phase fine-tuning: trace22 @ 200K → contrastive (n_decoys=8-16)

**Option B: If pure tFold fails (AUROC < 0.50)**
- Switch to hybrid tFold-ERGO (90% ERGO, 10% tFold)
- Use two-phase strategy: tFold warm-start → contrastive
- Consider ERGO-only warm-start (proven) → tFold fine-tuning

**Option C: If max_steps=4 wins**
- Adopt trace22 as new baseline
- Launch trace22-style two-phase: 200K warm-start → 200K contrastive
- Test even shorter horizons (max_steps=2)

---

## Key Insights from Historical Data

### What Worked (test41: 0.6243 AUROC)
- Two-phase training: warm-start → contrastive fine-tuning
- Strong warm-start (1M steps ERGO-only)
- More decoys = better specificity (16 decoys > 8 decoys)
- ESM-2 encoder (not lightweight)

### What Failed
- Pure ERGO from scratch: 0.48-0.61 AUROC (seed-dependent)
- Contrastive from scratch (no warm-start): 0.48 AUROC
- Lightweight encoder: 0.43 AUROC
- Large batch (n_envs=32): 0.49 AUROC (worse than n_envs=8)

### What's Unknown (Current Experiments)
- Pure tFold delta training (trace11) - **EVALUATING NOW**
- max_steps=4 vs max_steps=8 (trace22 vs trace11) - **EVALUATING NEXT**
- Active clipping (trace26) - Too early
- Absolute specificity reward (trace28) - Too early

---

## Critical Questions to Answer

1. **Does pure tFold delta training work?**
   - Hypothesis: tFold is more accurate than ERGO, so pure tFold should beat test41
   - Test: Compare trace11 @ 400K AUROC vs test41 @ 400K equivalent
   - Status: Evaluating now

2. **Is max_steps=4 better than max_steps=8?**
   - Hypothesis: Shorter horizon reduces credit assignment difficulty
   - Test: Compare trace22 @ 60K vs trace11 @ 60K equivalent
   - Status: Queued for evaluation

3. **Should we use two-phase training with tFold?**
   - Hypothesis: tFold warm-start → contrastive will beat pure tFold
   - Test: If trace11 < 0.62 AUROC, launch two-phase variant
   - Status: Waiting for trace11 evaluation

4. **Is active clipping helpful?**
   - Hypothesis: Training on best intermediate TCR reduces "edit into worse" problem
   - Test: Compare trace26 @ 50K vs trace11 @ 50K
   - Status: Too early (trace26 just started)

---

## Resource Optimization

### Current Usage
- 8 active experiments
- ~16 GPU-equivalents (trainer + tFold server per experiment)
- ~118 output directories (97 with saved configs)

### Recommended Consolidation
- Stop 3 low-value experiments → free 6 GPUs
- Keep 3 promising experiments (trace11, 22, 26)
- Launch 1-2 new experiments based on evaluation results
- **Net savings:** 6 GPUs for other users or faster training

---

## Next Steps (Ordered by Priority)

### Immediate (Today)
1. ✅ Wait for trace11 @ 400K ERGO evaluation to complete (~2-3 hours)
2. ⏳ Launch trace22 @ 60K ERGO evaluation
3. ⏳ Analyze results and write comparison report
4. ⏳ Make go/no-go decision on pure tFold approach

### Short-term (This Week)
5. Stop low-value experiments (trace24, 25, 27, 28) if evaluation shows no promise
6. Extend trace22 to 200K steps if evaluation is positive
7. Launch two-phase fine-tuning if trace11/trace22 show AUROC > 0.55

### Medium-term (Next Week)
8. Evaluate trace26 (active clipping) @ 50K steps
9. If two-phase launched, evaluate at 100K steps
10. Write final experiment report and update all_experiments_tracker.md

---

## Conclusion

You're at a critical decision point. You've invested significant GPU time in pure tFold experiments (trace11-28) but lack evaluation data to know if they work. The evaluation running now will determine whether to:

**A)** Continue pure tFold (if AUROC > 0.55)  
**B)** Switch to two-phase tFold (if 0.50 < AUROC < 0.62)  
**C)** Return to proven ERGO two-phase (if AUROC < 0.50)

**Recommendation:** Wait for evaluation results before launching any new experiments. Consolidate to 2-3 proven approaches and stop wasting GPU time on unproven variants.

---

**Document Status:** Living document - will be updated with evaluation results  
**Next Update:** After trace11 and trace22 evaluations complete
