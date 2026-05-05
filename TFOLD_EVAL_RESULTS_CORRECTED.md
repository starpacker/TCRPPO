# tFold Re-evaluation Results (CORRECTED - Bug Fixed)

**Date**: 2026-04-30  
**Evaluation**: Top 10 TCRs per target, 5 decoys per TCR  
**Models**: test41, test33, test39  
**Bug Fixed**: `cache_miss_score` parameter corrected from -0.5 to 0.5

## Critical Bug Fix (2026-04-30)

**Previous bug**: Used `cache_miss_score=-0.5`, which produced raw scores of -1.5 (outside valid range [-1, 0]). This caused:
1. Cache misses were not detected (falsely reported as "Cache hits: 10/10")
2. All cache-miss TCRs scored as -1.5, resulting in AUROC=0.5 (no discrimination)

**Fix**: Changed to `cache_miss_score=0.5` (default), which produces raw=-0.5 for cache misses. Cache miss detection now works correctly.

## Summary Table

| Model | ERGO AUROC | tFold AUROC | Difference | Status |
|-------|------------|-------------|------------|--------|
| test41 | 0.6243 | 0.4017 | -0.2227 | Best ERGO model |
| test33 | 0.5983 | 0.4367 | -0.1617 | Baseline |
| test39 | 0.6058 | 0.5000 | -0.1058 | Best tFold AUROC |

## Key Findings

### 1. tFold Shows Lower Mean AUROC Than ERGO

All three models show lower mean tFold AUROC compared to ERGO:
- test41: 0.4017 vs 0.6243 (ERGO better by 0.22)
- test33: 0.4367 vs 0.5983 (ERGO better by 0.16)
- test39: 0.5000 vs 0.6058 (ERGO better by 0.11)

### 2. Peptide-Specific Performance Varies Dramatically

**Peptides where tFold consistently outperforms ERGO:**
- **NLVPMVATV**: tFold 0.80-0.90 vs ERGO 0.29-0.49 (+0.31 to +0.61 improvement)
- **KLGGALQAK**: tFold 0.72-0.90 vs ERGO 0.61-0.63 (+0.10 to +0.29 improvement)
- **RLRAEAQVK**: tFold 0.76-0.98 vs ERGO 0.64-0.67 (+0.09 to +0.32 improvement)

**Peptides where tFold shows reversed discrimination:**
- **LLWNGPMAV**: tFold 0.02-0.16 vs ERGO 0.80-0.83 (tFold fails)
- **AVFDRKSDAK**: tFold 0.06-0.10 vs ERGO 0.71-0.73 (tFold fails)
- **IVTDFSVIK**: tFold 0.30-0.62 vs ERGO 0.91-0.92 (tFold worse)
- **YLQPRTFLL**: tFold 0.38-0.42 vs ERGO 0.89-0.91 (tFold worse)

### 3. Cache Coverage Impact

Most peptides had 0/10 cache hits for targets, requiring real-time tFold scoring:
- **GILGFVFTL**: Only peptide with consistent cache hits (10/10 for test41)
- **NLVPMVATV**: 1/10 cache hits (test41), 0/10 for others
- **All other peptides**: 0/10 cache hits

Despite low cache coverage, the evaluation completed successfully with the bug fix.

---

## Per-Model Results

### test41 (Best ERGO Model)

| Target | ERGO AUROC | tFold AUROC | Diff | Cache Hits |
|--------|-----------|------------|------|------------|
| GILGFVFTL | 0.3886 | 0.1200 | -0.2686 | 10/10 |
| **NLVPMVATV** | 0.4884 | **0.8000** | **+0.3116** | 1/10 |
| GLCTLVAML | 0.3538 | 0.1000 | -0.2538 | 0/10 |
| LLWNGPMAV | 0.8023 | 0.0400 | -0.7623 | 0/10 |
| YLQPRTFLL | 0.8978 | 0.3800 | -0.5178 | 0/10 |
| **FLYALALLL** | 0.4752 | **0.5400** | +0.0648 | 0/10 |
| **SLYNTVATL** | 0.6145 | **0.6200** | +0.0055 | 0/10 |
| **KLGGALQAK** | 0.6278 | **0.7800** | **+0.1522** | 0/10 |
| AVFDRKSDAK | 0.7334 | 0.1000 | -0.6334 | 0/10 |
| IVTDFSVIK | 0.9114 | 0.3000 | -0.6114 | 0/10 |
| SPRWYFYYL | 0.5272 | 0.2800 | -0.2472 | 0/10 |
| **RLRAEAQVK** | 0.6714 | **0.7600** | +0.0886 | 0/10 |
| **MEAN** | **0.6243** | **0.4017** | **-0.2227** | - |

**Highlights**:
- 5 peptides where tFold agrees/improves: NLVPMVATV (0.80!), KLGGALQAK (0.78), RLRAEAQVK (0.76), SLYNTVATL (0.62), FLYALALLL (0.54)
- 7 peptides with reversed discrimination (tFold scores decoys higher than targets)

### test33 (Baseline)

| Target | ERGO AUROC | tFold AUROC | Diff | Cache Hits |
|--------|-----------|------------|------|------------|
| GILGFVFTL | 0.4191 | 0.4000 | -0.0191 | 0/10 |
| **NLVPMVATV** | 0.3348 | **0.9000** | **+0.5652** | 0/10 |
| GLCTLVAML | 0.3360 | 0.0800 | -0.2560 | 0/10 |
| LLWNGPMAV | 0.8255 | 0.0200 | -0.8055 | 0/10 |
| YLQPRTFLL | 0.8856 | 0.3800 | -0.5056 | 0/10 |
| FLYALALLL | 0.4469 | 0.4200 | -0.0269 | 0/10 |
| SLYNTVATL | 0.5291 | 0.4400 | -0.0891 | 0/10 |
| **KLGGALQAK** | 0.6241 | **0.7200** | +0.0959 | 1/10 |
| AVFDRKSDAK | 0.7122 | 0.0600 | -0.6522 | 0/10 |
| IVTDFSVIK | 0.9069 | 0.4200 | -0.4869 | 0/10 |
| **SPRWYFYYL** | 0.5177 | **0.5400** | +0.0223 | 0/10 |
| **RLRAEAQVK** | 0.6423 | **0.8600** | **+0.2177** | 1/10 |
| **MEAN** | **0.5983** | **0.4367** | **-0.1617** | - |

**Highlights**:
- NLVPMVATV shows exceptional tFold AUROC of 0.90 (+0.57 vs ERGO)
- RLRAEAQVK: 0.86 tFold vs 0.64 ERGO (+0.22)
- 4 peptides where tFold improves over ERGO

### test39

| Target | ERGO AUROC | tFold AUROC | Diff | Cache Hits |
|--------|-----------|------------|------|------------|
| GILGFVFTL | 0.5495 | 0.3400 | -0.2095 | 0/10 |
| **NLVPMVATV** | 0.2943 | **0.9000** | **+0.6057** | 0/10 |
| GLCTLVAML | 0.3035 | 0.1800 | -0.1235 | 0/10 |
| LLWNGPMAV | 0.8317 | 0.1600 | -0.6717 | 0/10 |
| YLQPRTFLL | 0.9103 | 0.4200 | -0.4903 | 1/10 |
| FLYALALLL | 0.4860 | 0.4600 | -0.0260 | 0/10 |
| **SLYNTVATL** | 0.4500 | **0.6000** | +0.1500 | 0/10 |
| **KLGGALQAK** | 0.6125 | **0.9000** | **+0.2875** | 0/10 |
| AVFDRKSDAK | 0.7209 | 0.0800 | -0.6409 | 0/10 |
| IVTDFSVIK | 0.9195 | 0.6200 | -0.2995 | 0/10 |
| SPRWYFYYL | 0.5288 | 0.3600 | -0.1688 | 0/10 |
| **RLRAEAQVK** | 0.6630 | **0.9800** | **+0.3170** | 0/10 |
| **MEAN** | **0.6058** | **0.5000** | **-0.1058** | - |

**Highlights**:
- Best overall tFold mean AUROC (0.5000, exactly random baseline)
- RLRAEAQVK: exceptional 0.98 tFold AUROC (+0.32 vs ERGO)
- KLGGALQAK: 0.90 tFold AUROC (+0.29 vs ERGO)
- NLVPMVATV: 0.90 tFold AUROC (+0.61 vs ERGO, largest improvement)

---

## Interpretation

### 1. Why is tFold Mean AUROC Lower Than ERGO?

The lower mean tFold AUROC (0.40-0.50 vs ERGO 0.60-0.62) is driven by:

1. **Reversed discrimination on 7 peptides**: tFold scores decoys higher than targets on LLWNGPMAV, AVFDRKSDAK, IVTDFSVIK, YLQPRTFLL, GLCTLVAML, GILGFVFTL, SPRWYFYYL
2. **RL-generated TCRs may not be structurally plausible**: The RL model was trained on ERGO scores, optimizing for ERGO's criteria. tFold evaluates structural plausibility, which may penalize sequences that are "too optimized" for ERGO.
3. **V-region scaffold mismatch**: We use a generic V-region scaffold, but tFold expects natural V-CDR3 combinations. The mismatch may cause tFold to score these TCRs as unnatural.

### 2. Why Does tFold Excel on Some Peptides?

**NLVPMVATV, KLGGALQAK, RLRAEAQVK** show strong tFold performance (0.76-0.98 AUROC):
- These may be peptides where ERGO has poor discrimination (ERGO AUROC 0.29-0.67)
- tFold's structural evaluation provides complementary signal
- The RL-generated TCRs for these targets may happen to be structurally plausible

### 3. Comparison with PEPTIDE_SCORER_MAPPING.md

From `PEPTIDE_SCORER_MAPPING.md`, we know tFold's discriminator AUROC on real TCRs:

| Peptide | tFold Discriminator AUROC | tFold Eval AUROC (test41) | Match? |
|---------|---------------------------|---------------------------|--------|
| GILGFVFTL | 0.85 (Good) | 0.12 (Poor) | ❌ Mismatch |
| NLVPMVATV | 0.82 (Good) | 0.80 (Good) | ✅ Match |
| GLCTLVAML | 0.79 (Good) | 0.10 (Poor) | ❌ Mismatch |
| LLWNGPMAV | 0.81 (Good) | 0.04 (Poor) | ❌ Mismatch |

**Conclusion**: tFold's discriminator AUROC (measured on real TCRs) does NOT predict its performance on RL-generated TCRs. This suggests the RL-generated sequences are fundamentally different from natural TCRs.

---

## Recommendations

### 1. Use Peptide-Specific Scorer Selection

Based on these results, use:
- **NLVPMVATV, KLGGALQAK, RLRAEAQVK**: tFold (0.76-0.98 AUROC)
- **LLWNGPMAV, YLQPRTFLL, IVTDFSVIK**: ERGO (0.80-0.91 AUROC)
- **Others**: ERGO by default (tFold shows reversed discrimination)

### 2. Investigate Reversed Discrimination

Why does tFold score decoys higher than targets on 7 peptides?

**Hypothesis**: RL-generated TCRs lack structural plausibility. Even though they score well on ERGO, tFold's structure-aware evaluation penalizes them.

**Action**: Score known high-affinity TCRs from VDJdb with tFold to verify the scorer works correctly on natural sequences.

### 3. Consider Hybrid Training (ERGO + tFold)

Use both scorers during training:
- 90% episodes: ERGO reward (fast, good on most peptides)
- 10% episodes: tFold reward (accurate on NLVPMVATV, KLGGALQAK, RLRAEAQVK)

This prevents overfitting to ERGO's biases while maintaining training speed.

### 4. Build Complete tFold Cache

Current cache coverage is low (0-1 hits per 10 TCRs). To enable faster evaluation:

```bash
python scripts/warmup_tfold_cache.py \
    --targets GILGFVFTL,NLVPMVATV,GLCTLVAML,LLWNGPMAV,YLQPRTFLL,FLYALALLL,SLYNTVATL,KLGGALQAK,AVFDRKSDAK,IVTDFSVIK,SPRWYFYYL,RLRAEAQVK \
    --n_tcrs_per_target 200
```

This will take ~6-8 hours but enables <1ms cache-hit scoring.

---

## Files Generated

- `results/test41_eval/eval_results_with_tfold.json` (604 KB, updated 2026-04-30 14:10)
- `results/test33_eval/eval_results_with_tfold.json` (603 KB, updated 2026-04-30 16:03)
- `results/test39_eval/eval_results_with_tfold.json` (603 KB, updated 2026-04-30 17:55)

Each file contains:
- Original ERGO evaluation results
- New tFold evaluation results (under `specificity.tfold`)
- Per-TCR ranked details with target scores, AUROC, and composite metrics

---

## Comparison with Previous (Buggy) Evaluation

**Previous results (with cache_miss_score=-0.5 bug)**:
- test41: tFold AUROC 0.4842 (10 peptides showed 0.5000 due to cache miss bug)
- test33: tFold AUROC 0.5000 (all peptides 0.5000)
- test39: tFold AUROC 0.5000 (all peptides 0.5000)

**Corrected results (with cache_miss_score=0.5 fix)**:
- test41: tFold AUROC 0.4017 (real scores, cache misses properly detected)
- test33: tFold AUROC 0.4367 (real scores)
- test39: tFold AUROC 0.5000 (real scores, happens to be exactly 0.5)

The bug fix revealed the true tFold performance, which is lower than initially reported but now reflects actual discrimination ability.
