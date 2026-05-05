# tFold Evaluation Analysis: Cache Coverage Issue (RESOLVED)

**Date**: 2026-04-30  
**Status**: ✅ Bug fixed and re-evaluation completed  
**Issue**: Most peptides showed -1.5 scores (cache miss default) instead of real tFold scores  
**Resolution**: Fixed `cache_miss_score` parameter and re-ran evaluation with real tFold scoring

## Root Cause

The evaluation used `cache_miss_score=-0.5` which produces raw scores of `-1.5` for cache misses:
```python
default_raw = cache_miss_score - 1.0 = -0.5 - 1.0 = -1.5
```

The cache miss detection logic checks for `s == -0.5`, but with the wrong `cache_miss_score`, it should check for `s == -1.5`. This caused:
1. Cache misses were not detected (reported as "Cache hits: 10/10")
2. All cache-miss TCRs got scored as -1.5 (outside valid range [-1, 0])
3. AUROC became 0.5 (no discrimination when all scores are identical)

## Cache Coverage Per Target Peptide

| Peptide | Cache Entries | Coverage |
|---------|---------------|----------|
| **GILGFVFTL** | 255 | ✅ High |
| **NLVPMVATV** | 144 | ⚠️ Medium |
| GLCTLVAML | 71 | ❌ Low |
| LLWNGPMAV | 53 | ❌ Low |
| KLGGALQAK | 21 | ❌ Very Low |
| IVTDFSVIK | 19 | ❌ Very Low |
| FLYALALLL | 17 | ❌ Very Low |
| AVFDRKSDAK | 16 | ❌ Very Low |
| YLQPRTFLL | 15 | ❌ Very Low |
| SPRWYFYYL | 14 | ❌ Very Low |
| RLRAEAQVK | 22 | ❌ Very Low |
| SLYNTVATL | 6 | ❌ Very Low |

**Total cache size**: 15,486 entries  
**Top cached peptides**: GILGFVFTL (255), CLAGIVFIL (236), GLLGFVLDL (236), etc.

## Actual tFold Performance (Valid Scores Only)

### test41 - GILGFVFTL (255 cache entries)
- **tFold AUROC**: 0.2600 (below random!)
- **ERGO AUROC**: 0.3886
- **Mean target score**: -0.9051 (mapped: 0.0949)
- **Mean decoy score**: -0.8922 (mapped: 0.1078)
- **Gap**: -0.0130 (target scores LOWER than decoys)

**Interpretation**: tFold thinks the generated TCRs bind **worse** to the target than to decoys. This is reversed discrimination.

### test41 - NLVPMVATV (144 cache entries)
- **tFold AUROC**: 0.5500 (slightly above random)
- **ERGO AUROC**: 0.4884
- **Mean target score**: -1.4489 (mapped: 0.0511)
- **Mean decoy score**: -1.5000 (mapped: 0.0000, likely cache miss)
- **Gap**: +0.0511

**Interpretation**: Some TCRs had real scores, but many decoys were cache misses (scored as -1.5).

### All Other Peptides
- **All scores**: -1.5 (cache miss default)
- **AUROC**: 0.5000 (random, because all scores identical)
- **Conclusion**: No real tFold evaluation happened

## Why Cache Coverage is Low

The cache was built from:
1. **SAC training episodes** on 4 peptides (KLWASPLHV, FPRPWLHGL, KAFSPEVIPMF, HSKKKCDEL)
2. **Warmup cache script** for specific peptides
3. **Previous evaluations** (mainly GILGFVFTL)

The 12 McPAS test peptides were **not** systematically cached, so most generated TCRs are cache misses.

## tFold Scorer Performance Assessment

Based on the **only valid evaluation** (GILGFVFTL with 255 cache entries):

### GILGFVFTL: ❌ POOR (Reversed Discrimination)
- tFold AUROC: 0.26 (below random 0.5)
- Target scores: -0.91 (weak binding)
- Decoy scores: -0.89 (slightly better than target!)
- **Conclusion**: tFold thinks generated TCRs bind better to decoys than to target

### NLVPMVATV: ⚠️ INCONCLUSIVE
- tFold AUROC: 0.55 (slightly above random)
- Many decoys were cache misses (-1.5)
- Cannot draw reliable conclusions

### All Other 10 Peptides: ⚠️ NO DATA
- All scores are -1.5 (cache miss default)
- No real tFold evaluation occurred

## Comparison with PEPTIDE_SCORER_MAPPING.md

From `PEPTIDE_SCORER_MAPPING.md`, we know:

| Peptide | ERGO Reliability | tFold Reliability | Notes |
|---------|------------------|-------------------|-------|
| GILGFVFTL | ✅ Good (0.823) | ✅ Good (0.85) | Both scorers reliable |
| NLVPMVATV | ⚠️ Medium (0.698) | ✅ Good (0.82) | tFold better |
| GLCTLVAML | ❌ Poor (0.543) | ✅ Good (0.79) | tFold much better |
| LLWNGPMAV | ✅ Good (0.762) | ✅ Good (0.81) | Both reliable |

**Expected**: tFold should perform well on GILGFVFTL (0.85 discriminator AUROC)  
**Actual**: tFold AUROC = 0.26 (reversed discrimination)

**Hypothesis**: The RL-generated TCRs are **not natural-looking** according to tFold's criteria, even though they score well on ERGO. tFold may be penalizing:
1. Unnatural CDR3 sequences
2. Lack of proper V-region context (we use a generic scaffold)
3. Sequences optimized for ERGO but not for structural plausibility

## Recommendations

### 1. Fix the Evaluation Bug
```python
# In reevaluate_with_tfold_fast.py, line 272:
tfold_scorer_cache_only = AffinityTFoldScorer(
    device=args.device, 
    cache_only=True, 
    cache_miss_score=0.5  # NOT -0.5!
)

# And fix cache miss detection, line 113:
cache_miss_indices = [i for i, s in enumerate(target_scores_cached) if s == -0.5]
# This is correct when cache_miss_score=0.5 (raw = 0.5 - 1.0 = -0.5)
```

### 2. Build Complete Cache
Run warmup script for all 12 test peptides:
```bash
python scripts/warmup_tfold_cache.py \
    --targets GILGFVFTL,NLVPMVATV,GLCTLVAML,LLWNGPMAV,YLQPRTFLL,FLYALALLL,SLYNTVATL,KLGGALQAK,AVFDRKSDAK,IVTDFSVIK,SPRWYFYYL,RLRAEAQVK \
    --n_tcrs_per_target 200
```

This will take ~6-8 hours but is necessary for accurate evaluation.

### 3. Re-run Evaluation with Full Cache
After cache warmup:
```bash
python scripts/reevaluate_with_tfold_fast.py \
    --models test41,test33,test39 \
    --n_tcrs 50 \
    --n_decoys 20 \
    --device cuda:1
```

### 4. Investigate GILGFVFTL Reversed Discrimination

Why does tFold give **lower** scores to targets than decoys on GILGFVFTL?

Possible reasons:
- RL model optimized for ERGO, not tFold
- Generated sequences lack structural plausibility
- tFold's V-region scaffold doesn't match the generated CDR3s well
- tFold penalizes sequences that look "too optimized"

**Action**: Score known high-affinity GILGFVFTL binders from VDJdb with tFold to verify the scorer works correctly.

### 5. Consider Hybrid Training

If tFold and ERGO disagree systematically:
- Use **both** scorers during training (already implemented in SAC)
- Weight them based on per-peptide reliability from `PEPTIDE_SCORER_MAPPING.md`
- This prevents overfitting to ERGO's biases

## Summary

### Bug Fixed (2026-04-30)

✅ **Fixed `cache_miss_score` parameter**: Changed from -0.5 to 0.5 (default)  
✅ **Re-ran evaluation**: All 3 models (test41, test33, test39) completed successfully  
✅ **Cache miss detection working**: Properly detects and scores cache misses via tFold server  

### Corrected Results

**Mean tFold AUROC**:
- test41: 0.4017 (vs ERGO 0.6243, -0.22)
- test33: 0.4367 (vs ERGO 0.5983, -0.16)
- test39: 0.5000 (vs ERGO 0.6058, -0.11)

**Key findings**:
1. **tFold shows lower mean AUROC than ERGO** across all models
2. **Peptide-specific performance varies dramatically**:
   - **Strong tFold peptides**: NLVPMVATV (0.80-0.90), KLGGALQAK (0.72-0.90), RLRAEAQVK (0.76-0.98)
   - **Weak tFold peptides**: LLWNGPMAV (0.02-0.16), AVFDRKSDAK (0.06-0.10), IVTDFSVIK (0.30-0.62)
3. **7 peptides show reversed discrimination** (tFold scores decoys higher than targets)
4. **RL-generated TCRs may lack structural plausibility** according to tFold's criteria

### Interpretation

The lower tFold AUROC is NOT due to cache coverage issues (cache misses were properly scored). Instead:
- **RL model optimized for ERGO**, not tFold
- **Generated sequences may be "too optimized"** for ERGO's criteria, lacking natural structural features
- **V-region scaffold mismatch**: Generic scaffold may not match natural V-CDR3 combinations
- **tFold and ERGO evaluate different aspects**: ERGO focuses on sequence patterns, tFold on structural plausibility

### Recommendations

1. **Use peptide-specific scorer selection**: tFold for NLVPMVATV/KLGGALQAK/RLRAEAQVK, ERGO for others
2. **Investigate reversed discrimination**: Score known VDJdb binders with tFold to verify scorer correctness
3. **Consider hybrid training**: Mix ERGO (90%) and tFold (10%) rewards to prevent overfitting to ERGO biases
4. **Build complete cache** (optional): Warmup script for all 12 peptides to enable faster future evaluations

**See `TFOLD_EVAL_RESULTS_CORRECTED.md` for full results.**
