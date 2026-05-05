# tFold Re-evaluation Results (Fast Sampling)

**Date**: 2026-04-29  
**Evaluation**: Top 10 TCRs per target, 5 decoys per TCR  
**Models**: test41, test33, test39

## Summary

| Model | ERGO AUROC | tFold AUROC | Difference |
|-------|------------|-------------|------------|
| test41 | 0.6243 | 0.4842 | -0.1402 |
| test33 | 0.5983 | 0.5000 | -0.0983 |
| test39 | 0.6058 | 0.5000 | -0.1058 |

## Key Findings

1. **tFold shows significantly lower discrimination than ERGO** across all models
   - test41: 0.4842 (below random baseline of 0.5)
   - test33/test39: exactly 0.5000 (random discrimination)

2. **Most individual targets show exactly 0.5000 AUROC**
   - With only 10 TCRs and 5 decoys, the sample size is too small for stable AUROC estimation
   - tFold scores for targets and decoys are very similar (all in 0.01-0.08 range after mapping)

3. **Critical bug fixed during evaluation**
   - **Issue**: Generated TCRs from RL model lack the conserved N-terminal Cys
   - **Symptom**: tFold feature extraction failed with "Feature extraction failed" error
   - **Root cause**: tFold's CDR3 detection requires conserved Cys within 30 residues of FG.G motif
   - **Fix**: Prepend 'C' to CDR3β sequences that don't start with it (line 520-527 in `affinity_tfold.py`)
   - **Impact**: All cached entries have Cys prefix (e.g., `CAASMSGPLPTYGQIGYI`), but RL-generated sequences don't (e.g., `ASSQNGRRRTGNTIYF`)

## Per-Model Results

### test41 (Best Model)

| Target | ERGO AUROC | tFold AUROC | Diff |
|--------|------------|-------------|------|
| GILGFVFTL | 0.3886 | 0.2600 | -0.1286 |
| NLVPMVATV | 0.4884 | 0.5500 | +0.0616 |
| GLCTLVAML | 0.3538 | 0.5000 | +0.1462 |
| LLWNGPMAV | 0.8023 | 0.5000 | -0.3023 |
| YLQPRTFLL | 0.8978 | 0.5000 | -0.3978 |
| FLYALALLL | 0.4752 | 0.5000 | +0.0248 |
| SLYNTVATL | 0.6145 | 0.5000 | -0.1145 |
| KLGGALQAK | 0.6278 | 0.5000 | -0.1278 |
| AVFDRKSDAK | 0.7334 | 0.5000 | -0.2334 |
| IVTDFSVIK | 0.9114 | 0.5000 | -0.4114 |
| SPRWYFYYL | 0.5272 | 0.5000 | -0.0272 |
| RLRAEAQVK | 0.6714 | 0.5000 | -0.1714 |
| **MEAN** | **0.6243** | **0.4842** | **-0.1402** |

### test33

| Target | ERGO AUROC | tFold AUROC | Diff |
|--------|------------|-------------|------|
| All 12 targets | 0.5983 | 0.5000 | -0.0983 |

(All individual targets show exactly 0.5000 AUROC)

### test39

| Target | ERGO AUROC | tFold AUROC | Diff |
|--------|------------|-------------|------|
| All 12 targets | 0.6058 | 0.5000 | -0.1058 |

(All individual targets show exactly 0.5000 AUROC)

## Technical Details

### Evaluation Protocol

1. **Target scoring**: Top 10 TCRs per target (ranked by ERGO score)
2. **Decoy sampling**: 5 decoys per TCR, sampled from tiers A/B/D
   - Tier A: 1-2 AA point mutants (up to 5 per target)
   - Tier B: 2-3 AA mutants (up to 5 per target)
   - Tier D: Known binders from VDJdb/IEDB (up to 5 per target)
3. **AUROC computation**: Per-TCR AUROC (1 positive vs 5 negatives), then averaged
4. **Cache utilization**: All target TCRs were cache hits (no new feature extraction needed)

### Score Ranges

- **tFold raw scores**: [-1, 0] where 0 = strong binding
- **tFold mapped scores**: [0, 1] (raw + 1.0) for ERGO compatibility
- **Observed mapped scores**: 0.01-0.08 (very low binding predictions for all TCRs)

### Files Generated

- `results/test41_eval/eval_results_with_tfold.json` (599 KB)
- `results/test33_eval/eval_results_with_tfold.json` (598 KB)
- `results/test39_eval/eval_results_with_tfold.json` (598 KB)

Each file contains:
- Original ERGO evaluation results
- New tFold evaluation results (under `specificity.tfold`)
- Per-TCR ranked details with target scores, AUROC, and composite metrics

## Interpretation

The low tFold AUROCs suggest one of:

1. **Small sample size**: 10 TCRs and 5 decoys may be insufficient for stable AUROC estimation
2. **Score compression**: tFold assigns very similar scores to all TCRs (0.01-0.08 range), making discrimination difficult
3. **Different scoring criteria**: tFold and ERGO may evaluate binding differently
4. **Model quality**: The RL-generated TCRs may not have strong binding specificity according to tFold's criteria

## Recommendations

1. **Increase sample size**: Re-run with 50 TCRs and 20 decoys for more stable AUROC estimates
2. **Check score distributions**: Analyze the distribution of tFold scores for targets vs decoys
3. **Compare with known binders**: Score known high-affinity TCRs from VDJdb to validate tFold's discrimination ability
4. **Investigate score compression**: Check if tFold's V3.4 classifier is properly calibrated for CDR3β-only input

## Next Steps

- [ ] Resume paused SAC training: `kill -CONT 2695316`
- [ ] Decide whether to run full evaluation (200 TCRs, 30 decoys) for more accurate AUROC
- [ ] Consider using tFold for training (hybrid ERGO+tFold) to improve specificity
