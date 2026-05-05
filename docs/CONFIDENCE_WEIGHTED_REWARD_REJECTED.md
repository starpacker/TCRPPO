# Confidence-Weighted Reward: Validation Failed

**Date**: 2026-05-04  
**Status**: REJECTED - Do NOT implement test46  
**Validation Script**: `scripts/validate_ergo_confidence.py`  
**Results**: `results/validation/ergo_confidence_validation.json`

---

## Executive Summary

**Hypothesis**: Using `reward = score * confidence` where `confidence = 1 - MC_Dropout_std` will force the agent to design TCRs within ERGO's reliable domain, improving specificity.

**Validation Result**: **REJECTED**

**Critical Finding**: MC Dropout confidence has **strong NEGATIVE correlation** with specificity (r = -0.71). This means:
- **High confidence → Low specificity** (BAD)
- **Low confidence → High specificity** (GOOD)

Using confidence-weighted reward would make the agent prefer **low-specificity TCRs**, the exact opposite of our goal.

---

## Validation Methodology

### Experimental Design

1. **Sample 200 random TCRs** from TCRdb (representative of RL exploration space)
2. **Compute MC Dropout statistics** for each TCR on test peptide GILGFVFTL:
   - Run ERGO 10 times with dropout enabled
   - Extract mean score and std
   - Compute confidence = 1 - std
3. **Compute per-TCR AUROC** (specificity measure):
   - Score each TCR on target peptide (GILGFVFTL)
   - Score each TCR on 20 random decoy peptides (from 1900-peptide library)
   - AUROC = does ERGO score this TCR higher on target than decoys?
   - High AUROC = good specificity, low AUROC = poor specificity
4. **Analyze correlation** between confidence and AUROC

### Why This Validation is Correct

**Per-TCR AUROC** is the right metric because:
- It measures what we care about: does ERGO reliably score a TCR higher on target than decoys?
- High AUROC = ERGO's prediction is trustworthy for this TCR
- Low AUROC = ERGO's prediction is unreliable (scores decoys as high as target)

**If confidence is a good signal**, we expect:
- High confidence → High AUROC (ERGO is certain AND correct)
- Low confidence → Low AUROC (ERGO is uncertain AND unreliable)
- Positive correlation: r > 0.3

---

## Results

### MC Dropout std Distribution

| Statistic | Value |
|-----------|-------|
| Mean std | 0.0671 |
| Median std | 0.0464 |
| Std of std | 0.0625 |
| Min std | 0.0003 |
| Max std | 0.3569 |
| 5th percentile | 0.0054 |
| 95th percentile | 0.1865 |
| **Range** | **0.3566** |

**Interpretation**: MC Dropout std has good variance (range 0.36), so the signal is not flat. This rules out "confidence is always high" as the failure mode.

### Per-TCR AUROC Distribution

| Statistic | Value |
|-----------|-------|
| Mean AUROC | 0.4357 |
| Median AUROC | 0.3500 |
| Std AUROC | 0.3208 |
| Min AUROC | 0.0000 |
| Max AUROC | 1.0000 |

**Interpretation**: AUROC has wide variance (0.0-1.0), indicating some TCRs have excellent specificity while others have none. This is expected for random TCRs.

### Correlation Analysis

| Correlation | Value | Interpretation |
|-------------|-------|----------------|
| **corr(mean_score, AUROC)** | **+0.7194** | ✅ High ERGO score → High specificity (GOOD) |
| **corr(std_score, AUROC)** | **+0.7072** | ❌ High uncertainty → High specificity (OPPOSITE) |
| **corr(confidence, AUROC)** | **-0.7072** | ❌ High confidence → Low specificity (CATASTROPHIC) |

**Key Finding**: The correlation is **strong and NEGATIVE** (r = -0.71). This is not a weak signal - it's a strong signal in the **wrong direction**.

---

## Why This Happens: Root Cause Analysis

### Hypothesis 1: MC Dropout Measures Aleatoric Uncertainty, Not Epistemic

**Epistemic uncertainty** (what we want): "I don't know because I haven't seen data like this"
- Should correlate with OOD-ness
- Should predict unreliable predictions

**Aleatoric uncertainty** (what MC Dropout captures): "The data itself is noisy"
- Measures inherent variability in ERGO's training data
- High aleatoric uncertainty = ERGO saw many conflicting examples
- This does NOT mean the prediction is wrong

**Evidence**: 
- TCRs with high std (low confidence) have high AUROC (good specificity)
- This suggests high std = ERGO is uncertain but still discriminates well
- Low std = ERGO is confident but may be overconfident on easy/trivial cases

### Hypothesis 2: Low Std = Trivial/Universal Binders

**Observation**: Low std (high confidence) correlates with low AUROC (poor specificity).

**Possible explanation**:
- TCRs that score similarly across all peptides (universal binders) have low std
- ERGO gives them consistent scores regardless of peptide
- But this consistency is BAD - it means no specificity
- High std = ERGO's score varies across peptides = better discrimination

**Example from results**:
```json
{
  "tcr": "CSARVPLAGANEQFF",
  "mean_score": 0.2396,
  "std_score": 0.0480,  // Low std (high confidence)
  "confidence": 0.9520,
  "auroc": 0.45  // Poor specificity
}
```

vs

```json
{
  "tcr": "CSVAGIGVSGNTIYF",
  "mean_score": 0.2005,
  "std_score": 0.1177,  // High std (low confidence)
  "confidence": 0.8823,
  "auroc": 1.0  // Perfect specificity
}
```

### Hypothesis 3: MC Dropout is Not Calibrated for OOD Detection

MC Dropout was designed for **uncertainty quantification**, not **OOD detection**. These are related but not identical:

- **Uncertainty quantification**: "How confident is the model in this prediction?"
- **OOD detection**: "Is this input outside the training distribution?"

ERGO's MC Dropout measures the former, but we need the latter.

---

## Comparison with test45 Findings

### test45 (OOD Penalty with Additive Penalty)

- Used same MC Dropout uncertainty signal
- Applied penalty when uncertainty > threshold
- **Result**: AUROC 0.4031 (worse than baseline 0.4538)
- **OOD trigger rate increased** from 13% to 36.5%
- **Conclusion**: Agent learned to exploit high-uncertainty space

### This Validation (Confidence-Weighted Reward)

- Proposes multiplicative weighting: `reward = score * confidence`
- **Validation shows**: confidence is negatively correlated with specificity
- **Predicted outcome**: Agent would learn to prefer low-confidence (high-std) TCRs
- **But**: High std correlates with HIGH specificity, not low specificity!

### Paradox Resolution

**The paradox**: 
- test45 showed agent exploited high-uncertainty space (bad)
- This validation shows high uncertainty correlates with high specificity (good)

**Resolution**:
- In test45, agent learned to generate TCRs with high ERGO score AND high uncertainty
- These TCRs likely have high specificity (per this validation)
- But test45 trained on only 4 peptides, evaluated on 12 different ones (zero overlap)
- The poor AUROC (0.4031) was due to **training-evaluation mismatch**, not OOD penalty

**Implication**: The OOD penalty in test45 may have been working correctly (pushing agent toward high-uncertainty = high-specificity space), but the training-evaluation mismatch masked this.

---

## Alternative Interpretation: Should We Invert the Signal?

### Proposal: Use `reward = score * std` instead of `reward = score * (1 - std)`

**Rationale**: Since high std correlates with high specificity, maybe we should REWARD high uncertainty?

**Problems with this approach**:

1. **Counterintuitive**: Rewarding uncertainty goes against the principle of staying in-domain
2. **Correlation ≠ Causation**: High std may be a side effect of good TCRs, not the cause
3. **Overfitting to validation set**: This correlation may not hold on other peptides
4. **No theoretical justification**: Why would uncertainty be a good thing?

### Better Alternative: Use Mean Score Only

**Evidence from this validation**:
- `corr(mean_score, AUROC) = +0.7194` (strong positive)
- Mean ERGO score is a BETTER predictor of specificity than confidence

**Recommendation**: 
- Use `reward = mean_score` (pure ERGO, no confidence weighting)
- This is what test14 did (AUROC 0.6091)
- This is what test41 did in phase 1 (AUROC 0.6243 after contrastive fine-tuning)

---

## Final Verdict

### Do NOT Implement test46

**Reasons**:
1. **Strong negative correlation** (r = -0.71) between confidence and specificity
2. **Confidence-weighted reward would harm performance**, not improve it
3. **Mean ERGO score alone is a better signal** (r = +0.72 with specificity)
4. **No theoretical justification** for why MC Dropout std should predict OOD-ness

### Recommended Alternatives

1. ✅ **Pure ERGO reward** (test14: 0.6091, test41: 0.6243)
   - Use mean score only, ignore confidence
   - Already proven to work

2. ✅ **Two-phase training** (test41: 0.6243)
   - Phase 1: Pure ERGO warm-start (1M steps)
   - Phase 2: Contrastive fine-tuning with 16 decoys (1M steps)
   - Best result so far

3. ✅ **tFold cascade** (test44: in progress)
   - Use ERGO for fast scoring
   - Use tFold for validation when uncertainty is high
   - Structure-aware scoring is more reliable than uncertainty-based filtering

4. ✅ **Expand training peptides** (test44: 29 peptides)
   - Train on diverse peptide set (29 tFold-good peptides)
   - Reduces overfitting to specific peptides
   - Improves generalization

5. ❌ **Stronger OOD penalty** (test45 with higher weight)
   - Already failed with weight=1.0
   - Increasing weight won't fix the fundamental problem

6. ❌ **Inverted confidence signal** (`reward = score * std`)
   - No theoretical justification
   - Likely overfitting to this validation set

---

## Lessons Learned

1. **Always validate signals before using them in RL**
   - We assumed MC Dropout confidence would predict specificity
   - Validation showed the opposite
   - Saved us from wasting GPU time on test46

2. **MC Dropout uncertainty ≠ OOD detection**
   - MC Dropout measures model uncertainty, not input validity
   - High uncertainty can indicate good discrimination, not unreliability

3. **Correlation analysis is essential**
   - Per-TCR AUROC is the right metric for specificity
   - Correlation with confidence reveals signal quality
   - Strong negative correlation is worse than no correlation

4. **Mean score is a better signal than confidence**
   - `corr(mean_score, AUROC) = +0.72` vs `corr(confidence, AUROC) = -0.71`
   - Pure ERGO reward is better than confidence-weighted reward

5. **Training-evaluation mismatch is deadly**
   - test45 trained on 4 peptides, evaluated on 12 different ones
   - This masked the true effect of OOD penalty
   - Always train and evaluate on the same peptide set

---

## Appendix: Validation Script

**Location**: `scripts/validate_ergo_confidence.py`

**Key Functions**:
```python
def compute_mc_dropout_stats(ergo_scorer, tcrs, peptide, n_forward=10):
    """Compute MC Dropout mean and std for each TCR."""
    # Uses ergo_scorer.mc_dropout_score() (10 forward passes)
    # Returns mean_score, std_score, confidence = 1 - std

def compute_per_tcr_auroc(ergo_scorer, tcrs, peptide, decoy_peptides, n_decoys=20):
    """Compute per-TCR AUROC (specificity measure)."""
    # Score TCR on target and 20 random decoys
    # AUROC = does ERGO score TCR higher on target than decoys?
    # High AUROC = good specificity
```

**Outputs**:
- `figures/ergo_mc_dropout_std_distribution.png` - Histogram of MC Dropout std
- `figures/ergo_confidence_auroc_correlation.png` - Scatter plots of correlations
- `results/validation/ergo_confidence_validation.json` - Full results with per-TCR data

**Verdict Criteria**:
- PASS: `corr(confidence, AUROC) > 0.3` (positive correlation)
- FAIL: `corr(confidence, AUROC) < -0.1` (negative correlation)
- MARGINAL: `-0.1 < corr < 0.3` (weak correlation)

**Actual Result**: `corr(confidence, AUROC) = -0.71` → **FAIL**

---

**Document Author**: Claude Code  
**Last Updated**: 2026-05-04  
**Validation Completed**: 2026-05-04 07:59
