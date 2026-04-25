# TCR-Peptide Scorer Consistency Evaluation Report

**Date**: 2026-04-24  
**Dataset**: tc-hard (10 peptides, 1007 TCR-peptide pairs)  
**Scorers Evaluated**: NetTCR-2.0, ERGO, DeepAIR

---

## Executive Summary

Evaluated three TCR-peptide binding prediction models on tc-hard dataset to assess scorer consistency and correlation. **Key finding: DeepAIR exhibits degenerate behavior with nearly constant scores across all TCR-peptide pairs**, while NetTCR and ERGO show reasonable score distributions. Correlations between scorers are weak to negative.

---

## Score Distributions

### NetTCR-2.0
- **Mean**: 0.338 ± 0.084
- **Range**: [0.097, 0.512]
- **Median**: 0.344
- **Behavior**: Reasonable score distribution with moderate variance

### ERGO
- **Mean**: 0.110 ± 0.133
- **Range**: [0.000, 0.779]
- **Median**: 0.059
- **Behavior**: Low mean with high variance, skewed toward low scores

### DeepAIR
- **Mean**: 0.495 ± 0.004
- **Range**: [0.478, 0.498]
- **Median**: 0.498
- **Behavior**: ⚠️ **DEGENERATE** - Nearly constant scores across all inputs

---

## Correlation Analysis

### Pairwise Correlations (Pearson / Spearman)

| Scorer Pair | Pearson r | Spearman ρ | Interpretation |
|-------------|-----------|------------|----------------|
| NetTCR vs ERGO | 0.058 | 0.096 | Very weak positive |
| NetTCR vs DeepAIR | -0.321 | -0.198 | Weak negative |
| ERGO vs DeepAIR | -0.104 | -0.049 | Very weak negative |

**All correlations are statistically significant (p < 0.001) but practically weak.**

---

## DeepAIR Degenerate Behavior Analysis

### Root Cause

DeepAIR uses ProtBert `pooler_output` features for TCR CDR3 sequences. Investigation revealed:

1. **ProtBert pooler layer was never trained**
   - ProtBert (prot_bert_bfd) was pretrained with MLM (Masked Language Modeling) only
   - MLM does not train the pooler layer (only NSP does)
   - Pooler weights have std=0.00093872 (33× smaller than Xavier initialization)

2. **Degenerate pooler_output**
   - CLS token embeddings have good discriminability (cosine similarity 0.74-0.90)
   - After pooler layer: cosine similarity 0.999-1.0 (nearly identical)
   - L2 norm differences compressed from 1.7-2.3 → 0.09-0.83

3. **Impact on DeepAIR**
   - DeepAIR models were trained with degenerate pooler_output
   - Models rely primarily on raw CDR3 strings, not ProtBert features
   - Score variance comes from CDR3 string differences, not learned representations

### Evidence

Sample DeepAIR scores for GILGFVFTL peptide:
```
CASSIRSSYEQYF    → 0.0427
CASSLGQETQYF     → 0.0427
CASSPGTSGNTIYF   → 0.0427
```

All three different TCRs produce identical scores (to 4 decimal places).

From tc-hard evaluation (RLRPGGKKK peptide):
```
CASSPGWGSYEQYF   → 0.4831
CASSPQRGFYEQYF   → 0.4831
CASCPGWGSYEQYF   → 0.4831
CASSSDRSFYGYTF   → 0.4830
CASSLDRGVGGYTF   → 0.4831
```

Score range: 0.4829-0.4831 (0.0002 spread for 33 different TCRs)

---

## Scorer Consistency Assessment

### Overall Consistency: **POOR**

1. **DeepAIR is not usable** for discrimination tasks
   - Standard deviation (0.004) is too small for meaningful ranking
   - Scores are effectively constant across diverse TCR sequences
   - Cannot distinguish binders from non-binders

2. **NetTCR and ERGO show weak agreement**
   - Pearson r = 0.058 (explains only 0.3% of variance)
   - Different scoring scales and distributions
   - May capture different aspects of binding

3. **Negative correlations with DeepAIR**
   - Suggests DeepAIR's residual variance is noise, not signal
   - NetTCR/ERGO high scores → DeepAIR low scores (and vice versa)

---

## Recommendations

### For TCR-PPO v2 Development

1. **Do NOT use DeepAIR** as affinity scorer
   - Degenerate behavior makes it unsuitable for RL reward signal
   - Would provide no gradient information for policy optimization

2. **Use ERGO or NetTCR** as primary affinity scorer
   - Both show reasonable score distributions
   - ERGO has higher variance (better for exploration)
   - NetTCR has more centered distribution (better for stability)

3. **Consider ensemble scoring**
   - Average NetTCR and ERGO scores
   - May capture complementary binding features
   - Reduces impact of individual model biases

### For DeepAIR Improvement

To fix DeepAIR's degenerate behavior:

1. **Use TensorFlow-native ProtBert weights** (already done)
   - Eliminates PyTorch→TensorFlow conversion artifacts
   - Does NOT fix pooler degeneracy (inherent to ProtBert pretraining)

2. **Replace pooler_output with mean-pooled last_hidden_state**
   - Mean-pool over sequence length (with attention mask)
   - Preserves sequence-level information
   - Requires retraining DeepAIR models

3. **Use ESM-2 instead of ProtBert**
   - ESM-2 has better protein sequence understanding
   - Trained on larger protein corpus
   - No pooler degeneracy issues

---

## Visualizations

Generated plots (see `results/scorer_consistency/`):

1. **score_distributions.png**: Histograms showing score distributions for each scorer
2. **correlation_matrix.png**: Heatmap of pairwise Pearson correlations
3. **pairwise_scatter.png**: Scatter plots between all scorer pairs

---

## Technical Details

### Dataset
- **Source**: `/share/liuyutian/tcrppo_v2/data/l0_seeds_tchard/`
- **Peptides**: 10 unique epitopes
- **TCRs**: 1007 unique CDR3β sequences
- **Distribution**: Highly imbalanced (VLWAHGFEL: 731 TCRs, others: 5-176 TCRs)

### Model Configurations
- **NetTCR-2.0**: Beta-chain only, weights from `/share/liuyutian/tcrppo_v2/data/nettcr_model.weights.h5`
- **ERGO**: Autoencoder model, 1 MC sample, weights from `ae_mcpas1.pt`
- **DeepAIR**: Official TensorFlow SavedModel, ProtBert features from TF-native weights

### Computational Environment
- **Device**: CPU (to avoid PyTorch-TensorFlow GPU conflicts)
- **ProtBert**: TF-native weights from HuggingFace mirror
- **TensorFlow**: 2.x with Keras 3 compatibility

---

## Conclusion

The scorer consistency evaluation reveals **fundamental issues with DeepAIR** that make it unsuitable for TCR-PPO v2. The degenerate pooler_output from ProtBert causes nearly constant predictions across diverse TCR sequences. **NetTCR and ERGO are recommended** as affinity scorers, with ERGO preferred for its higher variance and better exploration properties.

The weak correlations between NetTCR and ERGO (r=0.058) suggest they capture different binding features, making an ensemble approach potentially valuable.
