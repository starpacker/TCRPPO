# OOD Penalty Redesign: Confidence-Weighted Reward

**Date**: 2026-05-03  
**Motivation**: test45 OOD penalty failed because it used additive penalty. Redesign to use multiplicative confidence weighting.

---

## Core Idea: Multiplicative Confidence Weighting

**Old approach (test45)**: `reward = score - penalty` where `penalty = (uncertainty - threshold) * weight`

**Problem**: 
- Agent can still get positive reward with high score + high uncertainty
- Penalty is additive, doesn't fundamentally change optimization objective
- Threshold and weight are arbitrary hyperparameters

**New approach**: `reward = score * confidence` where `confidence = 1 - uncertainty`

**Advantages**:
- **No hyperparameters**: No threshold, no weight, no mode selection
- **Natural optimization**: Agent must maximize BOTH score AND confidence
- **Interpretable**: Confidence acts as a quality gate on the score
- **Smooth gradient**: Differentiable everywhere, no hard cutoffs

---

## Mathematical Formulation

### ERGO MC Dropout

```python
# Run ERGO 10 times with dropout enabled
scores = [ergo.forward(tcr, peptide, dropout=True) for _ in range(10)]
mean_score = np.mean(scores)
std_score = np.std(scores)
confidence = 1.0 - std_score  # High std → low confidence
```

### Reward Computation

```python
# Old (test45): Additive penalty
if std_score > threshold:
    penalty = (std_score - threshold) * weight
    reward = mean_score - penalty
else:
    reward = mean_score

# New: Multiplicative confidence
confidence = 1.0 - std_score
reward = mean_score * confidence
```

### Example Scenarios

| Scenario | Score | Std | Confidence | Old Reward (t=0.15, w=1.0) | New Reward | Interpretation |
|----------|-------|-----|------------|---------------------------|------------|----------------|
| High score, high confidence | 0.8 | 0.1 | 0.9 | 0.8 (no penalty) | **0.72** | ✅ Desired |
| High score, low confidence | 0.8 | 0.7 | 0.3 | 0.25 (penalty=0.55) | **0.24** | ✅ Penalized |
| Low score, high confidence | 0.2 | 0.1 | 0.9 | 0.2 (no penalty) | **0.18** | ✅ Correct |
| Low score, low confidence | 0.2 | 0.7 | 0.3 | -0.35 (penalty=0.55) | **0.06** | ✅ Correct |

**Key difference**: Old approach can give negative reward (confusing signal). New approach always gives non-negative reward, with confidence acting as a multiplier.

---

## Why This Should Work Better Than test45

### Problem 1 (test45): Agent exploited high-uncertainty space

**Root cause**: Additive penalty was too weak. If `score=0.8, std=0.3`, penalty=0.15, final reward=0.65 (still high).

**Solution**: Multiplicative weighting. If `score=0.8, confidence=0.7`, final reward=0.56. If `score=0.8, confidence=0.3`, final reward=0.24 (much lower).

**Effect**: Agent cannot exploit high-uncertainty space because reward is **proportional** to confidence, not just reduced by a fixed penalty.

### Problem 2 (test45): OOD trigger rate increased

**Root cause**: Agent learned that high-uncertainty TCRs have high ERGO scores, and penalty was insufficient to discourage this.

**Solution**: No threshold, no "trigger". Every sample is weighted by confidence. Agent naturally learns to maximize confidence because it directly multiplies reward.

**Effect**: Agent should converge to high-confidence space because that's where the highest rewards are.

### Problem 3 (test45): Hyperparameter sensitivity

**Root cause**: Threshold (0.15) and weight (1.0) were arbitrary. Wrong values → catastrophic failure.

**Solution**: Zero hyperparameters. Confidence is simply `1 - std`, reward is simply `score * confidence`.

**Effect**: Robust to different peptides and training conditions.

---

## Implementation Plan

### Step 1: Modify RewardManager

```python
class RewardManager:
    def __init__(self, ..., use_confidence_weighting=False):
        self.use_confidence_weighting = use_confidence_weighting
    
    def compute_reward(self, tcr, peptide, ...):
        if self.use_confidence_weighting:
            # Use MC Dropout to get mean and std
            scores, confidences = self.affinity_scorer.score_batch([tcr], [peptide])
            aff_score = scores[0]
            confidence = confidences[0]  # Already 1 - std
            
            # Multiplicative weighting
            aff_score = aff_score * confidence
            
            components["affinity_raw"] = scores[0]
            components["confidence"] = confidence
            components["affinity_weighted"] = aff_score
        else:
            # Standard scoring (single forward pass)
            aff_score = self.affinity_scorer.score(tcr, peptide)
        
        # Rest of reward computation (decoy, naturalness, diversity)
        ...
```

### Step 2: Add CLI argument

```python
parser.add_argument("--use_confidence_weighting", action="store_true",
                    help="Weight affinity score by ERGO confidence (1 - MC Dropout std)")
```

### Step 3: Create new reward mode

```python
# Option A: New reward mode
reward_mode = "v1_ergo_confidence_weighted"

# Option B: Flag on existing mode
reward_mode = "v1_ergo_only"
use_confidence_weighting = True
```

### Step 4: Launch experiment

```bash
CUDA_VISIBLE_DEVICES=5 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test46_ergo_confidence_weighted \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --use_confidence_weighting \
    --affinity_scorer ergo \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/mcpas_12_targets.txt  # Use all 12 McPAS targets
```

---

## Expected Outcomes

### Success Criteria

1. **Mean confidence increases over training** (agent learns to stay in high-confidence space)
2. **Mean AUROC ≥ 0.60** (better than test45's 0.4031, comparable to test14's 0.6091)
3. **High ERGO score correlates with high confidence** (agent doesn't exploit low-confidence space)
4. **No hyperparameter tuning needed** (works out of the box)

### Monitoring Metrics

Log every 10K steps:
- `mean_confidence`: Average confidence across rollout
- `mean_affinity_raw`: Average raw ERGO score (before confidence weighting)
- `mean_affinity_weighted`: Average weighted score (after confidence weighting)
- `confidence_affinity_corr`: Correlation between confidence and raw score

**Expected trends**:
- `mean_confidence`: Should increase from ~0.5 to ~0.8
- `mean_affinity_raw`: Should increase (agent learns to bind)
- `mean_affinity_weighted`: Should increase faster than raw (confidence improves)
- `confidence_affinity_corr`: Should be positive (high score → high confidence)

### Comparison with test45

| Metric | test45 (additive penalty) | test46 (multiplicative confidence) | Expected |
|--------|--------------------------|-----------------------------------|----------|
| Mean AUROC | 0.4031 | ? | ≥ 0.60 |
| OOD trigger rate | 36.5% (increased) | N/A (no threshold) | N/A |
| Mean confidence | N/A | ? | ≥ 0.75 |
| Training peptides | 4 | 12 | Better generalization |

---

## Risk Analysis

### Risk 1: Confidence is too low initially

**Symptom**: Early in training, all confidences are ~0.5, so all rewards are halved. Agent struggles to learn.

**Mitigation**: 
- Use curriculum: Start with `reward = score` for first 100K steps, then switch to `reward = score * confidence`
- Or use soft transition: `reward = score * (0.5 + 0.5 * confidence)` initially, gradually increase to `reward = score * confidence`

### Risk 2: MC Dropout std is not calibrated

**Symptom**: ERGO's MC Dropout std is always very small (e.g., 0.05) or very large (e.g., 0.8), making confidence uninformative.

**Mitigation**: 
- Normalize std by its empirical distribution: `confidence = 1 - (std - mean_std) / std_std`
- Or use percentile-based confidence: `confidence = percentile_rank(std)`

**Check before training**: Run ERGO MC Dropout on 1000 random TCRs, plot std distribution. If std is always in [0.0, 0.1], confidence will always be [0.9, 1.0] (not useful).

### Risk 3: Confidence weighting is too aggressive

**Symptom**: Agent converges to very conservative TCRs with high confidence but low affinity.

**Mitigation**: 
- Use softer weighting: `reward = score * confidence^alpha` where `alpha < 1` (e.g., 0.5)
- Or use minimum confidence: `reward = score * max(confidence, 0.5)`

### Risk 4: Training on 12 peptides is too slow

**Symptom**: With n_envs=8, 12 peptides, training takes too long.

**Mitigation**: 
- Increase n_envs to 16 or 24
- Or use peptide sampling: Each rollout randomly selects 4 peptides from the 12

---

## Alternative Designs

### Design 1: Exponential confidence weighting

```python
reward = score * (confidence ** alpha)
```

**Pros**: `alpha > 1` makes confidence more important, `alpha < 1` makes it less important.

**Cons**: Adds a hyperparameter (alpha).

**Recommendation**: Start with `alpha=1` (simple multiplicative), tune if needed.

### Design 2: Soft threshold with sigmoid

```python
gate = sigmoid((confidence - threshold) * sharpness)
reward = score * gate
```

**Pros**: Smooth transition around threshold, avoids hard cutoff.

**Cons**: Adds two hyperparameters (threshold, sharpness).

**Recommendation**: Only use if simple multiplicative fails.

### Design 3: Confidence as a separate reward component

```python
reward = w_aff * score + w_conf * confidence
```

**Pros**: Can tune relative importance of score vs confidence.

**Cons**: Adds two hyperparameters, doesn't enforce "high score + high confidence" jointly.

**Recommendation**: Not recommended. Multiplicative is better for joint optimization.

### Design 4: Ensemble-based confidence

Instead of MC Dropout, train 3-5 ERGO models with different seeds, use ensemble disagreement as confidence:

```python
scores = [ergo_i.forward(tcr, peptide) for ergo_i in ensemble]
mean_score = np.mean(scores)
std_score = np.std(scores)
confidence = 1.0 - std_score
```

**Pros**: More reliable than MC Dropout (true epistemic uncertainty).

**Cons**: Requires training multiple ERGO models (expensive).

**Recommendation**: Future work if MC Dropout confidence is not reliable.

---

## Validation Before Full Training

### Step 1: Check ERGO MC Dropout std distribution

```python
# Sample 1000 random TCRs
tcrs = sample_random_tcrs(1000)
peptide = "GILGFVFTL"

# Compute MC Dropout std
stds = []
for tcr in tcrs:
    scores = [ergo.forward(tcr, peptide, dropout=True) for _ in range(10)]
    stds.append(np.std(scores))

# Plot distribution
plt.hist(stds, bins=50)
plt.xlabel("MC Dropout std")
plt.ylabel("Count")
plt.title("ERGO MC Dropout std distribution")
plt.savefig("ergo_mc_dropout_std_dist.png")

print(f"Mean std: {np.mean(stds):.4f}")
print(f"Median std: {np.median(stds):.4f}")
print(f"Std of std: {np.std(stds):.4f}")
print(f"Min std: {np.min(stds):.4f}")
print(f"Max std: {np.max(stds):.4f}")
```

**Expected**: std should be in [0.05, 0.3] with reasonable variance. If all stds are < 0.1, confidence will always be > 0.9 (not useful).

### Step 2: Check correlation between std and AUROC

```python
# For each of 12 McPAS targets
for peptide in mcpas_targets:
    # Generate 50 TCRs with test14 model (pure ERGO, known to work)
    tcrs = generate_tcrs(peptide, n=50)
    
    # Compute ERGO MC Dropout (mean, std)
    means, stds = [], []
    for tcr in tcrs:
        scores = [ergo.forward(tcr, peptide, dropout=True) for _ in range(10)]
        means.append(np.mean(scores))
        stds.append(np.std(scores))
    
    # Compute AUROC (specificity)
    aurocs = compute_auroc(tcrs, peptide)
    
    # Check correlation
    corr_mean_auroc = np.corrcoef(means, aurocs)[0, 1]
    corr_std_auroc = np.corrcoef(stds, aurocs)[0, 1]
    
    print(f"{peptide}: corr(mean, AUROC)={corr_mean_auroc:.3f}, corr(std, AUROC)={corr_std_auroc:.3f}")
```

**Expected**: 
- `corr(mean, AUROC)` should be positive (high score → high specificity)
- `corr(std, AUROC)` should be negative (high std → low specificity)

If `corr(std, AUROC)` is near zero or positive, MC Dropout std is not a useful signal.

---

## Summary

**Key innovation**: Replace additive penalty with multiplicative confidence weighting.

**Formula**: `reward = score * confidence` where `confidence = 1 - MC_Dropout_std`

**Advantages**:
- Zero hyperparameters
- Natural joint optimization of score and confidence
- Smooth gradients, no hard thresholds
- Interpretable and robust

**Next steps**:
1. Validate ERGO MC Dropout std distribution
2. Check correlation between std and AUROC
3. If validation passes, launch test46
4. Monitor mean_confidence and mean_affinity_weighted during training
5. Evaluate on 12 McPAS targets, compare with test14 (0.6091) and test45 (0.4031)

**Expected outcome**: Mean AUROC ≥ 0.60, mean confidence ≥ 0.75, no hyperparameter tuning needed.

---

**Document Author**: Claude Code  
**Last Updated**: 2026-05-03
