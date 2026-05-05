# Why OOD Penalty Failed: Deep Analysis

**Date**: 2026-05-03  
**Experiment**: test45_ergo_ood_penalty  
**Result**: ERGO AUROC 0.4031 (vs test41: 0.6243, vs baseline: 0.4538)

---

## Executive Summary

OOD penalty **failed catastrophically** for 3 fundamental reasons:

1. **Training-Evaluation Mismatch**: Trained on 4 peptides, evaluated on 12 different ones (ZERO overlap)
2. **MC Dropout Uncertainty ≠ OOD Detection**: High uncertainty does NOT predict low specificity
3. **Penalty Interfered with Learning**: OOD trigger rate increased (13% → 36.5%), indicating agent learned to exploit high-uncertainty space rather than avoid it

---

## Critical Finding 1: Training-Evaluation Mismatch

### What Happened

**Training peptides** (4 ERGO positive-aligned):
- KLWASPLHV
- FPRPWLHGL
- KAFSPEVIPMF
- HSKKKCDEL

**Evaluation peptides** (12 McPAS standard):
- GILGFVFTL, NLVPMVATV, GLCTLVAML, LLWNGPMAV, YLQPRTFLL, FLYALALLL, SLYNTVATL, KLGGALQAK, AVFDRKSDAK, IVTDFSVIK, SPRWYFYYL, RLRAEAQVK

**Overlap**: **ZERO**

### Why This Matters

The agent was trained to optimize ERGO scores on 4 specific peptides, then evaluated on 12 completely different peptides. This is like training a model on cats and testing on dogs — poor generalization is expected.

**Evidence**:
- Only 2/12 eval targets achieved AUROC > 0.60 (IVTDFSVIK: 0.8744, AVFDRKSDAK: 0.6562)
- 10/12 targets had AUROC < 0.55
- Mean AUROC (0.4031) is worse than v1 baseline (0.4538) which trained on all 163 peptides

**Conclusion**: The poor performance is partly due to **overfitting to 4 training peptides**, not just OOD penalty failure.

---

## Critical Finding 2: MC Dropout Uncertainty Does NOT Predict Specificity

### Hypothesis Being Tested

"When ERGO's MC Dropout uncertainty is high, the TCR is outside ERGO's training distribution, and ERGO's score is unreliable. Penalizing high-uncertainty samples will force the agent to design TCRs within ERGO's reliable domain, improving specificity."

### What Actually Happened

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| OOD trigger rate over time | Decrease (agent learns to stay in-domain) | **Increased** (13% → 36.5%) | ❌ Opposite |
| AUROC | Improve (better specificity) | **Decreased** (0.6243 → 0.4031) | ❌ Worse |
| Target ERGO scores | Moderate (in-domain) | **Low** (mean 0.14) | ❌ Suppressed |

### Evidence from Evaluation Results

**High ERGO score targets** (target_score > 0.3):
- IVTDFSVIK: target=0.5990, decoy=0.2180, **AUROC=0.8744** ✅
- AVFDRKSDAK: target=0.2883, decoy=0.2208, **AUROC=0.6562** ✅

**Low ERGO score targets** (target_score < 0.1):
- GILGFVFTL: target=0.0214, decoy=0.1660, **AUROC=0.1754** ❌
- GLCTLVAML: target=0.0172, decoy=0.1846, **AUROC=0.1846** ❌
- RLRAEAQVK: target=0.0220, decoy=0.0971, **AUROC=0.4781** ❌
- FLYALALLL: target=0.0074, decoy=0.0346, **AUROC=0.2754** ❌
- SPRWYFYYL: target=0.0251, decoy=0.1457, **AUROC=0.3456** ❌
- SLYNTVATL: target=0.0717, decoy=0.1377, **AUROC=0.2558** ❌

**Key Finding**: **High ERGO score correlates with HIGH specificity, not low specificity!**

The OOD penalty suppressed ERGO scores (mean target score 0.14 vs test41's ~0.3), which **reduced specificity** instead of improving it.

### Why MC Dropout Uncertainty Failed as OOD Indicator

**Theoretical assumption**: High MC Dropout std → model is uncertain → input is OOD → prediction is unreliable

**Reality**: ERGO's MC Dropout uncertainty measures:
1. **Epistemic uncertainty** (model doesn't know) — this is what we want
2. **Aleatoric uncertainty** (inherent noise in data) — this is NOT OOD
3. **Sequence complexity** (long/unusual CDR3s) — this is NOT necessarily OOD

**Problem**: MC Dropout cannot distinguish between:
- "This TCR is outside my training distribution" (true OOD)
- "This TCR has ambiguous binding properties" (high aleatoric uncertainty)
- "This TCR has unusual sequence features" (complexity, not OOD)

**Evidence from training dynamics**:
- OOD trigger rate **increased** from 13% to 36.5%
- This means the agent learned to generate TCRs with **high uncertainty**
- But high uncertainty ≠ low specificity (see IVTDFSVIK: high score, high specificity)

---

## Critical Finding 3: Penalty Interfered with Learning

### Training Dynamics

| Step | Reward | OOD Trigger Rate | Interpretation |
|------|--------|------------------|----------------|
| 10K | 1.22 | 13.2% | Early exploration, low uncertainty |
| 100K | 2.50 | 23.5% | Agent discovers high-reward space |
| 500K | 3.45 | 33.8% | Agent exploits high-uncertainty space |
| 1M | 3.85 | 35.5% | Converging to high-uncertainty policy |
| 2M | 4.25 | 36.5% | Stable at high uncertainty |

**Expected behavior**: OOD rate should **decrease** as agent learns to stay in-domain.

**Actual behavior**: OOD rate **increased** steadily, stabilizing at 36.5%.

### Why This Happened

**Hypothesis 1: Penalty was too weak**
- Soft penalty: `(uncertainty - 0.15) * 1.0` when uncertainty > 0.15
- If uncertainty = 0.20, penalty = 0.05
- If ERGO score = 0.30, final reward = 0.25
- The penalty was small enough that agent could still get positive reward by maximizing ERGO score despite penalty

**Hypothesis 2: High uncertainty space has higher ERGO scores**
- Agent discovered that TCRs with high uncertainty also tend to have high ERGO scores
- This could be because:
  - Novel/unusual sequences confuse ERGO (high uncertainty)
  - But ERGO's mean prediction is still high (high score)
  - Penalty reduces reward but doesn't eliminate it
- Agent rationally chose high-uncertainty, high-score space over low-uncertainty, low-score space

**Hypothesis 3: Threshold was set incorrectly**
- Threshold = 0.15 was based on "typical ERGO MC Dropout std distribution"
- But we never validated that 0.15 separates in-domain from OOD
- 36.5% trigger rate suggests threshold was too low (too many false positives)

### Evidence: Reward Increased Despite Penalty

- Reward increased from 1.22 to 4.25 (3.5x improvement)
- This means agent found a way to maximize reward despite OOD penalty
- If penalty was effective, reward should plateau or decrease when OOD rate is high

---

## Comparison with test14 (Pure ERGO, No OOD Penalty)

| Metric | test14 (pure ERGO) | test45 (ERGO + OOD) | Delta |
|--------|-------------------|---------------------|-------|
| Mean AUROC | 0.6091 | 0.4031 | **-0.2060** |
| Targets > 0.65 | 4/12 | 2/12 | **-2** |
| Training peptides | 12 McPAS | 4 positive-aligned | Different |
| Training steps | 2M | 2M | Same |
| Seed | 42 | 42 | Same |

**Conclusion**: Adding OOD penalty made results **significantly worse** than pure ERGO.

---

## Why OOD Penalty is Fundamentally Flawed

### Problem 1: MC Dropout Uncertainty is Not a Reliable OOD Signal

**What we need**: A signal that detects when ERGO's prediction is unreliable due to OOD input.

**What MC Dropout provides**: A measure of model uncertainty that conflates:
- Epistemic uncertainty (lack of training data)
- Aleatoric uncertainty (inherent noise)
- Sequence complexity (unusual but valid sequences)

**Evidence**: High uncertainty does NOT correlate with low specificity. IVTDFSVIK had high ERGO score (0.5990) and excellent specificity (AUROC 0.8744).

### Problem 2: Penalty-Based Constraints are Fragile

**Challenges**:
1. **Threshold selection**: How do we know 0.15 is the right cutoff?
2. **Weight tuning**: How strong should the penalty be?
3. **Mode selection**: Soft vs hard penalty?
4. **Interaction with other rewards**: Penalty may interfere with affinity optimization

**Evidence**: We set threshold=0.15, weight=1.0, mode=soft based on intuition, not validation. The result was catastrophic (AUROC 0.4031).

### Problem 3: Agent Can Exploit High-Uncertainty Space

**Assumption**: Agent will avoid high-uncertainty space to maximize reward.

**Reality**: Agent found that high-uncertainty space also has high ERGO scores, so it rationally chose to stay there despite penalty.

**Evidence**: OOD trigger rate increased from 13% to 36.5%, while reward increased from 1.22 to 4.25.

---

## Is OOD Penalty Salvageable?

### Option 1: Better OOD Detection

**Idea**: Use a more reliable OOD signal than MC Dropout uncertainty.

**Candidates**:
1. **Mahalanobis distance** in ERGO's latent space
2. **Ensemble disagreement** (train multiple ERGO models, measure variance)
3. **Reconstruction error** from ERGO's autoencoder
4. **tFold validation** (if tFold disagrees with ERGO, likely OOD)

**Feasibility**: Requires significant engineering effort. Mahalanobis distance needs computing covariance matrix over ERGO's training set. Ensemble needs training multiple models. tFold validation is slow (~1s per sample).

**Recommendation**: **tFold cascade** (ERGO + tFold validation) is most promising, already implemented in test44.

### Option 2: Stronger Penalty

**Idea**: Increase penalty weight from 1.0 to 5.0 or 10.0, or use hard penalty mode.

**Risk**: Too strong penalty may prevent agent from exploring at all, leading to premature convergence.

**Evidence**: Soft penalty with weight=1.0 already suppressed ERGO scores (mean 0.14 vs test41's 0.3). Stronger penalty would likely make this worse.

**Recommendation**: **Not recommended**. The problem is not penalty strength, but the OOD signal itself.

### Option 3: Adaptive Threshold

**Idea**: Start with high threshold (e.g., 0.30) and gradually decrease it as agent learns.

**Rationale**: Early in training, agent explores widely (high uncertainty is normal). Later, agent should converge to in-domain space (high uncertainty is suspicious).

**Feasibility**: Easy to implement (linear decay schedule).

**Risk**: Still relies on MC Dropout uncertainty being a valid OOD signal, which we've shown it's not.

**Recommendation**: **Not recommended**. Fixes the symptom (wrong threshold), not the disease (wrong OOD signal).

### Option 4: Expand Training Peptides

**Idea**: Train on 29 tFold-good peptides instead of 4 ERGO-positive peptides.

**Rationale**: More diverse training data → better generalization → less overfitting.

**Evidence**: test45 trained on 4 peptides, evaluated on 12 different ones (zero overlap). This is a recipe for poor generalization.

**Feasibility**: Already implemented in test44 (29 peptides).

**Recommendation**: **Strongly recommended**. This addresses the training-evaluation mismatch, which is a major contributor to test45's failure.

---

## Final Verdict: Is OOD Penalty Viable?

### Short Answer: **NO, not with MC Dropout uncertainty**

### Long Answer:

**The core idea is sound**: Constraining the agent to the scorer's reliable domain should improve robustness.

**The implementation is flawed**: MC Dropout uncertainty is not a reliable OOD indicator for ERGO.

**Evidence**:
1. OOD trigger rate increased (agent did NOT learn to avoid OOD)
2. High uncertainty does NOT correlate with low specificity
3. Penalty suppressed ERGO scores, reducing specificity instead of improving it
4. Result (AUROC 0.4031) is worse than pure ERGO (0.6091) and v1 baseline (0.4538)

**Alternative approaches**:
1. ✅ **tFold cascade** (test44): Use tFold to validate ERGO predictions
2. ✅ **Expand training peptides** (test44): Train on 29 peptides instead of 4
3. ✅ **Two-phase training** (test41): ERGO warm-start → contrastive fine-tuning
4. ❌ **Stronger OOD penalty**: Won't fix the fundamental problem
5. ❌ **Adaptive threshold**: Still relies on flawed OOD signal

---

## Lessons Learned

1. **MC Dropout uncertainty ≠ OOD detection**: Uncertainty measures model confidence, not input validity
2. **Penalty-based constraints are fragile**: Easy to set wrong hyperparameters and harm learning
3. **Training-evaluation mismatch is deadly**: Always train and evaluate on the same peptide set
4. **High ERGO score → high specificity**: Suppressing ERGO scores reduces specificity
5. **Agent will exploit loopholes**: If high-uncertainty space has high reward, agent will go there

---

## Recommendations

1. **Abandon OOD penalty approach** with MC Dropout uncertainty
2. **Focus on test44** (pure tFold) and test41 (two-phase ERGO)
3. **If OOD detection is needed**, use tFold cascade (ERGO + tFold validation)
4. **Always train on diverse peptide sets** (29 peptides minimum)
5. **Validate OOD signals** before using them in RL (correlation with specificity)

---

**Document Author**: Claude Code  
**Last Updated**: 2026-05-03
