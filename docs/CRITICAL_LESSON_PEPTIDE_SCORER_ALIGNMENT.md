# CRITICAL LESSON: Peptide-Scorer Alignment is Mandatory

**Date:** 2026-04-29  
**Discovered by:** SAC test3 evaluation  
**Impact:** CRITICAL — affects all future RL training

---

## Executive Summary

**Training on peptides where the scorer is unreliable produces reversed reward signals that teach the agent to bind decoys instead of targets.**

This is not a minor optimization — it is a **fundamental requirement** for successful RL training. Violating this principle wastes GPU time and produces models with worse-than-random performance.

---

## The Discovery: SAC test3

### Experiment Setup
- **Model:** SAC with ERGO scorer
- **Targets:** 7 ERGO-trainable peptides (AUC ≥ 0.7 on VDJdb/IEDB data)
- **Training:** 1M steps, ESM-2 encoder, reward_mode=v1_ergo_only
- **Hypothesis:** ERGO's high mapping AUC should produce good RL performance

### Results @ 1M Steps

| Peptide | ERGO Mapping AUC | RL AUROC | Gap (Target-Decoy) | Outcome |
|---------|------------------|----------|-------------------|---------|
| KLWASPLHV | 0.823 | 0.5656 | +0.027 | ✅ Correct learning |
| FPRPWLHGL | 0.762 | 0.6124 | +0.050 | ✅ Correct learning |
| KAFSPEVIPMF | 0.739 | 0.5912 | +0.046 | ✅ Correct learning |
| HSKKKCDEL | 0.763 | 0.6040 | +0.081 | ✅ Correct learning |
| RFYKTLRAEQASQ | 0.908 | 0.1244 | **-0.019** | ❌ **Reversed learning** |
| DRFYKTLRAEQASQEV | 0.786 | 0.2952 | **-0.010** | ❌ **Reversed learning** |
| FRCPRRFCF | 0.714 | 0.0960 | **-0.060** | ❌ **Reversed learning** |

**Mean AUROC: 0.4127** (worse than random 0.5)

### Key Insight

**Mapping AUC ≠ RL AUROC**

- **Mapping AUC**: Measures scorer's discrimination on **existing TCR-peptide pairs** from VDJdb/IEDB
- **RL AUROC**: Measures scorer's discrimination on **RL-generated novel TCRs**

**Problem**: RL generates TCRs outside the scorer's training distribution. When the scorer encounters these novel TCRs, its predictions become unreliable or even reversed.

---

## Why Reversed Reward Signals are Catastrophic

### Normal (Aligned) Peptide
```
Target TCR:  ERGO score = 0.20
Decoy TCRs:  ERGO score = 0.15 (mean)
Gap: +0.05 (positive)

→ Agent learns: "Bind target, avoid decoys" ✅
→ AUROC: 0.61 (good discrimination)
```

### Reversed Peptide
```
Target TCR:  ERGO score = 0.02
Decoy TCRs:  ERGO score = 0.04 (mean)
Gap: -0.02 (negative)

→ Agent learns: "Avoid target, bind decoys" ❌
→ AUROC: 0.12 (reversed discrimination)
```

**The agent is doing exactly what the reward signal tells it to do — but the reward signal is wrong.**

---

## Impact on Overall Performance

When training on mixed aligned/reversed peptides:

- **4 aligned peptides**: Pull mean AUROC up to 0.5933
- **3 reversed peptides**: Drag mean AUROC down to 0.1719
- **Net result**: 0.4127 (worse than random)

**Conclusion**: You cannot "average out" reversed peptides. They actively harm the model.

---

## The Solution: Mandatory Peptide Filtering

### Step 1: Consult PEPTIDE_SCORER_MAPPING.md

Before ANY experiment, read `PEPTIDE_SCORER_MAPPING.md` (project root) to find:
- Which peptides have AUC ≥ 0.7 for your chosen scorer
- Which scorer is optimal for each peptide
- Known failure cases and reversed peptides

### Step 2: Verify Reward-AUROC Alignment

For ERGO peptides, use only these 4 with verified positive alignment:
- KLWASPLHV (gap +0.027)
- FPRPWLHGL (gap +0.050)
- KAFSPEVIPMF (gap +0.046)
- HSKKKCDEL (gap +0.081)

**DO NOT USE** these 3 reversed peptides:
- RFYKTLRAEQASQ (gap -0.019)
- DRFYKTLRAEQASQEV (gap -0.010)
- FRCPRRFCF (gap -0.060)

### Step 3: Document Peptide Selection

In your experiment design doc, include:
```markdown
## Target Peptide Selection

**Scorer:** ERGO
**Peptides:** KLWASPLHV, FPRPWLHGL, KAFSPEVIPMF, HSKKKCDEL (4 peptides)

**Rationale:** 
- All 4 have ERGO mapping AUC ≥ 0.7
- All 4 have verified positive reward-AUROC alignment (gap > 0)
- Excluded 3 reversed peptides (RFYKTLRAEQASQ, DRFYKTLRAEQASQEV, FRCPRRFCF)

**Expected AUROC:** ~0.59 (based on test3 @ 1M on same 4 peptides)
```

---

## Hybrid Training: Correcting Scorer Errors

Even aligned peptides can have scorer errors on novel TCRs. **Hybrid tFold-ERGO training** addresses this:

- **90% episodes**: ERGO reward (fast, ~10ms)
- **10% episodes**: tFold reward (accurate, structure-based)
- **Mechanism**: tFold corrects ERGO's errors on novel TCRs

**Expected improvement**: 0.59 → >0.63 AUROC

**See:** `docs/TFOLD_HYBRID_TRAINING.md`

---

## Mandatory Checklist for All Future Experiments

Before launching ANY RL training:

- [ ] Read `PEPTIDE_SCORER_MAPPING.md`
- [ ] Filter peptides to AUC ≥ 0.7 for chosen scorer
- [ ] Verify no reversed peptides (check gap > 0)
- [ ] Document peptide selection rationale
- [ ] Update `CLAUDE.md` if new reversed peptides discovered

**Failure to follow this checklist will result in wasted GPU time and failed experiments.**

---

## References

- **Test3 full analysis:** `results/sac_test3_1m_ergo_7targets/ANALYSIS.md`
- **Peptide-scorer mapping:** `PEPTIDE_SCORER_MAPPING.md` (project root)
- **Hybrid training guide:** `docs/TFOLD_HYBRID_TRAINING.md`
- **CLAUDE.md section 6:** Scorer Selection and Hybrid Training (CRITICAL)
- **Experiment tracker:** `docs/all_experiments_tracker.md`

---

**This lesson cost us 1M training steps and 12 hours of GPU time. Do not repeat this mistake.**
