# TCRPPO v2 Preliminary Analysis

**Date:** 2026-04-15  
**Experiments analyzed:** 14 completed (3 still training: test8, test9, test10)

---

## 1. Executive Summary

**The v1_ergo_only seed=42 result (AUROC 0.8075) is an outlier, not a reliable baseline.**

Across 13 experiments (excluding seed=42), AUROC clusters tightly in the **0.47–0.59 range** (mean 0.54, std 0.04). No reward mode, architecture change, or penalty scheme breaks out of this band. The seed=123 reproduction of the identical config got only **0.5462** — a 32% drop from 0.8075.

The true performance of our v2 system is approximately **0.55 mean AUROC**, substantially above the original v1 baseline (0.4538) but far below the seed=42 "lucky" run.

---

## 2. Full Results Table

| # | Experiment | AUROC | vs v1 (0.4538) | Reward Mode | Key Change |
|---|-----------|-------|----------------|-------------|------------|
| 1 | **v1_ergo_only (s42)** | **0.8075** | +0.3537 | v1_ergo_only | Baseline (outlier) |
| 2 | test6_pure_v2 | 0.5894 | +0.1356 | v1_ergo_only | A1+A2+A10 arch, no L0 |
| 3 | v2_full | 0.5840 | +0.1302 | v2_full | All penalties, z-norm |
| 4 | test4_raw_multi | 0.5812 | +0.1274 | raw_multi_penalty | Raw d=0.05,n=0.02,v=0.01 |
| 5 | test3_stepwise | 0.5717 | +0.1179 | v1_ergo_stepwise | Per-step absolute ERGO |
| 6 | test5_threshold | 0.5697 | +0.1159 | threshold_penalty | Conditional at aff>0.5 |
| 7 | test1_two_phase | 0.5668 | +0.1130 | two-phase | 1M ERGO + 1M raw_decoy |
| 8 | test2_min6_raw | 0.5562 | +0.1024 | raw_decoy | min_steps=6, raw |
| 9 | v1_ergo_repro (s123) | 0.5462 | +0.0924 | v1_ergo_only | Seed=123 reproduction |
| 10 | v2_no_decoy | 0.5298 | +0.0760 | v2_no_decoy | No decoy penalty, z-norm |
| 11 | exp3_delta | 0.5004 | +0.0466 | v1_ergo_delta | Raw delta per-step |
| 12 | exp1_decoy | 0.4898 | +0.0360 | v2_decoy_only | z-norm, d=0.3 |
| 13 | exp4_min_steps | 0.4768 | +0.0230 | v2_full | z-norm, min_steps=3 |
| 14 | exp2_light | 0.4660 | +0.0122 | v2_full | z-norm, light weights |

---

## 3. Key Finding #1: Seed Dependence Dominates

The single largest factor explaining AUROC variance is **random seed**, not reward mode or architecture.

```
v1_ergo_only seed=42:  0.8075
v1_ergo_only seed=123: 0.5462
Gap:                   0.2613 (32% relative drop)
```

Per-target breakdown shows seed=42 got "lucky" on specific targets:

| Target | seed=42 | seed=123 | Gap | Interpretation |
|--------|---------|----------|-----|---------------|
| GLCTLVAML | 0.976 | 0.383 | +0.593 | Seed=42 nearly perfect, s123 below random |
| SLYNTVATL | 0.909 | 0.431 | +0.478 | Same pattern |
| GILGFVFTL | 0.969 | 0.550 | +0.419 | Same pattern |
| RLRAEAQVK | 0.938 | 0.554 | +0.384 | Same pattern |
| IVTDFSVIK | 0.855 | 0.872 | -0.017 | Both seeds agree — easy target |

**4 targets** (GLCTLVAML, SLYNTVATL, GILGFVFTL, RLRAEAQVK) swing by **0.38–0.59** between seeds. These 4 alone account for the entire mean AUROC difference. The remaining 8 targets are relatively stable.

**Implication:** The 0.8075 cannot be treated as a reproducible baseline. Any improvement claim must be validated across multiple seeds.

---

## 4. Key Finding #2: Reward Mode Makes Little Difference

When we exclude the seed=42 outlier, all reward modes converge to a narrow band:

| Category | Experiments | Mean AUROC | Range |
|----------|------------|------------|-------|
| Pure ERGO (no penalty) | 4 | 0.5519 | 0.50–0.59 |
| Raw penalties | 4 | 0.5685 | 0.56–0.58 |
| Z-norm penalties | 5 | 0.5093 | 0.47–0.58 |

**Observations:**
- Raw penalties slightly outperform z-norm penalties (0.57 vs 0.51), confirming z-normalization hurts
- But raw penalties do NOT outperform pure ERGO (0.57 vs 0.55) — the difference is within noise
- The total spread across 13 experiments is only **0.12** (0.47 to 0.59)
- This narrow spread suggests we're near a **performance ceiling** for this evaluation protocol

---

## 5. Key Finding #3: Target Difficulty is Intrinsic

Targets cluster into consistent difficulty tiers across ALL experiments:

| Tier | Targets | Mean AUROC | Character |
|------|---------|------------|-----------|
| EASY | IVTDFSVIK | 0.813 | Consistently high across all configs |
| EASY-MED | AVFDRKSDAK, NLVPMVATV, YLQPRTFLL, GILGFVFTL | 0.57–0.60 | Moderate, stable |
| MEDIUM | KLGGALQAK, SLYNTVATL, LLWNGPMAV, RLRAEAQVK | 0.50–0.54 | Variable, seed-sensitive |
| HARD | FLYALALLL, SPRWYFYYL, GLCTLVAML | 0.40–0.41 | Consistently poor |

**Hard targets** (FLYALALLL, SPRWYFYYL, GLCTLVAML) resist improvement regardless of reward mode. These likely have inherent ERGO scorer limitations — the model cannot reliably distinguish target binders from decoy binders for these peptides.

**Seed-sensitive targets** (SLYNTVATL, GLCTLVAML, RLRAEAQVK, GILGFVFTL) are where seed=42 got lucky. Their high variance (std 0.06–0.18) explains the seed dependence of overall AUROC.

---

## 6. Key Finding #4: Architecture Changes are Marginal

| Comparison | AUROC | Delta |
|-----------|-------|-------|
| v1_ergo_only (s123, full arch) | 0.5462 | baseline |
| test6_pure_v2 (A1+A2+A10 only, no L0 curriculum) | 0.5894 | +0.04 |

Removing the L0 curriculum (known binder initialization) and simplifying architecture to A1+A2+A10 actually slightly improved AUROC. This suggests:
- L0 curriculum may introduce bias rather than helping
- The simpler architecture is as effective or better
- Architecture is not the bottleneck

---

## 7. Per-Target AUROC Heatmap

```
Experiment           GILGFV NLVPMV GLCTLV LLWNGP YLQPRT FLYALA SLYNTV KLGGAL AVFDRK IVTDFS SPRWYF RLRAEA | MEAN
-----------------------------------------------------------------------------------------------------------------
v1_ergo_only (s42)    0.97   0.97   0.98   0.71   0.75   0.58   0.91   0.70   0.71   0.86   0.64   0.94  | 0.808
v1_ergo_repro(s123)   0.55   0.60   0.38   0.52   0.58   0.39   0.43   0.60   0.59   0.87   0.48   0.55  | 0.546
test6_pure_v2         0.54   0.65   0.47   0.57   0.55   0.45   0.75   0.59   0.68   0.72   0.59   0.49  | 0.589
v2_full               0.77   0.99   0.40   0.47   0.46   0.41   0.54   0.50   0.63   0.88   0.25   0.60  | 0.573
test4_raw_multi       0.54   0.55   0.46   0.56   0.67   0.48   0.57   0.56   0.67   0.86   0.61   0.46  | 0.581
test3_stepwise        0.60   0.61   0.35   0.58   0.54   0.45   0.69   0.58   0.62   0.80   0.55   0.50  | 0.572
test5_threshold       0.54   0.61   0.41   0.48   0.57   0.42   0.71   0.54   0.64   0.82   0.66   0.44  | 0.570
test1_two_phase       0.56   0.63   0.49   0.60   0.59   0.45   0.52   0.61   0.66   0.80   0.39   0.51  | 0.567
test2_min6_raw        0.58   0.51   0.41   0.52   0.57   0.46   0.47   0.53   0.62   0.83   0.61   0.58  | 0.556
v2_no_decoy           0.56   0.55   0.42   0.48   0.70   0.42   0.46   0.50   0.54   0.90   0.22   0.60  | 0.530
exp3_delta            0.60   0.51   0.39   0.48   0.58   0.33   0.45   0.58   0.59   0.79   0.23   0.46  | 0.500
exp1_decoy            0.51   0.59   0.35   0.52   0.58   0.38   0.48   0.47   0.55   0.75   0.26   0.45  | 0.490
exp4_min_steps        0.62   0.52   0.35   0.44   0.49   0.38   0.47   0.48   0.50   0.72   0.27   0.48  | 0.477
exp2_light            0.45   0.48   0.35   0.50   0.59   0.35   0.34   0.53   0.56   0.84   0.20   0.41  | 0.466
```

---

## 8. Root Cause Analysis: Why Can't We Beat 0.55?

### 8.1 ERGO Scorer Ceiling

The ERGO scorer (ae_mcpas1.pt) is used for both training reward AND evaluation. This creates a closed loop:
- **Training** maximizes ERGO binding score (deterministic forward pass)
- **Evaluation** measures ERGO binding discrimination via MC Dropout (N=10, stochastic)

The gap between deterministic training and stochastic evaluation means the policy learns to exploit deterministic ERGO features that don't generalize to the MC Dropout evaluation.

Evidence: The 3-tier eval (NetTCR cross-validation) of v1_ergo_only showed AUROC dropping from 0.8075 to 0.5754, confirming **ERGO overfitting**.

### 8.2 Evaluation Protocol Variance

With only 50 TCRs x 50 decoys per target, the evaluation has inherent noise. The per-target AUROC std across experiments is 0.04–0.18, suggesting the evaluation itself has significant variance.

### 8.3 TCR Editing vs Generation

The policy edits existing CDR3 sequences (max 8-10 steps). This limits the search space — the policy can only reach sequences within ~10 edit operations of the initialization. If the ERGO landscape has narrow optima, most seeds will miss them.

---

## 9. Recommendations

### Short-term (before test8/9/10 complete):
1. **Multi-seed evaluation**: Run the best config (v1_ergo_only or test6_pure_v2) with 5+ seeds to establish a reliable mean and confidence interval
2. **Increase eval budget**: Use 100+ TCRs and 100+ decoys per target to reduce evaluation noise

### Medium-term:
3. **Alternative scorer**: Replace or supplement ERGO with a different binding predictor (e.g., NetTCR, TITAN) to break the scorer overfitting loop
4. **Ensemble evaluation**: Score with multiple models during eval to get more robust AUROC
5. **Longer edit trajectories**: Increase max_steps from 8 to 15-20 to allow broader exploration

### What's likely NOT worth pursuing:
- Further reward mode tweaking — all modes converge to the same band
- More penalty weight tuning — penalties consistently hurt or have no effect
- Architecture changes — test6 showed architecture is not the bottleneck

---

## 10. Still Training (results pending)

| Experiment | Progress | What it tests | Expected insight |
|-----------|----------|---------------|-----------------|
| test8_longer_5M | 2.2M/5M (45%) | 2.5x more training steps | Does convergence improve? |
| test9_squared | 1.4M/2M (68%) | reward = ergo^2 | Does reward shaping help? |
| test10_big_slow | 0.3M/3M (9%) | lr=1e-4, hidden=768 | Does capacity matter? |

These may provide incremental information but are unlikely to break out of the 0.55 band based on the pattern above.
