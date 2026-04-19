# TCRPPO v2 Experiments: Comprehensive Report

**Generated:** 2026-04-19 12:17:11
**Total Experiments:** 24

## Executive Summary

| Exp | Status | Reward Mode | Scorer | Encoder | Seed | Steps | AUROC | Notes |
|-----|--------|-------------|--------|---------|------|-------|-------|-------|
| test11_nettcr | training | v1_ergo_only | nettcr | esm2 | 42 | 2000000 | N/A |  |
| test15_tcbind | training | v1_ergo_only | tcbind | esm2 | 42 | 2000000 | N/A |  |
| test9_squared | completed | v1_ergo_squared | ergo | esm2 | 42 | 2000000 | N/A | ERGO^2 (squared) reward mode. Lower training rewar |
| test2_min6_raw | completed | raw_decoy | ergo | esm2 | 42 | 2000000 | 0.5562 | Min 6 steps + raw decoy penalty (d=0.05, no z-norm |
| test3_stepwise | completed | v1_ergo_stepwise | ergo | esm2 | 42 | 2000000 | 0.5717 | Per-step absolute ERGO score (not delta, not termi |
| test10_big_slow | incomplete | v1_ergo_only | ergo | esm2 | 42 | 3000000 | N/A | Bigger architecture (hidden_dim=768 = 3.5M params  |
| test4_raw_multi | completed | raw_multi_penalty | ergo | esm2 | 42 | 2000000 | 0.5812 | Raw multi-penalty (d=0.05, n=0.02, v=0.01, no z-no |
| test5_threshold | completed | threshold_penalty | ergo | esm2 | 42 | 2000000 | 0.5697 | Conditional penalties at affinity>0.5 threshold. T |
| test8_longer_5M | completed | v1_ergo_only | ergo | esm2 | 42 | 5000000 | N/A | Extended 5M-step run of v1_ergo_only baseline. Kil |
| test11_nettcr_pure | training | v1_ergo_only | nettcr | esm2 | 42 | 2000000 | N/A |  |
| test1_two_phase_p2 | completed | v1_ergo_only -> raw_decoy | ergo | esm2 | 42 | 2000000 | 0.5668 | Two-phase: 1M pure ERGO, then 1M raw_decoy (d=0.05 |
| test6_pure_v2_arch | completed | v1_ergo_only | ergo | esm2 | 42 | 2000000 | 0.5894 | Pure v2 architecture (A1+A2+A10) without L0 curric |
| test7_v1ergo_repro | completed | v1_ergo_only | ergo | esm2 | 123 | 2000000 | 0.5462 | Reproduction of v1_ergo_only with seed=123. AUROC  |
| test18_v1ergo_seed7 | training | v1_ergo_only | ergo | esm2 | 7 | 2000000 | N/A |  |
| test14_bugfix_v1ergo | training | v1_ergo_only | ergo | esm2 | 42 | 2000000 | 0.6091 |  |
| test12_nettcr_seed123 | training | v1_ergo_only | nettcr | esm2 | 123 | 2000000 | N/A |  |
| test13_ensemble_reward | training | v1_ergo_only | ensemble | esm2 | 42 | 2000000 | N/A |  |
| test18_tfold_corrected | training | v1_ergo_only | ergo | esm2 | 42 | 2000000 | N/A |  |
| test19_v1ergo_seed2024 | training | v1_ergo_only | ergo | esm2 | 2024 | 2000000 | N/A |  |
| test16_ergo_lightweight | training | v1_ergo_only | ergo | esm2 | 42 | 2000000 | 0.4285 |  |
| test15_tcbind_lightweight | training | v1_ergo_only | tcbind | esm2 | 42 | 2000000 | 0.5245 |  |
| test13_ensemble_ergo_nettcr | training | v1_ergo_only | ensemble | esm2 | 42 | 2000000 | N/A |  |
| test16_ensemble_ergo_tcbind | training | v1_ergo_only | ensemble_ergo_tcbind | esm2 | 42 | 2000000 | N/A |  |
| test17_ergo_lightweight_s123 | training | v1_ergo_only | ergo | esm2 | 123 | 2000000 | 0.5148 |  |

## Results by Reward Mode

| Reward Mode | N | Mean AUROC | Std | Min | Max |
|-------------|---|------------|-----|-----|-----|
| raw_decoy | 1 | 0.5562 | 0.0000 | 0.5562 | 0.5562 |
| raw_multi_penalty | 1 | 0.5812 | 0.0000 | 0.5812 | 0.5812 |
| threshold_penalty | 1 | 0.5697 | 0.0000 | 0.5697 | 0.5697 |
| v1_ergo_only | 6 | 0.5354 | 0.0584 | 0.4285 | 0.6091 |
| v1_ergo_only -> raw_decoy | 1 | 0.5668 | 0.0000 | 0.5668 | 0.5668 |
| v1_ergo_stepwise | 1 | 0.5717 | 0.0000 | 0.5717 | 0.5717 |

## Results by Affinity Scorer

| Affinity Scorer | N | Mean AUROC | Std | Min | Max |
|-----------------|---|------------|-----|-----|-----|
| ergo | 10 | 0.5534 | 0.0481 | 0.4285 | 0.6091 |
| tcbind | 1 | 0.5245 | 0.0000 | 0.5245 | 0.5245 |

## Detailed Experiment Reports

### test11_nettcr

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: nettcr
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

---

### test15_tcbind

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: tcbind
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

---

### test9_squared

**Status:** completed

**Configuration:**
```
Reward Mode: v1_ergo_squared
Affinity Scorer: ergo
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

**Notes:** ERGO^2 (squared) reward mode. Lower training rewards than v1_ergo_only (~0.5 vs ~1.5) since scores get squared. Partial eval shows similar AUROC to other ERGO experiments (~0.53). max_tcr_len=27 (old default).

---

### test2_min6_raw

**Status:** completed

**Configuration:**
```
Reward Mode: raw_decoy
Affinity Scorer: ergo
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.05, nat=0.5, div=0.2
```

**Evaluation Results:**

- **Mean AUROC (ERGO):** 0.5562

Per-target AUROC (ERGO):

| Target | AUROC |
|--------|-------|
| GILGFVFTL | 0.5758 |
| NLVPMVATV | 0.5062 |
| GLCTLVAML | 0.4065 |
| LLWNGPMAV | 0.5198 |
| YLQPRTFLL | 0.5696 |
| FLYALALLL | 0.4645 |
| SLYNTVATL | 0.4732 |
| KLGGALQAK | 0.5275 |
| AVFDRKSDAK | 0.6200 |
| IVTDFSVIK | 0.8278 |
| SPRWYFYYL | 0.6052 |
| RLRAEAQVK | 0.5780 |

**Notes:** Min 6 steps + raw decoy penalty (d=0.05, no z-norm). Raw decoy penalty still hurts.

---

### test3_stepwise

**Status:** completed

**Configuration:**
```
Reward Mode: v1_ergo_stepwise
Affinity Scorer: ergo
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation Results:**

- **Mean AUROC (ERGO):** 0.5717

Per-target AUROC (ERGO):

| Target | AUROC |
|--------|-------|
| GILGFVFTL | 0.6024 |
| NLVPMVATV | 0.6085 |
| GLCTLVAML | 0.3532 |
| LLWNGPMAV | 0.5756 |
| YLQPRTFLL | 0.5410 |
| FLYALALLL | 0.4475 |
| SLYNTVATL | 0.6896 |
| KLGGALQAK | 0.5757 |
| AVFDRKSDAK | 0.6195 |
| IVTDFSVIK | 0.8020 |
| SPRWYFYYL | 0.5468 |
| RLRAEAQVK | 0.4985 |

**Notes:** Per-step absolute ERGO score (not delta, not terminal). Underperforms terminal reward.

---

### test10_big_slow

**Status:** incomplete

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ergo
State Encoder: esm2
Seed: 42
Total Steps: 3000000
Environments: 8
Learning Rate: 0.0001
Hidden Dim: 768
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

**Notes:** Bigger architecture (hidden_dim=768 = 3.5M params vs 2M standard) with lower LR (1e-4 vs 3e-4). Killed early due to GPU contention. 500K checkpoint available for eval. max_tcr_len=27 (old default).

---

### test4_raw_multi

**Status:** completed

**Configuration:**
```
Reward Mode: raw_multi_penalty
Affinity Scorer: ergo
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.05, nat=0.02, div=0.01
```

**Evaluation Results:**

- **Mean AUROC (ERGO):** 0.5812

Per-target AUROC (ERGO):

| Target | AUROC |
|--------|-------|
| GILGFVFTL | 0.5377 |
| NLVPMVATV | 0.5508 |
| GLCTLVAML | 0.4579 |
| LLWNGPMAV | 0.5552 |
| YLQPRTFLL | 0.6714 |
| FLYALALLL | 0.4747 |
| SLYNTVATL | 0.5691 |
| KLGGALQAK | 0.5614 |
| AVFDRKSDAK | 0.6695 |
| IVTDFSVIK | 0.8569 |
| SPRWYFYYL | 0.6100 |
| RLRAEAQVK | 0.4597 |

**Notes:** Raw multi-penalty (d=0.05, n=0.02, v=0.01, no z-norm). Multi-penalty still hurts vs pure ERGO.

---

### test5_threshold

**Status:** completed

**Configuration:**
```
Reward Mode: threshold_penalty
Affinity Scorer: ergo
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.05, nat=0.02, div=0.01
```

**Evaluation Results:**

- **Mean AUROC (ERGO):** 0.5697

Per-target AUROC (ERGO):

| Target | AUROC |
|--------|-------|
| GILGFVFTL | 0.5380 |
| NLVPMVATV | 0.6059 |
| GLCTLVAML | 0.4107 |
| LLWNGPMAV | 0.4828 |
| YLQPRTFLL | 0.5738 |
| FLYALALLL | 0.4199 |
| SLYNTVATL | 0.7103 |
| KLGGALQAK | 0.5391 |
| AVFDRKSDAK | 0.6439 |
| IVTDFSVIK | 0.8149 |
| SPRWYFYYL | 0.6564 |
| RLRAEAQVK | 0.4402 |

**Notes:** Conditional penalties at affinity>0.5 threshold. Threshold gating doesn't help.

---

### test8_longer_5M

**Status:** completed

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ergo
State Encoder: esm2
Seed: 42
Total Steps: 5000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

**Notes:** Extended 5M-step run of v1_ergo_only baseline. Killed at ~2.4M steps. Same ERGO-based v2 arch as other tests. max_tcr_len=27 (old default).

---

### test11_nettcr_pure

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: nettcr
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

---

### test1_two_phase_p2

**Status:** completed

**Configuration:**
```
Reward Mode: v1_ergo_only -> raw_decoy
Affinity Scorer: ergo
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.05, nat=0.5, div=0.2
```

**Evaluation Results:**

- **Mean AUROC (ERGO):** 0.5668

Per-target AUROC (ERGO):

| Target | AUROC |
|--------|-------|
| GILGFVFTL | 0.5621 |
| NLVPMVATV | 0.6334 |
| GLCTLVAML | 0.4919 |
| LLWNGPMAV | 0.5993 |
| YLQPRTFLL | 0.5878 |
| FLYALALLL | 0.4484 |
| SLYNTVATL | 0.5157 |
| KLGGALQAK | 0.6140 |
| AVFDRKSDAK | 0.6554 |
| IVTDFSVIK | 0.7958 |
| SPRWYFYYL | 0.3895 |
| RLRAEAQVK | 0.5087 |

**Notes:** Two-phase: 1M pure ERGO, then 1M raw_decoy (d=0.05). Decoy penalty in P2 degraded binding.

---

### test6_pure_v2_arch

**Status:** completed

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ergo
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.0, nat=0.0, div=0.0
```

**Evaluation Results:**

- **Mean AUROC (ERGO):** 0.5894

Per-target AUROC (ERGO):

| Target | AUROC |
|--------|-------|
| GILGFVFTL | 0.5360 |
| NLVPMVATV | 0.6544 |
| GLCTLVAML | 0.4656 |
| LLWNGPMAV | 0.5690 |
| YLQPRTFLL | 0.5532 |
| FLYALALLL | 0.4534 |
| SLYNTVATL | 0.7541 |
| KLGGALQAK | 0.5941 |
| AVFDRKSDAK | 0.6824 |
| IVTDFSVIK | 0.7223 |
| SPRWYFYYL | 0.5944 |
| RLRAEAQVK | 0.4944 |

**Notes:** Pure v2 architecture (A1+A2+A10) without L0 curriculum. Simpler arch performs same or better.

---

### test7_v1ergo_repro

**Status:** completed

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ergo
State Encoder: esm2
Seed: 123
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.0, nat=0.0, div=0.0
```

**Evaluation Results:**

- **Mean AUROC (ERGO):** 0.5462

Per-target AUROC (ERGO):

| Target | AUROC |
|--------|-------|
| GILGFVFTL | 0.5501 |
| NLVPMVATV | 0.5954 |
| GLCTLVAML | 0.3834 |
| LLWNGPMAV | 0.5234 |
| YLQPRTFLL | 0.5822 |
| FLYALALLL | 0.3912 |
| SLYNTVATL | 0.4312 |
| KLGGALQAK | 0.6001 |
| AVFDRKSDAK | 0.5891 |
| IVTDFSVIK | 0.8721 |
| SPRWYFYYL | 0.4822 |
| RLRAEAQVK | 0.5543 |

**Notes:** Reproduction of v1_ergo_only with seed=123. AUROC dropped to 0.5462 (from 0.8075 with seed=42). Confirms seed dependence.

---

### test18_v1ergo_seed7

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ergo
State Encoder: esm2
Seed: 7
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

---

### test14_bugfix_v1ergo

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ergo
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation Results:**

- **Mean AUROC (ERGO):** 0.6091

Per-target AUROC (ERGO):

| Target | AUROC |
|--------|-------|
| GILGFVFTL | 0.4583 |
| NLVPMVATV | 0.4874 |
| GLCTLVAML | 0.5671 |
| LLWNGPMAV | 0.6797 |
| YLQPRTFLL | 0.8264 |
| FLYALALLL | 0.5203 |
| SLYNTVATL | 0.7612 |
| KLGGALQAK | 0.6181 |
| AVFDRKSDAK | 0.6062 |
| IVTDFSVIK | 0.9281 |
| SPRWYFYYL | 0.3772 |
| RLRAEAQVK | 0.4786 |

---

### test12_nettcr_seed123

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: nettcr
State Encoder: esm2
Seed: 123
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

---

### test13_ensemble_reward

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ensemble
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

---

### test18_tfold_corrected

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ergo
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

---

### test19_v1ergo_seed2024

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ergo
State Encoder: esm2
Seed: 2024
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

---

### test16_ergo_lightweight

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ergo
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation Results:**

- **Mean AUROC (ERGO):** 0.4285

Per-target AUROC (ERGO):

| Target | AUROC |
|--------|-------|
| GILGFVFTL | 0.4235 |
| NLVPMVATV | 0.4286 |
| GLCTLVAML | 0.3895 |
| LLWNGPMAV | 0.5504 |
| YLQPRTFLL | 0.6129 |
| FLYALALLL | 0.2861 |
| SLYNTVATL | 0.2097 |
| KLGGALQAK | 0.4845 |
| AVFDRKSDAK | 0.5505 |
| IVTDFSVIK | 0.6966 |
| SPRWYFYYL | 0.1686 |
| RLRAEAQVK | 0.3411 |

---

### test15_tcbind_lightweight

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: tcbind
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation Results:**

- **Mean AUROC (ERGO):** 0.5245

Per-target AUROC (ERGO):

| Target | AUROC |
|--------|-------|
| GILGFVFTL | 0.7455 |
| NLVPMVATV | 0.7313 |
| GLCTLVAML | 0.6415 |
| LLWNGPMAV | 0.4297 |
| YLQPRTFLL | 0.4640 |
| FLYALALLL | 0.4970 |
| SLYNTVATL | 0.5530 |
| KLGGALQAK | 0.4396 |
| AVFDRKSDAK | 0.5026 |
| IVTDFSVIK | 0.4428 |
| SPRWYFYYL | 0.3274 |
| RLRAEAQVK | 0.5199 |

---

### test13_ensemble_ergo_nettcr

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ensemble
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

---

### test16_ensemble_ergo_tcbind

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ensemble_ergo_tcbind
State Encoder: esm2
Seed: 42
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation:** Not yet evaluated

---

### test17_ergo_lightweight_s123

**Status:** training

**Configuration:**
```
Reward Mode: v1_ergo_only
Affinity Scorer: ergo
State Encoder: esm2
Seed: 123
Total Steps: 2000000
Environments: 8
Learning Rate: 0.0003
Hidden Dim: 512
Max Steps: 8
Weights: aff=1.0, dec=0.8, nat=0.5, div=0.2
```

**Evaluation Results:**

- **Mean AUROC (ERGO):** 0.5148

Per-target AUROC (ERGO):

| Target | AUROC |
|--------|-------|
| GILGFVFTL | 0.6115 |
| NLVPMVATV | 0.6435 |
| GLCTLVAML | 0.6266 |
| LLWNGPMAV | 0.4460 |
| YLQPRTFLL | 0.8047 |
| FLYALALLL | 0.4684 |
| SLYNTVATL | 0.2591 |
| KLGGALQAK | 0.6940 |
| AVFDRKSDAK | 0.3205 |
| IVTDFSVIK | 0.8590 |
| SPRWYFYYL | 0.1310 |
| RLRAEAQVK | 0.3128 |

---
