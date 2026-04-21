# TCRPPO v2 Complete Experiment Tracker

**Last Updated:** 2026-04-21  
**Baseline:** v1_ergo_only = **0.8075 AUROC** (best result so far, but seed-dependent — repro got 0.5462)  
**Target:** Beat 0.8075 while adding specificity (decoy resistance)

---

## Experiment Summary Table

| # | Experiment Name | Status | Steps | Mean AUROC | vs Baseline | Reward Mode | Key Config | GPU | Notes |
|---|----------------|--------|-------|-----------|-------------|-------------|-----------|-----|-------|
| 1 | v1_ergo_only_ablation | ✅ DONE | 2M | **0.8075** | baseline | v1_ergo_only | Raw ERGO terminal, no penalties | - | **BEST** - seed=42, possibly lucky |
| 2 | v2_full_run1 | ✅ DONE | 2M | 0.5840 | -0.2235 | v2_full | d=0.8, n=0.5, v=0.2, z-norm | - | Early stopping (avg 2 steps) |
| 3 | exp1_decoy_only | ✅ DONE | 500K | 0.4898 | -0.3177 | v2_decoy_only | d=0.3, z-norm | 0 | Z-norm compressed affinity |
| 4 | exp2_light | ✅ DONE | 500K | 0.4660 | -0.3415 | v2_full | d=0.2, n=0.1, v=0.05, z-norm | 2 | Even light penalties fail with z-norm |
| 5 | exp3_ergo_delta | ✅ DONE | 500K | 0.5004 | -0.3071 | v1_ergo_delta | Raw delta per-step, no penalties | 1 | Long edits (7.9 steps) but poor binding |
| 6 | exp4_min_steps | ✅ DONE | 500K | 0.4768 | -0.3307 | v2_full | d=0.4, n=0.2, v=0.1, min_steps=3 | 3 | Min-steps doesn't fix z-norm issue |
| 7 | v2_no_decoy | 🔄 TRAINING | 1.35M/2M | TBD | TBD | v2_no_decoy | No decoy, has nat+div, z-norm | 7 | Ablation study |
| 8 | test1_two_phase | ✅ DONE | 2M | 0.5668 | -0.2407 | v1_ergo_only → raw_decoy | Phase1: 1M pure ERGO, Phase2: 1M +decoy | 0 | Decoy penalty in P2 degraded binding |
| 9 | test2_min6_raw | ✅ DONE | 2M | 0.5562 | -0.2513 | raw_decoy | min_steps=6, raw reward, d=0.05 | 1 | Raw decoy penalty still hurts |
| 10 | test3_stepwise | ✅ DONE | 2M | 0.5717 | -0.2358 | v1_ergo_stepwise | Raw ERGO per-step (absolute score) | 2 | Per-step reward underperforms terminal |
| 11 | test4_raw_multi | ✅ DONE | 2M | 0.5812 | -0.2263 | raw_multi_penalty | raw - 0.05d - 0.02n - 0.01v | 3 | Multi-penalty still hurts |
| 12 | test5_threshold | ✅ DONE | 2M | 0.5697 | -0.2378 | threshold_penalty | Conditional penalties at aff>0.5 | 4 | Threshold gating doesn't help |
| 13 | test6_pure_v2 | ✅ DONE | 2M | 0.5894 | -0.2181 | v1_ergo_only | A1+A2+A10 only, NO curriculum | 5 | Pure v2 arch without L0 curriculum |
| 14 | test7_v1ergo_repro | ✅ DONE | 2M | 0.5462 | -0.2613 | v1_ergo_only | seed=123, reproduction test | 2 | **Repro FAILED** — seed matters! |
| 15 | test8_longer_5M | 🔄 TRAINING | 2.2M/5M | TBD | TBD | v1_ergo_only | **5M steps** (2.5x longer) | 0 | ~44% complete |
| 16 | test9_squared | 🔄 TRAINING | 1.3M/2M | TBD | TBD | v1_ergo_squared | **reward=ergo^2** | 1 | ~66% complete |
| 17 | test10_big_slow | 🔄 TRAINING | 0.19M/3M | TBD | TBD | v1_ergo_only | **lr=1e-4, hidden=768, 3M** | 6 | ~6% complete |
| 18 | test11_nettcr_pure | ❌ CRASHED | 143K/2M | TBD | TBD | v1_ergo_only | **NetTCR as sole scorer**, seed=42 | 2 | Crashed early — relaunched as test11_nettcr |
| 19 | test12_nettcr_seed123 | ❌ CRASHED | 153K/2M | TBD | TBD | v1_ergo_only | **NetTCR scorer**, seed=123 | 3 | Crashed early |
| 20 | test13_ensemble_reward | ❌ CRASHED | 143K/2M | TBD | TBD | v1_ergo_only | **ERGO+NetTCR ensemble** (50/50), seed=42 | 4 | Crashed early — relaunched as test13_ensemble_ergo_nettcr |
| 21 | test14_bugfix_v1ergo | 🔄 TRAINING | 409K/2M | TBD | TBD | v1_ergo_only | **ERGO**, seed=42, bugfix run | 0 | Current best baseline rerun |
| 22 | test15_tcbind | 🔄 TRAINING | 0/2M | TBD | TBD | v1_ergo_only | **TCBind BiLSTM v2** scorer, seed=42 | 0 | tc-hard trained classifier, 0.5886 AUC |
| 23 | test11_nettcr | 🔄 TRAINING | 0/2M | TBD | TBD | v1_ergo_only | **NetTCR** scorer, seed=42 | 0 | Break ERGO train-eval coupling |
| 24 | test16_ensemble_ergo_tcbind | 🔄 TRAINING | 0/2M | TBD | TBD | v1_ergo_only | **ERGO+TCBind** (50/50), seed=42 | 0 | Dual-scorer: ERGO + sequence classifier |
| 25 | test13_ensemble_ergo_nettcr | 🔄 TRAINING | 0/2M | TBD | TBD | v1_ergo_only | **ERGO+NetTCR** (50/50), seed=42 | 0 | Dual-scorer: ERGO + NetTCR CNN |
| 26 | test16_ergo_lightweight | ✅ DONE | 2M | TBD | TBD | v1_ergo_only | Lightweight 256d, seed=42, ban_stop | - | Lightweight encoder baseline |
| 27 | test17_ergo_lightweight_s123 | ✅ DONE | 2M | TBD | TBD | v1_ergo_only | Lightweight 256d, seed=123, ban_stop | - | Seed=123 repro w/ lightweight |
| 28 | test18_v1ergo_seed7 | ✅ DONE | 2M | TBD | TBD | v1_ergo_only | seed=7, lightweight | - | Seed stability test |
| 29 | test19_v1ergo_seed2024 | ✅ DONE | 2M | TBD | TBD | v1_ergo_only | seed=2024, lightweight | - | Seed stability test |
| 30 | test20_ban_stop | ✅ DONE | 2M | TBD | TBD | v2_full | Lightweight, ban_stop, all penalties | - | R=-1.2, penalties destroy signal |
| 31 | test21_esm2_breakthrough | 🔄 TRAINING | 501K/2M | TBD | TBD | v1_ergo_shaped | ESM-2, ban_stop, shaped reward | - | R=0.15-0.19, shaped reward too weak |
| 32 | test22_tfold_cascade | 🔄 TRAINING | 153K/2M | TBD | TBD | v1_ergo_only | ESM-2, ban_stop, tFold cascade (t=0.15) | 2 | R=0.83-1.13, cascade works |
| 33 | test22b_ergo_only | 🔄 TRAINING | 153K/2M | TBD | TBD | v1_ergo_only | ESM-2, ban_stop, pure ERGO | 3 | R=0.98-1.31, ESM-2+ERGO strong |
| 34 | test23_contrastive_ergo | 📋 PLANNED | 0/2M | TBD | TBD | contrastive_ergo | **ERGO(target)-ERGO(decoys)**, ESM-2, ban_stop | 0 | Break ERGO train-eval coupling |
| 35 | test24_large_batch | 📋 PLANNED | 0/2M | TBD | TBD | v1_ergo_only | **n_envs=32**, seed=123, ESM-2, ban_stop | 1 | Stabilize failed seed via large batch |
| 36 | test26_curriculum_l0 | 📋 PLANNED | 0/2M | TBD | TBD | v1_ergo_only | **L0=0.5,L1=0.2,L2=0.3**, ESM-2, ban_stop | 4 | Better initialization via curriculum |

---

## Completed Experiments (1-6)

### Experiment 1: v1_ergo_only_ablation ⭐ BEST

**Goal:** Reproduce v1 baseline with v2 architecture (indel, ESM-2, L0 curriculum)

**Configuration:**
```yaml
reward_mode: v1_ergo_only
total_timesteps: 2000000
n_envs: 8
w_affinity: 1.0  # Not used (raw reward)
w_decoy: 0.0
w_naturalness: 0.0
w_diversity: 0.0
use_delta_reward: false  # Terminal reward only
```

**Reward Formula:**
```python
total = raw_affinity  # ERGO score (0-1), no normalization, terminal only
```

**Results:**
- **Mean AUROC:** 0.8075 (vs v1 baseline 0.4538, delta +0.3537)
- **Avg Steps:** 8.8
- **Training Speed:** 58 steps/sec (fast, no ESM/decoy overhead)
- **Target Score:** 0.445 avg
- **Decoy Score:** 0.110 avg
- **Best Target:** GILGFVFTL (0.9688), NLVPMVATV (0.9742), GLCTLVAML (0.9764)
- **Worst Target:** FLYALALLL (0.5792), SPRWYFYYL (0.6359)

**Key Insight:** Raw ERGO terminal reward works extremely well. No normalization = strong, clear signal.

**Caveat:** NetTCR cross-validation shows AUROC drops to 0.5754, suggesting possible ERGO scorer overfitting.

---

### Experiment 2: v2_full_run1

**Goal:** Full v2 pipeline with all 4 reward components

**Configuration:**
```yaml
reward_mode: v2_full
total_timesteps: 2000000
n_envs: 8
w_affinity: 1.0
w_decoy: 0.8
w_naturalness: 0.5
w_diversity: 0.2
use_delta_reward: true
```

**Reward Formula:**
```python
total = w_aff * z_norm(aff_delta) - w_d * z_norm(decoy) - w_n * z_norm(nat) - w_v * z_norm(div)
```

**Results:**
- **Mean AUROC:** 0.5840 (vs v1 baseline 0.4538, delta +0.1302)
- **Avg Steps:** ~2.0 (early stopping problem)
- **Training Speed:** 22 steps/sec (ESM + all scorers overhead)
- **Target Score:** 0.241 avg
- **Decoy Score:** 0.114 avg

**Key Issue:** Penalty weights too strong → policy learns to STOP early to avoid penalties → short sequences with poor binding.

---

### Experiment 3: exp1_decoy_only

**Goal:** Add ONLY decoy penalty at low weight (0.3), no naturalness/diversity

**Configuration:**
```yaml
reward_mode: v2_decoy_only
total_timesteps: 500000
w_affinity: 1.0
w_decoy: 0.3
```

**Reward Formula:**
```python
total = w_aff * z_norm(aff_delta) - w_d * z_norm(decoy)
```

**Results:**
- **Mean AUROC:** 0.4898 (FAILED)
- **Avg Steps:** 6.6-7.8
- **Training Speed:** 22 steps/sec
- **Target Score:** 0.130 avg
- **Decoy Score:** 0.118 avg

**Key Issue:** Even with light decoy penalty (0.3), z-score normalization compresses affinity signal.

---

### Experiment 4: exp2_light

**Goal:** Dramatically reduce penalty weights (10x reduction from v2_full)

**Configuration:**
```yaml
reward_mode: v2_full
total_timesteps: 500000
w_affinity: 1.0
w_decoy: 0.2
w_naturalness: 0.1
w_diversity: 0.05
```

**Results:**
- **Mean AUROC:** 0.4660 (FAILED)
- **Avg Steps:** 5.4
- **Target Score:** 0.141 avg
- **Decoy Score:** 0.138 avg (near-parity with target!)

**Key Issue:** Even 10x lighter penalties still fail catastrophically with z-score normalization.

---

### Experiment 5: exp3_ergo_delta

**Goal:** Use raw delta reward (per-step credit assignment) without normalization

**Configuration:**
```yaml
reward_mode: v1_ergo_delta
total_timesteps: 500000
w_affinity: N/A  # Raw delta, no weights
```

**Reward Formula:**
```python
total = aff_score - initial_affinity  # Raw delta, no z-norm
```

**Results:**
- **Mean AUROC:** 0.5004 (FAILED)
- **Avg Steps:** 7.9 (longest editing trajectories)
- **Training Speed:** 58 steps/sec (fast)
- **Target Score:** 0.165 avg
- **Decoy Score:** 0.145 avg

**Key Issue:** Raw delta without normalization produces many edits but poor binding. Per-step delta credit alone is insufficient.

---

### Experiment 6: exp4_min_steps

**Goal:** Force policy to take at least 3 steps before STOP

**Configuration:**
```yaml
reward_mode: v2_full
total_timesteps: 500000
w_affinity: 1.0
w_decoy: 0.4
w_naturalness: 0.2
w_diversity: 0.1
min_steps: 3
min_steps_penalty: -2.0
```

**Results:**
- **Mean AUROC:** 0.4768 (FAILED)
- **Avg Steps:** 5.0
- **Target Score:** 0.128 avg
- **Decoy Score:** 0.120 avg

**Key Issue:** Min-steps constraint prevents early termination but doesn't fix the fundamental z-norm compression problem.

---

## In-Progress Experiments (7)

### Experiment 7: v2_no_decoy

**Goal:** Ablation study — v2 without decoy penalty (isolate decoy contribution)

**Configuration:**
```yaml
reward_mode: v2_no_decoy
total_timesteps: 2000000
w_affinity: 1.0
w_decoy: 0.0
w_naturalness: 0.5
w_diversity: 0.2
```

**Status:** 1.31M/2M (65.5%), training on GPU 7

**Expected Completion:** ~6-8 hours

---

## Planned Experiments (8-12) — NEW TESTS

### Experiment 8: test1_two_phase (Two-Phase Training)

**Goal:** Train pure ERGO first, then add light decoy penalty via fine-tuning

**Phase 1 Configuration:**
```yaml
reward_mode: v1_ergo_only
total_timesteps: 1000000
n_envs: 8
```

**Phase 2 Configuration:**
```yaml
reward_mode: raw_decoy  # NEW MODE
total_timesteps: 1000000
resume_from: output/test1_two_phase/checkpoints/milestone_1000000.pt
resume_change_reward_mode: raw_decoy
w_decoy: 0.05  # Light absolute penalty
```

**Reward Formula (Phase 2):**
```python
total = raw_affinity - 0.05 * decoy_score  # No z-normalization
```

**Hypothesis:** Establish strong binding first, then gently add specificity without losing affinity signal.

**GPU:** 0  
**Estimated Time:** 12h (Phase 1) + 12h (Phase 2) = 24h total

---

### Experiment 9: test2_min6_raw (Min-Steps + Raw Reward)

**Goal:** Force at least 6 editing actions with raw reward (no z-norm)

**Configuration:**
```yaml
reward_mode: raw_decoy  # NEW MODE
total_timesteps: 2000000
n_envs: 8
w_decoy: 0.05
min_steps: 6
min_steps_penalty: -3.0
```

**Reward Formula:**
```python
total = raw_affinity - 0.05 * decoy_score
# If STOP before step 6: total += -3.0
```

**Hypothesis:** Combining min-steps constraint with raw (unnormalized) reward will maintain strong affinity while forcing exploration.

**GPU:** 1  
**Estimated Time:** 24h

---

### Experiment 10: test3_stepwise (Step-wise Raw Terminal Reward)

**Goal:** Apply raw ERGO score at EVERY step (not just terminal)

**Configuration:**
```yaml
reward_mode: v1_ergo_stepwise  # NEW MODE
total_timesteps: 2000000
n_envs: 8
```

**Reward Formula:**
```python
# At EVERY step (not just terminal):
total = raw_affinity  # Absolute ERGO score, no delta, no normalization
```

**Hypothesis:** Per-step absolute reward (not delta) may provide clearer credit assignment than terminal-only reward.

**Difference from exp3_ergo_delta:**
- exp3 used `aff_score - initial_affinity` (delta)
- This uses `aff_score` (absolute) per-step

**GPU:** 2  
**Estimated Time:** 24h

---

### Experiment 11: test4_raw_multi (Raw Multi-Penalty — Option A)

**Goal:** Implement Option A from fast_iteration_experiments.md conclusion

**Configuration:**
```yaml
reward_mode: raw_multi_penalty  # NEW MODE
total_timesteps: 2000000
n_envs: 8
w_decoy: 0.05
w_naturalness: 0.02
w_diversity: 0.01
```

**Reward Formula:**
```python
total = raw_affinity - 0.05*decoy - 0.02*naturalness - 0.01*diversity
# All raw scores, NO z-score normalization
```

**Hypothesis:** Small absolute penalties preserve strong affinity signal while gently discouraging bad behavior.

**GPU:** 3  
**Estimated Time:** 24h

---

### Experiment 12: test5_threshold (Threshold-Based Penalties — Option C)

**Goal:** Implement Option C from fast_iteration_experiments.md conclusion

**Configuration:**
```yaml
reward_mode: threshold_penalty  # NEW MODE
total_timesteps: 2000000
n_envs: 8
affinity_threshold: 0.5
w_decoy: 0.05
w_naturalness: 0.02
w_diversity: 0.01
```

**Reward Formula:**
```python
if raw_affinity < 0.5:
    total = raw_affinity  # Pure affinity signal
else:
    total = raw_affinity - 0.05*decoy - 0.02*nat - 0.01*div
```

**Hypothesis:** Policy learns binding first (below threshold), then specificity (above threshold).

**GPU:** 4  
**Estimated Time:** 24h

---

## Key Findings from Experiments 1-6

### Root Cause of Failures

**Z-score normalization + ANY penalty weights = compressed affinity signal**

The `RunningNormalizer` (window=10000, warmup=1000) compresses the affinity delta when combined with normalized penalty terms. Even very light penalty weights (exp2: d=0.2, n=0.1, v=0.05) cause catastrophic failure.

### Why v1_ergo_only Works

1. **Raw ERGO score (0-1 range)** — no normalization
2. **Terminal reward only** — clear credit assignment
3. **No penalties** — pure positive signal for binding improvement

### What Doesn't Work

1. **Z-score normalization** — compresses signal
2. **Raw delta without normalization** (exp3) — poor binding despite long edits
3. **Min-steps constraints alone** (exp4) — doesn't fix signal compression
4. **Light penalties with z-norm** (exp2) — still fails

### New Strategy for Tests 8-12

**Core principle:** Use RAW (unnormalized) rewards for all new tests

- Test 8: Two-phase training (establish binding, then add specificity)
- Test 9: Min-steps + raw reward (force exploration without signal compression)
- Test 10: Step-wise absolute reward (better credit assignment)
- Test 11: Raw multi-penalty (Option A — gentle penalties)
- Test 12: Threshold-based penalties (Option C — staged learning)

---

## Code Modifications Required

### New Reward Modes (reward_manager.py)

```python
# Mode 1: raw_decoy
if self.reward_mode == "raw_decoy":
    total = aff_score - 0.05 * decoy_score

# Mode 2: v1_ergo_stepwise
elif self.reward_mode == "v1_ergo_stepwise":
    total = aff_score  # Absolute score per-step

# Mode 3: raw_multi_penalty
elif self.reward_mode == "raw_multi_penalty":
    total = aff_score - 0.05*decoy - 0.02*nat - 0.01*div

# Mode 4: threshold_penalty
elif self.reward_mode == "threshold_penalty":
    if aff_score < 0.5:
        total = aff_score
    else:
        total = aff_score - 0.05*decoy - 0.02*nat - 0.01*div
```

### Two-Phase Training Support (ppo_trainer.py)

```python
parser.add_argument("--resume_from", help="Checkpoint to resume from")
parser.add_argument("--resume_change_reward_mode", help="New reward mode on resume")
parser.add_argument("--resume_reset_optimizer", action="store_true")
```

---

## Git Worktree Strategy

Each new test will use a separate worktree to isolate code changes:

```bash
# Test 8 (two-phase)
git worktree add .claude/worktrees/test1_two_phase -b test1_two_phase

# Test 9 (min6_raw)
git worktree add .claude/worktrees/test2_min6_raw -b test2_min6_raw

# Test 10 (stepwise)
git worktree add .claude/worktrees/test3_stepwise -b test3_stepwise

# Test 11 (raw_multi)
git worktree add .claude/worktrees/test4_raw_multi -b test4_raw_multi

# Test 12 (threshold)
git worktree add .claude/worktrees/test5_threshold -b test5_threshold
```

Each worktree will have its own reward_manager.py modifications for the specific reward mode.

---

## Evaluation Protocol

For each completed experiment:

```bash
CUDA_VISIBLE_DEVICES=<GPU> /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python -u \
    tcrppo_v2/test_tcrs.py \
    --checkpoint output/<run_name>/checkpoints/final.pt \
    --n_tcrs 50 --n_decoys 50
```

**Metrics to track:**
- Mean AUROC (primary metric)
- Per-target AUROC breakdown
- Avg target score
- Avg decoy score
- Avg steps per episode
- Training speed (steps/sec)
- Unique sequences generated

---

## Success Criteria

At least one of tests 8-12 should:
1. **Match or exceed v1_ergo_only AUROC (0.8075)**
2. **Show improved decoy resistance** (decoy score < 0.110)
3. **Maintain reasonable editing behavior** (avg steps 5-10)
4. **Cross-validate with NetTCR** (AUROC > 0.60 on Tier 2)

---

## Timeline

- **2026-04-11:** Experiments 1-6 completed (fast iteration phase)
- **2026-04-12:** Experiments 8-12 planned and ready to launch
- **2026-04-13-14:** Experiments 8-12 training (24h each)
- **2026-04-15:** Evaluation and results analysis
