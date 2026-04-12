# Fast Iteration Experiments — Improving Beyond v1_ergo_only

**Date:** 2026-04-11
**Baseline:** v1_ergo_only = **0.8075 mean AUROC** (2M steps, ~58 steps/sec)
**Goal:** Beat 0.8075 AUROC while adding specificity (decoy resistance)

---

## Problem Analysis

| Config | Mean AUROC | Avg Steps | Speed | Issue |
|--------|-----------|-----------|-------|-------|
| v1_ergo_only | 0.8075 | 8.8 | 58 s/s | No specificity — universal binder |
| v2_full (d=0.8,n=0.5,v=0.2) | 0.5733 | 4.2 | 22 s/s | Penalties too strong -> early termination |

**Root cause of v2_full failure:** Penalty weights (d=0.8, n=0.5, v=0.2) dominate
the z-normalized reward signal. Policy learns to STOP early to avoid accumulating
penalties, producing short sequences (4.2 steps avg) that lack specificity.

**Hypothesis:** v1_ergo_only works because the policy gets a clear positive signal
for binding improvement. Adding penalties is correct in principle but the weights
need drastic reduction so the affinity signal still dominates.

---

## Code Changes (DONE)

### 1. ppo_trainer.py — CLI weight overrides
Added `--w_affinity`, `--w_decoy`, `--w_naturalness`, `--w_diversity`,
`--min_steps`, `--min_steps_penalty` CLI args. Propagated to config dict.

### 2. reward_manager.py — New reward modes
- `v2_decoy_only`: affinity + decoy scorer active, NO naturalness/diversity.
  Total = w_aff * norm_aff - w_decoy * norm_decoy
- `v1_ergo_delta`: ERGO delta reward only (per-step credit, no penalties, no normalization).
  Total = aff_score - initial_affinity

### 3. env.py — Min-steps penalty
If STOP chosen before `min_steps` steps, `min_steps_penalty` added to reward.
Passed through VecTCREditEnv -> TCREditEnv.

---

## Running Experiments

### Runtime Tracking Table

| # | Name | GPU | PID | Log File | Status | Steps |
|---|------|-----|-----|----------|--------|-------|
| 1 | exp1_decoy_only | 0 | 404359 | `output/exp1_decoy_only_train.log` | **DONE** (AUROC 0.4898) | 501K/500K |
| 2 | exp2_light | 2 | 404361 | `output/exp2_light_train.log` | **DONE** (AUROC 0.4660) | 501K/500K |
| 3 | exp3_ergo_delta | 1 | 404360 | `output/exp3_ergo_delta_train.log` | **DONE** (AUROC 0.5004) | 501K/500K |
| 4 | exp4_min_steps | 3 | 404362 | `output/exp4_min_steps_train.log` | **DONE** (AUROC 0.4768) | 501K/500K |
| - | v2_no_decoy (2M) | 7 | 3419299 | `output/v2_no_decoy_2M_train.log` | TRAINING | 1.24M/2M |
| - | v2_full (2M) | - | - | `output/v2_full_2M_train.log` | **DONE** (AUROC 0.5840) | 2M/2M |

### How to check progress
```bash
# Quick status of all experiments
for f in output/exp*_train.log; do echo "=== $(basename $f) ==="; grep "^Step" "$f" | tail -1; done

# Check specific experiment
tail -5 output/exp1_decoy_only_train.log

# Check if any finished
ls output/exp*/checkpoints/final.pt 2>/dev/null

# Check GPU usage
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

### How to evaluate after training completes
```bash
# Quick eval (~1 min)
CUDA_VISIBLE_DEVICES=<GPU> python -u tcrppo_v2/test_tcrs.py \
    --checkpoint output/<run_name>/checkpoints/final.pt \
    --n_tcrs 20 --n_decoys 20

# Full eval (~2 min)
CUDA_VISIBLE_DEVICES=<GPU> python -u tcrppo_v2/test_tcrs.py \
    --checkpoint output/<run_name>/checkpoints/final.pt \
    --n_tcrs 50 --n_decoys 50
```

---

## Experiment Details

### Experiment 1: exp1_decoy_only

**Hypothesis:** Add ONLY decoy penalty at low weight (0.3). No naturalness/diversity.
The simplest way to add specificity without slowing training too much.

| Parameter | Value |
|-----------|-------|
| reward_mode | v2_decoy_only |
| w_affinity | 1.0 |
| w_decoy | 0.3 |
| w_naturalness | N/A (scorer not loaded) |
| w_diversity | N/A (scorer not loaded) |
| total_timesteps | 500,000 |
| seed | 42 |

**What to look for:** Does reward stay positive? Does avg steps stay >5?
If AUROC > 0.8 with any decoy resistance, this is a clear win.

**Result:** PENDING

---

### Experiment 2: exp2_light

**Hypothesis:** Keep all 4 reward components but with dramatically reduced penalty weights.
Affinity should dominate, penalties provide gentle guidance.

| Parameter | Value |
|-----------|-------|
| reward_mode | v2_full |
| w_affinity | 1.0 |
| w_decoy | 0.2 |
| w_naturalness | 0.1 |
| w_diversity | 0.05 |
| total_timesteps | 500,000 |
| seed | 42 |

**What to look for:** Slower than v1_ergo_only (~22 s/s due to ESM). If avg steps
stays high and AUROC competitive, we have a balanced reward.

**Result:** PENDING

---

### Experiment 3: exp3_ergo_delta

**Hypothesis:** Use delta reward (per-step credit assignment) instead of terminal-only
reward. Same as v1_ergo_only but with per-step delta. Does per-step credit help?

| Parameter | Value |
|-----------|-------|
| reward_mode | v1_ergo_delta |
| w_affinity | N/A (raw delta, no normalization) |
| w_decoy | N/A (no decoy) |
| w_naturalness | N/A (no naturalness) |
| w_diversity | N/A (no diversity) |
| total_timesteps | 500,000 |
| seed | 42 |

**What to look for:** If AUROC matches v1_ergo_only at 500K steps, delta reward
may enable faster convergence. Compare learning curves.

**First logged step:** Step 10,240 | R=0.001 | Len=4.6 | Ent=5.070

**Result:** PENDING

---

### Experiment 4: exp4_min_steps

**Hypothesis:** Force policy to take at least 3 editing steps before STOP is
allowed. This directly addresses early termination. Mid-range penalty weights.

| Parameter | Value |
|-----------|-------|
| reward_mode | v2_full |
| w_affinity | 1.0 |
| w_decoy | 0.4 |
| w_naturalness | 0.2 |
| w_diversity | 0.1 |
| min_steps | 3 |
| min_steps_penalty | -2.0 |
| total_timesteps | 500,000 |
| seed | 42 |

**What to look for:** Avg steps should be >=3. Does this produce better AUROC
than v2_full by preventing premature STOP?

**Result:** PENDING

---

## Results Summary (to be filled after training)

### Experiment 1: exp1_decoy_only
- **Status:** COMPLETE (500K steps)
- **Mean AUROC:** 0.4898 (vs baseline 0.8075 — **WORSE**)
- **Avg Steps:** ~6.6-7.8 (varies by target)
- **Training Speed:** ~22 s/s (ESM + decoy overhead)
- **Best target:** IVTDFSVIK (0.7471), worst: SPRWYFYYL (0.2575)
- **Target score:** 0.1216 avg, **Decoy score:** 0.1184 avg
- **Notes:** Decoy penalty (w=0.3) with z-normalized reward fails to reach v1_ergo_only affinity levels. Decoy scores are slightly lower than exp3 (suggesting some specificity), but the affinity signal is too weak. The z-score normalization compresses the affinity signal when combined with decoy penalty, even at low weight.

### Experiment 2: exp2_light
- **Status:** COMPLETE (500K steps)
- **Mean AUROC:** 0.4660 (vs baseline 0.8075 — **WORSE**)
- **Avg Steps:** ~5.4 (from training logs)
- **Training Speed:** ~22 s/s (ESM + all scorers overhead)
- **Best target:** IVTDFSVIK (0.8388), worst: SPRWYFYYL (0.2003)
- **Target score:** 0.1446 avg, **Decoy score:** 0.1215 avg
- **Notes:** Very light penalty weights (d=0.2, n=0.1, v=0.05) still fail catastrophically. Even with 10x reduction from v2_full, the z-score normalization compresses the affinity signal too much when combined with ANY penalties. AUROC barely above v1 baseline (0.4538) but far below v1_ergo_only (0.8075).

### Experiment 3: exp3_ergo_delta
- **Status:** COMPLETE (500K steps)
- **Mean AUROC:** 0.5004 (vs baseline 0.8075 — **WORSE**)
- **Avg Steps:** 7.9
- **Training Speed:** ~58 s/s (fastest — no ESM/decoy overhead)
- **Best target:** IVTDFSVIK (0.7862), worst: SPRWYFYYL (0.2323)
- **Target score:** 0.1647 avg, **Decoy score:** 0.1419 avg
- **Notes:** Raw delta reward (no z-norm) produces long editing trajectories but poor binding. Policy makes many edits (7.9 steps) but without normalized signal, it fails to learn strong affinity improvement. Per-step delta credit assignment alone is NOT sufficient — the z-score normalization in v1_ergo_only's terminal reward was crucial.

### Experiment 4: exp4_min_steps
- **Status:** COMPLETE (500K steps)
- **Mean AUROC:** 0.4768 (vs baseline 0.8075 — **WORSE**)
- **Avg Steps:** ~5.0 (from training logs)
- **Training Speed:** ~22 s/s (ESM + all scorers overhead)
- **Best target:** IVTDFSVIK (0.7159), worst: SPRWYFYYL (0.2664)
- **Target score:** 0.1344 avg, **Decoy score:** 0.1284 avg
- **Notes:** Min-steps penalty (min=3, penalty=-2.0) with mid-range weights (d=0.4, n=0.2, v=0.1) also fails. The min-steps constraint prevents early termination but doesn't fix the fundamental problem: z-score normalization + penalties = compressed affinity signal. AUROC slightly better than exp2 but still catastrophically below v1_ergo_only.

---

## Decision Matrix

| Experiment | AUROC | vs Baseline | Avg Steps | Decoy Score | Speed | Scale to 2M? |
|-----------|-------|-------------|-----------|-------------|-------|--------------|
| v1_ergo_only (baseline) | 0.8075 | -- | 8.8 | 0.1202 | 58 s/s | Already done |
| exp1_decoy_only | 0.4898 | -0.3177 | 6.6-7.8 | 0.1184 | 22 s/s | NO |
| exp2_light | 0.4660 | -0.3415 | 5.4 | 0.1215 | 22 s/s | NO |
| exp3_ergo_delta | 0.5004 | -0.3071 | 7.9 | 0.1419 | 58 s/s | NO |
| exp4_min_steps | 0.4768 | -0.3307 | 5.0 | 0.1284 | 22 s/s | NO |

**Winner:** NONE. All 4 experiments failed catastrophically.

**Root cause:** The combination of z-score normalization + ANY penalty weights (even very light ones) compresses the affinity signal too much. The policy cannot learn strong binding improvement when the normalized affinity delta is competing with normalized penalty terms.

**Key insight:** v1_ergo_only works because it uses RAW (unnormalized) ERGO terminal reward. The z-score normalization in the current reward_manager is the problem, not the solution.

---

## Timeline

- 2026-04-11 15:50 — All 4 experiments launched on GPUs 0-3
- 2026-04-11 15:58 — exp3_ergo_delta first training step (fastest, no ESM/decoy overhead)
- Exp3 estimated completion: ~2h (by 18:00)
- Exp1/2/4 estimated completion: ~3-4h (by 19:00-20:00, slower due to ESM/decoy scoring)

---

## Conclusion and Next Steps

**All 4 fast iteration experiments (500K steps each) FAILED to beat the v1_ergo_only baseline of 0.8075 AUROC.**

### What we learned:

1. **Z-score normalization is the problem**: Combining z-normalized affinity with z-normalized penalties compresses the affinity signal. Even with very light penalty weights (exp2: d=0.2, n=0.1, v=0.05), the policy cannot learn strong binding.

2. **Raw delta reward alone doesn't work**: exp3_ergo_delta used raw (unnormalized) per-step delta reward and achieved 0.5004 AUROC. Without normalization, the policy makes many edits but doesn't improve binding effectively.

3. **v1_ergo_only's success comes from RAW terminal reward**: It uses the raw ERGO score (0-1 range) as terminal reward with NO normalization and NO penalties. This gives a clear, strong signal for binding improvement.

4. **Min-steps penalty doesn't fix the core issue**: exp4 forced at least 3 steps but still failed (0.4768 AUROC). The problem isn't early termination — it's signal compression.

### Why v2_full (2M steps, AUROC 0.5840) also failed:

v2_full uses the same z-score normalization with aggressive penalty weights (d=0.8, n=0.5, v=0.2). It achieves better specificity than v1 baseline (0.4538) but sacrifices binding affinity catastrophically compared to v1_ergo_only (0.8075).

### Recommended next approach:

**Option A: Raw reward with light penalties (no normalization)**
- Use raw ERGO score (0-1) as affinity reward
- Add small ABSOLUTE penalties (not z-normalized):
  - Decoy: -0.05 * decoy_score
  - Naturalness: -0.02 * nat_score
  - Diversity: -0.01 * div_score
- Total = raw_affinity - 0.05*decoy - 0.02*nat - 0.01*div
- This preserves the strong affinity signal while gently discouraging bad behavior

**Option B: Separate affinity and specificity objectives**
- Phase 1: Train with raw ERGO only (reproduce v1_ergo_only)
- Phase 2: Fine-tune with light decoy penalty added
- This ensures we don't lose binding strength while adding specificity

**Option C: Reward shaping with affinity threshold**
- Only apply penalties when affinity > threshold (e.g., 0.5)
- Below threshold: pure affinity signal
- Above threshold: affinity + penalties
- This ensures the policy learns binding first, specificity second

**Immediate action:** The fast iteration approach has exhausted the "reduce penalty weights" strategy. We need a fundamentally different reward formulation. Recommend implementing Option A as the next experiment.

