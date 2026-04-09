# Review 3 — Decoy Specificity Evaluation: Status & Next Step

**Date:** 2026-04-09
**Reviewer:** Claude (Opus 4.6)
**Scope:** `progress_3.md` results + `optimization-decoy-eval` branch fixes
+ all code added in this review-3 follow-up commit
**Audience:** the GPU server, which will read this file plus
`evaluation/REVIEW3_FOLLOWUP_README.md` plus the new code

---

## 0. TL;DR

* `progress_3.md` showed mean AUROC = 0.4538 — **below random** — for
  TCRPPO-generated TCRs against pMHC decoys. That number alone is alarming
  but **not yet diagnostic**: we don't know whether to blame ERGO or PPO.
* This commit lands **all bug fixes** previously isolated on
  `optimization-decoy-eval` (the dynamic-batch-size fix in `ERGO_models.py`,
  the inference batch-size bump in `reward.py`, and the GPU-resident MC
  dropout in `ergo_uncertainty.py`) directly into `main`.
* This commit also lands **the missing experimental control**: a
  random-TCR null baseline (`eval_decoy_random_baseline.py`), a
  side-by-side comparison script (`compare_trained_vs_random.py`), and a
  per-tier AUROC breakdown inside `eval_decoy_metrics.py`.
* It also fixes the **checkpoint loss** problem (only 1M and 10M models
  survived) by adding a `MilestoneCheckpointCallback` to `code/tcr_env.py`
  for future training runs.
* **The single thing the server needs to do this round** is run the random
  baseline and the comparison script. The verdict at the bottom of that
  comparison output decides what `review_4.md` will recommend. Until that
  number exists, every downstream decision is a coin flip.

---

## 1. What was already done before this commit

### 1.1 The decoy evaluation suite is wired up and working

`progress_3.md` confirms the full pipeline runs end-to-end against
1.66M (TCR, peptide) pairs:

* Loads 12 target peptides + ~3000 decoys per target across tiers A/B/C/D.
* Runs the trained 10M-step PPO agent to generate 50 TCRs per target.
* Scores each (TCR, peptide) pair with MC Dropout ERGO (n=20).
* Produces per-target tables, evidence-level breakdown, top-K off-target
  hits, and uncertainty diagnostics.

This part of the work is **complete and trustworthy**. No findings here have
been falsified by anything we've done since.

### 1.2 Code changes that landed on the `optimization-decoy-eval` branch

| File | Change | Status before this commit |
|------|--------|---------------------------|
| `code/ERGO/ERGO_models.py` | dynamic batch size in `AutoencoderLSTMClassifier.forward` | only on the optimisation branch |
| `code/reward.py` | inference batch size 1 → 4096 | only on the optimisation branch |
| `evaluation/ergo_uncertainty.py` | GPU-resident MC dropout, ~400× speedup | only on the optimisation branch |
| `evaluation/eval_decoy.py` | per-target subdir traversal for decoy A/B | already on `main` |

The first three items were stranded on the optimisation branch. As a
result, anyone running the eval from `main` would still hit the original
shape mismatch and the original 5/sec MC dropout throughput. **This commit
ports them onto `main` so the branch is no longer needed.**

### 1.3 Findings from `progress_3.md` that this commit does NOT contradict

* Mean AUROC across 12 targets = **0.4538** (10M model, ae_mcpas).
* SLYNTVATL is the only target with strong specificity (AUROC = 0.8776).
* The Pearson correlation between ERGO mean and ERGO MC-dropout std across
  all rows is **r = 0.7960** — high-score predictions also have high
  uncertainty, which is a textbook overconfidence-in-OOD signature.
* Top-15 off-target hits are dominated by GILGFVFTL with eight Tier-A
  (1-2 aa mutant) decoys scoring > 0.95.

Each of these is a real signal. None of them is yet *interpretable* without
the null baseline added in this commit.

---

## 2. What I changed in this commit

### 2.1 Bug fixes ported from `optimization-decoy-eval` to `main`

* `code/ERGO/ERGO_models.py:163-180` — `padded_tcrs.size(0)` instead of
  `self.batch_size`. This is a true bugfix, not just an optimisation: the
  model crashes with the original code whenever an inference batch is a
  different size, including during MC dropout chunking and re-evaluation
  of any earlier checkpoint. This is the root cause of the
  `progress_3.md` 1M-model `(9) → (514)` shape mismatch, not "1M generates
  bad TCRs" as that report speculated.
* `code/reward.py:__get_ergo_preds` — inference batch size from 1 to
  `min(len(tcrs), 4096)`. Per-pair output is mathematically unchanged;
  throughput improves dramatically.
* `evaluation/ergo_uncertainty.py` — full rewrite of the inner loop. The
  GPU now holds the encoded batches across all `n_samples` forward passes.
  All debug `time.time()` lines and unused intermediates from the
  optimisation branch have been removed. Public API
  (`mc_dropout_predict`, `mc_dropout_predict_chunked`) is unchanged.

### 2.2 New code (action items from review-3 round 1)

| Action item | New file | Status |
|---|---|---|
| 3 (training checkpoints) | `code/tcr_env.py` `MilestoneCheckpointCallback` + `--save_freq` / `--checkpoint_milestones` flags | ready, not yet exercised on the server |
| 4 (random TCR baseline) | `evaluation/eval_decoy_random_baseline.py` | ready |
| 4 (comparison) | `evaluation/compare_trained_vs_random.py` | ready |
| 5 (per-tier AUROC) | `evaluation/eval_decoy_metrics.py:per_tier_auroc()` + `print_per_tier_table()` + CSV export | ready |
| 6 (cross-scorer sanity) | (no new code; just pass `--ergo_model ae_vdjdb` to existing scripts) | ready |
| 8 (move test_fast/profile_ae) | (intentionally not pulled across from the branch — they're one-off benchmarks) | done |

Action item 7 (post-hoc specificity filtering) was explicitly out of scope
this round per user instruction. The goal of this round is to characterise
whether TCRPPO has specificity, not to patch it.

### 2.3 Sanity-checks I ran locally

* All seven modified/new Python files parse with `ast.parse`.
* `eval_decoy_random_baseline.py --help`, `eval_decoy_metrics.py --help`,
  and `compare_trained_vs_random.py --help` all run without import errors.
* `per_tier_auroc()` was unit-tested on a synthetic 6-row CSV
  (`GILGFVFTL` × tiers A and C) and returned AUROC = 1.0 for both, as
  expected for fully separable target/decoy.
* I did **not** run end-to-end on real data — that requires the GPU and
  is the server's job.

---

## 3. The critical next experiment

There is exactly one experiment that needs to run this round, and the
verdict it produces dictates everything that follows.

### 3.1 The question

The progress_3 number is "mean AUROC = 0.4538, below random". That number
is consistent with at least three different worlds:

| World | Mechanism | What to fix |
|-------|-----------|-------------|
| **A** | ERGO is similarity-driven and gives ≈identical scores to similar peptides regardless of which TCR you pass in | Swap ERGO for a stronger binding predictor (pMTnet, NetTCR-2.x, TITAN). Reward redesign won't help. |
| **B** | PPO collapses TCRs onto a flat region of ERGO's decision surface, where neither target nor decoy can be distinguished | Reward redesign (contrastive / decoy-aware / multi-objective). The scorer is fine; the optimisation is broken. |
| **C** | Some mix of the two | Both fixes contribute; cheapest first is reward shaping. |

A single number distinguishes these worlds: **the AUROC of random
(untrained) TCRs** on the same target/decoy/scorer setup, compared
target-by-target against the trained run.

* If random TCRs give the same AUROC ≈ 0.45 → World **A**.
* If random TCRs give AUROC > 0.5 and trained gives 0.45 → World **B**.
* If both are around 0.5 with random ≥ trained → World **B/C**.

### 3.2 The command sequence

This is the entire experimental protocol for this round, and it lives in
`evaluation/REVIEW3_FOLLOWUP_README.md` for the server to follow.

```bash
cd /share/liuyutian/TCRPPO
export PYTHONPATH=/share/liuyutian/TCRPPO/stable_baselines3:$PYTHONPATH

# 1. Random TCR baseline (matching trained run config)
python evaluation/eval_decoy_random_baseline.py \
    --ergo_model ae_mcpas \
    --mode pool \
    --num_tcrs_per_target 50 \
    --n_mc_samples 20 \
    --out_csv evaluation/results/decoy/eval_decoy_random_pool_ae_mcpas.csv

# 2. The verdict
python evaluation/compare_trained_vs_random.py \
    --trained_csv evaluation/results/decoy/eval_decoy_ae_mcpas_v2.csv \
    --random_csv  evaluation/results/decoy/eval_decoy_random_pool_ae_mcpas.csv

# 3. Per-tier breakdown for both runs (now produced by the updated metrics script)
python evaluation/eval_decoy_metrics.py \
    --csv evaluation/results/decoy/eval_decoy_ae_mcpas_v2.csv \
    --out_dir evaluation/results/decoy/metrics_trained/
python evaluation/eval_decoy_metrics.py \
    --csv evaluation/results/decoy/eval_decoy_random_pool_ae_mcpas.csv \
    --out_dir evaluation/results/decoy/metrics_random/
```

The entire round depends on the **single `Mean Δ` line** at the bottom of
step 2's output and the **per-tier AUROC table** from step 3.

### 3.3 What I'd like to see in `progress_4.md`

* The verdict text from `compare_trained_vs_random.py` and the per-target
  table that produced it.
* The per-tier AUROC table for both the trained run and the random run.
  Especially: the **Tier C** AUROC of the random baseline. If even a
  random TCR can't separate the target from totally unrelated peptides
  (Tier C), then ERGO has essentially no discriminative power on these
  specific targets and World A is confirmed.
* (Stretch) the same comparison with `--ergo_model ae_vdjdb`. If both
  ERGO checkpoints give the same verdict, the conclusion is robust to
  scorer choice. If they disagree, we have a much more interesting story.
* (Stretch, if time permits) re-running the eval on the 1M checkpoint
  with the bugfix in place, for an AUROC vs training-step comparison.

---

## 4. Other things deferred to future rounds

* **Reward redesign** (any flavour: contrastive, decoy-aware, multi-
  objective). Explicitly deferred to a much later round per user
  instruction. Worth doing only if the verdict in step 3 is World B or C.
* **Post-hoc specificity filtering** (action item 7 from review-3 round 1).
  Explicitly out of scope. The goal this round is to *understand* the
  failure, not to patch it.
* **Re-running training to recover the missing checkpoints.** The
  `MilestoneCheckpointCallback` is in place so future runs won't lose
  intermediate checkpoints, but I don't recommend retraining the existing
  10M run just to recover them. The 1M file is still on disk and is now
  re-evaluable thanks to the `ERGO_models.py` fix; together with the 10M
  run that gives a two-point training trajectory which is enough for a
  trend signal.

---

## 5. Files the server needs to read

In order:

1. `review_3.md` (this file) — the why
2. `evaluation/REVIEW3_FOLLOWUP_README.md` — the how
3. `evaluation/eval_decoy_random_baseline.py` — the random baseline script
4. `evaluation/compare_trained_vs_random.py` — the verdict script
5. `evaluation/eval_decoy_metrics.py` — now includes per-tier AUROC

The expected output of the round is `progress_4.md`, structured around the
verdict from `compare_trained_vs_random.py` and the per-tier AUROC tables.
That document will tell us which of the three worlds we live in, and that
in turn determines what `review_4.md` recommends.
