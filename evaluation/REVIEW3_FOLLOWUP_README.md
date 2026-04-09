# Review-3 follow-up — null-baseline & checkpoint evaluation

This README documents the code added in response to `review_3.md`. It covers:

1. **Bug fixes** that were applied to `main` (the `optimization-decoy-eval`
   branch is no longer needed — everything is in `main` now).
2. **`eval_decoy_random_baseline.py`** — random-TCR null hypothesis baseline.
3. **`compare_trained_vs_random.py`** — side-by-side AUROC diff.
4. **Per-tier AUROC** added to `eval_decoy_metrics.py`.
5. **`MilestoneCheckpointCallback`** added to `code/tcr_env.py` so we get
   permanent named checkpoints during training.

The over-arching goal of this round is to **answer one question** with high
confidence:

> Is TCRPPO's failure to discriminate target from decoy pMHCs a property of
> the trained agent, or is it a property of the underlying ERGO scorer?

The answer determines what to fix next. If ERGO is the bottleneck, redesigning
the PPO reward will not help — we need a different binding predictor. If PPO
training is the bottleneck, the reward function is the right place to
intervene. Until we know which is true, picking either intervention is a coin
flip.

---

## 1. Bug fixes already applied (no action needed)

| File | Fix |
|------|-----|
| `code/ERGO/ERGO_models.py` | `concat = padded_tcrs.view(self.batch_size, ...)` → `padded_tcrs.view(padded_tcrs.size(0), ...)`. The hard-coded `self.batch_size` crashes whenever the actual incoming batch is a different size — exactly what happens during MC dropout chunking and when re-evaluating an early checkpoint. **This is a real bugfix, not just an optimisation.** Training behaviour is unchanged because at training time `padded_tcrs.size(0) == self.batch_size` by construction. |
| `code/reward.py` | Inference batch size 1 → `min(len(tcrs), 4096)`. Pure throughput change; numerical outputs are identical. |
| `evaluation/ergo_uncertainty.py` | Batches built once and pre-loaded to GPU; the inner MC-sample loop just runs forward passes on already-resident tensors. Switched from per-batch `preds.extend([t[0] for t in probs.cpu().data.tolist()])` to a single `torch.cat → .cpu()` at the end. Replaced the fragile trailing-batch trim logic with an explicit `expected_n` truncation. ~400× speedup. Debug `time` / `t0`/`t1` lines from the optimisation branch were removed. |

The `optimization-decoy-eval` branch is now redundant — all of its
functionally-meaningful changes are on `main`. The branch's `test_fast.py`,
`profile_ae.py`, and `run_eval_pipeline.sh` were one-off benchmarks that don't
belong in the long-lived tree, so they were intentionally not pulled across.

---

## 2. The new evaluation pipeline

The decoy evaluation is now a 4-step pipeline. Steps 1-3 are unchanged from
the previous round; **step 4 is new and is the most important step in this
round.**

```
            ┌──────────────────────────────────┐
            │  Step 1: trained PPO eval        │
            │  evaluation/eval_decoy.py        │
            │  → eval_decoy_<ergo>.csv         │
            └──────────────────┬───────────────┘
                               │
                               ▼
            ┌──────────────────────────────────┐
            │  Step 2: metrics (NOW per-tier)  │
            │  evaluation/eval_decoy_metrics.py│
            │  → console + per_*.csv exports   │
            └──────────────────────────────────┘

            ┌──────────────────────────────────┐
            │  Step 3: plots                   │
            │  evaluation/eval_decoy_visualize │
            │  → figures/decoy/*.png           │
            └──────────────────────────────────┘

            ┌──────────────────────────────────┐
            │  Step 4: NULL BASELINE  (NEW)    │
            │  evaluation/eval_decoy_random_   │
            │     baseline.py                  │
            │  → eval_decoy_random_<mode>_     │
            │     <ergo>.csv                   │
            └──────────────────┬───────────────┘
                               │
                               ▼
            ┌──────────────────────────────────┐
            │  Step 5: trained vs random       │
            │  evaluation/compare_trained_vs_  │
            │     random.py                    │
            │  → verdict                       │
            └──────────────────────────────────┘
```

### 2.1 Re-run the trained eval (only if not already done)

```bash
cd /share/liuyutian/TCRPPO
export PYTHONPATH=/share/liuyutian/TCRPPO/stable_baselines3:$PYTHONPATH

python evaluation/eval_decoy.py \
    --ergo_model ae_mcpas \
    --num_tcrs_per_target 50 \
    --n_mc_samples 20 \
    --num_envs 8 \
    --out_csv evaluation/results/decoy/eval_decoy_ae_mcpas.csv
```

If you already have `eval_decoy_ae_mcpas_v2.csv` from progress_3, you can
skip this step and use that file in the comparison below.

### 2.2 Run the random-TCR null baseline

```bash
python evaluation/eval_decoy_random_baseline.py \
    --ergo_model ae_mcpas \
    --mode pool \
    --num_tcrs_per_target 50 \
    --n_mc_samples 20 \
    --out_csv evaluation/results/decoy/eval_decoy_random_pool_ae_mcpas.csv
```

The `pool` mode samples real human TCRs from
`data/tcrdb/test_uniq_tcr_seqs.txt`. These are biologically valid CDR3βs that
the trained agent has never seen — the cleanest "untrained but plausible"
baseline.

For a model-free baseline, run again with `--mode synthetic`. Synthetic TCRs
are uniform-random AAs and ignore CDR3 conserved residues, so they are not
biologically realistic, but they form a true zero-prior null distribution.
Comparing pool vs synthetic isolates how much of ERGO's apparent
discrimination is just sequence-statistics.

### 2.3 Run per-target metrics on each CSV separately

```bash
python evaluation/eval_decoy_metrics.py \
    --csv evaluation/results/decoy/eval_decoy_ae_mcpas.csv \
    --out_dir evaluation/results/decoy/metrics_trained/

python evaluation/eval_decoy_metrics.py \
    --csv evaluation/results/decoy/eval_decoy_random_pool_ae_mcpas.csv \
    --out_dir evaluation/results/decoy/metrics_random/
```

`eval_decoy_metrics.py` now also prints a **PER-TIER AUROC BREAKDOWN** table
and exports `per_tier_auroc.csv` to `--out_dir`. Read the table this way:

* **Tier C (sequence-unrelated decoys) AUROC ≪ 1.0** → ERGO can't even
  separate the target from random irrelevant peptides. Major problem.
* **Tier C high but Tier A/B low** → ERGO has *some* binding signal but is
  fooled by sequence neighbours. The expected failure mode for a similarity-
  driven scorer.
* **All tiers near 0.5** → No discrimination at all. Most likely cause is the
  trained PPO collapsing onto a region where ERGO has no opinion.

### 2.4 Direct comparison and verdict

```bash
python evaluation/compare_trained_vs_random.py \
    --trained_csv evaluation/results/decoy/eval_decoy_ae_mcpas.csv \
    --random_csv  evaluation/results/decoy/eval_decoy_random_pool_ae_mcpas.csv
```

This prints a per-target table with three columns of interest:

| Column | What it means |
|--------|---------------|
| `AUROC_RL` | trained PPO's AUROC for this target (the number from progress_3) |
| `AUROC_RND` | random-TCR baseline's AUROC for the same target |
| `Delta` | RL minus random — **the actual experimental result** |

The script ends with a verdict text. The three possible verdicts and their
implications are:

| Mean Δ | Verdict | Implication |
|--------|---------|-------------|
| ≈ 0 (within ±0.02) | **ERGO is the bottleneck.** | Reward redesign is wasted effort. The fix is to swap ERGO for a stronger binding predictor (pMTnet, NetTCR-2.x, TITAN). |
| Significantly < 0 | **PPO actively destroys specificity.** | The reward function (binding-only) is collapsing TCRs into ERGO's flat region. Reward redesign (contrastive, multi-objective, decoy-aware) is the right intervention. |
| Significantly > 0 | **PPO does real work; both contribute.** | Cheapest first improvement is reward shaping; longer-term, also evaluate scorer choice. |

These are the three branches of the next-step decision tree. Until we have
this number, we shouldn't pick a branch.

---

## 3. Checkpoint capture during training

`code/tcr_env.py` now exposes two new CLI flags and adds a custom callback so
the checkpoint problem from progress_3 (only the 1M and 10M checkpoints
survived) can't recur:

| Flag | Default | Purpose |
|------|---------|---------|
| `--save_freq` | `50000` | Existing SB3 `CheckpointCallback` interval. Was hard-coded; now configurable. |
| `--checkpoint_milestones` | `1000000,2000000,5000000,10000000` | Comma-separated list of timesteps at which to **also** write a permanently named checkpoint `rl_model_milestone_<step>_steps.zip`. These files are intended to survive cleanup. Set to empty string to disable. |

Two callbacks now run together via SB3's `CallbackList`:

```python
checkpoint_callback = CheckpointCallback(
    save_freq=args.save_freq, save_path=path + '/', name_prefix='rl_model')
milestone_callback  = MilestoneCheckpointCallback(
    milestones=[1_000_000, 2_000_000, 5_000_000, 10_000_000],
    save_path=path + '/')
model.learn(total_timesteps=args.steps,
            callback=CallbackList([checkpoint_callback, milestone_callback]))
```

The milestone callback writes files with a distinct prefix
(`rl_model_milestone_*`) so they're easy to spot in the directory and easy to
exclude from cleanup scripts. It also prints a one-line `[milestone-ckpt]
saved ...` log line on every save so the run log shows exactly which
checkpoints made it to disk.

### 3.1 Use case: re-running training to recover the lost checkpoints

If you decide to retrain to recover the missing checkpoints:

```bash
python code/tcr_env.py \
    --num_envs 20 \
    --ergo_model ./code/ERGO/models/ae_mcpas1.pt \
    --peptide_path ./data/test_peptides/ae_mcpas_test_peptides.txt \
    --bad_ratio 0.0 \
    --hidden_dim 256 \
    --latent_dim 128 \
    --gamma 0.90 \
    --save_freq 100000 \
    --checkpoint_milestones 1000000,2000000,5000000,10000000 \
    --path ./output/ae_mcpas_
```

`--save_freq 100000` writes a regular checkpoint every 100K steps (100 files
total at 10M steps × 23MB ≈ 2.3GB) — half the disk usage of the default
50000. The four milestone files are extra and tagged so you know not to
delete them.

### 3.2 Use case: evaluating just the 1M and 10M checkpoints we already have

If retraining is too expensive, this is the fallback. The original 1M
checkpoint failed during MC dropout scoring with a shape mismatch — that's
the bug fixed in `ERGO_models.py` (see section 1). With the fix in place,
1M should now work. Run:

```bash
python evaluation/eval_decoy.py \
    --model_path output/ae_mcpas_mcpas_0.5_0.0_0.9_256_None/rl_model_1000000_steps \
    --ergo_model ae_mcpas \
    --out_csv evaluation/results/decoy/eval_decoy_ae_mcpas_step_1M.csv

python evaluation/eval_decoy.py \
    --model_path output/ae_mcpas_mcpas_0.5_0.0_0.9_256_None/rl_model_10000000_steps \
    --ergo_model ae_mcpas \
    --out_csv evaluation/results/decoy/eval_decoy_ae_mcpas_step_10M.csv
```

Then run `eval_decoy_metrics.py` on each to get the per-tier AUROC at both
training stages. A meaningful AUROC change between 1M and 10M would mean
training is doing something. No change means training is irrelevant for
specificity (which would be consistent with the "ERGO is the bottleneck"
hypothesis).

---

## 4. Recommended order of operations on the server

Assuming you already have `evaluation/results/decoy/eval_decoy_ae_mcpas_v2.csv`
from progress_3:

```bash
cd /share/liuyutian/TCRPPO
export PYTHONPATH=/share/liuyutian/TCRPPO/stable_baselines3:$PYTHONPATH

# 1. Random TCR baseline (pool mode) — same num_tcrs/n_mc_samples as the trained run
python evaluation/eval_decoy_random_baseline.py \
    --ergo_model ae_mcpas \
    --mode pool \
    --num_tcrs_per_target 50 \
    --n_mc_samples 20 \
    --out_csv evaluation/results/decoy/eval_decoy_random_pool_ae_mcpas.csv

# 2. Side-by-side comparison + verdict
python evaluation/compare_trained_vs_random.py \
    --trained_csv evaluation/results/decoy/eval_decoy_ae_mcpas_v2.csv \
    --random_csv  evaluation/results/decoy/eval_decoy_random_pool_ae_mcpas.csv

# 3. Per-tier breakdown for the trained run (to see WHERE it fails)
python evaluation/eval_decoy_metrics.py \
    --csv evaluation/results/decoy/eval_decoy_ae_mcpas_v2.csv \
    --out_dir evaluation/results/decoy/metrics_trained/

# 4. Per-tier breakdown for the random baseline
python evaluation/eval_decoy_metrics.py \
    --csv evaluation/results/decoy/eval_decoy_random_pool_ae_mcpas.csv \
    --out_dir evaluation/results/decoy/metrics_random/

# 5. (Optional) Re-evaluate the 1M checkpoint with the bugfix in place
python evaluation/eval_decoy.py \
    --model_path output/ae_mcpas_mcpas_0.5_0.0_0.9_256_None/rl_model_1000000_steps \
    --ergo_model ae_mcpas \
    --out_csv evaluation/results/decoy/eval_decoy_ae_mcpas_step_1M.csv

python evaluation/eval_decoy_metrics.py \
    --csv evaluation/results/decoy/eval_decoy_ae_mcpas_step_1M.csv

# 6. (Optional) Cross-scorer sanity check
python evaluation/eval_decoy.py \
    --ergo_model ae_vdjdb \
    --out_csv evaluation/results/decoy/eval_decoy_ae_vdjdb.csv
python evaluation/eval_decoy_random_baseline.py \
    --ergo_model ae_vdjdb \
    --out_csv evaluation/results/decoy/eval_decoy_random_pool_ae_vdjdb.csv
python evaluation/compare_trained_vs_random.py \
    --trained_csv evaluation/results/decoy/eval_decoy_ae_vdjdb.csv \
    --random_csv  evaluation/results/decoy/eval_decoy_random_pool_ae_vdjdb.csv
```

The most important output of this entire round is the **single `Mean Δ`
number** at the bottom of step 2. That number determines what `review_4.md`
will recommend.

---

## 5. File index — what's in this commit

| Path | Type | Purpose |
|------|------|---------|
| `code/ERGO/ERGO_models.py` | bugfix | Dynamic batch size in `AutoencoderLSTMClassifier.forward` |
| `code/reward.py` | perf | ERGO inference batch size 1 → 4096 |
| `code/tcr_env.py` | new feature | `--save_freq`, `--checkpoint_milestones`, `MilestoneCheckpointCallback` |
| `evaluation/ergo_uncertainty.py` | rewrite | GPU-resident MC dropout, 400× speedup |
| `evaluation/eval_decoy_random_baseline.py` | NEW | Random-TCR null hypothesis baseline |
| `evaluation/compare_trained_vs_random.py` | NEW | AUROC diff + verdict text |
| `evaluation/eval_decoy_metrics.py` | extension | New `per_tier_auroc()` + `print_per_tier_table()` and CSV export |
| `evaluation/REVIEW3_FOLLOWUP_README.md` | NEW | This file |

Out of scope (per user instruction):

* Post-hoc specificity filtering (action item 7 in `review_3.md`) — the goal
  this round is to *characterise* whether TCRPPO has specificity, not to
  patch it.
* Reward redesign / contrastive training — explicitly deferred to a much
  later round.
