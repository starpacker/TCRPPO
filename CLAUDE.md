# CLAUDE.md — TCRPPO v2 Autonomous Implementation

**This file governs the entire v2 implementation session. Read it completely
before writing a single line of code. Every instruction is mandatory.**

---

## 0. Mission

You are building **TCRPPO v2** — a reinforcement-learning pipeline that designs
TCRs with **both** high on-target binding affinity **and** specificity (low
cross-reactivity against decoy/self peptides). The complete design specification
is at `docs/2026-04-09-tcrppo-v2-design.md`. That document is the **single
source of truth** for architecture, hyperparameters, and module interfaces.

### Success criteria (non-negotiable)

1. **Mean AUROC > 0.65** across all 12 McPAS targets on decoy specificity eval
   (v1 baseline: 0.45 — worse than random).
2. **Mean target affinity score >= v1 level** (no regression on binding).
3. All components pass unit tests and integration tests before declaring a phase complete.
4. Results are reproducible — random seeds, configs, and checkpoints are saved.

### V1 code reuse strategy (selective copy)

v2 does NOT start from scratch. The following v1 components are **copied into
this repo** and reused directly:

| Copied to | Source | How to use |
|-----------|--------|------------|
| `tcrppo_v2/ERGO/` | `TCRPPO/code/ERGO/` | ERGO model loading, inference, pretrained weights — used by `affinity_ergo.py` scorer. Adapt for Python 3.10 if needed, but do NOT rewrite the core logic. |
| `evaluation/` | `TCRPPO/evaluation/eval_decoy*.py`, `ergo_uncertainty.py` | Decoy specificity evaluation pipeline + MC Dropout. Use the SAME eval code for both v1 and v2 to ensure fair comparison. Adapt paths/imports as needed. |
| `ref/` | `TCRPPO/code/data_utils.py`, `config.py`, `reward.py` | Reference files. Read these to understand v1 patterns, then extract what you need into `tcrppo_v2/utils/`. Do NOT import from `ref/` at runtime. |

The following must be **written from scratch** (v1 architecture is incompatible):

- `env.py` — indel action space, ESM-2 state, per-step delta reward
- `policy.py` — 3-head autoregressive actor-critic with action masking
- `ppo_trainer.py` — custom PPO (SB3 cannot handle autoregressive masking)
- `reward_manager.py` — 4-component reward with running normalization
- `scorers/decoy.py`, `scorers/naturalness.py`, `scorers/diversity.py` — new components

**Runtime imports**: v2 code may import from `tcrppo_v2/ERGO/` (it is part of
this repo now) but must NEVER import from `/share/liuyutian/TCRPPO/code/`.

### What you are NOT allowed to do

- **Do NOT stop, pause, or ask for permission** between phases. Run continuously
  until all phases are complete. If context grows too large, compact and continue.
- **Do NOT fake results.** Every metric must come from actual model runs with
  actual data. If a test fails, fix it — do not skip it.
- **Do NOT deviate from the design spec** without documenting why in
  `progress_v2.md` and updating the spec accordingly.
- **Do NOT import from `/share/liuyutian/TCRPPO/code/` at runtime.** All reused
  code has been copied locally. The v1 repo is READ-ONLY reference.

---

## 1. Environment and Paths

### 1.1 Hardware

- **GPU**: 5x NVIDIA A800-SXM4-80GB (use `CUDA_VISIBLE_DEVICES` to select)
- **Python**: Use Python 3.10+ (system python or create a new conda env)
- **Conda env**: Create `tcrppo_v2` — do NOT use the `tcrppo` env (Python 3.6)

### 1.2 Critical Paths

```
/share/liuyutian/tcrppo_v2/             # THIS project root
/share/liuyutian/TCRPPO/                # v1 baseline (READ-ONLY reference)
/share/liuyutian/pMHC_decoy_library/    # Decoy library (READ-ONLY)
  data/
    candidate_targets.json
    decoy_a/<TARGET>/...     # Tier A: 1-2 AA point mutants
    decoy_b/<TARGET>/...     # Tier B: 2-3 AA mutants
    decoy_c/decoy_library.json  # Tier C: 1900 unrelated peptides
    decoy_d/<TARGET>/...     # Tier D: known binders from VDJdb/IEDB

/share/liuyutian/TCRPPO/code/ERGO/models/     # ERGO pretrained weights
/share/liuyutian/TCRPPO/data/tcrdb/           # TCRdb CDR3beta sequences
/share/liuyutian/TCRPPO/data/test_peptides/   # Test peptide files
/share/liuyutian/TCRPPO/output/               # v1 checkpoints (for comparison)
```

### 1.3 Conda Environment Setup

```bash
conda create -n tcrppo_v2 python=3.10 -y
conda activate tcrppo_v2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install fair-esm transformers numpy scipy scikit-learn pandas matplotlib \
    pyyaml tensorboard wandb tqdm pytest
```

Verify ESM-2 loads before writing any code:
```python
import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
print("ESM-2 loaded successfully")
```

---

## 2. Implementation Phases (Execute Sequentially, Do NOT Skip)

### Phase 0: Project Scaffolding and Environment Validation

**Estimated: ~30 min**

Pre-copied files already in place:
- `tcrppo_v2/ERGO/` — ERGO model code + pretrained weights (from v1)
- `evaluation/` — decoy eval pipeline (from v1)
- `ref/` — v1 utility files for reference

Steps:
1. Create remaining directory layout per design spec section 2.1 (scorers/, data/, utils/, configs/, tests/, output/, results/, figures/)
2. Set up conda env `tcrppo_v2` with all dependencies (Python 3.10, torch, fair-esm, etc.)
3. Verify ESM-2 loads on GPU
4. Verify ERGO weights load from local `tcrppo_v2/ERGO/models/`
5. Adapt ERGO code for Python 3.10 if needed (v1 was Python 3.6)
6. Verify decoy library is readable at `/share/liuyutian/pMHC_decoy_library/`
7. Verify TCRdb data is accessible at `/share/liuyutian/TCRPPO/data/tcrdb/`
8. Extract reusable constants from `ref/config.py` and `ref/data_utils.py` into `tcrppo_v2/utils/constants.py` and `tcrppo_v2/utils/encoding.py`
9. Write `configs/default.yaml` with all hyperparameters from design spec section 10
10. Initialize git repo, make initial commit
11. **Checkpoint**: `progress_v2.md` Phase 0 entry, git commit + push

### Phase 1: Scorer Modules

**Estimated: ~2 hours**

Build and unit-test each scorer independently:

1. `scorers/base.py` — Abstract `BaseScorer` with `score() -> (float, float)`
2. `scorers/affinity_ergo.py` — ERGO binding scorer with MC Dropout (N=10)
   - Wrap the copied `tcrppo_v2/ERGO/` code + `evaluation/ergo_uncertainty.py` MC Dropout logic
   - Model weights at `tcrppo_v2/ERGO/models/ae_mcpas1.pt` (already copied locally)
   - Reference `ref/reward.py` for how v1 loaded and called ERGO
   - Unit test: score known binders vs random sequences -> binders score higher
3. `scorers/decoy.py` — LogSumExp contrastive penalty
   - Loads decoys from `/share/liuyutian/pMHC_decoy_library/`
   - Tier-weighted sampling (A:3, B:3, D:2, C:1)
   - Unit test: TCR similar to target peptide should have lower penalty than universal binder
4. `scorers/naturalness.py` — ESM perplexity with CDR3 z-score
   - Offline compute mean/std perplexity over TCRdb CDR3beta sample (10K seqs)
   - Save stats to `data/cdr3_ppl_stats.json`
   - Unit test: real CDR3 sequences score better than random AA strings
5. `scorers/diversity.py` — Recent-buffer similarity penalty
   - Unit test: submitting identical sequences triggers penalty
6. `reward_manager.py` — Combine all scorers with running normalization
   - Unit test: raw scores are z-normalized after warmup period
7. **Checkpoint**: All scorer unit tests pass, `progress_v2.md` Phase 1 entry, git commit + push

### Phase 2: Data Pipeline

**Estimated: ~1 hour**

1. `data/pmhc_loader.py` — Load target peptides + HLA pseudosequences
   - Parse `candidate_targets.json` from decoy library
   - HLA pseudosequence encoding (34 AAs, NetMHC-style)
2. `data/tcr_pool.py` — TCRdb loading + curriculum sampler (L0/L1/L2)
   - L0: VDJdb known binders with 3-5 random mutations
   - L1: Pre-compute top-500 TCRdb seqs per target by ERGO score (one-time offline)
   - L2: Random TCRdb sequences
   - Curriculum schedule from design spec section 6.2
3. `data/decoy_sampler.py` — Tiered decoy sampling with unlock schedule
   - Phase-aware: only sample from unlocked tiers based on training step
4. `utils/esm_cache.py` — Frozen ESM-2 inference with per-sequence caching
   - Cache pMHC embeddings (computed once per target)
   - Cache TCR embeddings with LRU eviction
5. **Checkpoint**: Data pipeline tests pass, L1 seeds generated, `progress_v2.md` Phase 2 entry, git commit + push

### Phase 3: Environment

**Estimated: ~2 hours**

1. `env.py` — Gym environment with:
   - Two-head autoregressive action space (op_type, position, token)
   - Action masking (length bounds, PAD positions, no step-0 STOP)
   - SUB/INS/DEL/STOP sequence editing
   - ESM-2 state encoding (TCR re-encoded per step, pMHC cached)
   - Per-step delta reward from RewardManager
   - Curriculum-aware reset (L0/L1/L2 sampling)
2. Integration test: Run 100 random episodes, verify:
   - Sequence lengths stay in [8, 27]
   - STOP terminates episodes
   - State dimensions are consistent
   - Rewards are finite
3. **Checkpoint**: Env tests pass, `progress_v2.md` Phase 3 entry, git commit + push

### Phase 4: Policy and PPO Trainer

**Estimated: ~3 hours**

1. `policy.py` — Actor-Critic with:
   - Shared MLP backbone (state -> hidden)
   - 3 action heads (op_type, position, token) — autoregressive
   - Action masking integration
   - Value head for critic
2. `ppo_trainer.py` — Custom PPO implementation with:
   - VecEnv support (n_envs=20)
   - Autoregressive rollout collection
   - GAE advantage estimation
   - PPO clipped objective with entropy bonus
   - Online SpecificityCallback (eval every 100K steps)
   - Checkpointing at milestones (500K, 1M, 2M, 5M, 10M)
   - TensorBoard/wandb logging
   - Ablation support via `reward_mode` config switch
3. Smoke test: Train for 1000 steps with n_envs=2
   - Verify loss decreases
   - Verify rewards are logged
   - Verify checkpoint saves/loads correctly
4. **Checkpoint**: Smoke test passes, `progress_v2.md` Phase 4 entry, git commit + push

### Phase 5: Full Training Run

**Estimated: ~8-12 hours wall time**

1. Launch full 10M-step training:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 python tcrppo_v2/ppo_trainer.py \
       --config configs/default.yaml \
       --run_name v2_full_run1 \
       --seed 42
   ```
2. Monitor via TensorBoard/wandb:
   - Target affinity should trend upward
   - AUROC from SpecificityCallback should exceed 0.5 by 2M steps
   - If AUROC < 0.40 after warmup (500K), the callback should abort — investigate and fix
3. Save all milestone checkpoints
4. **Checkpoint**: Training complete, `progress_v2.md` Phase 5 entry, git commit + push

### Phase 6: Evaluation and Baseline Comparison

**Estimated: ~2 hours**

1. Generate 50 TCRs per target using trained v2 model
2. Run full decoy specificity evaluation (same protocol as v1 eval):
   - MC Dropout scoring against targets and all decoy tiers
   - Compute per-target AUROC
   - Compute mean AUROC across 12 targets
3. Run identical evaluation on v1 model for fair comparison:
   - v1 checkpoint: `/share/liuyutian/TCRPPO/output/ae_mcpas_mcpas_0.5_0.0_0.9_256_None/ppo_tcr`
4. Produce comparison table and plots:

   | Metric | v1 Baseline | v2 Full | Delta |
   |--------|-------------|---------|-------|
   | Mean AUROC | 0.45 | ? | ? |
   | Mean Target Score | ? | ? | ? |
   | Mean Decoy Score | ? | ? | ? |
   | Diversity (unique seqs) | ? | ? | ? |

5. **Checkpoint**: Eval complete, comparison documented, `progress_v2.md` Phase 6 entry, git commit + push

### Phase 7: Ablation Studies

**Estimated: ~6-8 hours each, parallelizable across GPUs**

Run 3 ablation configs:

1. `v1_ergo_only` — ERGO terminal reward only (should reproduce v1-like behavior)
2. `v2_no_decoy` — v2 without decoy penalty (isolate decoy contribution)
3. `v2_no_curriculum` — v2 with random init only (isolate curriculum contribution)

Evaluate each with same protocol as Phase 6. Document in `progress_v2.md`.

4. **Checkpoint**: Ablations complete, full comparison table, git commit + push

---

## 3. Git Workflow

### Repository setup

This project lives within the TCRPPO repo as a `tcrppo_v2/` subdirectory, or
as its own repo — depending on user preference. Default: subdirectory of TCRPPO.

```bash
cd /share/liuyutian/tcrppo_v2
git init
git remote add origin https://github.com/starpacker/TCRPPO.git
```

### Commit conventions

- Commit after EVERY phase completion (not just at the end)
- Commit message format: `v2(phaseN): brief description`
  - e.g., `v2(phase1): implement and test all scorer modules`
- Push after every commit: `git push origin main`
- Tag milestones: `git tag v2.0-phase{N}`

---

## 4. Progress Logging

Maintain `progress_v2.md` in the project root. After each phase:

```markdown
## Phase N: Title — [COMPLETE|IN PROGRESS|FAILED]

**Date:** YYYY-MM-DD HH:MM
**Duration:** X hours

### What was done
- Bullet points of concrete deliverables

### Test results
- Paste actual test output (no screenshots, text only)
- Include key metrics with actual numbers

### Issues encountered
- What broke and how it was fixed

### Next step
- What Phase N+1 will do
```

---

## 5. Execution Discipline

### 5.1 No Faking

- Every test result must come from actual execution on this machine
- Every metric must come from actual model inference with actual data
- If a test fails, you MUST fix the code and re-run — not skip or comment out
- Do not hardcode expected outputs into tests to make them "pass"

### 5.2 Continuous Operation

- Do NOT stop between phases. After completing Phase N, immediately begin Phase N+1.
- If context window fills up, use `/compact` to compress and continue.
- Re-read this CLAUDE.md after any context compaction to restore orientation.
- If a long training run is needed (Phase 5), launch it in background (`nohup`),
  then continue with other preparatory work while monitoring.

### 5.3 Error Recovery

- If a phase fails, diagnose root cause before retrying
- Document all failures in `progress_v2.md` — they are valuable information
- If stuck on a single issue for >30 minutes, document it, skip to the next
  independent phase, and return later
- NEVER silently swallow errors or exceptions

### 5.4 Resource Management

- Use `CUDA_VISIBLE_DEVICES` to avoid conflicts with other users' jobs
- Check `nvidia-smi` before launching GPU-intensive work
- For training, use at most 2 GPUs unless the machine is idle
- Kill stale processes you started before exiting

### 5.5 Code Quality

- Type hints on all function signatures
- Docstrings on all public functions (one-line is fine)
- No wildcard imports (`from x import *`)
- Config values come from `default.yaml`, not hardcoded in source
- All file paths are relative to project root or configurable via config/CLI args
- Random seeds are set and logged for reproducibility

### 5.6 Verification Before Moving On

Before declaring any phase complete, verify:
1. All unit tests for that phase pass (`pytest -v`)
2. No regressions in previously passing tests
3. Progress file updated with actual results
4. Git commit created and pushed

---

## 6. V1 Baseline Reference (READ-ONLY)

The v1 codebase at `/share/liuyutian/TCRPPO/` is your reference for:
- ERGO model loading patterns (`code/ERGO/`, `code/reward.py`)
- TCRdb data format (`data/tcrdb/`)
- Test peptide format (`data/test_peptides/`)
- The 10M-step v1 checkpoint for comparison

**Do NOT modify any files in `/share/liuyutian/TCRPPO/`.**

### V1 baseline numbers (from progress_3.md)

| Target | AUROC |
|--------|-------|
| GILGFVFTL | 0.3200 |
| NLVPMVATV | 0.4022 |
| GLCTLVAML | 0.6778 |
| LLWNGPMAV | 0.3472 |
| YLQPRTFLL | 0.3028 |
| FLYALALLL | 0.4133 |
| SLYNTVATL | 0.8776 |
| KLGGALQAK | 0.5200 |
| AVFDRKSDAK | 0.4561 |
| IVTDFSVIK | 0.3022 |
| SPRWYFYYL | 0.6056 |
| RLRAEAQVK | 0.2311 |
| **Mean** | **0.4538** |

Your v2 model must beat this across the board.

---

## 7. Session Startup Checklist

Every time you start or resume a session in this directory:

1. Read this `CLAUDE.md` completely
2. Read `progress_v2.md` to find where you left off
3. Read `docs/2026-04-09-tcrppo-v2-design.md` if you need architecture details
4. Activate conda env: `conda activate tcrppo_v2`
5. Check GPU availability: `nvidia-smi`
6. Run existing tests to confirm nothing is broken: `pytest -v`
7. Continue from the last incomplete phase

---

## 8. File Layout (Target State After All Phases)

```
tcrppo_v2/
  CLAUDE.md                    # This file
  progress_v2.md               # Phase-by-phase progress log
  docs/
    2026-04-09-tcrppo-v2-design.md  # Design specification
  configs/
    default.yaml             # All hyperparameters
  ref/                           # V1 reference files (NOT imported at runtime)
    data_utils.py
    config.py
    reward.py
  evaluation/                    # Decoy eval pipeline (copied from v1, adapted)
    eval_decoy.py
    eval_decoy_metrics.py
    eval_decoy_visualize.py
    eval_decoy_random_baseline.py
    ergo_uncertainty.py
  tcrppo_v2/                   # Source code package
    __init__.py
    env.py                   # Gym environment (NEW)
    policy.py                # Actor-Critic network (NEW)
    ppo_trainer.py           # Training entrypoint (NEW)
    test_tcrs.py             # Inference script (NEW)
    reward_manager.py        # Reward combination + normalization (NEW)
    ERGO/                    # ERGO model code + weights (COPIED from v1)
      ERGO_models.py
      ae_utils.py
      lstm_utils.py
      models/                # Pretrained weights (ae_mcpas1.pt, etc.)
      TCR_Autoencoder/
    scorers/
      __init__.py
      base.py                # NEW
      affinity_ergo.py       # NEW (wraps ERGO/)
      decoy.py               # NEW
      naturalness.py         # NEW
      diversity.py           # NEW
    data/
      __init__.py
      pmhc_loader.py         # NEW
      tcr_pool.py            # NEW
      decoy_sampler.py       # NEW
    utils/
      __init__.py
      constants.py           # Extracted from ref/config.py
      encoding.py            # Extracted from ref/data_utils.py
      esm_cache.py           # NEW
  tests/
    test_scorers.py
    test_data.py
    test_env.py
    test_policy.py
    test_integration.py
  data/
    l1_seeds/                # Pre-computed L1 curriculum seeds
    cdr3_ppl_stats.json      # ESM perplexity stats for naturalness
  output/                      # Training checkpoints
  results/                     # Evaluation results
  figures/                     # Plots and visualizations
```

---

## 9. Key Design Decisions Quick Reference

These are the most important design choices from the spec. Consult the full
design doc for details.

| Component | Decision | Why |
|-----------|----------|-----|
| Action space | 3-head autoregressive (op/pos/token) | Supports indel, not just substitution |
| State encoder | ESM-2 650M frozen | Deep biochemical understanding, no training needed |
| Reward | 4-component: affinity + decoy + naturalness + diversity | v1 had only affinity -> universal binder |
| Credit assignment | Per-step delta reward | v1 had only terminal reward -> broken credit |
| Decoy penalty | LogSumExp over K=32 sampled decoys | Smooth, differentiable contrastive signal |
| Decoy tiers | A>B>D>C with phased unlocking | Curriculum from easy to hard negatives |
| TCR init | L0/L1/L2 curriculum | Reduces wasted exploration on zero-affinity starts |
| PPO implementation | Custom (not SB3) | SB3 cannot handle autoregressive action masking |
| Naturalness | ESM perplexity z-score | Better than AE+GMM, CDR3-aware normalization |

---

## 10. Emergency Contacts and References

- **Design spec**: `docs/2026-04-09-tcrppo-v2-design.md`
- **v1 progress history**: `/share/liuyutian/TCRPPO/progress.md`, `progress_2.md`, `progress_3.md`
- **v1 architecture audit**: `/share/liuyutian/TCRPPO/query.md`
- **Decoy library docs**: `/share/liuyutian/pMHC_decoy_library/README.md`
- **ERGO paper**: IdoSpringer/ERGO on GitHub
- **ESM-2**: `fair-esm` package, model `esm2_t33_650M_UR50D`
