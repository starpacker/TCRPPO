# TCRPPO v2 Progress Log

## Phase 0: Project Scaffolding and Environment Validation — COMPLETE

**Date:** 2026-04-09 
**Duration:** ~30 min

### What was done
- Created full directory layout: `tcrppo_v2/scorers/`, `data/`, `utils/`, `tests/`, `configs/`, `output/`, `results/`, `figures/`
- Created conda env `tcrppo_v2` (Python 3.10, PyTorch 2.7.1+cu118, fair-esm, etc.)
- Verified ESM-2 (esm2_t33_650M_UR50D, 651M params) loads on GPU 0
- Verified ERGO AE model loads from local `tcrppo_v2/ERGO/models/ae_mcpas1.pt`
- Adapted ERGO code for Python 3.10 / PyTorch 2.x:
  - Fixed `lstm_pass` device mismatch (unperm_idx must be on CPU for indexing)
  - Fixed `F.sigmoid` -> `torch.sigmoid` (deprecated in PyTorch 2.x)
  - Fixed `torch.tensor()` copy warning in ae_utils.py
- Verified ERGO inference: known binder CASSIRSSYEQYF scores 0.9997 for GILGFVFTL
- Verified decoy library readable at `/share/liuyutian/pMHC_decoy_library/`
- Verified TCRdb data accessible (7.28M sequences)
- Extracted constants from v1 `config.py` -> `tcrppo_v2/utils/constants.py`
- Extracted encoding utils from v1 `data_utils.py` -> `tcrppo_v2/utils/encoding.py`
- Created `configs/default.yaml` with all hyperparameters from design spec

### Test results
```
ESM-2 loaded on GPU successfully (651M params)
ERGO AE model loaded on GPU successfully
ERGO predictions: [0.195, 0.074, 0.9997]  (random, random, known binder)
```

### Issues encountered
- ERGO `lstm_pass` failed on PyTorch 2.x due to `unperm_idx` being on GPU while `lengths` returns on CPU from `pad_packed_sequence`. Fixed by adding `.cpu()` on `unperm_idx`.
- `F.sigmoid` deprecated in PyTorch 2.x. Replaced with `torch.sigmoid`.

### Next step
- Phase 2: Data pipeline

---

## Phase 1: Scorer Modules — COMPLETE

**Date:** 2026-04-09
**Duration:** ~1 hour

### What was done
- `scorers/base.py` — Abstract `BaseScorer` with `score()` and `score_batch()`
- `scorers/affinity_ergo.py` — ERGO binding scorer with MC Dropout (N=10)
  - Self-contained ERGO loading, no v1 imports
  - MC Dropout via enable/disable on nn.Dropout modules
  - GPU batch building reused across MC samples for efficiency
- `scorers/decoy.py` — LogSumExp contrastive penalty
  - Loads all 4 tiers from decoy library (A/B/C/D)
  - Tier-weighted sampling with configurable weights
  - Unlock schedule support for curriculum
- `scorers/naturalness.py` — ESM perplexity with CDR3 z-score
  - Pseudo-perplexity computation via ESM-2 logits
  - Offline stats computation from TCRdb CDR3beta sequences
  - Threshold-based penalty
- `scorers/diversity.py` — Recent-buffer Levenshtein similarity penalty
- `reward_manager.py` — 4-component reward with running z-score normalization
  - Supports reward_mode: v2_full, v1_ergo_only, v2_no_decoy, v2_no_curriculum

### Test results
```
19 passed in 70.79s

Key metrics:
- Known binder (CASSIRSSYEQYF) ERGO score: 0.997 (conf=0.995)
- Random TCR mean ERGO score: 0.109
- MC Dropout std: 0.009-0.038
- Decoy counts for GILGFVFTL: A=591, B=50, C=1900, D=827
- Strong binder decoy penalty: 0.659 vs weak TCR: 0.370
- Diversity penalty for identical seq: -0.15
```

### Issues encountered
- `ERGO_models` import failed at module level: `sys.path.insert` in `affinity_ergo.py` ran before `constants.py` resolved `ERGO_DIR` correctly. Fixed by computing paths from `__file__` with correct directory depth (3 levels up from `utils/constants.py`).
- `scorers/__init__.py` eagerly imported all scorers, triggering the ERGO import cascade. Fixed by removing eager imports.

### Next step
- Phase 2: Data pipeline (pmhc_loader, tcr_pool, decoy_sampler, esm_cache)

---

## Phase 2: Data Pipeline — COMPLETE

**Date:** 2026-04-10
**Duration:** ~1.5 hours

### What was done
- `data/pmhc_loader.py` — PMHCLoader with 12 eval targets, HLA-allele mapping, pseudosequence encoding
  - Hardcoded HLA alleles for all 12 McPAS targets (mostly HLA-A*02:01)
  - Supports uniform and difficulty-weighted target sampling
- `data/tcr_pool.py` — TCRPool with TCRdb loading (7.28M CDR3beta sequences) + L0/L1/L2 curriculum
  - L0: Known binders from decoy tier D with 3-5 random mutations
  - L1: Pre-computed ERGO top-500 per target (loaded from data/l1_seeds/)
  - L2: Random TCRdb sequences
  - Curriculum schedule from design spec (4 phases)
- `data/decoy_sampler.py` — DecoySampler with tier-weighted sampling and unlock schedule
  - Loads all 4 tiers (A/B/C/D) per target
  - Phase-aware unlock: A(0-2M) -> +B(2-5M) -> +D(5-8M) -> +C(8M+)
  - Tier C: 1900 unrelated peptides (shared globally)
- `utils/esm_cache.py` — ESMCache: Frozen ESM-2 (esm2_t33_650M_UR50D, 1280-dim)
  - pMHC embeddings cached permanently (computed once per target)
  - TCR embeddings with LRU cache (4096 entries)
  - Batch encoding with partial cache hits
- `data/generate_l1_seeds.py` — L1 seed generation script
  - Generated L1 seeds for all 12 targets (500 seqs each, 50K TCRdb sample)
- `tests/test_data.py` — 21 unit tests for all data pipeline modules

### Phase 3: Environment — COMPLETE (done in same session)

- `env.py` — TCREditEnv with:
  - 3-head autoregressive action space (op/pos/token)
  - SUB/INS/DEL/STOP sequence editing
  - Action masking (length bounds, PAD positions, step-0 no STOP)
  - ESM-2 state encoding (TCR re-encoded per step, pMHC cached)
  - Per-step delta reward from RewardManager
  - VecTCREditEnv wrapper for parallel environments
- `tests/test_env.py` — 12 tests including 100-episode random integration test

### Test results
```
52 passed in 239.86s

Key metrics:
- TCRdb loaded: 7,280,430 sequences
- Tier C global: 1,900 decoys
- GILGFVFTL tiers: A=591, B=50, C=1900, D=827
- ESM embed dim: 1280
- ESM cache: first call 0.017s, cached call 0.000004s
- Obs dim: 2562 (1280*2 + 2)
- L1 seeds: 12 targets x 500 seqs each

100 random episodes:
  Total steps: 496
  Mean ep reward: 3.6350
  Mean final TCR len: 14.6
  TCR len range: [9, 19]
```

### L1 seed generation results
| Target | Best ERGO Score | Worst Top-500 Score |
|--------|----------------|-------------------|
| GILGFVFTL | 0.7627 | 0.0044 |
| NLVPMVATV | 0.9022 | 0.1072 |
| GLCTLVAML | 0.9459 | 0.0010 |
| YLQPRTFLL | 0.8765 | 0.0010 |
| SLYNTVATL | 0.8919 | 0.1259 |
| KLGGALQAK | 0.8363 | 0.0030 |
| AVFDRKSDAK | 0.7934 | 0.0004 |
| IVTDFSVIK | 0.8157 | 0.0622 |
| SPRWYFYYL | 0.7654 | 0.0015 |
| RLRAEAQVK | 0.7781 | 0.1515 |

### Issues encountered
- DecoyScorer test `test_universal_binder_higher_penalty` was flaky with K=8 due to stochastic sampling. Fixed by increasing K=32 and using fixed seed (rng=42).
- L1 seed generation background job initially used wrong python path (`/share/liuyutian/miniconda3/` instead of `/home/liuyutian/server/miniconda3/`).

### Next step
- Phase 4: Policy and PPO Trainer

---

## Phase 4: Policy and PPO Trainer — COMPLETE

**Date:** 2026-04-09
**Duration:** ~1.5 hours

### What was done
- `policy.py` — ActorCritic with 3-head autoregressive action space:
  - Shared MLP backbone (obs_dim -> 512 -> 512)
  - Head 1: op_type (4-way categorical)
  - Head 2: position (max_tcr_len-way), conditioned on op embedding
  - Head 3: token (20-way), conditioned on op + position embeddings
  - Value head (512 -> 256 -> 1)
  - Action masking integration
  - Token log-prob masked to 0 for DEL/STOP ops
  - Orthogonal weight initialization (0.01 gain for policy heads)
- `ppo_trainer.py` — Custom PPO implementation:
  - RolloutBuffer with pre-allocated arrays, GAE computation, minibatch shuffling
  - PPO clipped objective with entropy bonus
  - VecEnv integration with autoregressive rollout collection
  - Milestone checkpointing (500K, 1M, 2M, 5M, 10M)
  - TensorBoard logging (reward, episode length, pg_loss, vf_loss, entropy)
  - CLI with --config, --run_name, --seed, --reward_mode overrides
  - Full setup() builds ERGO, ESM, PMHCLoader, TCRPool, DecoySampler, RewardManager, VecEnv
- `tests/test_policy.py` — 12 unit tests:
  - Sampling: output shapes, action ranges, op masking, pos masking, value-only
  - Evaluation: output shapes, log-probs negative, entropy non-negative, token masking for DEL/STOP
  - Gradient flow through all heads
  - RolloutBuffer: add/GAE computation, batch generation

### Test results
```
64 passed in 197.90s

Policy tests (12):
  - Sample output shapes: PASS
  - Action ranges valid: PASS
  - Op masking enforced: PASS
  - Pos masking enforced: PASS
  - Value-only: PASS
  - Evaluate shapes: PASS
  - Log-probs <= 0: PASS
  - Entropy >= 0: PASS
  - Token log-prob masked for DEL/STOP: PASS
  - Gradient flow: PASS
  - RolloutBuffer GAE: PASS
  - RolloutBuffer batches: PASS

Smoke test (1024 steps, 2 envs, v1_ergo_only):
  172 episodes completed
  Mean reward: 0.615
  Mean episode length: 5.6
  Policy: 841,108 parameters
  Checkpoints: latest.pt, final.pt, milestone_512.pt, milestone_1024.pt
  Checkpoint load + inference: verified OK
```

### Issues encountered
- Missing `import torch.nn.functional as F` in `ppo_trainer.py` — would have caused runtime error on vf_loss line. Fixed before first run.

### Next step
- Phase 5: Full Training Run (10M steps)
