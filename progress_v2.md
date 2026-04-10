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
