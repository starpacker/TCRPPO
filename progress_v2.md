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
- Phase 1: Build and test all scorer modules
