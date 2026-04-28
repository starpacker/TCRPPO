# DeepAIR Integration - Current Status

**Date:** 2026-04-23  
**Time:** 01:45 UTC  
**Phase:** Waiting for model downloads

---

## Summary

DeepAIR affinity scorer has been **partially deployed** with a placeholder implementation. The placeholder uses a custom Transformer architecture and produces random predictions. Full deployment requires downloading the actual DeepAIR pretrained weights (~12GB) and ProtBert model (~1.5GB), then rebuilding the scorer with the official implementation.

---

## What's Complete

### ✅ 1. Placeholder Implementation
- **File:** `tcrppo_v2/scorers/affinity_deepair.py` (273 lines)
- **Status:** Working but uses random weights
- **Architecture:** Custom Transformer (embedding + self-attention + cross-attention + MLP)
- **Interface:** Fully compliant with BaseScorer
- **Functionality:** Can score single pairs and batches, but predictions are meaningless

### ✅ 2. Ensemble Integration
- **File:** `tcrppo_v2/scorers/affinity_ensemble.py` (unchanged)
- **Status:** Ready to use
- **Supports:** NetTCR-2.0 + ERGO + DeepAIR (placeholder)
- **Tested:** Yes, ensemble works with all three scorers

### ✅ 3. Consistency Evaluation
- **Script:** `scripts/eval_scorer_consistency.py` (287 lines)
- **Dataset:** tc-hard (10 peptides, 1,007 TCR-peptide pairs)
- **Results:** Generated and saved to `results/scorer_consistency/`
- **Findings:** 
  - NetTCR vs ERGO: Low correlation (r=0.058) → complementary
  - DeepAIR: Random predictions (mean=0.495, std=0.004) → needs training

### ✅ 4. Documentation
- **English:** `docs/DEEPAIR_INTEGRATION.md` (500+ lines)
- **Chinese:** `docs/DEEPAIR_INTEGRATION_CN.md` (400+ lines)
- **Quick Start:** `docs/DEEPAIR_README.md`
- **Delivery Summary:** `docs/DELIVERY_SUMMARY.md`
- **Rebuild Plan:** `docs/DEEPAIR_REBUILD_PLAN.md` (NEW)
- **Status:** All documents note that DeepAIR uses random weights

### ✅ 5. Usage Examples
- **Script:** `scripts/example_ensemble_usage.py` (150+ lines)
- **Examples:** Single scorer, ensemble, batch processing
- **Status:** All examples work with placeholder

### ✅ 6. Testing Infrastructure
- **Evaluation script:** Ready
- **Monitoring script:** `scripts/monitor_downloads.sh` (NEW)
- **Verification script:** `scripts/verify_downloads.sh` (NEW)
- **Status:** Ready to re-run after rebuild

---

## What's In Progress

### 🔄 1. DeepAIR Weights Download
- **Source:** Zenodo (https://zenodo.org/records/7792621)
- **File:** DeepAIR.zip (~12GB)
- **Location:** `/share/liuyutian/tcrppo_v2/models/deepair/DeepAIR.zip`
- **Progress:** 3.0GB / ~12GB (25%)
- **ETA:** ~26 minutes
- **PID:** 900757 (wget running in background)

### 🔄 2. ProtBert Model Download
- **Source:** HuggingFace Mirror (https://hf-mirror.com/Rostlab/prot_bert_bfd)
- **File:** pytorch_model.bin (~1.5GB expected)
- **Location:** `/share/liuyutian/tcrppo_v2/models/protbert/pytorch_model.bin`
- **Progress:** 251MB / ~1.5GB (15%)
- **ETA:** ~42 minutes
- **PID:** 905555 (wget running in background)
- **Note:** Config files already downloaded (config.json, vocab.txt, tokenizer_config.json)

---

## What's Pending

### ⏳ 1. Verify Downloads
- **Script:** `scripts/verify_downloads.sh`
- **Actions:**
  - Check DeepAIR.zip integrity with `unzip -t`
  - Verify ProtBert pytorch_model.bin size (~1.5GB expected)
  - Check file types (ensure not HTML error pages)
- **Estimated Time:** 5 minutes

### ⏳ 2. Extract DeepAIR
- **Command:** `cd models/deepair && unzip DeepAIR.zip`
- **Expected Contents:**
  - Model architecture code (*.py)
  - Pretrained weights (*.pt, *.pth, *.ckpt)
  - Config files
  - Example scripts
- **Estimated Time:** 5 minutes

### ⏳ 3. Examine DeepAIR Structure
- **Goal:** Understand how to integrate into BaseScorer interface
- **Key Questions:**
  - What is the main model class?
  - How are sequences encoded (ProtBert usage)?
  - What is input/output format?
  - How to load pretrained weights?
  - Does it require 3D structure or work with sequence only?
- **Estimated Time:** 30-60 minutes

### ⏳ 4. Rebuild `affinity_deepair.py`
- **Action:** Replace placeholder with actual DeepAIR implementation
- **Requirements:**
  - Load ProtBert from `/share/liuyutian/tcrppo_v2/models/protbert/`
  - Load DeepAIR weights from extracted directory
  - Maintain BaseScorer interface compatibility
  - Support batch processing
- **Estimated Time:** 1-2 hours

### ⏳ 5. Test Rebuilt Scorer
- **Quick Test:**
  ```python
  from tcrppo_v2.scorers.affinity_deepair import AffinityDeepAIRScorer
  scorer = AffinityDeepAIRScorer(device='cuda')
  score, conf = scorer.score('CASSIRSSYEQYF', 'GILGFVFTL')
  assert score != 0.495  # Not random anymore
  ```
- **Estimated Time:** 10 minutes

### ⏳ 6. Re-run Consistency Evaluation
- **Command:** `python scripts/eval_scorer_consistency.py`
- **Expected Changes:**
  - DeepAIR mean: NOT ~0.495 (should be in [0.1, 0.9] range)
  - DeepAIR std: > 0.05 (currently 0.004)
  - Correlations: Positive with NetTCR and ERGO (currently negative)
- **Estimated Time:** 5 minutes

### ⏳ 7. Update Documentation
- **Files to Update:**
  - `docs/DEEPAIR_INTEGRATION.md`
  - `docs/DEEPAIR_INTEGRATION_CN.md`
  - `docs/DEEPAIR_README.md`
  - `docs/DELIVERY_SUMMARY.md`
- **Changes:**
  - Remove "random initialization" warnings
  - Add actual architecture details
  - Update performance statistics
  - Add ProtBert dependency info
- **Estimated Time:** 20 minutes

---

## Timeline

### Current Status (2026-04-23 01:45)
- Downloads in progress
- Placeholder implementation complete
- Documentation complete (with caveats)
- Waiting for downloads to finish

### Estimated Completion
- **Downloads complete:** ~42 minutes (by 02:27)
- **Rebuild complete:** +2-3 hours (by 05:27)
- **Final delivery:** ~3 hours from now

---

## Monitoring Commands

```bash
# Check download progress
bash scripts/monitor_downloads.sh

# Check if downloads complete
ps aux | grep wget | grep -E "DeepAIR|pytorch_model"

# Verify downloads (run after completion)
bash scripts/verify_downloads.sh

# Extract DeepAIR (run after verification)
cd /share/liuyutian/tcrppo_v2/models/deepair
unzip DeepAIR.zip
ls -la DeepAIR/
```

---

## Fallback Plans

### If ProtBert Download Fails
1. Try direct HuggingFace (if accessible): `https://huggingface.co/Rostlab/prot_bert_bfd`
2. Try ModelScope mirror (China): `https://modelscope.cn/models/Rostlab/prot_bert_bfd`
3. Use alternative protein LM: ESM-2 (already available in environment)

### If DeepAIR Integration Too Complex
1. **Option A:** Train placeholder model on NetTCR-2.0 data (document as "DeepAIR-inspired")
2. **Option B:** Use two-scorer ensemble (NetTCR + ERGO only)
3. **Option C:** Find alternative scorer (TITAN, ImRex)

### If Downloads Corrupt
1. Re-download with `wget -c` (resume)
2. Try alternative mirrors
3. Download on different machine and transfer

---

## Key Files Reference

### Implementation
- `tcrppo_v2/scorers/affinity_deepair.py` - Scorer (placeholder, needs rebuild)
- `tcrppo_v2/scorers/affinity_ensemble.py` - Ensemble (ready)
- `tcrppo_v2/scorers/affinity_nettcr.py` - NetTCR-2.0 (working)
- `tcrppo_v2/scorers/affinity_ergo.py` - ERGO (working)

### Scripts
- `scripts/eval_scorer_consistency.py` - Evaluation (ready)
- `scripts/example_ensemble_usage.py` - Examples (ready)
- `scripts/monitor_downloads.sh` - Monitor (NEW)
- `scripts/verify_downloads.sh` - Verify (NEW)

### Documentation
- `docs/DEEPAIR_INTEGRATION.md` - Full docs (English)
- `docs/DEEPAIR_INTEGRATION_CN.md` - Full docs (Chinese)
- `docs/DEEPAIR_README.md` - Quick start
- `docs/DELIVERY_SUMMARY.md` - Delivery summary
- `docs/DEEPAIR_REBUILD_PLAN.md` - Rebuild plan (NEW)
- `docs/DEEPAIR_STATUS.md` - This file (NEW)

### Results
- `results/scorer_consistency/scorer_results.csv` - Raw scores
- `results/scorer_consistency/summary.json` - Statistics
- `results/scorer_consistency/*.png` - Plots

### Models (downloading)
- `models/deepair/DeepAIR.zip` - DeepAIR weights (3GB / 12GB)
- `models/protbert/pytorch_model.bin` - ProtBert (251MB / 1.5GB)
- `models/protbert/config.json` - ProtBert config
- `models/protbert/vocab.txt` - ProtBert vocab
- `models/protbert/tokenizer_config.json` - Tokenizer config

---

## Contact Information

**Project:** TCRPPO v2  
**Location:** `/share/liuyutian/tcrppo_v2/`  
**User:** liuyutian  
**Date:** 2026-04-23

---

**Status:** 🔄 In Progress - Waiting for Downloads  
**Next Action:** Monitor downloads, then verify and rebuild  
**ETA:** ~3 hours to full completion
