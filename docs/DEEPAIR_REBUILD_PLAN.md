# DeepAIR Scorer Rebuild Plan

**Date:** 2026-04-23  
**Status:** Waiting for downloads to complete  
**Current:** Placeholder implementation with random weights  
**Target:** Full DeepAIR integration with pretrained weights

---

## Download Status

### DeepAIR Weights (Zenodo)
- **URL:** https://zenodo.org/records/7792621
- **File:** DeepAIR.zip (~12GB)
- **Location:** `/share/liuyutian/tcrppo_v2/models/deepair/DeepAIR.zip`
- **Status:** Downloading (13% complete, ~30 min remaining)

### ProtBert (HuggingFace Mirror)
- **URL:** https://hf-mirror.com/Rostlab/prot_bert_bfd
- **Files:** 
  - `pytorch_model.bin` (downloading, ~123MB so far, ~47 min remaining)
  - `config.json` (361 bytes, downloaded)
  - `vocab.txt` (81 bytes, downloaded)
  - `tokenizer_config.json` (86 bytes, downloaded)
- **Location:** `/share/liuyutian/tcrppo_v2/models/protbert/`
- **Status:** Downloading

**Note:** The config files seem suspiciously small. Need to verify after download completes.

---

## Rebuild Steps (Execute After Downloads Complete)

### Step 1: Verify Downloads

```bash
# Check DeepAIR.zip integrity
cd /share/liuyutian/tcrppo_v2/models/deepair
unzip -t DeepAIR.zip

# Check ProtBert files
cd /share/liuyutian/tcrppo_v2/models/protbert
ls -lh
# Expected: pytorch_model.bin should be ~1.5GB for ProtBert
# If only 123MB, the download may be incomplete or wrong file
```

### Step 2: Extract and Examine DeepAIR

```bash
cd /share/liuyutian/tcrppo_v2/models/deepair
unzip DeepAIR.zip
ls -la DeepAIR/

# Look for:
# - Model architecture files (*.py)
# - Pretrained weights (*.pt, *.pth, *.ckpt)
# - Config files
# - Example usage scripts
```

### Step 3: Understand DeepAIR Architecture

**From the paper (Science Advances 2023):**
- Uses ProtBert for sequence encoding
- Integrates 3D structure information (optional)
- Transformer-based architecture
- Trained on TCR-peptide binding data

**Key questions to answer:**
1. What is the main model class name?
2. How are TCR and peptide sequences encoded?
3. What is the input format? (raw sequences, tokenized, embedded?)
4. What is the output format? (binding probability, logits, score?)
5. How are pretrained weights loaded?
6. Does it require 3D structure or can it work with sequence only?

### Step 4: Rebuild `affinity_deepair.py`

**Current placeholder structure:**
```python
class DeepAIRModel(nn.Module):
    # Custom transformer (NOT the real DeepAIR)
    pass

class AffinityDeepAIRScorer(BaseScorer):
    def __init__(self, model_path=None, device='cuda'):
        # Loads placeholder model
        pass
    
    def score(self, tcr, peptide):
        # Returns random predictions
        pass
```

**Target structure:**
```python
# Import actual DeepAIR modules
from transformers import BertModel, BertTokenizer
# Or: sys.path.append('/path/to/DeepAIR'); from deepair import ...

class AffinityDeepAIRScorer(BaseScorer):
    def __init__(self, 
                 deepair_weights_path='/share/liuyutian/tcrppo_v2/models/deepair/DeepAIR/...',
                 protbert_path='/share/liuyutian/tcrppo_v2/models/protbert/',
                 device='cuda'):
        """
        Initialize DeepAIR scorer with pretrained weights.
        
        Args:
            deepair_weights_path: Path to DeepAIR model weights
            protbert_path: Path to ProtBert model directory
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        self.device = device
        
        # Load ProtBert
        self.tokenizer = BertTokenizer.from_pretrained(protbert_path)
        self.protbert = BertModel.from_pretrained(protbert_path)
        self.protbert.to(device)
        self.protbert.eval()
        
        # Load DeepAIR model
        # TODO: Determine exact loading method from extracted code
        self.model = load_deepair_model(deepair_weights_path)
        self.model.to(device)
        self.model.eval()
    
    def score(self, tcr: str, peptide: str) -> Tuple[float, float]:
        """
        Score TCR-peptide binding affinity.
        
        Returns:
            (score, confidence): Binding probability and confidence
        """
        # TODO: Implement based on actual DeepAIR inference code
        pass
    
    def score_batch(self, tcrs: list, peptides: list) -> Tuple[list, list]:
        """Batch scoring for efficiency."""
        # TODO: Implement batch inference
        pass
```

### Step 5: Test New Implementation

```bash
# Quick test
cd /share/liuyutian/tcrppo_v2
python -c "
from tcrppo_v2.scorers.affinity_deepair import AffinityDeepAIRScorer
scorer = AffinityDeepAIRScorer(device='cuda')
score, conf = scorer.score('CASSIRSSYEQYF', 'GILGFVFTL')
print(f'Score: {score:.4f}, Confidence: {conf:.4f}')
assert 0.0 <= score <= 1.0, 'Score out of range'
assert score != 0.495, 'Still using random weights!'
print('✓ DeepAIR scorer working with pretrained weights')
"
```

### Step 6: Re-run Consistency Evaluation

```bash
cd /share/liuyutian/tcrppo_v2
python scripts/eval_scorer_consistency.py
```

**Expected changes:**
- DeepAIR mean score: Should NOT be ~0.495 (random)
- DeepAIR std: Should be > 0.004 (currently near-constant)
- Correlations: Should show positive correlation with NetTCR and ERGO (not negative)

### Step 7: Update Documentation

Update the following files to reflect actual DeepAIR implementation:
- `docs/DEEPAIR_INTEGRATION.md`
- `docs/DEEPAIR_INTEGRATION_CN.md`
- `docs/DEEPAIR_README.md`
- `docs/DELIVERY_SUMMARY.md`

Key changes:
- Remove "DeepAIR uses random initialization" warnings
- Add actual model architecture details
- Update performance statistics
- Add ProtBert dependency information

---

## Potential Issues and Solutions

### Issue 1: ProtBert File Size Mismatch

**Symptom:** `pytorch_model.bin` is only ~123MB, but ProtBert should be ~1.5GB

**Diagnosis:**
```bash
file /share/liuyutian/tcrppo_v2/models/protbert/pytorch_model.bin
head -100 /share/liuyutian/tcrppo_v2/models/protbert/pytorch_model.bin
```

**Solutions:**
1. Check if it's an HTML error page (common with download failures)
2. Try alternative mirror: `https://hf-mirror.com` → `https://huggingface.co` (if accessible)
3. Use `git lfs` to clone the full repo
4. Download from alternative source (ModelScope, etc.)

### Issue 2: DeepAIR Code Not Compatible with Current Environment

**Symptom:** Import errors, version conflicts, missing dependencies

**Solutions:**
1. Check DeepAIR requirements.txt and install missing packages
2. Adapt code to work with current PyTorch/transformers versions
3. Create isolated conda environment if needed
4. Vendor DeepAIR code into `tcrppo_v2/deepair/` directory

### Issue 3: DeepAIR Requires 3D Structure

**Symptom:** Model expects structure features, not just sequence

**Solutions:**
1. Check if sequence-only mode is available
2. Use dummy structure features (zeros) if optional
3. Integrate AlphaFold or ESM-Fold for structure prediction (future work)
4. For now, document limitation and use sequence-only predictions

### Issue 4: DeepAIR Output Format Incompatible

**Symptom:** Output is not a binding probability [0, 1]

**Solutions:**
1. Apply sigmoid if output is logits
2. Normalize if output is unbounded score
3. Add calibration layer if needed
4. Document transformation in scorer code

---

## Fallback Plan

If DeepAIR integration proves too complex or downloads fail:

### Option A: Use Simplified DeepAIR Architecture
- Keep current transformer-based architecture
- Train on NetTCR-2.0 dataset
- Use ProtBert embeddings instead of custom embeddings
- Document as "DeepAIR-inspired" rather than "DeepAIR"

### Option B: Use Alternative Scorer
- TITAN (if available)
- ImRex (if available)
- Train custom transformer on TCR data
- Document substitution in delivery notes

### Option C: Two-Scorer Ensemble
- Use only NetTCR-2.0 + ERGO
- Document DeepAIR as "future work"
- Deliver current results with two-scorer ensemble

**Preference:** Try Option A before falling back to B or C.

---

## Success Criteria

The rebuild is successful when:

1. ✅ DeepAIR scorer loads pretrained weights without errors
2. ✅ Scores are NOT centered at 0.5 (mean should be in [0.1, 0.9] range)
3. ✅ Score variance is meaningful (std > 0.05)
4. ✅ Positive correlation with NetTCR and ERGO (r > 0.1)
5. ✅ Batch scoring works efficiently
6. ✅ All tests pass
7. ✅ Documentation updated

---

## Timeline Estimate

- **Step 1-2 (Verify & Extract):** 10 minutes
- **Step 3 (Understand):** 30-60 minutes (depends on code complexity)
- **Step 4 (Rebuild):** 1-2 hours (depends on compatibility issues)
- **Step 5 (Test):** 10 minutes
- **Step 6 (Re-evaluate):** 5 minutes (script already exists)
- **Step 7 (Update docs):** 20 minutes

**Total:** 2-3 hours after downloads complete

---

## Next Actions (Automated)

Once downloads complete:
1. Check download completion status
2. Verify file integrity
3. Extract DeepAIR.zip
4. Examine code structure
5. Begin rebuild of `affinity_deepair.py`
6. Test and validate
7. Update documentation

**Current Status:** Waiting for downloads (~30-47 min remaining)

---

**Last Updated:** 2026-04-23 01:40 UTC
