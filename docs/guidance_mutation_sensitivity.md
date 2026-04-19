# Guidance: ERGO & NetTCR-2.0 Single-Point Mutation Sensitivity Analysis

## Objective

Test how sensitive ERGO and NetTCR-2.0 are to single amino acid mutations in CDR3b sequences. This reveals whether each model captures residue-level binding contributions or relies on coarse sequence features.

---

## 1. ERGO (Autoencoder-LSTM Classifier)

### 1.1 Architecture

```
Input: CDR3b sequence (string) + peptide sequence (string)

CDR3b path:
  CDR3b string + "X" terminator
  → One-hot encoding (21-dim: 20 AA + X), padded to max_len=28
  → Frozen PaddingAutoencoder encoder:
      Linear(28*21=588 → 300) → ELU → Dropout(0.1)
      → Linear(300 → 100) → ELU → Dropout(0.1)
      → Linear(100 → 100)
      = 100-dim TCR embedding

Peptide path:
  Peptide string
  → Integer encoding (1-indexed, 0=PAD for 21 classes)
  → nn.Embedding(21, 10, padding_idx=0)
  → 2-layer LSTM(input=10, hidden=100, dropout=0.1)
  → Last hidden state
  = 100-dim peptide embedding

Combined:
  Concat(TCR_emb[100], Pep_emb[100]) = 200-dim
  → Linear(200 → 100) → LeakyReLU → Dropout(0.1)
  → Linear(100 → 1) → Sigmoid
  = binding probability ∈ [0, 1]
```

### 1.2 Key Properties

| Property | Value |
|---|---|
| Framework | PyTorch |
| Input | CDR3b (raw AA string) + peptide (raw AA string) |
| CDR3b encoding | One-hot 21-dim (20 AA + X terminator), pad to 28 |
| Peptide encoding | Integer index (1-indexed, 0=PAD), Embedding(21,10) |
| CDR3b max length | 28 (ERGO_MAX_LEN) |
| TCR representation | Frozen autoencoder encoder → 100-dim |
| Output | Sigmoid probability [0, 1] |
| Training data | McPAS-TCR database (positive pairs) + negative sampling |
| Weights file | `tcrppo_v2/ERGO/models/ae_mcpas1.pt` |
| AE weights | `tcrppo_v2/ERGO/TCR_Autoencoder/tcr_ae_dim_100.pt` |
| MC Dropout | 10 forward passes with Dropout(0.1) enabled |
| Device | GPU (cuda) or CPU |

### 1.3 Amino Acid Encoding Details

```python
# TCR uses 21-class one-hot (20 AA + X terminator), 0-indexed
ERGO_TCR_ATOX = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
    'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20
}

# Peptide uses integer encoding, 0=PAD, 1-indexed
ERGO_PEP_ATOX = {
    'PAD': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7,
    'G': 8, 'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14,
    'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20
}
```

### 1.4 How to Use for Scoring

```python
import sys
sys.path.insert(0, "tcrppo_v2/ERGO")

from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer

# Initialize
scorer = AffinityERGOScorer(
    model_file="tcrppo_v2/ERGO/models/ae_mcpas1.pt",
    ae_file="tcrppo_v2/ERGO/TCR_Autoencoder/tcr_ae_dim_100.pt",
    device="cuda",       # or "cpu"
    mc_samples=10,       # MC Dropout samples (set 1 for fast, no uncertainty)
)

# --- Single pair ---
score, confidence = scorer.score("CASSIRSSYEQYF", "GILGFVFTL")
# score: float [0,1], confidence: float [0,1] (1 - MC std)

# --- Batch scoring (with MC Dropout) ---
tcrs = ["CASSIRSSYEQYF", "CASSLGQAYEQYF", "CASSYSADTQYF"]
peps = ["GILGFVFTL",     "GILGFVFTL",     "GILGFVFTL"]
scores, confidences = scorer.score_batch(tcrs, peps)
# scores: List[float], confidences: List[float]

# --- Fast batch (no MC Dropout, single forward pass) ---
scores = scorer.score_batch_fast(tcrs, peps)
# scores: List[float]
```

### 1.5 Internal Data Flow for Scoring

```python
# What happens inside score_batch_fast():
# 1. Deep copy input lists (ERGO mutates them in-place!)
# 2. ae_utils.get_full_batches():
#    a. TCR: each string → append "X" → one-hot pad to (28, 21) tensor
#    b. Peptide: each string → integer list via ERGO_PEP_ATOX → pad_batch()
#    c. Group into batches of 4096
# 3. model.forward(tcr_tensor, padded_peps, pep_lens):
#    a. TCR tensor → flatten → autoencoder.encoder → 100-dim
#    b. Pep indices → embedding(21,10) → LSTM(10,100) → last cell → 100-dim
#    c. Concat → MLP → sigmoid → probability
```

**Important**: ERGO's `ae_utils.convert_data()` **mutates the input lists in-place** (converts strings to tensors). The scorer always deep-copies inputs before calling.

---

## 2. NetTCR-2.0 (Multi-Kernel CNN)

### 2.1 Architecture

```
Input: CDR3b sequence (string) + peptide sequence (string)

CDR3b path:
  CDR3b string
  → BLOSUM50 encoding (20-dim per residue), padded to max_len=30
  → 5 parallel Conv1D branches (kernel sizes 1, 3, 5, 7, 9):
      Each: Conv1D(20→16, kernel=k, padding='same', activation='sigmoid')
            → GlobalMaxPooling1D() → 16-dim
  → Concatenate 5 branches = 80-dim

Peptide path:
  Peptide string
  → BLOSUM50 encoding (20-dim per residue), padded to max_len=15
  → 5 parallel Conv1D branches (same structure as CDR3b)
  → Concatenate 5 branches = 80-dim

Combined:
  Concat(CDR3b_feat[80], Pep_feat[80]) = 160-dim
  → Dense(160 → 32, activation='sigmoid')
  → Dense(32 → 1, activation='sigmoid')
  = binding probability ∈ [0, 1]
```

### 2.2 Key Properties

| Property | Value |
|---|---|
| Framework | TensorFlow/Keras |
| Input | CDR3b (raw AA string) + peptide (raw AA string) |
| CDR3b encoding | BLOSUM50 (20-dim per residue), zero-padded to 30 |
| Peptide encoding | BLOSUM50 (20-dim per residue), zero-padded to 15 |
| CDR3b max length | 30 (MAX_CDR3_LEN) |
| Peptide max length | 15 (MAX_PEP_LEN) |
| CNN kernels | [1, 3, 5, 7, 9], 16 filters each |
| Parameters | ~50K (very lightweight) |
| Output | Sigmoid probability [0, 1] |
| Training data | 232K train / 41K test rows (CDR3b, peptide, binder) |
| Weights file | `data/nettcr_model.weights.h5` |
| Training CSV | `data/nettcr_train.csv`, `data/nettcr_test.csv` |
| Uncertainty | None built-in |
| Device | CPU only (TF GPU hidden to avoid PyTorch conflict) |

### 2.3 BLOSUM50 Encoding

Each amino acid is encoded as a 20-dimensional vector from the BLOSUM50 substitution matrix:

```python
BLOSUM50_20AA = {
    'A': [ 5,-2,-1,-2,-1,-1,-1, 0,-2,-1,-2,-1,-1,-3,-1, 1, 0,-3,-2, 0],
    'R': [-2, 7,-1,-2,-4, 1, 0,-3, 0,-4,-3, 3,-2,-3,-3,-1,-1,-3,-1,-3],
    'N': [-1,-1, 7, 2,-2, 0, 0, 0, 1,-3,-4, 0,-2,-4,-2, 1, 0,-4,-2,-3],
    # ... (20 amino acids total, each a 20-dim integer vector)
}
```

Key property of BLOSUM encoding: **biochemically similar amino acids have similar vectors**. This means NetTCR inherently encodes amino acid similarity, unlike ERGO's one-hot encoding where all mutations are equidistant.

### 2.4 Training Data Format

```csv
CDR3b,peptide,binder
CASSYLPGQGDHYSNQPQHF,FLKEKGGL,1
CASSFEAGQGFFSNQPQHF,FLKEKGGL,1
...
```

- 232,117 training pairs, 41,149 test pairs
- Binary labels: 1 = binder, 0 = non-binder
- Source: tc-hard filtered dataset

### 2.5 How to Use for Scoring

```python
from tcrppo_v2.scorers.affinity_nettcr import AffinityNetTCRScorer

# Initialize (auto-loads weights from data/nettcr_model.weights.h5)
scorer = AffinityNetTCRScorer(
    model_path=None,     # None = use default path
    device="cpu",        # TF always runs on CPU in this setup
    batch_size=256,
)

# --- Single pair ---
score, confidence = scorer.score("CASSIRSSYEQYF", "GILGFVFTL")
# score: float [0,1], confidence: always 1.0 (no uncertainty)

# --- Batch scoring ---
tcrs = ["CASSIRSSYEQYF", "CASSLGQAYEQYF", "CASSYSADTQYF"]
peps = ["GILGFVFTL",     "GILGFVFTL",     "GILGFVFTL"]
scores, confidences = scorer.score_batch(tcrs, peps)
# scores: List[float], confidences: List[float] (all 1.0)

# --- Fast batch (same as score_batch but returns only scores) ---
scores = scorer.score_batch_fast(tcrs, peps)
# scores: List[float]
```

Or use the lower-level `NetTCRScorer` directly:

```python
from tcrppo_v2.evaluation.nettcr_scorer import NetTCRScorer

scorer = NetTCRScorer(model_path="data/nettcr_model.weights.h5")

# Single score
score = scorer.score("CASSIRSSYEQYF", "GILGFVFTL")  # float

# Batch scores
scores = scorer.score_batch(["CASSIRSSYEQYF"], ["GILGFVFTL"])  # np.ndarray
```

---

## 3. Key Differences for Mutation Sensitivity

| Aspect | ERGO | NetTCR-2.0 |
|---|---|---|
| **CDR3b encoding** | One-hot (21-dim, all AA equidistant) | BLOSUM50 (20-dim, encodes AA similarity) |
| **CDR3b processing** | Autoencoder → 100-dim dense vector | Multi-kernel CNN → 80-dim |
| **Peptide encoding** | Integer → Embedding(10) → LSTM | BLOSUM50 → Multi-kernel CNN |
| **Expected mutation sensitivity** | May be less sensitive to conservative mutations (one-hot treats A→V same as A→W) | Should distinguish conservative vs. radical mutations (BLOSUM captures similarity) |
| **Position sensitivity** | AE flattens → position info partially lost | CNN with variable kernels → captures local motifs |
| **Framework** | PyTorch | TensorFlow/Keras |
| **Speed** | ~2200 pairs/sec (CPU, mc=1) | Comparable (CPU-only) |

### Implications for Mutation Sensitivity Testing

1. **ERGO's autoencoder bottleneck**: The TCR autoencoder compresses CDR3b into a 100-dim vector. If a single mutation doesn't significantly change this compressed representation, ERGO may be insensitive to it. The autoencoder was trained to reconstruct sequences, so it may average out single residue changes.

2. **NetTCR's BLOSUM advantage**: Because BLOSUM50 encodes biochemical similarity, a mutation from A→V (similar, BLOSUM score=0) will produce a smaller input perturbation than A→W (dissimilar, BLOSUM score=-3). This is biologically meaningful.

3. **NetTCR's CNN kernels**: The multi-kernel design (1,3,5,7,9) captures different-scale patterns. A mutation within a motif captured by a size-3 kernel will have different impact than one in a less structured region.

---

## 4. Suggested Experiment Protocol

### 4.1 Test Set Selection

Choose well-studied CDR3b-peptide pairs with known binding. Recommended:

```python
# Known strong binders from McPAS/VDJdb
test_pairs = [
    ("CASSIRSSYEQYF",  "GILGFVFTL"),   # Influenza M1
    ("CASSLGQAYEQYF",  "NLVPMVATV"),   # CMV pp65
    ("CASSYSADTQYF",   "GLCTLVAML"),   # EBV BMLF1
    # ... add more from EVAL_TARGETS in pmhc_loader.py
]
```

### 4.2 Single-Point Mutation Scan

For each (CDR3b, peptide) pair:

```python
AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")

def mutation_scan(scorer, cdr3b, peptide):
    """Score all single-point mutations of CDR3b against peptide."""
    wt_score = scorer.score(cdr3b, peptide)  # or score_batch_fast
    results = []
    for pos in range(len(cdr3b)):
        for mut_aa in AMINO_ACIDS:
            if mut_aa == cdr3b[pos]:
                continue  # skip wild-type
            mutant = cdr3b[:pos] + mut_aa + cdr3b[pos+1:]
            mut_score = scorer.score(mutant, peptide)
            results.append({
                'position': pos,
                'wt_aa': cdr3b[pos],
                'mut_aa': mut_aa,
                'wt_score': wt_score,
                'mut_score': mut_score,
                'delta': mut_score - wt_score,
            })
    return results
```

### 4.3 Batch-Optimized Version

For efficiency, generate all mutants first, then score in a single batch:

```python
def mutation_scan_batch(scorer, cdr3b, peptide):
    """Batch-optimized mutation scan."""
    mutants = []
    meta = []
    for pos in range(len(cdr3b)):
        for mut_aa in AMINO_ACIDS:
            if mut_aa == cdr3b[pos]:
                continue
            mutant = cdr3b[:pos] + mut_aa + cdr3b[pos+1:]
            mutants.append(mutant)
            meta.append((pos, cdr3b[pos], mut_aa))

    # Batch score all mutants at once
    peptides = [peptide] * len(mutants)

    # For ERGO:
    scores = scorer.score_batch_fast(mutants, peptides)

    # For NetTCR (use lower-level scorer):
    # scores = scorer._scorer.score_batch(mutants, peptides)

    wt_score = scorer.score_batch_fast([cdr3b], [peptide])[0]

    results = []
    for i, (pos, wt_aa, mut_aa) in enumerate(meta):
        results.append({
            'position': pos,
            'wt_aa': wt_aa,
            'mut_aa': mut_aa,
            'wt_score': wt_score,
            'mut_score': scores[i],
            'delta': scores[i] - wt_score,
        })
    return results
```

### 4.4 Metrics to Compute

1. **Mean |delta|** per position — identifies which positions matter most
2. **Mean |delta|** per mutation type — identifies which substitutions matter most
3. **Correlation of deltas between ERGO and NetTCR** — measures agreement
4. **Variance of deltas across positions** — measures position sensitivity spread
5. **Delta distribution** — histogram of score changes, compare distributions

### 4.5 Visualization Ideas

- **Heatmap**: Position (x-axis) vs. mutant AA (y-axis), colored by delta score
- **Position importance bar chart**: Mean |delta| per position, overlay for both models
- **ERGO vs NetTCR scatter**: Each point = one mutation, x = ERGO delta, y = NetTCR delta
- **Substitution matrix**: 20x20 mean delta grouped by (wt_aa, mut_aa) pairs

---

## 5. File Reference

| File | Description |
|---|---|
| `tcrppo_v2/scorers/affinity_ergo.py` | ERGO scorer wrapper (BaseScorer interface) |
| `tcrppo_v2/scorers/affinity_nettcr.py` | NetTCR scorer wrapper (BaseScorer interface) |
| `tcrppo_v2/ERGO/ERGO_models.py` | ERGO model classes (AutoencoderLSTMClassifier, PaddingAutoencoder) |
| `tcrppo_v2/ERGO/ae_utils.py` | ERGO data encoding + batch utilities |
| `tcrppo_v2/evaluation/nettcr_scorer.py` | NetTCR model + training + scoring |
| `tcrppo_v2/ERGO/models/ae_mcpas1.pt` | ERGO weights (trained on McPAS) |
| `tcrppo_v2/ERGO/models/ae_vdjdb1.pt` | ERGO weights (trained on VDJdb, alternative) |
| `tcrppo_v2/ERGO/TCR_Autoencoder/tcr_ae_dim_100.pt` | ERGO TCR autoencoder weights |
| `data/nettcr_model.weights.h5` | NetTCR trained weights |
| `data/nettcr_train.csv` | NetTCR training data (232K rows) |
| `data/nettcr_test.csv` | NetTCR test data (41K rows) |
| `tcrppo_v2/utils/constants.py` | Shared constants (AA encodings, paths, max lengths) |

---

## 6. Environment Setup

```bash
conda activate tcrppo_v2

# ERGO needs PyTorch
python -c "import torch; print(torch.__version__)"

# NetTCR needs TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Quick verification
python -c "
from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
s = AffinityERGOScorer('tcrppo_v2/ERGO/models/ae_mcpas1.pt', device='cpu', mc_samples=1)
print('ERGO:', s.score('CASSIRSSYEQYF', 'GILGFVFTL'))
"

python -c "
from tcrppo_v2.scorers.affinity_nettcr import AffinityNetTCRScorer
s = AffinityNetTCRScorer(device='cpu')
print('NetTCR:', s.score('CASSIRSSYEQYF', 'GILGFVFTL'))
"
```
