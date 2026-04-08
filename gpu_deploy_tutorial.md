# TCRPPO GPU Deployment & Reproduction Tutorial

**Target**: Linux server with NVIDIA GPU (CUDA 11.3+)
**Source**: Windows local deployment at `C:\Users\30670\Desktop\TCRPPO`
**Goal**: Full GPU training + reproduce paper results

---

## Step 0: Upload Files to Server

```bash
# On local machine (Windows), pack the folder
# Exclude __pycache__, .git, and the quick-trained autoencoder
cd C:\Users\30670\Desktop
tar -czf TCRPPO.tar.gz --exclude='__pycache__' TCRPPO/

# Upload to server
scp TCRPPO.tar.gz user@server:/home/user/

# On server
cd /home/user
tar -xzf TCRPPO.tar.gz
cd TCRPPO
```

### Verify critical files exist after upload

```bash
# These must all exist:
ls code/tcr_env.py                          # Training script
ls code/test_RL_tcrs.py                     # Testing script
ls code/reward.py                           # Reward model
ls code/ERGO/models/ae_mcpas1.pt            # Pre-trained ERGO (AE+McPAS)
ls code/ERGO/models/ae_vdjdb1.pt            # Pre-trained ERGO (AE+VDJdb)
ls code/ERGO/ERGO_models.py                 # ERGO model definitions
ls code/ERGO/ae_utils.py                    # AE utilities
ls code/ERGO/lstm_utils.py                  # LSTM utilities
ls code/ERGO/TCR_Autoencoder/train_tcr_autoencoder.py
ls code/ERGO/TCR_Autoencoder/BM_data_CDR3s/ # AE training data
ls code/reward/ae_model                     # Reward AE model
ls code/reward/gmm.pkl                      # GMM model
ls code/like_ratio/semantic_model           # Likelihood model
ls code/like_ratio/background_model         # Likelihood model
ls data/tcrdb/train_uniq_tcr_seqs.txt       # 114MB TCR training data
ls data/tcrdb/test_uniq_tcr_seqs.txt        # Test TCR sequences
ls data/tcrdb/length_dist.txt               # Length distribution
ls data/test_peptides/ae_mcpas_test_peptides.txt
ls data/test_peptides/ae_vdjdb_test_peptides.txt
ls data/test_peptides/lstm_mcpas_test_peptides.txt
ls data/test_peptides/lstm_vdjdb_test_peptides.txt
ls code/ERGO/data/McPAS-TCR/peptides.txt    # Required by test script default
ls stable_baselines3/                       # Vendored SB3 (DO NOT pip install)
```

---

## Step 1: Create Conda Environment (GPU)

```bash
# Create environment
conda create -n tcrppo python=3.6.13 -y
conda activate tcrppo

# Install PyTorch 1.10.2 with CUDA 11.3
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install other dependencies
pip install gym==0.21.0 scikit-learn==0.24.2 numpy==1.19.5 \
    pandas==1.1.5 matplotlib==3.3.4

# Verify CUDA is available
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:
```
CUDA: True
Device: NVIDIA A100-SXM4-80GB    # (or your GPU name)
```

> **Important**: Do NOT `pip install stable-baselines3`. The repo ships a vendored & modified version in `stable_baselines3/`. Installing the PyPI version will break things.

---

## Step 2: Train TCR Autoencoder (Full, GPU)

The local deployment used a quick-trained autoencoder (5000 sequences, 50 epochs). For reproduction, train the full version.

```bash
cd /home/user/TCRPPO

# Train with full BM_data_CDR3s dataset, 300 epochs, encoding_dim=100
python code/ERGO/TCR_Autoencoder/train_tcr_autoencoder.py \
    code/ERGO/TCR_Autoencoder/BM_data_CDR3s \
    cuda:0 \
    code/ERGO/TCR_Autoencoder/tcr_ae_dim_100.pt \
    100
```

**What this does**:
- Loads ~559K TCR CDR3 sequences from BM_data_CDR3s/
- Trains PaddingAutoencoder with encoding_dim=100 for 300 epochs
- Saves to `tcr_ae_dim_100.pt` with keys: `amino_to_ix`, `ix_to_amino`, `batch_size`, `max_len`, `enc_dim`, `model_state_dict`

**Critical**: The saved `max_len` must be >= 28 (reward.py hardcodes `max_len=28` at line 64). The BM_data_CDR3s dataset should naturally produce max_len >= 28. Verify after training:

```python
import torch
ckpt = torch.load('code/ERGO/TCR_Autoencoder/tcr_ae_dim_100.pt')
print(f"max_len={ckpt['max_len']}, enc_dim={ckpt['enc_dim']}")
# Must show: max_len=28 (or close), enc_dim=100
```

> **If max_len != 28**: The ERGO model in reward.py creates `AutoencoderLSTMClassifier(..., 28, 21, 100, ...)`. If max_len from training differs, you'll get a shape mismatch. See Troubleshooting section below.

**Estimated time**: ~10-30 min on GPU (vs hours on CPU)

---

## Step 3: Verify Setup with Quick Training Run

Before committing to full 10M-step training, do a sanity check:

```bash
cd /home/user/TCRPPO

python ./code/tcr_env.py \
    --num_envs 4 \
    --ergo_model ./code/ERGO/models/ae_mcpas1.pt \
    --peptide_path ./data/test_peptides/ae_mcpas_test_peptides.txt \
    --bad_ratio 0.0 \
    --hidden_dim 256 \
    --latent_dim 128 \
    --gamma 0.90 \
    --steps 1024 \
    --n_steps 256 \
    --path ./output/sanity_check_
```

Expected: completes in ~1-2 minutes, prints training stats, saves model. If this fails, fix before proceeding.

---

## Step 4: Full Training (Reproduce Paper)

The paper uses 4 configurations: 2 ERGO architectures (AE, LSTM) x 2 datasets (McPAS, VDJdb).
However, the repo **only ships AE models** (`ae_mcpas1.pt`, `ae_vdjdb1.pt`). LSTM ERGO models are not provided.

### 4.1 AE + McPAS (primary configuration)

```bash
python ./code/tcr_env.py \
    --num_envs 20 \
    --ergo_model ./code/ERGO/models/ae_mcpas1.pt \
    --peptide_path ./data/test_peptides/ae_mcpas_test_peptides.txt \
    --bad_ratio 0.0 \
    --hidden_dim 256 \
    --latent_dim 128 \
    --gamma 0.90 \
    --path ./output/ae_mcpas_
```

### 4.2 AE + VDJdb

```bash
python ./code/tcr_env.py \
    --num_envs 20 \
    --ergo_model ./code/ERGO/models/ae_vdjdb1.pt \
    --peptide_path ./data/test_peptides/ae_vdjdb_test_peptides.txt \
    --bad_ratio 0.0 \
    --hidden_dim 256 \
    --latent_dim 128 \
    --gamma 0.90 \
    --path ./output/ae_vdjdb_
```

### Training Parameters (from README, matches paper)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_envs` | 20 | Parallel environments, scale to CPU cores |
| `bad_ratio` | 0.0 | README example uses 0.0 (default is 0.5) |
| `hidden_dim` | 256 | README example (default is 128) |
| `latent_dim` | 128 | README example (default is 64) |
| `gamma` | 0.90 | README example (default is 0.99) |
| `steps` | 10,000,000 | Default, ~10M timesteps |
| `n_steps` | 256 | Rollout steps per env per update |
| `beta` | 0.5 | Reward weighting (ERGO vs GMM) |
| `ent_coef` | 0.01 | Entropy bonus for exploration |
| `clip` | 0.2 | PPO clipping range |
| `max_step` | 8 | Max modifications per episode |

### Output Path Convention

The script auto-generates a subdirectory name:
```
{path}{dataset}_{beta}_{bad_ratio}_{gamma}_{n_steps}_{latent_dim}/
```

For AE+McPAS with above args:
```
output/ae_mcpas_mcpas_0.5_0.0_0.9_256_128/ppo_tcr.zip
```

### Checkpoints

Saved every 50,000 steps in the output directory:
```
output/ae_mcpas_mcpas_0.5_0.0_0.9_256_128/
├── rl_model_50000_steps.zip
├── rl_model_100000_steps.zip
├── ...
└── ppo_tcr.zip              # Final model
```

### Estimated Training Time

| Setup | ~Time for 10M steps |
|-------|---------------------|
| NVIDIA A100 | 3-6 hours |
| NVIDIA V100 | 6-12 hours |
| NVIDIA 3090 | 4-8 hours |
| CPU only | Days (impractical) |

> **Tip**: Monitor with `nvidia-smi` to confirm GPU utilization. The RL environments run on CPU (multiprocessing), but reward computation (ERGO forward pass) uses GPU.

---

## Step 5: Testing / Inference

After training completes, generate optimized TCR sequences:

### 5.1 Test AE + McPAS

```bash
python ./code/test_RL_tcrs.py \
    --num_envs 4 \
    --out ./results/ae_mcpas_results.txt \
    --ergo_model ./code/ERGO/models/ae_mcpas1.pt \
    --peptides ./data/test_peptides/ae_mcpas_test_peptides.txt \
    --rollout 1 \
    --tcrs ./data/tcrdb/test_uniq_tcr_seqs.txt \
    --path ./output/ae_mcpas_mcpas_0.5_0.0_0.9_256_128/ppo_tcr \
    --hour 5 \
    --max_size 50000
```

### 5.2 Test AE + VDJdb

```bash
python ./code/test_RL_tcrs.py \
    --num_envs 4 \
    --out ./results/ae_vdjdb_results.txt \
    --ergo_model ./code/ERGO/models/ae_vdjdb1.pt \
    --peptides ./data/test_peptides/ae_vdjdb_test_peptides.txt \
    --rollout 1 \
    --tcrs ./data/tcrdb/test_uniq_tcr_seqs.txt \
    --path ./output/ae_vdjdb_vdjdb_0.5_0.0_0.9_256_128/ppo_tcr \
    --hour 5 \
    --max_size 50000
```

### Test Script Notes

- `--path`: Must point to the model **file** (without `.zip`), NOT the directory
- `--hour 5`: Time limit in hours (default). Testing iterates over all peptide×TCR×rollout combinations
- `--max_size 50000`: Max results per peptide before stopping
- `--rollout 1`: Number of rollouts per peptide-TCR pair (increase for more samples)

### Output Format

File: `results/ae_mcpas_results.txt`
```
<peptide> <init_tcr> <final_tcr> <ergo_score> <seq_edit_dist> <gmm_likelihood>
```

Each line is one TCR optimization trajectory result:
- `peptide`: Target peptide the TCR was optimized for
- `init_tcr`: Original TCR CDR3β sequence (from test set)
- `final_tcr`: Optimized TCR after RL agent modifications
- `ergo_score`: ERGO binding prediction (0-1, higher = stronger binding)
- `seq_edit_dist`: 1 - normalized edit distance (higher = more similar to original)
- `gmm_likelihood`: GMM TCR-likeness score (higher = more realistic TCR)

---

## Step 6: Evaluate Results (Reproduce Paper Metrics)

The paper reports these key metrics for generated TCRs:
1. **ERGO Score**: Predicted binding affinity (should be > 0.9 for successful optimization)
2. **TCR-likeness (GMM)**: Whether generated TCRs resemble real TCRs
3. **Diversity**: Unique sequences among generated TCRs
4. **Edit distance**: How many mutations were needed

### Analysis Script

Create `analyze_results.py`:

```python
#!/usr/bin/env python3
"""Analyze TCRPPO test results to reproduce paper metrics."""
import sys
import numpy as np
from collections import defaultdict

def analyze(result_file):
    peptide_results = defaultdict(list)
    
    with open(result_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            peptide = parts[0]
            init_tcr = parts[1]
            final_tcr = parts[2]
            ergo_score = float(parts[3])
            seq_edit_dist = float(parts[4])
            gmm_likelihood = float(parts[5])
            peptide_results[peptide].append({
                'init': init_tcr,
                'final': final_tcr,
                'ergo': ergo_score,
                'edit_dist': seq_edit_dist,
                'gmm': gmm_likelihood
            })
    
    print(f"{'Peptide':<25} {'Count':>6} {'AvgERGO':>8} {'ERGO>0.9':>9} "
          f"{'AvgGMM':>8} {'AvgEdit':>8} {'Unique':>7} {'Changed':>8}")
    print("-" * 95)
    
    all_ergos = []
    all_gmms = []
    all_edits = []
    total_unique = 0
    total_count = 0
    total_changed = 0
    
    for peptide in sorted(peptide_results.keys()):
        results = peptide_results[peptide]
        n = len(results)
        ergos = [r['ergo'] for r in results]
        gmms = [r['gmm'] for r in results]
        edits = [r['edit_dist'] for r in results]
        unique_tcrs = len(set(r['final'] for r in results))
        changed = sum(1 for r in results if r['init'] != r['final'])
        high_ergo = sum(1 for e in ergos if e >= 0.9)
        
        all_ergos.extend(ergos)
        all_gmms.extend(gmms)
        all_edits.extend(edits)
        total_unique += unique_tcrs
        total_count += n
        total_changed += changed
        
        print(f"{peptide:<25} {n:>6} {np.mean(ergos):>8.4f} {high_ergo/n*100:>8.1f}% "
              f"{np.mean(gmms):>8.4f} {np.mean(edits):>8.4f} {unique_tcrs:>7} {changed:>8}")
    
    print("-" * 95)
    print(f"{'OVERALL':<25} {total_count:>6} {np.mean(all_ergos):>8.4f} "
          f"{sum(1 for e in all_ergos if e >= 0.9)/len(all_ergos)*100:>8.1f}% "
          f"{np.mean(all_gmms):>8.4f} {np.mean(all_edits):>8.4f} "
          f"{total_unique:>7} {total_changed:>8}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <result_file>")
        sys.exit(1)
    analyze(sys.argv[1])
```

Usage:
```bash
python analyze_results.py results/ae_mcpas_results.txt
python analyze_results.py results/ae_vdjdb_results.txt
```

### What to expect (paper-level results)

For a well-trained model (10M steps), the paper reports:
- **ERGO Score > 0.9** for a significant fraction of generated TCRs
- **GMM likelihood** scores indicating TCR-like sequences
- **Low edit distance** from original TCRs (the agent makes few, targeted mutations)
- Generated TCRs should be **diverse** (not collapsing to a few templates)

---

## Step 7: Running All Experiments (Batch Script)

```bash
#!/bin/bash
# run_all.sh — Train and test all available configurations
set -e
cd /home/user/TCRPPO
mkdir -p output results

echo "=== Step 1: Train TCR Autoencoder (full) ==="
python code/ERGO/TCR_Autoencoder/train_tcr_autoencoder.py \
    code/ERGO/TCR_Autoencoder/BM_data_CDR3s \
    cuda:0 \
    code/ERGO/TCR_Autoencoder/tcr_ae_dim_100.pt \
    100

echo "=== Step 2: Verify autoencoder ==="
python -c "
import torch
ckpt = torch.load('code/ERGO/TCR_Autoencoder/tcr_ae_dim_100.pt')
print(f'max_len={ckpt[\"max_len\"]}, enc_dim={ckpt[\"enc_dim\"]}')
assert ckpt['enc_dim'] == 100, 'enc_dim must be 100'
assert ckpt['max_len'] >= 28, f'max_len={ckpt[\"max_len\"]} < 28, will cause shape mismatch'
print('Autoencoder OK')
"

echo "=== Step 3: Train AE+McPAS ==="
python ./code/tcr_env.py \
    --num_envs 20 \
    --ergo_model ./code/ERGO/models/ae_mcpas1.pt \
    --peptide_path ./data/test_peptides/ae_mcpas_test_peptides.txt \
    --bad_ratio 0.0 \
    --hidden_dim 256 \
    --latent_dim 128 \
    --gamma 0.90 \
    --path ./output/ae_mcpas_

echo "=== Step 4: Train AE+VDJdb ==="
python ./code/tcr_env.py \
    --num_envs 20 \
    --ergo_model ./code/ERGO/models/ae_vdjdb1.pt \
    --peptide_path ./data/test_peptides/ae_vdjdb_test_peptides.txt \
    --bad_ratio 0.0 \
    --hidden_dim 256 \
    --latent_dim 128 \
    --gamma 0.90 \
    --path ./output/ae_vdjdb_

echo "=== Step 5: Test AE+McPAS ==="
python ./code/test_RL_tcrs.py \
    --num_envs 4 \
    --out ./results/ae_mcpas_results.txt \
    --ergo_model ./code/ERGO/models/ae_mcpas1.pt \
    --peptides ./data/test_peptides/ae_mcpas_test_peptides.txt \
    --rollout 1 \
    --tcrs ./data/tcrdb/test_uniq_tcr_seqs.txt \
    --path ./output/ae_mcpas_mcpas_0.5_0.0_0.9_256_128/ppo_tcr \
    --hour 5 \
    --max_size 50000

echo "=== Step 6: Test AE+VDJdb ==="
python ./code/test_RL_tcrs.py \
    --num_envs 4 \
    --out ./results/ae_vdjdb_results.txt \
    --ergo_model ./code/ERGO/models/ae_vdjdb1.pt \
    --peptides ./data/test_peptides/ae_vdjdb_test_peptides.txt \
    --rollout 1 \
    --tcrs ./data/tcrdb/test_uniq_tcr_seqs.txt \
    --path ./output/ae_vdjdb_vdjdb_0.5_0.0_0.9_256_128/ppo_tcr \
    --hour 5 \
    --max_size 50000

echo "=== Step 7: Analyze results ==="
python analyze_results.py results/ae_mcpas_results.txt
python analyze_results.py results/ae_vdjdb_results.txt

echo "=== ALL DONE ==="
```

```bash
chmod +x run_all.sh
nohup bash run_all.sh > run_all.log 2>&1 &
tail -f run_all.log
```

---

## Troubleshooting

### 1. Autoencoder max_len mismatch

**Symptom**: `RuntimeError: Error(s) in loading state_dict ... size mismatch for encoder.0.weight`

**Cause**: The trained autoencoder's `max_len` differs from 28 (hardcoded in `reward.py` line 64: `AutoencoderLSTMClassifier(10, device, 28, 21, 100, 1, ae_file, False)`).

**Fix**: Force max_len=28 during training. Modify the training script or train with a wrapper:

```python
# force_train_ae.py
import torch, sys, os
sys.path.insert(0, 'code/ERGO/TCR_Autoencoder')
from train_tcr_autoencoder import *

amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
amino_to_ix = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
ix_to_amino = {index: amino for index, amino in enumerate(amino_acids + ['X'])}
batch_size = 50

tcrs = load_all_data('code/ERGO/TCR_Autoencoder/BM_data_CDR3s')
train, test, _, _ = train_test_split(tcrs, tcrs, test_size=0.2)

max_len = max(find_max_len(tcrs), 28)  # Force at least 28
print(f"Using max_len={max_len}")

train_batches = get_batches(train, amino_to_ix, batch_size, max_len)
test_batches = get_batches(test, amino_to_ix, batch_size, max_len)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
encoding_dim = 100
model = train_model(train_batches, batch_size, max_len,
                    encoding_dim=encoding_dim, epochs=300, device=device)
evaluate(test_batches, batch_size, model, ix_to_amino, device)
torch.save({
    'amino_to_ix': amino_to_ix, 'ix_to_amino': ix_to_amino,
    'batch_size': batch_size, 'max_len': max_len,
    'enc_dim': encoding_dim, 'model_state_dict': model.state_dict(),
}, 'code/ERGO/TCR_Autoencoder/tcr_ae_dim_100.pt')
print("Saved tcr_ae_dim_100.pt")
```

### 2. CUDA out of memory

**Fix**: Reduce `--num_envs` (e.g., 10 instead of 20). The environments use CPU multiprocessing, but ERGO reward computation batches GPU forward passes.

### 3. `ModuleNotFoundError: No module named 'pandas'` / `'matplotlib'`

```bash
pip install pandas matplotlib
```

### 4. SubprocVecEnv hangs on startup

**Cause**: Too many `num_envs` for available CPU cores.
**Fix**: Set `--num_envs` to number of CPU cores minus 2.

### 5. Test script `--path` error

The `--path` argument must point to the model **file** (without `.zip` extension):
```bash
# WRONG:
--path ./output/ae_mcpas_mcpas_0.5_0.0_0.9_256_128/
# WRONG:
--path ./output/ae_mcpas_mcpas_0.5_0.0_0.9_256_128/ppo_tcr.zip
# CORRECT:
--path ./output/ae_mcpas_mcpas_0.5_0.0_0.9_256_128/ppo_tcr
```

### 6. Missing ERGO source files

If `code/ERGO/` only has `models/` directory:
```bash
git clone https://github.com/IdoSpringer/ERGO.git /tmp/ERGO
cp /tmp/ERGO/ERGO_models.py code/ERGO/
cp /tmp/ERGO/ae_utils.py code/ERGO/
cp /tmp/ERGO/lstm_utils.py code/ERGO/
cp -r /tmp/ERGO/TCR_Autoencoder code/ERGO/
rm -rf /tmp/ERGO
```

### 7. Output directory naming

The training script auto-generates subdirectory names. If your `--path` ends with `/`, the generated name may differ. Check:
```bash
ls output/  # See actual generated directory names
```

---

## Key Architecture Notes

### How TCRPPO Works

1. **Environment**: Each parallel env holds a (TCR, peptide) pair
2. **Action**: Agent picks (position ∈ [0,27], amino_acid ∈ [0,20]) to mutate one residue
3. **Observation**: Concatenated TCR (27 positions) + Peptide (25 positions), encoded as amino acid indices
4. **Reward**: ERGO binding score + β × GMM TCR-likeness
5. **Termination**: ERGO ≥ 0.9 AND GMM ≥ 1.2577, or after 8 steps
6. **PPO**: Modified stable-baselines3 with bad-sample replay buffer

### Vendored stable_baselines3

The `stable_baselines3/` directory is a **modified** version (v0.11.0a7). Key modifications:
- Custom reward model integration (reward computed externally, not from env)
- Bad-sample replay buffer for harder exploration
- Modified rollout collection

**Never** install stable-baselines3 from pip — it will override the vendored version.

---

## File Reference

| File | Purpose |
|------|---------|
| `code/tcr_env.py` | Training entry point + TCREnv gym environment |
| `code/test_RL_tcrs.py` | Testing/inference with trained model |
| `code/reward.py` | Reward: ERGO binding + GMM TCR-likeness |
| `code/ppo.py` | Modified PPO (n_epochs=10, batch_size=64) |
| `code/policy.py` | PolicyNet: Actor-Critic with SeqEmbed features |
| `code/seq_embed.py` | TCR/peptide sequence feature extraction |
| `code/config.py` | Global constants (device, paths, dimensions) |
| `code/data_utils.py` | BLOSUM encoding, sequence utilities |
| `code/ERGO/ERGO_models.py` | AutoencoderLSTMClassifier, DoubleLSTMClassifier |
| `code/ERGO/ae_utils.py` | AE model batch construction & prediction |
| `code/ERGO/lstm_utils.py` | LSTM model batch construction & prediction |
| `stable_baselines3/` | Vendored & modified SB3 (DO NOT pip install) |
