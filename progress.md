# TCRPPO Deployment Progress

**Repository**: https://github.com/ninglab/TCRPPO
**Date**: 2026-04-07
**Platform**: Windows 11 Pro, no CUDA GPU

---

## 1. Environment Setup

### Conda Environment
```bash
conda create -n tcrppo python=3.6.13 -y
conda activate tcrppo
```

### Installed Packages
- **PyTorch 1.10.2+cpu** (CPU-only, no CUDA available)
- **gym 0.21.0** (classic OpenAI Gym, not gymnasium)
- **scikit-learn 0.24.2**
- **numpy 1.19.5**
- **pandas 1.1.5**
- **matplotlib 3.3.4**
- **gdown 4.7.3** (for Google Drive data download)

```bash
pip install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install gym==0.21.0 scikit-learn pandas matplotlib gdown
```

> **[CUDA SKIP]** No CUDA GPU detected. PyTorch installed as CPU-only. Training and inference will run on CPU (significantly slower than GPU). For CUDA support, install:
> ```bash
> pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
> ```

---

## 2. Repository Structure

```
TCRPPO/
├── code/
│   ├── tcr_env.py          # Main training script (PPO RL environment)
│   ├── test_RL_tcrs.py     # Testing/inference script
│   ├── reward.py           # Reward model (ERGO + GMM)
│   ├── config.py           # Configuration constants
│   ├── ppo.py              # Modified PPO implementation
│   ├── policy.py           # Policy network
│   ├── seq_embed.py        # Sequence feature extractor
│   ├── data_utils.py       # Data utilities (BLOSUM, seq encoding)
│   ├── env_util.py         # Environment utilities
│   ├── ERGO/               # ERGO binding predictor
│   │   ├── models/         # Pre-trained ERGO models (ae_mcpas1.pt, ae_vdjdb1.pt)
│   │   ├── ERGO_models.py  # Model definitions (cloned from IdoSpringer/ERGO)
│   │   ├── ae_utils.py     # Autoencoder utilities
│   │   ├── lstm_utils.py   # LSTM utilities
│   │   └── TCR_Autoencoder/# TCR autoencoder (trained dim=100 version)
│   ├── reward/             # Reward sub-models
│   │   ├── ae_model        # Pre-trained AE model for reward
│   │   └── gmm.pkl         # GMM model for TCR-likeness
│   └── like_ratio/         # Likelihood ratio models
│       ├── semantic_model   # Semantic likelihood model
│       └── background_model # Background likelihood model
├── data/
│   ├── test_peptides/      # Test peptide files (4 sets: ae/lstm x mcpas/vdjdb)
│   └── tcrdb/              # TCRdb data (downloaded from Google Drive)
│       ├── train_uniq_tcr_seqs.txt  # 114MB, training TCR sequences
│       ├── test_uniq_tcr_seqs.txt   # 784KB, test TCR sequences
│       └── length_dist.txt          # TCR length distribution
├── stable_baselines3/      # Vendored & modified Stable-Baselines3 0.11.0a7
├── output/                 # Model checkpoints (created)
└── results/                # Test results (created)
```

---

## 3. Data Preparation

### 3.1 ERGO Source Code
The repo only ships `code/ERGO/models/` (pre-trained weights). We cloned the full ERGO repo (https://github.com/IdoSpringer/ERGO) and copied the required source files:
- `ERGO_models.py` — Model class definitions
- `ae_utils.py` — Autoencoder utility functions
- `lstm_utils.py` — LSTM utility functions
- `TCR_Autoencoder/` — Autoencoder training code and data

### 3.2 TCR Autoencoder (dim=100)
The code expects `TCR_Autoencoder/tcr_ae_dim_100.pt` (encoding_dim=100, max_len=28), but the ERGO repo only ships a dim=30 version. We trained a compatible model:

```python
# Trained with: 5000 TCR sequences, 50 epochs, encoding_dim=100, max_len=28
# Loss converged: 0.0105 at epoch 50
# File: code/ERGO/TCR_Autoencoder/tcr_ae_dim_100.pt (1.74MB)
```

> **Note**: This is a quick-trained model for functional validation. For production use, train on the full dataset (559K sequences) for 300 epochs using the original script:
> ```bash
> cd code/ERGO/TCR_Autoencoder
> python train_tcr_autoencoder.py BM_data_CDR3s cuda:0 tcr_ae_dim_100.pt 100
> ```

### 3.3 TCRdb Dataset
Downloaded from Google Drive (https://drive.google.com/drive/folders/1l5Pf50-7sDcKodeIo-VMHRlODu_ruGtM):
- `train_uniq_tcr_seqs.txt` — 114MB unique TCR CDR3beta sequences
- `test_uniq_tcr_seqs.txt` — Test TCR sequences
- `length_dist.txt` — TCR length distribution

### 3.4 McPAS Peptides File
Created `code/ERGO/data/McPAS-TCR/peptides.txt` (required by test script's default `--peptide_path`), populated with the same peptides from `data/test_peptides/ae_mcpas_test_peptides.txt`.

---

## 4. Training Validation

### Command
```bash
python ./code/tcr_env.py \
    --num_envs 2 \
    --ergo_model ./code/ERGO/models/ae_mcpas1.pt \
    --peptide_path ./data/test_peptides/ae_mcpas_test_peptides.txt \
    --bad_ratio 0.0 \
    --hidden_dim 256 \
    --latent_dim 128 \
    --gamma 0.90 \
    --steps 512 \
    --n_steps 128 \
    --path ./output/test_run
```

### Result: SUCCESS
```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 8           |
|    ep_rew_mean          | -0.423      |
| time/                   |             |
|    fps                  | 27          |
|    iterations           | 2           |
|    time_elapsed         | 18          |
|    total_timesteps      | 512         |
| train/                  |             |
|    approx_kl            | 0.017871803 |
|    clip_fraction        | 0.326       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.62       |
|    explained_variance   | -0.065      |
|    learning_rate        | 0.0003      |
|    loss                 | -0.174      |
|    n_updates            | 10          |
|    policy_gradient_loss | -0.0796     |
|    value_loss           | 0.0234      |
-----------------------------------------
finish training in 76.0724
saving model.....
```

Model saved to: `output/test_runmcpas_0.5_0.0_0.9_128_None/ppo_tcr.zip`

---

## 5. Testing/Inference Validation

### Command
```bash
python ./code/test_RL_tcrs.py \
    --num_envs 2 \
    --out ./results/test_output.txt \
    --ergo_model ./code/ERGO/models/ae_mcpas1.pt \
    --peptides ./data/test_peptides/ae_mcpas_test_peptides.txt \
    --rollout 1 \
    --tcrs ./data/tcrdb/test_uniq_tcr_seqs.txt \
    --path ./output/test_runmcpas_0.5_0.0_0.9_128_None/ppo_tcr \
    --hour 0 \
    --max_size 20 \
    --device cpu
```

### Result: SUCCESS
The test pipeline ran end-to-end without errors. The RL agent successfully:
1. Loaded the trained PPO model
2. Initialized TCR environments
3. Performed TCR sequence modifications guided by the policy
4. Computed ERGO binding scores and GMM TCR-likeness scores
5. Output results (empty due to minimal training — expected for 512-step model)

Sample output from inference:
```
terminal: False; action: 5,17; old_tcr: CSVALAEYNEQFF; new_tcr: CSVALTEYNEQFF;
init_tcr: CSVALAEYNEQFF; peptide: SSLENFRAYV; rewards: 0.0000; score: 0.0001;
score1: 1.0000; score2: 0.8873;
```

---

## 6. Full Training Command (for production)

> **[CUDA SKIP]** The following commands require GPU for reasonable training time. CPU training is functional but extremely slow for full runs.

```bash
# AE + McPAS (recommended starting point)
python ./code/tcr_env.py \
    --num_envs 20 \
    --ergo_model ./code/ERGO/models/ae_mcpas1.pt \
    --peptide_path ./data/test_peptides/ae_mcpas_test_peptides.txt \
    --bad_ratio 0.0 \
    --hidden_dim 256 \
    --latent_dim 128 \
    --gamma 0.90 \
    --path ./output/ae_mcpas

# AE + VDJdb
python ./code/tcr_env.py \
    --num_envs 20 \
    --ergo_model ./code/ERGO/models/ae_vdjdb1.pt \
    --peptide_path ./data/test_peptides/ae_vdjdb_test_peptides.txt \
    --hidden_dim 256 \
    --latent_dim 128 \
    --gamma 0.90 \
    --path ./output/ae_vdjdb
```

---

## 7. Issues Encountered & Fixes

| Issue | Fix |
|-------|-----|
| Missing `pandas` module | `pip install pandas` |
| Missing `matplotlib` module | `pip install matplotlib` |
| ERGO source code not in repo (only `models/` shipped) | Cloned IdoSpringer/ERGO, copied required `.py` files |
| `tcr_ae_dim_100.pt` not available (ERGO ships dim=30 only) | Trained dim=100 autoencoder with max_len=28 |
| Autoencoder shape mismatch (max_len=23 vs expected 28) | Retrained with fixed max_len=28 |
| Missing `code/ERGO/data/McPAS-TCR/peptides.txt` for test script | Created from test_peptides data |
| Test script `--path` expects model file, not directory | Pass `path/ppo_tcr` (without .zip) |
| No CUDA available | Installed CPU-only PyTorch; all code has CPU fallback |

---

## 8. Summary

| Item | Status |
|------|--------|
| Conda env (tcrppo, Python 3.6.13) | Done |
| PyTorch + dependencies | Done (CPU-only) |
| ERGO source code | Done |
| TCR Autoencoder (dim=100) | Done (quick-trained, functional) |
| TCRdb data download | Done |
| Training pipeline | Validated (512 steps) |
| Testing pipeline | Validated (end-to-end) |
| CUDA/GPU support | SKIPPED (no GPU detected) |
| Full production training | SKIPPED (requires GPU for practical runtime) |
