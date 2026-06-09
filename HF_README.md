# TCRPPO v2 — TCR Design with RL

**Designing T-Cell Receptors with High Affinity, Specificity, and Naturalness**

[![GitHub](https://img.shields.io/badge/GitHub-starpacker%2FTCRPPO-blue)](https://github.com/starpacker/TCRPPO)
[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-red)]()

---

## 🎯 Overview

TCRPPO v2 uses **Proximal Policy Optimization (PPO)** reinforcement learning to design T-cell receptor (TCR) CDR3β sequences that simultaneously satisfy three critical constraints:

1. **High Target Affinity** — Strong binding to target peptide-MHC complexes
2. **High Specificity** — Low cross-reactivity with decoy/self peptides
3. **High Naturalness** — Biologically plausible sequences

### Key Innovations
- **tFold FP32 Affinity Scorer** — Structure-based binding prediction (735M params)
- **AE+GMM Naturalness** — Effective poly-C pattern detection
- **Online Dynamic Pool** — Curriculum learning from elite TCRs
- **Triple Constraint Gating** — Progressive multi-objective optimization

---

## 📦 Repository Contents

### Checkpoints

| File | Trace | Steps | Description |
|------|-------|-------|-------------|
| `checkpoints/trace104_5M.pt` | trace104 | 5M | Triple constraint (nat>0.7, target>0.6, decoy<0.6) |
| `checkpoints/trace104_latest.pt` | trace104 | Latest | Same as above, most recent |
| `checkpoints/trace98_200K.pt` | trace98 | 200K | Finetune from trace61 baseline |
| `checkpoints/trace99_800K.pt` | trace99 | 800K | High naturalness weight (w=5.0) |
| `checkpoints/trace61_baseline.pt` | trace61 | 700K | FP32 baseline model |

### Results

- `all_traces_qualifying.json` — 3 qualifying TCRs (affinity > 0.6, decoy violation = 0)
- `logs/alive_traces_affinity_summary_v2.csv` — Training statistics across all traces
- `docs/tcrppo_v2_report.html` — Interactive results dashboard

---

## 🚀 Quick Start

### 1. Load a Trained Model

```python
import torch
from tcrppo_v2.policy import ActorCritic

# Load checkpoint
checkpoint = torch.load("checkpoints/trace104_5M.pt")
policy = ActorCritic(...)  # Initialize with same config
policy.load_state_dict(checkpoint['policy_state_dict'])

# Generate TCRs for a target peptide
tcrs = generate_tcrs(policy, target_peptide="GILGFVFTL", n_samples=50)
```

### 2. Evaluate Affinity with tFold

```python
from tcrppo_v2.scorers.affinity_tfold import TFoldScorer

scorer = TFoldScorer()
affinity = scorer.score(tcr="CASSLAPGTQYF", peptide="GILGFVFTL", hla="HLA-A*02:01")
print(f"Binding affinity logit: {affinity:.3f}")
```

---

## 📊 Current Results (as of 2026-06-10)

### Best Performing Traces

| Trace | Steps | Target Aff ↑ | Decoy Viol ↓ | Naturalness ↑ | Status |
|-------|-------|-------------|--------------|---------------|--------|
| trace104 | 5.1M | **0.20** | 3.0 | 0.0 | 🔄 Learning A>0 |
| trace88 | 923K | **0.73** | 0.0 | — | ✅ Best affinity |
| trace79 | 686K | **0.69** | 0.0 | — | ✅ 2 qualifying |

### Qualifying TCRs

**LLLDRLNQL** (best, affinity = 0.728):
- `CALNMGVRTEAAFYYCCCCF` (trace88, step 923K)

**LLWRGSIYKL** (affinity = 0.692):
- `CAISIDHGSGNTQYFCCCCN` (trace79, step 686K)
- `CAISIDHGSGNAQYFCCCCK` (trace79, step 686K)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Environment                                             │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ TCR String │→ │ ESM-2 650M   │→ │ State Vector    │ │
│  │ SUB/INS/DEL│  │ (Frozen)     │  │ [TCR|pMHC|meta] │ │
│  └────────────┘  └──────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Policy (Actor-Critic)                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  3-Head Autoregressive:                         │   │
│  │   1. op_type  → {SUB, INS, DEL, STOP}          │   │
│  │   2. position → [0, L-1]                        │   │
│  │   3. token    → [0, 19] (AA alphabet)          │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Reward Manager                                          │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ tFold Scorer │  │ Decoy Penalty│  │ AE+GMM Nat  │  │
│  │ (FP32, 735M) │  │ (LogSumExp)  │  │ (poly-C det)│  │
│  └──────────────┘  └─────────────┘  └──────────────┘  │
│         R = w_aff * target - w_dec * decoys - w_nat    │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Training Configuration

### Hyperparameters (trace104)

```yaml
# Environment
max_steps: 8
max_tcr_len: 20
min_tcr_len: 8
ban_stop: true
terminal_reward_only: true

# Training
n_envs: 8
batch_size: 256
learning_rate: 0.0001
clip_range: 0.2
ent_coef: 0.001

# Reward weights
w_affinity: 2.0
w_decoy: 1.5
w_naturalness: 3.0

# Affinity scorer
affinity_scorer_type: "tfold"

# Naturalness scorer
naturalness_scorer_type: "ae_gmm"
naturalness_ae_threshold: 0.7

# Decoy library
decoy_K: 8
decoy_tau: 10.0
decoy_difficulty: "medium"
```

---

## 📚 Citation

If you use this code or models, please cite:

```bibtex
@article{tcrppo_v2_2026,
  title={TCRPPO v2: Reinforcement Learning for TCR Design with Affinity, Specificity, and Naturalness},
  author={[Your Name]},
  journal={bioRxiv},
  year={2026}
}
```

---

## 🔗 Links

- **GitHub**: [starpacker/TCRPPO](https://github.com/starpacker/TCRPPO)
- **Paper**: Coming soon
- **Contact**: [Your Email]

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- **tFold**: Structure-based TCR-pMHC binding prediction
- **ESM-2**: Protein language model by Meta AI
- **ERGO**: Baseline TCR-pMHC binding predictor
- **McPAS-TCR**: T-cell receptor database
