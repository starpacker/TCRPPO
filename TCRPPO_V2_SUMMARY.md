# TCRPPO v2 Project Summary

**Date**: 2026-06-10  
**Status**: Active Training (4 traces running)  
**Goal**: Design TCRs with high affinity + specificity + naturalness

---

## 🎯 Current Best Results

### Qualifying TCRs (Target Affinity > 0.6, Decoy Violation = 0)
- **Total peptides**: 2 (LLWRGSIYKL, LLLDRLNQL)
- **Total TCRs**: 3 candidates
- **Best affinity**: 0.728 (LLLDRLNQL, trace88)

### Active Training Traces (as of 2026-06-10)
| Trace | Steps | Status | Key Feature |
|-------|-------|--------|-------------|
| trace104_triple_constraint | 5.1M | ✅ Learning A>0 | Triple constraint (nat>0.7, target>0.6, decoy<0.6) |
| trace98_finetune | 216K | 🔄 Training | Finetune from trace61 |
| trace99_finetune_nat5 | 800K+ | 🔄 Training | High naturalness weight (5.0) |
| trace100_cross_attn | Running | 🔄 Training | Cross-attention architecture |

---

## 🏗️ Architecture

### Core Components
- **Environment**: Gym env with SUB/INS/DEL/STOP actions
- **Policy**: 3-head autoregressive actor-critic
- **State Encoder**: ESM-2 650M (frozen)
- **Affinity Scorer**: tFold (FP32, 735M params)
- **Naturalness**: AE+GMM combined scorer
- **Decoy Library**: 4-tier contrastive learning

### Key Innovations
1. **Terminal Reward**: Cleaner credit assignment
2. **Online Dynamic Pool**: Curriculum learning from elite TCRs
3. **Triple Constraint Gating**: Simultaneous optimization of 3 objectives
4. **AE+GMM Naturalness**: Effective poly-C detection

---

## 📊 Training Statistics

### trace104 (Most Promising)
- **Target Affinity**: 0.0 → 0.2 (learned to bind!)
- **Decoy Violation**: High (~3.0) - still learning specificity
- **Naturalness**: 0.0 (not activated yet)
- **Strategy**: Progressive gate raising (currently learning A>0)

### Historical Best (test41)
- **Mean AUROC**: 0.6243 (12 peptides)
- **Method**: Two-phase (ERGO warm-start + contrastive fine-tune)
- **Decoys**: 16 per target

---

## 📦 Repository Structure

```
tcrppo_v2/
├── tcrppo_v2/           # Source code
│   ├── env.py
│   ├── policy.py
│   ├── ppo_trainer.py
│   ├── reward_manager.py
│   └── scorers/
├── configs/             # Experiment configs
├── scripts/             # Training scripts
├── docs/                # Documentation
├── output/              # Checkpoints (13GB)
├── logs/                # Training logs (1.6GB)
├── results/             # Evaluation results (204MB)
└── data/                # Datasets + cache (219GB)
```

---

## 🚀 Next Steps

### Immediate (1-2 weeks)
1. **Adversarial Decoy Generation**: Dynamic hard negatives
2. **Uncertainty-Aware Exploration**: Bayesian RL with MC Dropout

### Medium-term (2-4 weeks)
3. **Multi-Objective Pareto Optimization**: Trade-off diversity
4. **Counterfactual Data Augmentation**: Causal learning

### Long-term (1-2 months)
5. **Meta-Learning**: Fast adaptation to new peptides
6. **Structure-Aware Design**: AlphaFold2 constraints
7. **Hierarchical RL**: Macro-actions for faster exploration

---

## 📚 Key Documents

- `CLAUDE.md`: Project governance
- `docs/2026-04-09-tcrppo-v2-design.md`: Architecture spec
- `docs/all_experiments_tracker.md`: Complete experiment history
- `PEPTIDE_SCORER_MAPPING.md`: Scorer reliability per peptide
- `docs/tcrppo_v2_report.html`: Current results dashboard

