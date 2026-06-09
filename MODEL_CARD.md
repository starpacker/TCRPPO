---
language: en
license: mit
tags:
  - tcr-design
  - reinforcement-learning
  - ppo
  - biology
  - immunology
  - protein-design
datasets:
  - mcpas-tcr
  - tcrdb
metrics:
  - auroc
  - binding-affinity
model-index:
  - name: TCRPPO-v2-trace104
    results:
      - task:
          type: tcr-design
          name: TCR-pMHC Binding Design
        metrics:
          - type: target-affinity
            value: 0.20
            name: Mean Target Affinity (tFold logit)
          - type: qualifying-tcrs
            value: 3
            name: Qualifying TCRs (aff>0.6, decoy=0)
---

# TCRPPO v2 Model Card

## Model Description

TCRPPO v2 is a reinforcement learning system for designing T-cell receptor (TCR) CDR3β sequences with:
- **High target affinity** (strong binding to target peptide-MHC)
- **High specificity** (low cross-reactivity with decoys)
- **High naturalness** (biologically plausible sequences)

### Model Architecture

- **Base Model**: PPO (Proximal Policy Optimization)
- **State Encoder**: ESM-2 650M (frozen, from Meta AI)
- **Policy Network**: 3-head autoregressive actor-critic
  - Hidden dim: 512
  - 3 action heads: op_type (4-way), position (L-way), token (20-way)
- **Affinity Scorer**: tFold (FP32, 735M feature extractor + 1.57M classifier)
- **Naturalness Scorer**: Autoencoder + GMM (poly-C detection)

### Training Data

- **Target peptides**: 45 peptides from McPAS-TCR (HLA-A*02:01)
- **Init TCRs**: TCRdb (700K+ CDR3β sequences) + VDJdb known binders
- **Decoy library**: 4-tier contrastive negatives
  - Tier A: 1-2 AA point mutants (hardest)
  - Tier B: 2-3 AA mutants
  - Tier C: 1900 unrelated peptides
  - Tier D: Known binders to other targets
- **Training steps**: 5M+ (trace104), 200K-800K (trace98/99)

## Intended Use

### Primary Use Case
Design TCR sequences for therapeutic applications:
- Cancer immunotherapy (tumor antigen targeting)
- Infectious disease (viral epitope targeting)
- Autoimmune disease research

### Out-of-Scope Uses
- Clinical diagnosis without experimental validation
- Direct therapeutic use without wet-lab testing
- Non-HLA-A*02:01 targets (model trained only on A*02:01)

## Model Performance

### trace104 (5M steps) — Triple Constraint Training

| Metric | Value | Status |
|--------|-------|--------|
| Target Affinity (mean) | 0.20 | ✅ Learned A>0 |
| Decoy Violation | 3.0 | 🔄 Still learning |
| Naturalness Score | 0.0 | ⏸️ Not activated |
| Training Strategy | Progressive gate raising | Phase 1: Learn A>0 first |

**Key Achievement**: Model successfully learned to bind targets (affinity crossed 0 from negative), but specificity (decoy discrimination) still needs improvement.

### Historical Best (test41)

| Metric | Value |
|--------|-------|
| Mean AUROC (12 peptides) | 0.6243 |
| Peptides with AUROC > 0.65 | 5/12 |
| Training Method | Two-phase (ERGO warm-start + contrastive fine-tune) |

### Qualifying TCRs (Affinity > 0.6, Decoy Violation = 0)

**Peptide: LLLDRLNQL**
- `CALNMGVRTEAAFYYCCCCF` (trace88, affinity=0.728)

**Peptide: LLWRGSIYKL**
- `CAISIDHGSGNTQYFCCCCN` (trace79, affinity=0.692)
- `CAISIDHGSGNAQYFCCCCK` (trace79, affinity=0.659)

## Limitations

1. **HLA Restriction**: Only trained on HLA-A*02:01 (most common allele, ~50% population)
2. **Decoy Specificity**: Current best model (trace104) still has high decoy violation (~3.0)
3. **Naturalness**: AE+GMM scorer effective for poly-C but may miss other unnatural patterns
4. **Computational Cost**: tFold inference ~1s per TCR (on cache miss)
5. **Dataset Bias**: Trained on existing VDJdb/McPAS data (may inherit biases)
6. **No 3D Structure Validation**: Predicted affinities not validated against AlphaFold2 structures

## Ethical Considerations

### Dual Use
TCR design technology could potentially be misused for:
- Creating autoreactive TCRs (autoimmune disease)
- Evading immune surveillance

**Mitigation**: Models are released for research purposes only. Wet-lab validation required before any clinical application.

### Bias and Fairness
- Training data (VDJdb/McPAS) primarily from European/North American cohorts
- HLA-A*02:01 focus may not generalize to other populations
- Consider population-specific HLA distributions in downstream applications

## Training Details

### Hyperparameters (trace104)

```yaml
Environment:
  max_steps: 8
  max_tcr_len: 20
  min_tcr_len: 8
  ban_stop: true
  terminal_reward_only: true

Training:
  n_envs: 8
  batch_size: 256
  n_epochs: 4
  learning_rate: 0.0001
  clip_range: 0.2
  ent_coef: 0.001
  gamma: 0.99
  gae_lambda: 0.95

Reward:
  w_affinity: 2.0
  w_decoy: 1.5
  w_naturalness: 3.0
  
Affinity Scorer:
  type: tfold
  precision: fp32
  
Naturalness:
  type: ae_gmm
  threshold: 0.7
  
Decoy:
  K: 8 decoys per target
  tau: 10.0 (LogSumExp temperature)
  difficulty: medium
```

### Compute Infrastructure
- **GPU**: 4x NVIDIA A800-SXM4-80GB (80GB VRAM each)
- **Training Time**: ~5 days (trace104, 5M steps)
- **Throughput**: ~150-200 steps/min with tFold cache
- **Total GPU Hours**: ~480 GPU-hours (single A800)

## Environmental Impact

- **Total Training Time**: ~20 days (all traces combined)
- **Estimated CO₂ Emissions**: ~200 kg CO₂eq (assuming 0.5 kg CO₂/GPU-hour)
- **Power Consumption**: ~400W per A800 GPU

## Citation

```bibtex
@article{tcrppo_v2_2026,
  title={TCRPPO v2: Reinforcement Learning for TCR Design with Triple Constraints},
  author={[Your Name]},
  journal={bioRxiv},
  year={2026},
  url={https://huggingface.co/starpacker/tcrppo-v2}
}
```

## Contact

For questions or collaborations:
- **GitHub Issues**: https://github.com/starpacker/TCRPPO/issues
- **Email**: [Your Email]

## Acknowledgments

- **tFold**: Yu et al., "tFold: Structure-based TCR-pMHC binding prediction"
- **ESM-2**: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure"
- **ERGO**: Springer et al., "Prediction of specific TCR-peptide binding"
- **McPAS-TCR**: Tickotsky et al., "McPAS-TCR: a manually curated catalogue"
- **VDJdb**: Shugay et al., "VDJdb: a database of T-cell receptor sequences"

## Version History

- **v2.0 (2026-06-10)**: Initial release
  - trace104 (5M steps): Triple constraint training
  - trace98/99: Finetune experiments
  - 3 qualifying TCRs across 2 peptides

## License

MIT License — See LICENSE file for details
