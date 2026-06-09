# Cross-Attention Architecture Experiment (trace48)

**Date:** 2026-05-24  
**Goal:** Test if cross-attention improves peptide pathway contribution

---

## Background

诊断发现所有训练过的 policy 都存在 **peptide pathway 严重衰减**：
- Peptide sensitivity (var ratio): 0.08-0.16 (期望 ~1.0)
- Peptide-centroid collapse: cos-sim = 0.76 (期望 <0.5)
- 仅 3-5% 神经元对 peptide 有强响应

**根本原因：** Episode 内 peptide embedding 是常量，网络学会走捷径（只看 TCR 变化）。

---

## Solution: Cross-Attention Architecture

### Old Architecture (trace29)
```
obs = [TCR_emb | pep_emb | scalars] → Linear(2562→512) → ReLU → ...
```
- 简单 concatenation，peptide 信息容易被忽略

### New Architecture (trace48)
```
TCR pathway:    TCR_emb → Linear(1280→256) → ReLU
Peptide pathway: pep_emb → Linear(1280→256) → ReLU
Cross-attention: TCR attends to peptide (4 heads)
Fusion: [TCR_features | attended_pep | pep_features | scalars] → backbone
```
- **强制显式建模 TCR-peptide 交互**
- Peptide 有独立表征通路
- Cross-attention 确保 TCR 特征"看到" peptide 信息

**Parameters:** 2,008,365 (vs 旧架构 ~1.8M)

---

## Experiment Setup

### Config: `configs/trace48_cross_attn.yaml`
- Based on trace29 (test62_simple_target_gated_decoy)
- **Policy:** `ActorCriticCrossAttn` (new)
  - `use_cross_attn: true`
  - `n_attn_heads: 4`
- **Training:** From scratch (架构不兼容，无法从旧 checkpoint 迁移)
- **Reward:** Same as trace29 (v2_simple_target_gated_decoy)
- **Targets:** 20 peptides from `data/tfold_excellent_peptides.txt`

### Launch
```bash
./launch_trace48_cross_attn.sh
```

---

## Real ESM-2 Peptide Embedding Analysis

**Key Finding:** 训练集的 20 个 peptide 的 ESM-2 embedding **高度相似**：
- Mean pairwise cosine similarity: **0.9685**
- Min: 0.9285, Max: 0.9876
- PC1+PC2 只解释 36.5% 方差

**Implication:** 
1. Peptide 信息确实分散在高维空间（需要 12 个 PC 才能达到 90% 方差）
2. 简单 MLP 很难从这种高相似度的 embedding 中提取区分性特征
3. Cross-attention 的显式交互机制可能更有效

**Visualization:** `results/peptide_embedding_viz_real/peptide_embeddings_real_esm2.png`

---

## Expected Outcomes

### Success Metrics
1. **Peptide sensitivity ratio > 0.3** (vs 当前 0.08-0.16)
2. **Peptide-centroid collapse cos-sim < 0.6** (vs 当前 0.76)
3. **>10% L2 neurons with >50% pep-explained variance** (vs 当前 3-5%)

### Diagnosis Plan
训练完成后，运行：
```bash
python diagnose_pep_synthetic.py \
  --checkpoints \
    trace48_cross_attn=output/trace48_cross_attn/checkpoints/latest.pt \
    trace29_baseline=output/test62_simple_target_gated_decoy_trace29_simple_target_gated_decoy/checkpoints/latest.pt \
  --output results/trace48_vs_trace29_diagnosis
```

---

## Files Created

### Architecture
- `tcrppo_v2/policy_cross_attn.py` — Cross-attention policy implementation
- `configs/trace48_cross_attn.yaml` — Experiment config

### Scripts
- `launch_trace48_cross_attn.sh` — Launch script
- `convert_to_cross_attn.py` — Checkpoint converter (unused, training from scratch)
- `diagnose_pep_contribution.py` — Dead neuron + ablation + gradient flow analysis
- `diagnose_pep_synthetic.py` — Per-neuron sensitivity + collapse analysis
- `visualize_peptide_embeddings_real.py` — Real ESM-2 embedding visualization

### Results
- `PEPTIDE_PATHWAY_DIAGNOSIS_REPORT.md` — Full diagnosis report
- `results/pep_synthetic_diagnosis/` — Baseline diagnosis (trace29/43/46/47)
- `results/peptide_embedding_viz_real/` — Real ESM-2 peptide embedding plots

---

## Next Steps

1. **Launch trace48:** `./launch_trace48_cross_attn.sh`
2. **Monitor training:** `tail -f logs/trace48_cross_attn_train.log`
3. **Diagnose after training:** Compare trace48 vs trace29 peptide sensitivity
4. **If successful:** Consider additional improvements:
   - Peptide classification auxiliary loss
   - TCR dropout regularization
   - Multi-peptide episodes

---

**Status:** Ready to launch  
**Estimated training time:** ~同 trace29 (2M steps, ~数小时取决于 GPU)
