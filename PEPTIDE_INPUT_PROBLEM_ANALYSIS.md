# Peptide Input Representation 问题分析与改进方案

**Date:** 2026-05-24  
**Critical Finding:** ESM-2 peptide embeddings 缺乏区分度（cos-sim = 0.9685）

---

## 🔴 问题：Peptide Input 本身就高度相似

### 实验数据

**20 个训练 peptide 的 ESM-2 embedding：**
- **Mean pairwise cosine similarity: 0.9685**
- Min: 0.9285, Max: 0.9876
- PC1+PC2 只解释 36.5% 方差
- 需要 12 个 PC 才能达到 90% 方差

**Peptide 序列特征：**
- 长度：8-10 aa（13 个 9-mer，6 个 10-mer，1 个 8-mer）
- 都是 HLA-A*02:01 结合肽
- 组成相似：疏水性 40-78%，带电 0-56%

### 为什么 ESM-2 embedding 如此相似？

1. **ESM-2 是通用蛋白质语言模型**
   - 在整个蛋白质序列空间上训练
   - 对于 8-10 aa 的短肽，信息量有限
   - 主要捕获"氨基酸共现模式"，而非 peptide-specific 特征

2. **HLA-A*02:01 结合肽的序列约束**
   - P2 位置偏好 L/M（anchor）
   - P9/P10 位置偏好 L/V/I（anchor）
   - 这些约束导致序列空间高度受限

3. **当前 input = peptide + HLA_pseudosequence**
   - HLA pseudosequence 对所有 peptide 都相同（都是 A*02:01）
   - 进一步稀释了 peptide 之间的差异

### 结论

**即使 cross-attention 架构也无法从如此相似的 input 中学到有意义的 peptide-specific 策略。**

这不是架构问题，而是 **input representation 问题**。

---

## 💡 改进方案

### Option 1: 更丰富的 Peptide Representation

#### 1A. 多尺度 Peptide Features
```python
peptide_input = [
    ESM_embedding(peptide),           # 1280-dim
    physicochemical_features(peptide), # 疏水性、电荷、芳香性等 profile
    position_specific_features(peptide), # P1-P10 的 one-hot + properties
    binding_motif_features(peptide),   # HLA anchor 位置的显式编码
]
```

**优势：** 显式编码 peptide 的结构和化学特性，增加区分度

#### 1B. Peptide-Specific Learned Embedding
```python
# 为每个 peptide 学习一个 low-dim embedding
peptide_id_embedding = nn.Embedding(num_peptides, 64)
peptide_input = [ESM_embedding, peptide_id_embedding[peptide_id]]
```

**优势：** 网络可以学习 peptide-specific 的表征  
**劣势：** 无法泛化到新 peptide

#### 1C. Contrastive Peptide Encoder
```python
# 用 contrastive learning 预训练一个 peptide encoder
# 目标：最大化不同 peptide 的 embedding 距离
peptide_encoder = ContrastivePeptideEncoder(pretrained=True)
peptide_input = peptide_encoder(peptide_sequence)
```

**优势：** 专门优化 peptide 区分度  
**劣势：** 需要额外的预训练

---

### Option 2: 改变训练信号

#### 2A. Multi-Peptide Episodes
```python
# 在同一个 episode 内切换 peptide
t=0-3:  edit TCR for peptide_A
t=4-7:  edit TCR for peptide_B (same TCR, different target)
```

**优势：** 
- Temporal credit assignment 依赖 peptide 变化
- 强制网络学习 peptide-conditional 策略

**实现：** 修改 `env.py` 的 `step()` 函数

#### 2B. Peptide Classification Auxiliary Task
```python
# 在训练中加入辅助任务：从 features 预测 peptide identity
peptide_logits = PeptideClassifier(features)
loss += alpha * CrossEntropy(peptide_logits, peptide_id)
```

**优势：** 强制 features 保留 peptide 信息

#### 2C. Peptide-Contrastive Reward
```python
# Reward 显式依赖 peptide 区分度
reward = affinity(TCR, target_peptide) - mean(affinity(TCR, other_peptides))
```

**优势：** 让 reward 信号直接依赖 peptide 特异性

---

### Option 3: 数据层面改进

#### 3A. 增加 Peptide 多样性
- 当前：20 个 HLA-A*02:01 肽（高度相似）
- 改进：包含多个 HLA 等位基因的肽
  - HLA-A*02:01, A*01:01, A*03:01, B*07:02, ...
  - 不同 HLA 的结合肽序列空间差异更大

#### 3B. 使用 Peptide Clusters
- 将 peptide 按序列相似度聚类
- 每个 cluster 采样 1-2 个代表
- 确保训练集 peptide 之间有足够差异

---

## 🎯 推荐方案（短期 + 长期）

### 短期（立即可测试）

**1. Multi-Peptide Episodes (Option 2A)**
- 最小改动，最大收益
- 直接解决"episode 内 peptide 不变"的核心问题
- 预期效果：peptide sensitivity 从 0.08 提升到 0.3-0.5

**2. Peptide Classification Auxiliary Loss (Option 2B)**
- 实现简单（~50 行代码）
- 与 cross-attention 架构互补
- 预期效果：进一步提升 peptide pathway 贡献

### 中期（需要一些工程）

**3. 多尺度 Peptide Features (Option 1A)**
- 增加 physicochemical + position-specific features
- 提升 input 区分度
- 预期效果：cos-sim 从 0.97 降到 0.7-0.8

### 长期（研究方向）

**4. Contrastive Peptide Encoder (Option 1C)**
- 专门为 TCR-peptide binding 优化的 peptide encoder
- 可能需要大规模 TCR-peptide binding 数据预训练

**5. 多 HLA 训练集 (Option 3A)**
- 扩展到多个 HLA 等位基因
- 提升模型泛化能力

---

## 📊 实验计划

### Experiment A: Multi-Peptide Episodes + Auxiliary Loss
```yaml
# trace49_multi_pep_episode.yaml
policy_class: "ActorCriticCrossAttn"  # 保留 cross-attention
max_steps_per_episode: 8
peptide_switch_interval: 4  # 每 4 步切换 peptide
auxiliary_peptide_classification: true
auxiliary_loss_weight: 0.1
```

### Experiment B: Enhanced Peptide Features
```yaml
# trace50_enhanced_pep_features.yaml
peptide_feature_mode: "multi_scale"  # ESM + physicochemical + positional
peptide_feature_dim: 1280 + 20 + 100  # = 1400
```

### 对比实验
- **Baseline:** trace29 (simple concat, 单 peptide episode)
- **Cross-attn:** trace48 (cross-attention, 单 peptide episode)
- **Multi-pep:** trace49 (cross-attention + multi-peptide episode + aux loss)
- **Enhanced:** trace50 (enhanced features + multi-peptide episode)

---

## 🔬 诊断指标

除了之前的 peptide sensitivity，还需要：

1. **Peptide discrimination accuracy**
   ```python
   # 从 features 预测 peptide identity 的准确率
   peptide_pred_acc = accuracy(peptide_classifier(features), true_peptide_id)
   ```

2. **Cross-peptide generalization**
   ```python
   # 在 peptide A 上训练的 TCR，在 peptide B 上的 affinity
   cross_pep_transfer = affinity(TCR_trained_on_A, peptide_B)
   ```

3. **Peptide-conditional policy divergence**
   ```python
   # 同一个 TCR state，不同 peptide 下的 action distribution 差异
   policy_divergence = KL(π(·|TCR, pep_A), π(·|TCR, pep_B))
   ```

---

## 💭 深层思考

### 这个问题揭示了什么？

1. **Representation matters more than architecture**
   - 即使最好的架构，也无法从无区分度的 input 中学到有意义的策略
   - "Garbage in, garbage out"

2. **Domain-specific representation 的重要性**
   - ESM-2 是通用模型，不是为 TCR-peptide binding 优化的
   - 需要 task-specific 的 peptide representation

3. **训练信号的设计至关重要**
   - Episode 内 peptide 不变 → 网络学不到 peptide-conditional 策略
   - 需要让 temporal credit assignment 依赖 peptide

### 与 Shapley 贡献度的联系

你最初的直觉是对的：
- 如果 peptide input 本身就缺乏信息（cos-sim 0.97），那么它对输出的 Shapley 贡献度**理论上限**就很低
- 这不是网络的问题，而是 input 的问题

**类比：**
- 如果你给模型 20 个几乎相同的图片（只有 3% 差异），然后问"为什么模型学不到图片之间的区别？"
- 答案：因为图片本身就几乎没有区别

---

## ✅ 下一步行动

1. **暂停 trace48** — cross-attention 无法解决 input 缺乏区分度的问题

2. **实现 Multi-Peptide Episodes (trace49)**
   - 修改 `env.py` 支持 episode 内切换 peptide
   - 加入 peptide classification auxiliary loss
   - 预期：这是最有可能成功的方案

3. **重新评估 peptide 选择**
   - 分析当前 20 个 peptide 的序列相似度
   - 考虑替换为更多样化的 peptide 集合

4. **长期：开发 task-specific peptide encoder**
   - 用 contrastive learning 或 binding data 预训练
   - 目标：cos-sim < 0.7

---

**结论：你的直觉完全正确 — peptide input 需要重新考虑。Cross-attention 只是治标，改进 input representation 和训练信号才是治本。**
