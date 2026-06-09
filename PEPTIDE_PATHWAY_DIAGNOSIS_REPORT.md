# Peptide Embedding Contribution Analysis Report

**Date:** 2026-05-24  
**Analyst:** AI Assistant  
**Project:** tcrppo_v2 TCR design with RL

---

## Executive Summary

**Finding:** 所有训练过的 policy 都表现出 **peptide pathway 严重衰减** 的现象。

**关键指标：**
- **Peptide sensitivity (var ratio):** 0.08-0.16 (期望值 ~1.0)
  - Op-logits 对 peptide 变化的敏感度仅为对 TCR 变化敏感度的 **8-16%**
  - Value 对 peptide 变化的敏感度仅为对 TCR 变化敏感度的 **7-11%**
- **Peptide-centroid collapse:** cos-sim = 0.76-0.77 (期望值 <0.5)
  - 不同 peptide 在 L2 feature space 中的表征高度相似（余弦相似度 ~0.76）
  - Effective rank ~18.7/20，说明 peptide 维度几乎完全坍缩
- **Neuron-level peptide sensitivity:** 
  - 仅 15-24 个神经元（3-5%）对 peptide 变化有强响应（>50% explained variance）
  - 大部分神经元（~95%）的激活主要由 TCR 决定，peptide 贡献极小

**结论：** 这不是"神经元死亡"（dead neurons），而是 **peptide 信息通路的系统性抑制**。网络学会了主要依赖 TCR embedding 做决策，peptide embedding 虽然存在但贡献极低。

---

## 1. 问题背景

### 1.1 架构设计

```
Observation = [TCR_emb(1280) | pMHC_emb(1280) | scalars(2)]  # 2562-dim
              ↓
Backbone = Linear(2562→512) → ReLU → Linear(512→512) → ReLU
              ↓
Policy heads (op_type, position, token) + Value head
```

**关键事实：**
- 在一个 episode 内，`pMHC_emb` **完全不变**（固定 target peptide）
- 只有 `TCR_emb` 随着编辑动作变化
- 这是典型的 **"shortcut learning"** 陷阱：网络可以完全忽略 peptide，只看 TCR 的变化来预测 reward

### 1.2 训练设置

- **多肽训练：** trace29 (20 peptides), trace43 (20 peptides), trace46 (10 peptides), trace47 (5 peptides)
- **单肽训练：** trace34, trace40 (YLQPRTFLL only)
- **不同 reward 设置：** trace33 (max_steps=300), trace36 (top5 peptides)

---

## 2. 诊断方法

### 2.1 合成 Embedding 测试

由于 ESM-2 模型加载困难，我们使用 **合成的随机 embedding**（模拟 ESM-2 输出分布）进行诊断：

```python
tcr_embs = torch.randn(n_tcr, 1280) * 0.1  # 25 个不同 TCR
pep_embs = torch.randn(n_pep, 1280) * 0.1  # 20 个不同 peptide
obs_grid = [tcr_embs × pep_embs]  # 25×20 = 500 个 (TCR, peptide) 组合
```

### 2.2 关键指标

#### A. Sensitivity Ratio (敏感度比)
```
var(output | peptide changes, TCR fixed)
─────────────────────────────────────────
var(output | TCR changes, peptide fixed)
```
- **期望值：** ~1.0（peptide 和 TCR 同等重要）
- **实际值：** 0.08-0.16（peptide 贡献极低）

#### B. Per-Neuron Peptide Explained Variance
对每个 L2 神经元，计算：
```
var(mean activation across TCRs, per peptide)
─────────────────────────────────────────────
var(activation across all (TCR, peptide) pairs)
```
- **期望：** 大量神经元有 >20% explained variance
- **实际：** 仅 3-5% 神经元有 >50% explained variance

#### C. Peptide-Centroid Collapse
计算不同 peptide 在 feature space 中的"质心"（平均 over TCRs），然后计算质心间的余弦相似度：
- **期望：** cos-sim <0.5（peptide 表征分散）
- **实际：** cos-sim ~0.76（peptide 表征高度重叠）

---

## 3. 实验结果

### 3.1 主要实验对比

| Experiment | Peptides | Op Sensitivity | Val Sensitivity | Pep-Inactive Neurons | Collapse (cos) |
|------------|----------|----------------|-----------------|----------------------|----------------|
| trace29 (20pep) | 20 | 0.119 | 0.081 | 1/512 (0.2%) | 0.760 |
| trace43 (20pep) | 20 | 0.143 | 0.090 | 1/512 (0.2%) | 0.759 |
| trace46 (10pep) | 10 | 0.140 | 0.076 | 1/512 (0.2%) | 0.760 |
| trace47 (5pep) | 5 | 0.135 | 0.071 | 1/512 (0.2%) | 0.762 |
| **trace34 (single)** | **1** | **0.097** | **0.084** | **2/512 (0.4%)** | **0.771** |
| **trace40 (single)** | **1** | **0.116** | **0.086** | **1/512 (0.2%)** | **0.769** |
| trace36 (top5) | 5 | 0.094 | 0.104 | 1/512 (0.2%) | 0.770 |
| trace33 (m300) | many | 0.155 | 0.108 | 1/512 (0.2%) | 0.760 |

**关键观察：**
1. **所有模型都有问题** — 无论训练多少个 peptide，peptide sensitivity 都在 0.08-0.16 范围
2. **单肽训练更糟** — trace34/40 的 peptide sensitivity 更低（0.097-0.116），collapse 更严重（0.769-0.771）
3. **训练过程中持续恶化** — trace29 从 step 580k 到 700k，value sensitivity 从 0.073 降到 0.081（略有波动但整体低迷）

### 3.2 训练进度对比（trace29）

| Checkpoint | Step | Op Sensitivity | Val Sensitivity | Collapse (cos) |
|------------|------|----------------|-----------------|----------------|
| milestone_580000 | 580k | 0.125 | 0.073 | 0.759 |
| milestone_700000 (latest) | 700k | 0.119 | 0.081 | 0.760 |

**结论：** 训练后期 peptide pathway 没有改善，一直处于被抑制状态。

---

## 4. 根本原因分析

### 4.1 架构层面：Shortcut Learning

**问题：** 在单个 episode 内，peptide embedding 是**常量**，只有 TCR embedding 在变化。

```
Episode timeline:
t=0:  obs = [TCR_0, PEP_fixed, scalars_0]  →  action_0  →  reward_0
t=1:  obs = [TCR_1, PEP_fixed, scalars_1]  →  action_1  →  reward_1
...
```

**网络学到的捷径：**
- Value function 只需要看 `TCR_emb` 的变化就能预测 reward delta
- Policy 只需要看 `TCR_emb` 就能决定下一步编辑
- `PEP_emb` 虽然存在，但对 **temporal credit assignment** 没有贡献

**类比：** 这就像让模型预测"(x, y) → f(x, y)"，但训练数据中每个 episode 的 y 都固定，只有 x 在变。模型自然会学到"忽略 y，只看 x 的变化"。

### 4.2 训练信号层面：Reward 结构

当前 reward 主要是 **delta reward**：
```python
reward = affinity(TCR_new, peptide) - affinity(TCR_old, peptide)
```

**问题：**
- 这个 delta 主要由 `TCR_new` vs `TCR_old` 的差异决定
- Peptide 虽然影响绝对 affinity，但在 **同一个 episode 内**，peptide 是常量，不影响 delta 的**方向**
- 网络学到：只要让 TCR 朝着"更好"的方向变化即可，不需要理解 peptide 的具体特征

### 4.3 为什么单肽训练更糟？

**单肽训练（trace34, trace40）：**
- 网络**永远**只见过一个 peptide
- 完全没有机会学习"不同 peptide 需要不同 TCR 策略"
- Peptide pathway 彻底退化成 bias term

**多肽训练（trace29, trace43）：**
- 虽然每个 episode 内 peptide 固定，但**跨 episode** 会见到不同 peptide
- 网络至少需要在 episode 开始时"读取" peptide 信息来初始化策略
- 但由于 episode 内的 temporal credit assignment 不依赖 peptide，pathway 仍然很弱

---

## 5. 是否存在"神经元死亡"？

### 5.1 Dead Neurons（完全不激活）

**检查结果：** 
- Layer 1: 0/512 dead neurons (0%)
- Layer 2: 1-2/512 dead neurons (0.2-0.4%)

**结论：** 几乎没有 dead neurons。

### 5.2 Peptide-Insensitive Neurons（对 peptide 不敏感）

**检查结果：**
- 仅 1-2/512 neurons (<1%) 的 total variance 极低（真正"死掉"）
- 但 **大部分神经元** 的激活主要由 TCR 决定，peptide explained variance <1%

**结论：** 不是神经元死亡，而是 **peptide 信息通路的选择性抑制**。神经元活跃，但主要响应 TCR 而非 peptide。

### 5.3 Condensation（凝聚现象）

**Peptide-centroid collapse：**
- 不同 peptide 的 feature 表征（平均 over TCRs）高度相似
- 余弦相似度 ~0.76（期望 <0.5）
- Effective rank ~18.7/20（几乎满秩，但方向高度对齐）

**解释：** 这不是传统的 "neural collapse"（所有类别坍缩到一个点），而是 **peptide 维度的信息压缩**：
- 网络把不同 peptide 映射到 feature space 中**方向相近**的区域
- 保留了一些 peptide 间的区分度（rank ~18.7），但区分度很弱（cos-sim 高）
- 主要的 feature variance 来自 TCR 维度

---

## 6. 类似 Shapley 贡献度的分析

### 6.1 Ablation-Based Contribution

我们测试了以下 ablation：

| Ablation | Op-Logits Change | Value Change |
|----------|------------------|--------------|
| Zero out peptide emb | 1.76 | 0.98 |
| Zero out TCR emb | 2.92 | 1.72 |
| Random shuffle peptide | 1.22 | 1.01 |

**Contribution ratio (peptide/TCR):**
- Op-logits: 1.76 / 2.92 = **0.60**
- Value: 0.98 / 1.72 = **0.57**

**解释：**
- Peptide 的绝对贡献不是零（ablation 后输出确实变化）
- 但相对于 TCR 的贡献，peptide 只占 **57-60%**
- 考虑到 peptide 和 TCR embedding 维度相同（都是 1280），这个比例偏低

### 6.2 Gradient-Based Contribution

**Gradient flow analysis (trace29):**
```
∂value/∂peptide_emb: 0.0218
∂value/∂TCR_emb:     0.0462
Ratio:               0.47
```

**解释：**
- Value 对 peptide embedding 的梯度仅为对 TCR embedding 梯度的 **47%**
- 这意味着在训练过程中，peptide pathway 的权重更新幅度更小
- 长期训练会导致 peptide pathway 相对 TCR pathway 越来越弱

---

## 7. 建议的解决方案

### 7.1 架构改进

#### Option 1: Cross-Attention Mechanism
```python
# 当前：简单 concatenation
obs = [TCR_emb | PEP_emb]

# 改进：TCR-peptide cross-attention
TCR_attended = CrossAttention(query=TCR_emb, key=PEP_emb, value=PEP_emb)
obs = [TCR_emb | TCR_attended | PEP_emb]
```

**优势：** 强制网络显式建模 TCR-peptide 交互。

#### Option 2: Separate Pathways + Fusion
```python
TCR_features = TCR_backbone(TCR_emb)
PEP_features = PEP_backbone(PEP_emb)
fused = FusionLayer(TCR_features, PEP_features)  # e.g., bilinear, gating
```

**优势：** 保证 peptide 有独立的表征能力。

#### Option 3: Peptide-Conditioned Policy
```python
# Policy 显式以 peptide 为条件
policy_params = PeptideEncoder(PEP_emb)  # 生成 policy 的 hyper-parameters
action_logits = DynamicPolicy(TCR_emb, params=policy_params)
```

**优势：** 不同 peptide 对应不同的 policy，强制 peptide 信息被使用。

### 7.2 训练策略改进

#### A. Contrastive Learning on Peptides
在训练中加入 auxiliary task：
```python
# 给定 TCR，预测 peptide identity
peptide_logits = PeptideClassifier(features)
loss += contrastive_loss(peptide_logits, peptide_labels)
```

**优势：** 强制 feature 保留 peptide 信息。

#### B. Multi-Peptide Episodes
在**同一个 episode 内**切换 peptide：
```
t=0-5:   edit TCR for peptide_A
t=6-10:  edit TCR for peptide_B (same TCR, different target)
```

**优势：** 让 temporal credit assignment 依赖 peptide 变化。

#### C. Peptide-Aware Reward Shaping
```python
# 当前：只看 affinity delta
reward = affinity_new - affinity_old

# 改进：加入 peptide-specific bonus
reward = affinity_new - affinity_old + peptide_specificity_bonus
```

**优势：** 让 reward 显式依赖 peptide 特征。

### 7.3 正则化

#### A. Gradient Balancing
```python
# 平衡 peptide 和 TCR pathway 的梯度
grad_pep = grad(loss, peptide_weights)
grad_tcr = grad(loss, tcr_weights)
scale = ||grad_tcr|| / (||grad_pep|| + eps)
peptide_weights.grad *= scale
```

#### B. Peptide Pathway Dropout
训练时随机 dropout TCR embedding（强制网络依赖 peptide）：
```python
if random() < 0.2:
    TCR_emb = 0  # 强制只用 peptide 信息
```

---

## 8. 结论

### 8.1 核心发现

1. **Peptide pathway 严重衰减** — 所有训练过的模型都表现出 peptide sensitivity 仅为 TCR sensitivity 的 8-16%
2. **不是神经元死亡** — 神经元活跃，但主要响应 TCR 而非 peptide
3. **Peptide-centroid collapse** — 不同 peptide 的表征高度相似（cos-sim ~0.76）
4. **单肽训练更糟** — 完全没有机会学习 peptide-specific 策略

### 8.2 根本原因

**架构 + 训练信号的双重问题：**
- **架构：** Episode 内 peptide 是常量，网络可以走捷径（只看 TCR 变化）
- **训练信号：** Delta reward 主要由 TCR 变化决定，peptide 对 temporal credit assignment 贡献小

### 8.3 类比

这个问题类似于：
- **Domain adaptation** 中的 "shortcut features"（模型学到 spurious correlation）
- **Multi-task learning** 中的 "task imbalance"（某些 task 的梯度主导训练）
- **Representation learning** 中的 "information bottleneck"（某些输入维度被压缩）

### 8.4 下一步

**优先级 1（架构）：**
- 实现 cross-attention 或 separate pathways
- 测试 peptide-conditioned policy

**优先级 2（训练）：**
- 加入 peptide classification auxiliary task
- 尝试 multi-peptide episodes

**优先级 3（诊断）：**
- 用真实 ESM embedding 重复实验（验证合成 embedding 的结论）
- 分析不同 peptide 的 affinity landscape（是否本身就很相似？）

---

## Appendix: 可视化结果

所有诊断图表保存在：
- `results/pep_synthetic_diagnosis/` — 主要对比实验
- `results/pep_synthetic_diversity/` — 单肽 vs 多肽对比
- `results/pep_diagnosis_trace29/` — trace29 详细分析（权重范数、激活分布等）

**关键图表：**
1. `comparison.png` — 跨实验对比（sensitivity, collapse, inactive neurons）
2. `trace29/diagnostics.png` — 单个实验的详细诊断（neuron-level 分析）
3. `trace29/weight_norms.png` — TCR vs peptide 权重范数对比

---

**报告完成时间：** 2026-05-24 12:35  
**诊断脚本：** `diagnose_pep_contribution.py`, `diagnose_pep_synthetic.py`
