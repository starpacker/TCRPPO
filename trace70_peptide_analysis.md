# Trace70 在 20 个 Peptide 上的最佳亲和力分析

## 实验设置
- **实验**: trace70_gate_m1p5_from_trace61
- **Gate阈值**: -1.5 (target_decoy_gate_logit)
- **训练步数**: 从 612,096 (trace61 checkpoint) 继续训练
- **目标**: 20 个 tfold_excellent_peptides

---

## 整体结果总结

### 通过率统计
- **通过 -0.8 gate (trace71标准)**: **19/20 peptides (95%)**
- **通过 -1.5 gate (trace70标准)**: **20/20 peptides (100%)**
- **通过 -2.0 gate (trace61标准)**: **20/20 peptides (100%)**

### 关键发现
1. **所有 20 个 peptide 都能达到 -1.5 以上的亲和力**
2. **19 个 peptide 能达到 -0.8 以上** (仅 TPRVTGGGAM 未达到，最佳为 -0.8988)
3. **5 个 peptide 达到正亲和力** (>0.0)，最高达到 +0.7018

---

## 详细结果表

| Rank | Peptide | Best Affinity | Mean Affinity | Std | Samples | Pass -0.8 | Pass -1.5 |
|------|---------|---------------|---------------|-----|---------|-----------|-----------|
| 1 | **LLLDRLNQL** | **+0.7018** | -1.4286 | 1.8760 | 390 | ✓ | ✓ |
| 2 | **YLQPRTFLL** | **+0.5542** | -1.7646 | 1.8914 | 376 | ✓ | ✓ |
| 3 | **NLVPMVATV** | **+0.3487** | -1.7586 | 2.0910 | 434 | ✓ | ✓ |
| 4 | **ELAGIGILTV** | **+0.2213** | -1.5340 | 2.0031 | 430 | ✓ | ✓ |
| 5 | **AVFDRKSDAK** | **+0.1953** | -2.0065 | 1.7358 | 404 | ✓ | ✓ |
| 6 | SLFNTVATLY | +0.1379 | -1.7032 | 1.9489 | 468 | ✓ | ✓ |
| 7 | GILGFVFTL | +0.1318 | -2.0040 | 1.9814 | 400 | ✓ | ✓ |
| 8 | GLCTLVAML | +0.0869 | -2.1710 | 1.8803 | 396 | ✓ | ✓ |
| 9 | CINGVCWTV | +0.0862 | -2.1486 | 1.8245 | 414 | ✓ | ✓ |
| 10 | KLGGALQAK | +0.0597 | -1.9287 | 2.0477 | 434 | ✓ | ✓ |
| 11 | LLWNGPMAV | -0.0796 | -1.9419 | 1.8515 | 398 | ✓ | ✓ |
| 12 | IMNDMPIYM | -0.2106 | -2.1701 | 1.9224 | 342 | ✓ | ✓ |
| 13 | RLRPGGKKK | -0.2423 | -2.1934 | 1.8966 | 404 | ✓ | ✓ |
| 14 | RAKFKQLL | -0.2754 | -2.6187 | 2.0898 | 422 | ✓ | ✓ |
| 15 | RLRAEAQVK | -0.3299 | -2.3173 | 1.9355 | 380 | ✓ | ✓ |
| 16 | FLASKIGRLV | -0.3645 | -2.2183 | 1.8026 | 430 | ✓ | ✓ |
| 17 | IPSINVHHY | -0.5175 | -2.4366 | 2.0210 | 390 | ✓ | ✓ |
| 18 | KRWIILGLNK | -0.5973 | -2.4428 | 1.8181 | 340 | ✓ | ✓ |
| 19 | ATDALMTGY | -0.6558 | -2.9279 | 1.9489 | 412 | ✓ | ✓ |
| 20 | TPRVTGGGAM | -0.8988 | -3.0959 | 1.8239 | 384 | **✗** | ✓ |

---

## Top 5 Peptides - 最佳 TCR 序列

### 1. LLLDRLNQL (Best: +0.7018)
- **Best CDR3β**: `CALSIEEGNGYATYYCCC`
- **Samples**: 390
- **Mean affinity**: -1.4286 ± 1.8760
- **特点**: 唯一达到 +0.7 以上的 peptide，显著超过所有 gate 阈值

### 2. YLQPRTFLL (Best: +0.5542)
- **Best CDR3β**: `CALRRPYGGRAAEYFCCC`
- **Samples**: 376
- **Mean affinity**: -1.7646 ± 1.8914
- **特点**: 第二高亲和力，稳定超过 -0.8 gate

### 3. NLVPMVATV (Best: +0.3487)
- **Best CDR3β**: `CALSLGPLDTQAYYYCCC`
- **Samples**: 434
- **Mean affinity**: -1.7586 ± 2.0910
- **特点**: 样本数最多的 top peptide 之一

### 4. ELAGIGILTV (Best: +0.2213)
- **Best CDR3β**: `CALSVARTHLTATYYCCC`
- **Samples**: 430
- **Mean affinity**: -1.5340 ± 2.0031
- **特点**: 平均亲和力接近 -1.5 gate

### 5. AVFDRKSDAK (Best: +0.1953)
- **Best CDR3β**: `CSLSFEGTDTQAYYYCCC`
- **Samples**: 404
- **Mean affinity**: -2.0065 ± 1.7358
- **特点**: 虽然最佳亲和力高，但平均值较低，说明优化难度较大

---

## 关键洞察

### 1. Gate 阈值的可达性

**trace70 (gate=-1.5) 的结果证明了这是一个合理的阈值**:
- 100% 的 peptides 能够达到或超过 -1.5
- 95% 的 peptides 甚至能达到更严格的 -0.8
- 这解释了为什么 trace70 的通过率高达 74.5%

**trace71 (gate=-0.8) 的困难**:
- 虽然 19/20 peptides 的**最佳**结果能达到 -0.8
- 但这些是在整个训练过程中的**最好**结果
- 在单个 episode 中稳定达到 -0.8 仍然极其困难
- 这解释了为什么 trace71 的通过率仅 9.8%

### 2. Peptide 难度分层

#### 容易优化 (Best > 0.0): 10 个
- LLLDRLNQL, YLQPRTFLL, NLVPMVATV, ELAGIGILTV, AVFDRKSDAK
- SLFNTVATLY, GILGFVFTL, GLCTLVAML, CINGVCWTV, KLGGALQAK

#### 中等难度 (-0.8 < Best < 0.0): 9 个
- LLWNGPMAV, IMNDMPIYM, RLRPGGKKK, RAKFKQLL, RLRAEAQVK
- FLASKIGRLV, IPSINVHHY, KRWIILGLNK, ATDALMTGY

#### 困难 (Best < -0.8): 1 个
- **TPRVTGGGAM** (Best: -0.8988)
  - 这是唯一无法达到 -0.8 的 peptide
  - 平均亲和力 -3.0959，是所有 peptide 中最低的
  - 可能需要特殊的优化策略

### 3. 训练稳定性

**样本数分布均匀** (340-468 samples):
- 说明模型对所有 peptide 都进行了充分探索
- 没有明显的偏好或忽略某些 peptide

**标准差较大** (1.7-2.1):
- 说明每个 peptide 的亲和力分布很广
- 最佳结果和平均结果差距显著
- 这是强化学习探索-利用权衡的正常现象

### 4. 与 trace71 的对比

**为什么 trace70 成功而 trace71 失败？**

1. **目标可达性**:
   - trace70: 所有 peptide 都能达到 -1.5 → 频繁的正反馈
   - trace71: 虽然最佳结果能达到 -0.8，但稳定达到极难 → 稀疏的正反馈

2. **学习信号质量**:
   - trace70: 74.5% 通过率 → 丰富的成功经验
   - trace71: 9.8% 通过率 → 几乎没有成功经验

3. **探索空间**:
   - trace70: 在 -1.5 附近探索和优化
   - trace71: 无法建立稳定的策略，持续随机探索

---

## 建议

### 1. 对于 trace71 类型的严格 gate

如果要使用 -0.8 这样的严格阈值，建议：

**方案 A: 课程学习**
```python
# 逐步提高难度
gates = [-2.0, -1.5, -1.0, -0.8]
# 每个阶段训练到稳定后再提高
```

**方案 B: 软门控**
```python
# 使用连续的奖励而非二元判断
gate_weight = sigmoid((affinity - gate) / temperature)
bonus = gate_weight * target_pass_bonus
```

**方案 C: 分层目标**
```python
# 对不同难度的 peptide 设置不同的 gate
easy_peptides = ["LLLDRLNQL", "YLQPRTFLL", ...]  # gate = -0.8
hard_peptides = ["TPRVTGGGAM", ...]  # gate = -1.2
```

### 2. 针对 TPRVTGGGAM 的特殊处理

这个 peptide 明显比其他更难优化：
- 考虑增加其训练权重
- 或者从训练集中暂时移除，专注于其他 19 个
- 或者为其设计专门的奖励函数

### 3. 利用已发现的高亲和力 TCR

Top 5 peptides 的最佳 TCR 序列可以作为：
- 模仿学习的示例
- 初始化 TCR pool 的种子
- 验证 tFold 预测准确性的基准

---

## 结论

**trace70 的成功证明了 -1.5 是一个合理且可达成的目标阈值**:
- 100% 的 peptides 能够达到
- 95% 的 peptides 甚至能达到更严格的 -0.8
- 提供了丰富的学习信号，使模型能够稳定优化

**trace71 的失败不是因为目标不可达，而是因为目标难以稳定达到**:
- 虽然 19/20 peptides 的最佳结果能达到 -0.8
- 但在单个 episode 中稳定达到 -0.8 需要更复杂的训练策略
- 二元的 gate 机制在这种情况下提供的学习信号过于稀疏

**建议**: 如果要追求 -0.8 级别的高亲和力，应该采用渐进式课程学习或软门控机制，而不是直接设置硬阈值。
