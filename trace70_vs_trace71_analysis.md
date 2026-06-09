# Trace70 vs Trace71 性能差异分析

## 实验配置对比

### 共同点
- **基础模型**: 都从 trace61 checkpoint (step 612,096) 继续训练
- **奖励模式**: `v2_simple_target_gated_decoy`
- **环境设置**: 8 envs, max_steps=8, terminal_reward_only=True
- **权重**: affinity=1.0, decoy=0.3, naturalness=0.05, diversity=0.02
- **训练目标**: 20 peptides from tfold_excellent_peptides.txt
- **其他参数**: 完全相同的学习率、网络结构、优化器设置

### 唯一差异：target_decoy_gate_logit (目标亲和力阈值)

| 实验 | Gate阈值 | 含义 |
|------|---------|------|
| **trace70** | **-1.5** | 更宽松的阈值，TCR affinity ≥ -1.5 即可"通过" |
| **trace71** | **-0.8** | 更严格的阈值，TCR affinity ≥ -0.8 才能"通过" |

---

## 性能结果对比

### 整体统计

| 指标 | Trace70 (gate=-1.5) | Trace71 (gate=-0.8) | 差异 |
|------|---------------------|---------------------|------|
| **总episodes** | 4,016 | 4,488 | - |
| **平均奖励** | **-0.923 ± 1.029** | **-3.745 ± 2.188** | **+2.82** ✓ |
| **平均亲和力** | **-1.216 ± 0.907** | **-3.775 ± 2.131** | **+2.56** ✓ |
| **平均TargetSat** | **0.506** | **0.035** | **+0.47** ✓ |
| **通过率** | **74.5%** (2990/4016) | **9.8%** (440/4488) | **+64.6%** ✓ |

### 最后500 episodes表现

| 指标 | Trace70 | Trace71 | 差异 |
|------|---------|---------|------|
| **平均奖励** | **-0.918** | **-5.577** | **+4.66** ✓ |
| **平均亲和力** | **-1.216** | **-5.574** | **+4.36** ✓ |
| **通过率** | **75.8%** (379/500) | **0.2%** (1/500) | **+75.6%** ✓ |

---

## 奖励机制分析

### v2_simple_target_gated_decoy 奖励函数

```python
def _simple_target_gated_decoy_reward(self, aff_score, ...):
    target_gate = self.target_decoy_gate_logit  # trace70: -1.5, trace71: -0.8
    target_passed = 1.0 if aff_score >= target_gate else 0.0
    
    # 基础奖励：总是给予亲和力分数
    total = self.weights["affinity"] * aff_score  # w_affinity = 1.0
    
    # 如果通过阈值：
    if target_passed:
        total += self.target_pass_bonus  # 额外奖励 (默认 +1.0)
        total += decoy_term  # 激活 decoy 惩罚机制
    
    # 辅助项
    total += self.weights["naturalness"] * nat_score
    total += self.weights["diversity"] * div_score
    
    return total
```

### 关键差异

#### Trace70 (gate=-1.5) - 容易通过
- **通过条件**: affinity ≥ -1.5
- **典型episode**: 
  - 初始 affinity: -6.77 → 终止 affinity: -0.83
  - **通过阈值** ✓
  - 获得 pass_bonus: +1.0
  - 激活 decoy 惩罚: DecViol=2.16, DecA=-0.84
  - **TargetSat = 0.67** (超过阈值 0.67)
  - **最终奖励 R = -0.48**

#### Trace71 (gate=-0.8) - 极难通过
- **通过条件**: affinity ≥ -0.8
- **典型episode**:
  - 初始 affinity: -7.26 → 终止 affinity: -0.84
  - **未通过阈值** ✗ (差 0.04)
  - 无 pass_bonus
  - 无 decoy 惩罚激活
  - **TargetSat = 0.0**
  - **TargetShort = 0.04** (距离阈值的差距)
  - **最终奖励 R = -0.84** (仅为 affinity 本身)

---

## 为什么效果差距如此巨大？

### 1. **阈值难度的指数级差异**

tFold affinity logit 的分布特性：
- **-2.0 到 -1.0**: 中等亲和力区间，较容易达到
- **-1.0 到 0.0**: 高亲和力区间，需要较好的序列
- **0.0 以上**: 极高亲和力，非常罕见

**trace70 的 -1.5 阈值**:
- 位于中等亲和力区间
- 通过率 74.5% 说明这是一个**可达成的目标**
- 模型能够获得频繁的正反馈（pass_bonus）

**trace71 的 -0.8 阈值**:
- 位于高亲和力区间
- 通过率仅 9.8%，最后500 episodes 降至 0.2%
- 这是一个**几乎不可达成的目标**
- 模型几乎从未获得 pass_bonus

### 2. **奖励信号的质量差异**

#### Trace70 的奖励信号
```
Episode 1: R=-0.48 (A=-0.83, TargetSat=0.67, DecViol=2.16)
Episode 2: R=-0.21 (A=-0.54, TargetSat=0.96, DecViol=2.24)
Episode 3: R=+0.22 (A=-0.21, TargetSat=1.29, DecViol=1.79)  ← 正奖励！
```
- **丰富的奖励梯度**: 从 -0.48 到 +0.22
- **明确的成功信号**: TargetSat > 0 表示超过阈值
- **decoy 机制激活**: 引导模型优化特异性

#### Trace71 的奖励信号
```
Episode 1: R=-0.84 (A=-0.84, TargetSat=0.0, TargetShort=0.04)
Episode 2: R=-1.12 (A=-1.12, TargetSat=0.0, TargetShort=0.32)
Episode 3: R=-0.87 (A=-0.84, TargetSat=0.0, TargetShort=0.04)
...
Episode 4434: R=-7.52 (A=-7.50, TargetSat=0.0, TargetShort=6.70)  ← 崩溃
```
- **单调的负奖励**: 几乎所有 episode 都是 TargetSat=0.0
- **无成功信号**: 模型不知道"通过"是什么样子
- **奖励 = 亲和力本身**: R ≈ A，缺乏额外的引导信号
- **后期崩溃**: 最后500 episodes 平均奖励降至 -5.58

### 3. **学习动力学的崩溃**

#### Trace70: 正常的强化学习循环
1. 模型探索 → 达到 -1.5 阈值
2. 获得 pass_bonus (+1.0) → 正反馈
3. 激活 decoy 惩罚 → 学习特异性
4. 持续优化 → 稳定在 -1.2 左右

#### Trace71: 学习信号缺失
1. 模型探索 → 达到 -0.84 (非常接近！)
2. **但未通过 -0.8 阈值** → 无 bonus
3. 奖励 = -0.84 → 负反馈
4. 模型认为这是"失败" → 尝试其他策略
5. 新策略更差 (-2.0, -3.0, ...) → 更负的奖励
6. **陷入恶性循环** → 最终崩溃至 -5.5

**关键问题**: -0.84 vs -0.8 的差距仅 0.04，但在二元奖励机制下，这是"完全失败"和"成功"的区别。

### 4. **探索-利用困境**

**Trace70**:
- 阈值可达 → 模型能够找到"成功"的策略
- 在成功策略附近探索 → 逐步优化
- 通过率稳定在 75% → 健康的探索-利用平衡

**Trace71**:
- 阈值几乎不可达 → 模型找不到"成功"的策略
- 随机探索 → 所有尝试都"失败"
- 无法建立价值函数 → 策略崩溃
- 通过率从 9.8% 降至 0.2% → 放弃探索

---

## 结论

### 核心原因

**trace71 效果极差的根本原因是 gate=-0.8 设置了一个几乎不可达成的目标，导致：**

1. **奖励信号稀疏**: 99.8% 的 episodes 无法获得 pass_bonus
2. **无正反馈**: 模型从未学到"成功"是什么样子
3. **二元陷阱**: 即使达到 -0.84 (非常好的结果)，仍被判定为"失败"
4. **学习崩溃**: 缺乏引导信号，策略逐渐退化

### 设计建议

1. **阈值应该是可达成的**: 
   - trace70 的 -1.5 是合理的，通过率 ~75%
   - trace71 的 -0.8 太严格，通过率 <1%

2. **考虑渐进式课程学习**:
   ```python
   # 从容易到困难
   curriculum_gates = [-2.0, -1.5, -1.0, -0.8]
   ```

3. **使用软门控而非硬门控**:
   ```python
   # 替代二元判断
   gate_weight = sigmoid((aff_score - gate) / temperature)
   bonus = gate_weight * target_pass_bonus
   ```

4. **提供中间奖励**:
   ```python
   # 即使未通过，也奖励接近阈值的尝试
   if aff_score < gate:
       proximity_bonus = max(0, 1 - (gate - aff_score) / 2.0)
   ```

### 数据支持

| 指标 | Trace70 | Trace71 | 说明 |
|------|---------|---------|------|
| 通过率 | 74.5% | 9.8% | trace71 几乎无法通过 |
| 最终通过率 | 75.8% | 0.2% | trace71 完全崩溃 |
| 奖励差异 | +2.82 | - | trace70 显著更好 |
| 亲和力差异 | +2.56 | - | trace70 学到更好的策略 |

**trace70 和 trace71 的巨大差距不是因为模型能力，而是因为 trace71 的目标设置超出了当前训练范式的能力范围。**
