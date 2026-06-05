# Curriculum Reward 设计方案：逐步爬升到 Affinity > 0.5

## 背景分析

### 现有实验表现

| 实验 | Reward Mode | Max Affinity | >0% | >0.5% | 评价 |
|------|-------------|--------------|-----|-------|------|
| **trace29** | v2_simple_target_gated_decoy | **1.30** | 1.24% | ~0.1% | 最高峰值，但不稳定 |
| trace43 | 同 trace29 复现 | 0.90 | 0.92% | ~0% | 复现成功但略低 |
| **trace11** | v2_no_decoy_delta | **-2.54** | 0% | 0% | 纯 target 优化失败 |
| trace8 | v2_no_decoy | 0.10 | 99.6% | 0% | 稳定但低 |

### 关键观察

1. **达到 0.5 非常困难**：trace29 全 12K episodes 中只有极少数达到 > 0.5
2. **trace11 (纯 target delta) 完全失败**：Max 只有 -2.54，说明 delta reward 对高 affinity 无效
3. **trace29 虽然有 max=1.30，但 mean=-1.83**：说明只是偶尔"撞大运"
4. **当前 reward 机制的问题**：
   - Gate-based reward 让模型满足于"踩线"（A > -2）
   - Decoy penalty 过早引入，分散了 target 优化的注意力
   - 没有明确的"爬升激励"

---

## 设计理念

### 核心思想（你的建议）

1. **首要目标**：让 target affinity 稳定达到 **0.5**
2. **分阶段策略**：
   - Phase 1: 爬到 -0.2（相对容易）
   - Phase 2: 爬到 0.0
   - Phase 3: 爬到 0.2
   - Phase 4: 爬到 0.5
3. **Decoy 延后**：只有当 A > 0.5 后才考虑 decoy
4. **Gap 奖励不惩罚**：A > 0.5 后，奖励大的 (A - DecA)，但不惩罚小的 gap

### 关键原则

- **正向激励为主**：每个阶段都有明确的奖励，避免纯惩罚
- **平滑过渡**：阶段切换用 soft gate，避免 reward 跳变
- **保护下界**：即使 A < 0.5，也不因 decoy 而被惩罚

---

## Reward 公式设计

### Phase 1-4: Target Climbing (A < 0.5)

```python
# Curriculum gates
gates = [-0.2, 0.0, 0.2, 0.5]
bonuses = [0.5, 1.0, 1.5, 2.0]

# 计算当前处于哪个阶段
current_bonus = 0
for i, gate in enumerate(gates):
    if A >= gate:
        current_bonus = bonuses[i]

# Reward = 基础 affinity + 阶段奖励
R = w_affinity * A + current_bonus + w_nat * Nat + w_div * Div

# 不考虑 decoy
```

**特点**：
- 每跨过一个 gate，获得额外奖励
- 鼓励持续爬升
- 简单直接，无 decoy 干扰

### Phase 5: Specificity Optimization (A >= 0.5)

```python
# 只有 A >= 0.5 才激活 decoy
if A >= 0.5:
    gap = A - DecA_mean
    
    # 奖励大的 gap，但不惩罚小的 gap
    gap_bonus = w_gap * max(0, gap - margin)  # margin = 1.0
    
    R = w_affinity * A + 2.0 + gap_bonus + w_nat * Nat + w_div * Div
else:
    # A < 0.5 时，完全不考虑 decoy
    R = (同 Phase 1-4)
```

**特点**：
- `max(0, gap - margin)` 确保只有 gap > 1.0 才有额外奖励
- Gap < 1.0 时，gap_bonus = 0，不会因 decoy 被惩罚
- 保护机制：即使 decoy 很高，只要 A >= 0.5，仍能拿到基础奖励

---

## 完整 Reward 实现

### Python 伪代码

```python
def curriculum_climbing_reward(
    A: float,              # Target affinity
    DecA: float,           # Decoy affinity (mean)
    Nat: float,            # Naturalness
    Div: float,            # Diversity
    w_affinity: float = 1.0,
    w_gap: float = 1.0,
    w_nat: float = 0.05,
    w_div: float = 0.02,
    gap_margin: float = 1.0
) -> float:
    """
    Curriculum reward: 逐步爬升到 A > 0.5，然后优化 specificity
    """
    # Curriculum gates and bonuses
    gates = [-0.2, 0.0, 0.2, 0.5]
    bonuses = [0.5, 1.0, 1.5, 2.0]
    
    # 计算阶段奖励
    stage_bonus = 0.0
    for gate, bonus in zip(gates, bonuses):
        if A >= gate:
            stage_bonus = bonus
    
    # 基础 reward
    R = w_affinity * A + stage_bonus
    
    # Phase 5: Specificity (只有 A >= 0.5 才激活)
    if A >= 0.5:
        gap = A - DecA
        gap_bonus = w_gap * max(0.0, gap - gap_margin)
        R += gap_bonus
    
    # 辅助项
    R += w_nat * Nat + w_div * Div
    
    return R
```

### 关键参数

```python
w_affinity = 1.0      # Target affinity 权重
w_gap = 1.0           # Specificity gap 权重
gap_margin = 1.0      # Gap 阈值（只有 gap > 1 才奖励）
gates = [-0.2, 0.0, 0.2, 0.5]
bonuses = [0.5, 1.0, 1.5, 2.0]
```

---

## 训练策略

### Stage 1: 爬到 -0.2 (0-200K steps)

**目标**: 让 50%+ episodes 达到 A > -0.2

**预期**:
- 从 trace29 数据看，A > -0.2 的比例约 30%
- 通过 bonus=0.5 的激励，应该能提升到 50%+

**监控指标**:
- Mean A
- % A > -0.2
- Max A

### Stage 2: 爬到 0.0 (200K-500K steps)

**目标**: 让 20%+ episodes 达到 A > 0.0

**预期**:
- trace29 中 A > 0 的比例只有 1.24%
- 通过 bonus=1.0 的激励，目标提升到 20%

### Stage 3: 爬到 0.2 (500K-800K steps)

**目标**: 让 10%+ episodes 达到 A > 0.2

**预期**:
- 这是关键阶段，需要模型真正学会生成高亲和力 TCR

### Stage 4: 爬到 0.5 (800K-1.2M steps)

**目标**: 让 5%+ episodes 达到 A > 0.5

**预期**:
- 如果能达到这个目标，说明模型已经掌握了高亲和力设计

### Stage 5: 优化 Specificity (1.2M-2M steps)

**目标**: 在保持 A > 0.5 的前提下，最大化 A - DecA

**预期**:
- Gap > 1: 30%+
- Gap > 2: 10%+

---

## 与现有方法对比

### vs. trace29 (v2_simple_target_gated_decoy)

| 维度 | trace29 | Curriculum Climbing |
|------|---------|---------------------|
| Target gate | -2.0（固定） | -0.2 → 0.0 → 0.2 → 0.5（渐进） |
| Decoy 激活 | A > -2 时（67% episodes） | A > 0.5 时（预期 5%） |
| Decoy penalty | -0.3 * max(0, DecA+3) | 0（A<0.5）或 +1.0*max(0, gap-1)（A>=0.5） |
| Pass bonus | +1.0（固定） | +0.5 → +1.0 → +1.5 → +2.0（渐进） |
| 问题 | 过早引入 decoy，target 优化不足 | 延后 decoy，专注 target |

### vs. trace11 (v2_no_decoy_delta)

| 维度 | trace11 | Curriculum Climbing |
|------|---------|---------------------|
| Reward 类型 | Delta (A - A_init) | Absolute A + bonus |
| 最高 A | -2.54 | 目标 > 0.5 |
| 问题 | Delta 对高 affinity 无效 | 用 absolute + bonus 持续激励 |

---

## 实现细节

### 1. Soft Gate（可选优化）

为了避免 reward 跳变，可以用 sigmoid 平滑过渡：

```python
def soft_gate_bonus(A, gate, bonus, temperature=0.5):
    """Soft gate: 平滑过渡"""
    return bonus * sigmoid((A - gate) / temperature)

# 总 bonus
stage_bonus = sum(soft_gate_bonus(A, g, b) for g, b in zip(gates, bonuses))
```

### 2. Adaptive Gate（可选优化）

根据训练进度动态调整 gate：

```python
# 训练初期降低 gate，后期提高
current_gate = base_gate + progress_factor * gate_increment
```

### 3. Decoy Sampling 策略

当 A >= 0.5 激活 decoy 时：
- 使用 **hard negative mining**：选择 affinity 最高的 decoy
- 或使用 **adaptive sampling**：根据 A 的值选择难度相当的 decoy

---

## 预期效果

### 乐观估计

| 阶段 | Steps | 目标 | 预期达成率 |
|------|-------|------|-----------|
| Stage 1 | 200K | A > -0.2 | 50%+ |
| Stage 2 | 500K | A > 0.0 | 20%+ |
| Stage 3 | 800K | A > 0.2 | 10%+ |
| Stage 4 | 1.2M | A > 0.5 | 5%+ |
| Stage 5 | 2M | Gap > 1 | 30%+ |

### 保守估计

| 阶段 | Steps | 目标 | 预期达成率 |
|------|-------|------|-----------|
| Stage 1 | 200K | A > -0.2 | 40%+ |
| Stage 2 | 500K | A > 0.0 | 10%+ |
| Stage 3 | 800K | A > 0.2 | 5%+ |
| Stage 4 | 1.2M | A > 0.5 | 2%+ |
| Stage 5 | 2M | Gap > 1 | 10%+ |

---

## 风险与缓解

### 风险 1: 卡在某个 gate

**症状**: 某个阶段长时间无法突破（如 A 一直在 -0.1 ~ 0.0 徘徊）

**缓解**:
- 降低该 gate 的阈值（如 0.0 → -0.1）
- 增加该阶段的 bonus（如 1.0 → 1.5）
- 增加 exploration（提高 entropy_coef）

### 风险 2: 达到 0.5 后 decoy 失控

**症状**: A > 0.5 但 DecA 也很高（> 0），导致 gap < 1

**缓解**:
- 由于我们用的是 `max(0, gap - 1)`，不会惩罚 gap < 1
- 模型仍能拿到 `w_affinity * A + 2.0` 的基础奖励
- 如果长期无法提升 gap，可以考虑增加 w_gap

### 风险 3: 训练不稳定

**症状**: A 在不同阶段来回震荡

**缓解**:
- 使用 soft gate 平滑过渡
- 降低 learning rate
- 增加 batch size

---

## 实验计划

### Experiment: trace_curriculum_v1

```yaml
name: trace_curriculum_v1
reward_mode: v2_curriculum_climbing
seed: 42
total_timesteps: 2000000
n_envs: 8
learning_rate: 0.0003
hidden_dim: 512
max_steps: 8
affinity_scorer: tfold
ban_stop: true
terminal_reward_only: true

weights:
  affinity: 1.0
  gap: 1.0
  naturalness: 0.05
  diversity: 0.02

curriculum_gates: [-0.2, 0.0, 0.2, 0.5]
curriculum_bonuses: [0.5, 1.0, 1.5, 2.0]
gap_margin: 1.0
decoy_activation_threshold: 0.5

train_targets: data/tfold_excellent_peptides.txt
```

### 监控指标

每 10K steps 记录：
- Mean A, Median A, Max A
- % A > -0.2, % A > 0.0, % A > 0.2, % A > 0.5
- Mean DecA, Mean Gap (仅 A > 0.5 的 episodes)
- Reward 分布

### 成功标准

**Minimal Success**:
- 2M steps 后，5%+ episodes 达到 A > 0.5
- Mean A > -0.5

**Target Success**:
- 2M steps 后，10%+ episodes 达到 A > 0.5
- 其中 30%+ 达到 Gap > 1

**Stretch Goal**:
- 2M steps 后，20%+ episodes 达到 A > 0.5
- 其中 50%+ 达到 Gap > 2

---

## 总结

这个 curriculum reward 设计的核心是：

1. **分阶段爬升**：-0.2 → 0.0 → 0.2 → 0.5，每个阶段都有明确奖励
2. **延后 decoy**：只有 A > 0.5 才考虑 specificity
3. **正向激励**：用 bonus 而非 penalty，避免模型"躺平"
4. **保护下界**：即使 gap 小，只要 A 高就有奖励

相比 trace29 的 gate-based reward，这个方案更注重**持续爬升**而非"踩线即止"。

---

**设计者**: Based on your insights  
**日期**: 2026-05-23  
**参考实验**: trace29, trace11, trace43
