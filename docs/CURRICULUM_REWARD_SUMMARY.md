# Curriculum Climbing Reward - 实现摘要

## 🎯 核心思想

**首要目标**: 让 target affinity 稳定达到 **0.5**  
**策略**: 分阶段爬升 → 延后 decoy → 只奖励不惩罚

---

## 📊 现状分析

| 实验 | Max A | Mean A | >0.5% | 问题 |
|------|-------|--------|-------|------|
| **trace29** | 1.30 | -1.83 | ~0.1% | 过早引入 decoy，target 优化不足 |
| **trace11** | -2.54 | -5.68 | 0% | Delta reward 对高 affinity 无效 |
| **trace43** | 0.90 | -2.10 | ~0% | 同 trace29 |

**关键问题**: 
- trace29 虽有 max=1.30，但只是偶尔"撞大运"，不稳定
- 67% episodes 达到 gate (-2) 后就满足于"踩线"
- Decoy penalty 过早引入，分散了 target 优化注意力

---

## 🚀 Reward 公式

### Phase 1-4: Target Climbing (A < 0.5)

```python
# Curriculum gates and bonuses
gates = [-0.2, 0.0, 0.2, 0.5]
bonuses = [0.5, 1.0, 1.5, 2.0]

# 计算当前阶段奖励
stage_bonus = 0
for gate, bonus in zip(gates, bonuses):
    if A >= gate:
        stage_bonus = bonus

# Reward (不考虑 decoy)
R = w_affinity * A + stage_bonus + w_nat * Nat + w_div * Div
```

### Phase 5: Specificity (A >= 0.5)

```python
if A >= 0.5:
    gap = A - DecA_mean
    gap_bonus = w_gap * max(0, gap - margin)  # margin = 1.0
    R = w_affinity * A + 2.0 + gap_bonus + w_nat * Nat + w_div * Div
else:
    R = (同 Phase 1-4，不考虑 decoy)
```

**关键特性**:
- ✅ `max(0, gap - margin)` 确保 gap < 1 时不惩罚
- ✅ A < 0.5 时完全不考虑 decoy
- ✅ 即使 decoy 高，只要 A >= 0.5 仍有基础奖励

---

## 📈 训练路线图

| Stage | Steps | 目标 | 预期达成率 | 策略 |
|-------|-------|------|-----------|------|
| 1 | 0-200K | A > -0.2 | 50%+ | Bonus +0.5 激励 |
| 2 | 200K-500K | A > 0.0 | 20%+ | Bonus +1.0 激励 |
| 3 | 500K-800K | A > 0.2 | 10%+ | Bonus +1.5 激励 |
| 4 | 800K-1.2M | A > 0.5 | 5%+ | Bonus +2.0 激励 |
| 5 | 1.2M-2M | Gap > 1 | 30%+ | Gap bonus 激励 |

---

## ⚙️ 参数配置

```yaml
reward_mode: v2_curriculum_climbing

weights:
  affinity: 1.0      # Target affinity 权重
  gap: 1.0           # Specificity gap 权重
  naturalness: 0.05
  diversity: 0.02

curriculum_gates: [-0.2, 0.0, 0.2, 0.5]
curriculum_bonuses: [0.5, 1.0, 1.5, 2.0]
gap_margin: 1.0                          # Gap 阈值
decoy_activation_threshold: 0.5          # Decoy 激活阈值

# 其他
learning_rate: 0.0003
entropy_coef: 0.02
n_envs: 8
max_steps: 8
```

---

## 🎯 成功标准

### Minimal Success (可接受)
- 2M steps 后，**5%+ episodes 达到 A > 0.5**
- Mean A > -0.5

### Target Success (目标)
- 2M steps 后，**10%+ episodes 达到 A > 0.5**
- 其中 30%+ 达到 Gap > 1

### Stretch Goal (理想)
- 2M steps 后，**20%+ episodes 达到 A > 0.5**
- 其中 50%+ 达到 Gap > 2

---

## 🔧 实现要点

### 1. RewardManager 新增方法

```python
def _curriculum_climbing_reward(
    self,
    aff_score: float,
    nat_score: float,
    div_score: float,
    components: Dict[str, float],
) -> float:
    """Curriculum climbing: 逐步爬升到 A > 0.5"""
    
    # Stage bonus
    gates = self.curriculum_gates  # [-0.2, 0.0, 0.2, 0.5]
    bonuses = self.curriculum_bonuses  # [0.5, 1.0, 1.5, 2.0]
    
    stage_bonus = 0.0
    for gate, bonus in zip(gates, bonuses):
        if aff_score >= gate:
            stage_bonus = bonus
    
    # Base reward
    total = self.weights["affinity"] * aff_score + stage_bonus
    
    # Specificity (only if A >= 0.5)
    if aff_score >= self.decoy_activation_threshold:
        decoy_mean = components.get("decoy_final_mean", 0.0)
        gap = aff_score - decoy_mean
        gap_bonus = self.weights["gap"] * max(0.0, gap - self.gap_margin)
        total += gap_bonus
        
        components["gap"] = gap
        components["gap_bonus"] = gap_bonus
    
    total += self.weights["naturalness"] * nat_score
    total += self.weights["diversity"] * div_score
    
    components["stage_bonus"] = stage_bonus
    components["affinity_stage"] = sum(1 for g in gates if aff_score >= g)
    
    return total
```

### 2. Config 新增字段

```python
# reward_manager.py __init__
curriculum_gates: List[float] = None,
curriculum_bonuses: List[float] = None,
gap_margin: float = 1.0,
decoy_activation_threshold: float = 0.5,
```

### 3. Decoy Scorer 调用逻辑

```python
# 只有 A >= 0.5 才调用 decoy scorer
if self.reward_mode == "v2_curriculum_climbing":
    if aff_score >= self.decoy_activation_threshold:
        # 调用 decoy scorer
        decoy_scores = self.decoy_scorer.score_batch(...)
    else:
        # 不调用，节省计算
        decoy_scores = None
```

---

## 📊 监控指标

每 10K steps 记录：

```python
# Target affinity
- mean_A, median_A, max_A, std_A
- pct_A_gt_m02, pct_A_gt_0, pct_A_gt_02, pct_A_gt_05

# Decoy & Gap (仅 A > 0.5 的 episodes)
- mean_DecA, mean_Gap
- pct_gap_gt_1, pct_gap_gt_2

# Stage 分布
- stage_0_pct (A < -0.2)
- stage_1_pct (-0.2 <= A < 0.0)
- stage_2_pct (0.0 <= A < 0.2)
- stage_3_pct (0.2 <= A < 0.5)
- stage_4_pct (A >= 0.5)
```

---

## 🆚 与 trace29 对比

| 维度 | trace29 | Curriculum Climbing |
|------|---------|---------------------|
| **Target gate** | -2.0（固定） | -0.2 → 0.5（渐进） |
| **Decoy 激活** | A > -2 (67%) | A > 0.5 (预期 5%) |
| **Decoy 处理** | 惩罚 DecA > -3 | 奖励 gap > 1，不惩罚 gap < 1 |
| **Pass bonus** | +1.0（固定） | +0.5 → +2.0（渐进） |
| **Max A** | 1.30 | 目标 > 0.5 稳定 |
| **Mean A** | -1.83 | 目标 > -0.5 |
| **>0.5%** | ~0.1% | 目标 5-10% |

---

## 🚨 风险缓解

### 风险 1: 卡在某个 gate
- **缓解**: 降低 gate 阈值或增加 bonus
- **监控**: 如果某个 stage 停留 > 200K steps，触发调整

### 风险 2: 达到 0.5 后 decoy 失控
- **缓解**: 用 `max(0, gap-1)` 不惩罚，保护基础奖励
- **监控**: 如果 gap < 1 的比例 > 80%，考虑增加 w_gap

### 风险 3: 训练不稳定
- **缓解**: 使用 soft gate 平滑过渡
- **监控**: 如果 std_A 持续 > 2.0，降低 learning rate

---

## 📝 下一步

1. **实现 reward mode**: 在 `reward_manager.py` 中添加 `_curriculum_climbing_reward`
2. **创建 config**: `configs/curriculum_climbing_v1.yaml`
3. **启动训练**: `launch_curriculum_v1.sh`
4. **密切监控**: 每 50K steps 检查 stage 分布和 max A
5. **动态调整**: 如果卡住，及时调整 gates/bonuses

---

**设计完成**: 2026-05-23  
**完整文档**: `/share/liuyutian/tcrppo_v2/docs/CURRICULUM_REWARD_DESIGN.md`
