# TCR-PPO 策略全面分析报告

**生成时间**: 2026-05-25  
**目标**: 找到能稳定达到 **Target Affinity > 0.6** 的训练策略  
**当前最佳**: trace29 max=1.30, 但极不稳定（mean=-1.83）

---

## 🎯 核心问题定义

你说得对！**我们应该先不管 decoy，专注把 target affinity 提上去！**

当前的困境：
- **trace11** (v2-no-decoy-delta): 走了 67K episodes / 577K steps，Last100 R=2.402，但 **Last100 A=-5.542**（affinity 仍然很差）
- **trace29** (v2-simple-target-gated-decoy): max A=1.30，但 mean A=-1.83，只有 **1.24% episodes 达到 A>0**
- **test41** (ERGO two-phase): 平均 AUROC=0.624，但这是 ERGO scorer，不是 tFold

**关键洞察**: 
1. Delta reward (trace11) 对提升 absolute affinity 无效
2. 过早引入 decoy 会分散注意力
3. 需要找到能让 **absolute target affinity 稳定 > 0.6** 的方法

---

## 📊 实验方法全景图

### 方法分类

| 类别 | Reward 模式 | 代表实验 | 核心思想 | 效果 |
|------|------------|---------|---------|------|
| **A. Pure Target Absolute** | v2_no_decoy | trace10, trace13 | 直接优化 target affinity logit | ⚠️ 数据不足 |
| **B. Target Delta** | v2_no_decoy_delta | trace11, trace22, trace23, trace24 | 优化 final - initial delta | ❌ 对 absolute 无效 |
| **C. Target + Gated Decoy** | v2_simple_target_gated_decoy | trace29 | Target 达标后引入 decoy | ⚠️ 不稳定 |
| **D. Delta - Decoy** | v2_delta_minus_decoy | trace20, trace21 | Target delta 减 decoy delta | ❌ Decoy 也增长 |
| **E. Stepwise** | tfold_stepwise | trace17, trace18 | 每步给 delta reward | ⚠️ 数据不足 |
| **F. ERGO Two-Phase** | ergo → contrastive | test33, test41 | ERGO warm-start + contrastive | ✅ ERGO 上成功 |
| **G. Curriculum** | v2_curriculum_climbing | trace50, trace51 | 分阶段爬升 | 🔄 进行中 |

---

## 🔍 详细实验分析

### A. Pure Target Absolute (v2_no_decoy)

**代表**: trace10, trace13

**Reward 公式**:
```python
R = w_affinity * A + w_nat * Nat + w_div * Div
```

**数据**:
- trace10: 数据不完整（pre-trace11）
- trace13: Last100 R=0.651, Last100 A=-7.348（❌ 很差）

**问题**: 
- 没有足够的激励机制让 A 快速增长
- 可能需要更长时间或更大的 w_affinity

**潜力**: ⭐⭐ 需要更多实验验证

---

### B. Target Delta (v2_no_decoy_delta)

**代表**: trace11 (主线), trace22 (max_steps=4), trace23 (允许 STOP), trace24 (sub_only)

**Reward 公式**:
```python
R = w_affinity * (A_final - A_initial) + w_nat * Nat + w_div * Div
```

**数据**:

| Trace | Episodes | Steps | Last100 R | Last100 A | Max A | 评价 |
|-------|----------|-------|-----------|-----------|-------|------|
| trace11 | 67,640 | 577K | 2.402 | -5.542 | 0.509 | ❌ R 高但 A 差 |
| trace22 | 29,592 | 118K | 1.419 | -6.526 | -2.587 | ❌ 更差 |
| trace23 | 4,949 | 36K | 1.114 | - | - | ❌ 数据不足 |
| trace24 | 4,776 | 38K | 0.728 | -7.317 | - | ❌ 最差 |

**关键问题**:
1. **Delta reward 与 absolute affinity 脱钩**: R=2.4 但 A=-5.5
2. 模型学会了"从很差变到稍微不那么差"，但绝对值仍然很低
3. 即使训练 577K steps，max A 只有 0.509（远低于 trace29 的 1.30）

**结论**: ❌ **Delta reward 对提升 absolute affinity 无效，应该放弃**

---

### C. Target + Gated Decoy (v2_simple_target_gated_decoy)

**代表**: trace29 ⭐ **当前最佳**

**Reward 公式**:
```python
if A >= target_gate (-2.0):
    R = w_affinity * A + target_pass_bonus (1.0) + decoy_penalty + w_nat*Nat + w_div*Div
else:
    R = w_affinity * A + w_nat*Nat + w_div*Div  # 不考虑 decoy
```

**数据** (12,384 episodes / 674K steps):

| 指标 | 值 | 评价 |
|------|-----|------|
| **Max A** | **1.30** | ⭐⭐⭐ 全局最高！ |
| Mean A | -1.83 | ⚠️ 平均很低 |
| Median A | -1.49 | ⚠️ 中位数也低 |
| **A > 0 比例** | **1.24%** | ❌ 极少 |
| A > -1 比例 | 26.67% | ⚠️ 只有 1/4 |
| **A > -2 比例** | **67.32%** | ✅ 大部分达标 |

**按 peptide 分析** (Top 5):

| Peptide | Max A | Mean A | A>0 count | 评价 |
|---------|-------|--------|-----------|------|
| YLQPRTFLL | **1.30** | -2.71 | 95/2082 (4.6%) | ⭐ 最好 |
| LLLDRLNQL | 0.83 | -2.76 | 77/1928 (4.0%) | ⭐ 很好 |
| AVFDRKSDAK | 0.66 | -3.31 | 7/2048 (0.3%) | ⚠️ 不稳定 |
| GILGFVFTL | 0.66 | -3.04 | 23/1942 (1.2%) | ⚠️ 不稳定 |
| SLFNTVATLY | 0.65 | -3.13 | 16/1932 (0.8%) | ⚠️ 不稳定 |

**关键发现**:
1. ✅ **能达到高 affinity**: max=1.30 远超其他方法
2. ✅ **某些 peptide 效果好**: YLQPRTFLL 有 4.6% episodes > 0
3. ❌ **极不稳定**: 平均 -1.83，只有 1.24% > 0
4. ❌ **Decoy 问题**: 平均 DecA=-0.89（太高），Gap=-0.76（负值！）

**问题根源**:
1. **Target gate 太低 (-2.0)**: 67% episodes 达标后就"满足"了，不再努力提升
2. **Decoy weight 太小 (0.3)**: 无法有效压制 decoy
3. **Pass bonus 太小 (1.0)**: 没有足够激励继续提升

**潜力**: ⭐⭐⭐⭐ **最有潜力，但需要调参**

---

### D. Delta - Decoy (v2_delta_minus_decoy)

**代表**: trace20 (from trace11), trace21 (scratch)

**Reward 公式**:
```python
R = w_affinity * (A_final - A_initial) - w_decoy * (DecA_final - DecA_initial)
```

**数据**:

| Trace | Episodes | Last100 R | Last100 DecDelta | 评价 |
|-------|----------|-----------|------------------|------|
| trace20 | - | 1.582 | 1.134 | ❌ Decoy 也增长 |
| trace21 | - | 0.812 | 0.671 | ❌ Decoy 也增长 |

**问题**: DecDelta > 0 说明 decoy 也在增强，没有 specificity

**结论**: ❌ **失败，应该放弃**

---

### E. Stepwise (tfold_stepwise)

**代表**: trace17, trace18

**Reward 公式**: 每步给 delta reward

**数据**:

| Trace | Episodes | Steps | Last100 R | Last100 A | 评价 |
|-------|----------|-------|-----------|-----------|------|
| trace17 | 13,160 | 104K | 1.290 | -6.339 | ⚠️ A 很差 |
| trace18 | 6,689 | 46K | 0.651 | -7.348 | ❌ 更差 |

**问题**: 与 delta 类似，R 高但 A 差

**结论**: ⚠️ **不如 trace29，优先级低**

---

### F. ERGO Two-Phase (ergo → contrastive)

**代表**: test33, test41 ⭐ **ERGO 上的最佳**

**方法**:
1. Phase 1: 2M steps 纯 ERGO binding
2. Phase 2: 1.5M steps contrastive (target - mean(8 decoys))

**数据** (ERGO scorer, Best-of-200):

| Model | 平均 AUROC | Top1 Binding | Top1 AUROC | 评价 |
|-------|-----------|--------------|------------|------|
| test33 | 0.598 | 0.877 | 0.919 | ⭐⭐⭐ |
| test41 | **0.624** | 0.848 | 0.931 | ⭐⭐⭐⭐ |

**关键**: 
- ✅ 在 ERGO scorer 上非常成功
- ❌ 但 ERGO 和 tFold 是不同的 scorer，不能直接迁移
- ⚠️ 需要验证这个方法在 tFold 上是否有效

**潜力**: ⭐⭐⭐⭐ **如果能迁移到 tFold，可能是最佳方案**

---

### G. Curriculum (v2_curriculum_climbing)

**代表**: trace50, trace51

**方法**: 分阶段爬升，逐步提高 gate

**数据**: 🔄 进行中，数据不足

**潜力**: ⭐⭐⭐ **理论上合理，但需要验证**

---

## 🎯 策略推荐

### 🥇 推荐 1: 改进 trace29 (最快见效)

**核心思想**: trace29 已经证明能达到 max=1.30，只是不稳定。通过调参让它更稳定。

**具体改进**:

1. **提高 target gate**: -2.0 → **-1.0** 或 **-0.5**
   - 让模型不满足于"刚过线"
   - 只有真正高 affinity 才能获得 pass bonus

2. **增大 pass bonus**: 1.0 → **3.0** 或 **5.0**
   - 强烈激励达到高 affinity

3. **暂时移除 decoy penalty** (或设置很小的 weight)
   - 专注提升 target affinity
   - 等 target 稳定 > 0 后再引入 decoy

4. **增加 affinity weight**: 1.0 → **2.0**
   - 让 affinity 成为主导信号

**新 Reward 公式**:
```python
if A >= -0.5:
    R = 2.0 * A + 5.0 + 0.05*Nat + 0.02*Div
else:
    R = 2.0 * A + 0.05*Nat + 0.02*Div
```

**预期效果**:
- A > 0 比例: 1.24% → **10%+**
- Mean A: -1.83 → **-0.5+**
- Max A: 1.30 → **1.5+**

**优点**: 
- ✅ 基于已验证的最佳方法
- ✅ 改动小，风险低
- ✅ 快速见效（预计 200K steps 见效）

**缺点**:
- ⚠️ 仍然是调参，不是根本性改进

---

### 🥈 推荐 2: tFold Two-Phase (借鉴 ERGO 成功经验)

**核心思想**: 复制 test41 的成功经验，但用 tFold scorer

**Phase 1: Pure Target Binding** (1M steps)
```yaml
reward_mode: v2_no_decoy_absolute_amplified
R = 3.0 * A + 0.05*Nat + 0.02*Div
learning_rate: 3e-4
entropy_coef: 0.05 → 0.01
```

**Phase 2: Contrastive Fine-tuning** (1M steps)
```yaml
reward_mode: v2_contrastive_mean
R = A - mean(DecA_8) + 0.05*Nat + 0.02*Div
learning_rate: 1.5e-4
entropy_coef: 0.01 → 0.005
```

**关键**:
- Phase 1 专注建立 binding 能力，不管 specificity
- Phase 2 在 binding 基础上强化 specificity
- 使用 mean aggregation（test40 证明 max 会崩溃）

**预期效果**:
- Phase 1 结束: Mean A > 0, Max A > 1.5
- Phase 2 结束: Mean AUROC > 0.6

**优点**:
- ✅ 在 ERGO 上已验证成功
- ✅ 理论上更稳健
- ✅ 分阶段训练，风险可控

**缺点**:
- ⚠️ 需要 2M steps，时间长
- ⚠️ tFold 和 ERGO 可能有差异

---

### 🥉 推荐 3: Curriculum with Soft Gates (理论最优)

**核心思想**: 不用 hard gate，用 soft bonus 平滑过渡

**Reward 公式**:
```python
# Soft bonus: 随 A 增长平滑增加
bonus = 0
for gate, bonus_val in [(-1.0, 1.0), (-0.5, 2.0), (0.0, 3.0), (0.5, 5.0)]:
    if A >= gate:
        bonus = bonus_val

R = 2.0 * A + bonus + 0.05*Nat + 0.02*Div
```

**训练计划**:
- 0-500K: 目标 A > -1 (30%+)
- 500K-1M: 目标 A > -0.5 (20%+)
- 1M-1.5M: 目标 A > 0 (10%+)
- 1.5M-2M: 目标 A > 0.5 (5%+)

**优点**:
- ✅ 理论上最优雅
- ✅ 避免 hard gate 的"踩线"问题
- ✅ 平滑过渡，训练稳定

**缺点**:
- ⚠️ 需要 2M steps
- ⚠️ 调参空间大

---

## 📋 实验优先级

### 立即启动 (本周)

1. **trace29_improved_v1**: 改进 trace29，提高 gate 和 bonus
   - GPU: 1 个
   - 预计: 500K steps (2-3 天)
   - 成功标准: A>0 比例 > 5%

2. **trace29_improved_v2**: 移除 decoy，纯 target
   - GPU: 1 个
   - 预计: 500K steps (2-3 天)
   - 成功标准: Mean A > -1.0

### 下周启动

3. **tfold_two_phase_v1**: 复制 test41 方法
   - GPU: 1 个
   - 预计: 2M steps (7-10 天)
   - 成功标准: Phase 1 结束 Mean A > 0

4. **curriculum_soft_v1**: Soft gate curriculum
   - GPU: 1 个
   - 预计: 2M steps (7-10 天)
   - 成功标准: 500K steps 时 A>-1 比例 > 30%

---

## 🔬 关键假设验证

### 假设 1: Pure absolute reward 能提升 affinity

**验证方法**: trace29_improved_v2 (移除 decoy)

**如果成功**: 说明 decoy 确实分散注意力，应该延后引入

**如果失败**: 说明需要更强的激励机制（更大的 bonus）

### 假设 2: ERGO two-phase 方法可迁移到 tFold

**验证方法**: tfold_two_phase_v1

**如果成功**: 这将是最稳健的方案

**如果失败**: 说明 tFold 和 ERGO 的 reward landscape 差异太大

### 假设 3: Soft gate 优于 hard gate

**验证方法**: curriculum_soft_v1 vs trace29_improved_v1

**如果成功**: 未来所有实验都应该用 soft gate

**如果失败**: 说明 hard gate 的"明确目标"反而更有效

---

## 📊 成功标准

### Minimal Success (可接受)
- **Mean A > -0.5**
- **A > 0 比例 > 5%**
- Max A > 1.0

### Target Success (目标)
- **Mean A > 0**
- **A > 0 比例 > 10%**
- **A > 0.5 比例 > 5%**
- Max A > 1.5

### Stretch Goal (理想)
- **Mean A > 0.3**
- **A > 0.5 比例 > 10%**
- **A > 0.6 比例 > 5%**
- Max A > 2.0

---

## 🎯 最终建议

**你说得对！我们应该先专注 target affinity，不管 decoy！**

**立即行动**:

1. **启动 trace29_improved_v2** (纯 target，无 decoy)
   ```yaml
   reward_mode: v2_pure_target_amplified
   R = 3.0 * A + bonus(A) + 0.05*Nat + 0.02*Div
   bonus: A>-0.5 → +5.0, A>0 → +10.0, A>0.5 → +20.0
   ```

2. **同时启动 trace29_improved_v1** (提高 gate)
   ```yaml
   reward_mode: v2_simple_target_gated_decoy
   target_gate: -0.5 (原 -2.0)
   target_pass_bonus: 5.0 (原 1.0)
   w_decoy: 0.1 (原 0.3，降低)
   ```

3. **密切监控**: 每 50K steps 检查 A 分布
   - 如果 Mean A 不增长，立即调整 bonus
   - 如果 Max A 增长但 Mean 不动，增加 exploration (entropy)

**预期**: 500K steps 内看到明显改善，Mean A > -1.0

---

**报告完成时间**: 2026-05-25  
**下一步**: 等待你的决策，立即启动实验
