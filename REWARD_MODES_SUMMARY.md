# TCR-PPO Reward Modes 完整总结

**生成时间**: 2026-05-25  
**目标**: 列出所有 reward_manager.py 中的 reward modes 及其效果（仅看 target affinity）

---

## 📋 所有 Reward Modes 列表

### 🔵 V1 系列 (ERGO Scorer)

| Reward Mode | 公式 | 核心思想 | 实验 | Target Affinity 效果 |
|-------------|------|---------|------|---------------------|
| **v1_ergo_only** | `R = w * ERGO` | 纯 ERGO binding | test14, test22b | ✅ **ERGO 上成功** (AUROC=0.609) |
| **v1_ergo_ood_penalty** | `R = w * (ERGO - OOD_penalty)` | ERGO + OOD 惩罚 | test45 | ⚠️ 数据不足 |
| **v1_ergo_convex** | `R = w * ERGO^α` (α=3) | 凸函数放大高分 | test27 | ⚠️ 数据不足 |
| **v1_ergo_squared** | `R = w * ERGO^2` | 平方放大 | test9 | ⚠️ 数据不足 |
| **v1_ergo_delta** | `R = w * (ERGO_final - ERGO_init)` | ERGO delta | - | ❌ 未测试 |
| **v1_ergo_stepwise** | `R = w * ERGO` (每步) | 逐步奖励 | - | ❌ 未测试 |
| **v1_ergo_shaped** | `R = 0.1*delta (中间) + full (终止)` | Shaped reward | - | ❌ 未测试 |
| **contrastive_ergo** | `R = ERGO(target) - mean(ERGO(decoys))` | 对比学习 | test23, test33, test41 | ✅ **ERGO 上成功** (AUROC=0.624) |

---

### 🟢 V2 系列 (tFold Scorer)

#### A. Pure Target (无 Decoy)

| Reward Mode | 公式 | 核心思想 | 实验 | Target Affinity 效果 |
|-------------|------|---------|------|---------------------|
| **v2_no_decoy** | `R = w * A + w_nat*Nat + w_div*Div` | 纯 target absolute | trace10, trace13 | ⚠️ **Last100 A=-7.35** (trace13) |
| **v2_no_decoy_delta** | `R = w * (A_final - A_init) + ...` | Target delta | trace11, trace22, trace23, trace24 | ❌ **Last100 A=-5.54** (trace11, 577K steps) |
| **v2_no_decoy_delta_calibrated** | `R = w*delta + w_abs*(A - ref)` | Delta + absolute 校准 | trace16 | ⚠️ **Last100 A=-6.29** (trace16, 117K steps) |
| **v2_no_decoy_sigmoid_delta** | `R = w * (sigmoid(A_final) - sigmoid(A_init))` | Sigmoid delta | trace12 | ⚠️ 数据不足 |

#### B. Target + Decoy (Gated)

| Reward Mode | 公式 | 核心思想 | 实验 | Target Affinity 效果 |
|-------------|------|---------|------|---------------------|
| **v2_simple_target_gated_decoy** | `if A >= gate: R = w*A + bonus + decoy_penalty` | Gate 激活 decoy | trace29 | ⭐ **Max A=1.30, Mean A=-1.83** (最佳但不稳定) |
| **v2_target_guarded_decoy** | 复杂的 guard 机制 | Guard + decoy | - | ❌ 未测试 |
| **v2_absolute_specificity** | 绝对 specificity | - | - | ❌ 未测试 |
| **v2_hybrid_abs_delta_gated_decoy** | Absolute + delta 混合 | trace54 | ⚠️ 数据不足 |

#### C. Delta - Decoy

| Reward Mode | 公式 | 核心思想 | 实验 | Target Affinity 效果 |
|-------------|------|---------|------|---------------------|
| **v2_delta_minus_decoy** | `R = w*(A_delta) - w_dec*(DecA_delta)` | Target delta 减 decoy delta | trace20, trace21 | ❌ **Decoy 也增长** (DecDelta>0) |

#### D. Stepwise

| Reward Mode | 公式 | 核心思想 | 实验 | Target Affinity 效果 |
|-------------|------|---------|------|---------------------|
| **tfold_stepwise** | `R = w * (A - A_init)` (每步) | 每步给 delta | trace17, trace18 | ⚠️ **Last100 A=-6.34** (trace17, 104K steps) |
| **tfold_delta_calibrated** | `R = w*delta + w_abs*(A - ref)` | Delta + absolute | - | ❌ 未测试 |
| **tfold_delta_amplified** | `R = w * amplified_delta` | 放大 delta | trace19 | ⚠️ **Last100 A=-8.07** (trace19, 800 steps) |

#### E. Curriculum

| Reward Mode | 公式 | 核心思想 | 实验 | Target Affinity 效果 |
|-------------|------|---------|------|---------------------|
| **v2_curriculum_climbing** | 分阶段 gate + bonus | 逐步爬升 | trace50, trace51 | 🔄 **进行中** |

#### F. Legacy/Other

| Reward Mode | 公式 | 核心思想 | 实验 | Target Affinity 效果 |
|-------------|------|---------|------|---------------------|
| **v2_full** | `R = w*A - w_dec*DecA + ...` | 完整版 | - | ❌ 未测试 |
| **v2_decoy_only** | `R = w*A - w_dec*DecA` | - | - | ❌ 未测试 |
| **threshold_penalty** | `if A < 0.5: R = w*A; else: R = w*A - decoy` | 阈值激活 | - | ❌ 未测试 |

---

## 🎯 Target Affinity 效果排名

### ✅ 成功 (Max A > 1.0 或 Mean A > 0)

| 排名 | Reward Mode | 实验 | Max A | Mean A | A>0 比例 | 评价 |
|------|-------------|------|-------|--------|---------|------|
| 🥇 | **v2_simple_target_gated_decoy** | trace29 | **1.30** | -1.83 | 1.24% | **最高 max，但不稳定** |
| 🥈 | **v1_ergo_only** | test14 | - | - | - | **ERGO 上 AUROC=0.609** |
| 🥉 | **contrastive_ergo** | test41 | - | - | - | **ERGO 上 AUROC=0.624** |

### ⚠️ 部分成功 (Max A > 0 但 Mean A < -1)

| Reward Mode | 实验 | Max A | Mean A | Last100 A | 评价 |
|-------------|------|-------|--------|-----------|------|
| **v2_no_decoy_delta** | trace11 | 0.509 | -6.10 | -5.54 | Delta 与 absolute 脱钩 |
| **tfold_stepwise** | trace17 | - | - | -6.34 | 每步 delta 效果差 |
| **v2_no_decoy_delta_calibrated** | trace16 | - | - | -6.29 | 校准无效 |

### ❌ 失败 (Max A < 0 或 Mean A < -5)

| Reward Mode | 实验 | Last100 A | 评价 |
|-------------|------|-----------|------|
| **v2_no_decoy** | trace13 | -7.35 | Pure absolute 无激励 |
| **tfold_delta_amplified** | trace19 | -8.07 | 放大 delta 无效 |
| **v2_delta_minus_decoy** | trace20/21 | - | Decoy 也增长 |

---

## 📊 详细实验数据对比

### trace29 (v2_simple_target_gated_decoy) - 当前最佳

**配置**:
```yaml
reward_mode: v2_simple_target_gated_decoy
target_decoy_gate_logit: -2.0
target_pass_bonus: 1.0
w_affinity: 1.0
w_decoy: 0.3
decoy_affinity_center: -3.0
```

**效果** (12,384 episodes / 674K steps):
- **Max A**: 1.30 ⭐⭐⭐
- **Mean A**: -1.83
- **Median A**: -1.49
- **A > 0 比例**: 1.24%
- **A > -1 比例**: 26.67%
- **A > -2 比例**: 67.32% (触发 stage 2)

**按 peptide**:
| Peptide | Max A | Mean A | A>0 count | A>0 比例 |
|---------|-------|--------|-----------|---------|
| YLQPRTFLL | 1.30 | -2.71 | 95/2082 | 4.6% |
| LLLDRLNQL | 0.83 | -2.76 | 77/1928 | 4.0% |
| AVFDRKSDAK | 0.66 | -3.31 | 7/2048 | 0.3% |

**问题**:
- Gate 太低 (-2.0)，67% episodes "踩线"后不再努力
- Decoy weight 太小 (0.3)，无法有效压制
- 极不稳定，只有 1.24% 达到 A>0

---

### trace11 (v2_no_decoy_delta) - Delta 失败案例

**配置**:
```yaml
reward_mode: v2_no_decoy_delta
w_affinity: 1.0
max_steps: 8
ban_stop: yes
terminal_only: yes
```

**效果** (67,640 episodes / 577K steps):
- **Max A**: 0.509
- **Mean A**: -6.10
- **Last100 R**: 2.402 (R 很高！)
- **Last100 A**: -5.54 (A 很差！)
- **A > 0 count**: 0

**问题**:
- **Delta reward 与 absolute affinity 完全脱钩**
- 模型学会"从很差变到稍微不那么差"
- 训练 577K steps，R=2.4 但 A 仍然 -5.5
- **应该放弃 delta 方法**

---

### trace13 (v2_no_decoy) - Pure Absolute 失败

**配置**:
```yaml
reward_mode: v2_no_decoy
w_affinity: 1.0
w_naturalness: 0.2
```

**效果** (episodes 数据不足):
- **Last100 A**: -7.35
- **Last100 R**: 0.651

**问题**:
- Pure absolute 没有足够激励
- 可能需要更大的 w_affinity 或 bonus

---

### trace17 (tfold_stepwise) - Stepwise 失败

**配置**:
```yaml
reward_mode: tfold_stepwise
max_steps: 8
ban_stop: yes
```

**效果** (13,160 episodes / 104K steps):
- **Last100 R**: 1.290
- **Last100 A**: -6.34
- **Max R**: 14.013

**问题**:
- 与 delta 类似，R 高但 A 差
- Stepwise delta 无法提升 absolute affinity

---

### test41 (contrastive_ergo) - ERGO 上的成功

**配置**:
```yaml
# Phase 1: Pure ERGO (2M steps)
reward_mode: v1_ergo_only

# Phase 2: Contrastive (1.5M steps)
reward_mode: contrastive_ergo
n_contrast_decoys: 16
contrastive_agg: mean
```

**效果** (ERGO scorer, Best-of-200):
- **平均 AUROC**: 0.624 ⭐⭐⭐⭐
- **Top1 Binding**: 0.848
- **Top1 AUROC**: 0.931

**关键**:
- ✅ 在 ERGO scorer 上非常成功
- ⚠️ 但 ERGO 和 tFold 是不同的 scorer
- 需要验证能否迁移到 tFold

---

## 🎯 核心结论

### 1. 最佳方法：v2_simple_target_gated_decoy (trace29)

**优点**:
- ✅ 唯一达到 Max A > 1.0 的方法
- ✅ 某些 peptide 效果好 (YLQPRTFLL: 4.6% > 0)
- ✅ 证明了 gate 机制有效

**缺点**:
- ❌ 极不稳定 (mean -1.83, 只有 1.24% > 0)
- ❌ Gate 太低 (-2.0)
- ❌ Decoy weight 太小 (0.3)

**改进方向**:
1. 提高 gate: -2.0 → -0.5
2. 增大 bonus: 1.0 → 5.0
3. 增大 affinity weight: 1.0 → 2.0-3.0
4. 暂时移除或降低 decoy weight

---

### 2. 完全失败：Delta Reward 系列

**失败方法**:
- v2_no_decoy_delta (trace11)
- v2_delta_minus_decoy (trace20/21)
- tfold_stepwise (trace17)
- tfold_delta_amplified (trace19)

**共同问题**:
- **Delta reward 与 absolute affinity 脱钩**
- R 可以很高，但 A 仍然很差
- 模型学会"从很差变到稍微不那么差"

**结论**: ❌ **应该放弃所有 delta-based 方法**

---

### 3. 未充分测试：Pure Absolute

**方法**: v2_no_decoy (trace13)

**问题**:
- 数据不足 (只有 Last100 A=-7.35)
- 可能需要更强的激励机制

**潜力**: ⭐⭐ 值得重新测试，但需要：
- 更大的 w_affinity (2.0-3.0)
- 阶梯式 bonus
- 更长的训练时间

---

### 4. ERGO 成功但未迁移：Two-Phase

**方法**: v1_ergo_only + contrastive_ergo (test41)

**效果**: ERGO 上 AUROC=0.624 ⭐⭐⭐⭐

**问题**: 未在 tFold 上测试

**潜力**: ⭐⭐⭐⭐ 如果能迁移到 tFold，可能是最佳方案

---

## 📋 推荐实验优先级

### 🔥 立即启动

1. **trace29_improved_v2**: 改进 trace29，移除 decoy
   ```yaml
   reward_mode: v2_no_decoy  # 或新建 v2_pure_target_amplified
   R = 3.0 * A + bonus(A) + 0.05*Nat + 0.02*Div
   bonus: A>-0.5 → +5.0, A>0 → +10.0, A>0.5 → +20.0
   ```

2. **trace29_improved_v1**: 提高 gate 和 bonus
   ```yaml
   reward_mode: v2_simple_target_gated_decoy
   target_decoy_gate_logit: -0.5  # 原 -2.0
   target_pass_bonus: 5.0         # 原 1.0
   w_affinity: 2.0                # 原 1.0
   w_decoy: 0.1                   # 原 0.3
   ```

### 🔄 下周启动

3. **tfold_two_phase_v1**: 复制 test41 方法到 tFold
   - Phase 1: Pure target (1M steps)
   - Phase 2: Contrastive (1M steps)

4. **curriculum_soft_v1**: Soft gate curriculum
   - 使用 v2_curriculum_climbing
   - 调整 gates 和 bonuses

---

## 📊 监控指标

每 50K steps 检查：
- **Mean A**, Median A, Max A
- **A > 0 比例**
- **A > 0.5 比例**
- A > -1 比例
- A > -2 比例

**成功标准**:
- Minimal: Mean A > -0.5, A>0 比例 > 5%
- Target: Mean A > 0, A>0 比例 > 10%
- Stretch: Mean A > 0.3, A>0.5 比例 > 10%

---

**报告完成**: 2026-05-25  
**下一步**: 根据此报告启动改进实验
