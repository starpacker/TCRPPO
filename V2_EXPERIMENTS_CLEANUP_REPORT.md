# V2 系列实验日志清理和状态报告

**生成时间**: 2026-05-25  
**数据来源**: recent_experiment_report_20260519.md + trace29 分析  
**目标**: 专注 Target Affinity 提升，Decoy 放在最后

---

## 📊 总体统计

基于现有日志分析，V2 系列实验分类：

- **总实验数**: ~25 个 trace
- **短期失败** (<10K steps): ~8 个 → **建议删除**
- **中期实验** (10K-100K steps): ~10 个
- **长期实验** (>100K steps): ~7 个 → **重点分析**

---

## 🗑️ 建议删除：短期失败实验

以下实验运行步数 < 10K，数据不足，**建议删除日志文件**：

### 确认删除列表

| Trace | Reward Mode | Steps | Episodes | 原因 |
|-------|-------------|-------|----------|------|
| trace3 | (未知) | < 1K | < 100 | 极早期失败 |
| trace4 | v2_no_decoy | < 5K | < 500 | 数据不足 |
| trace5 | v2_no_decoy_delta | < 5K | < 500 | 数据不足 |
| trace6 | v2_no_decoy_delta | < 5K | < 500 | 数据不足 |
| trace7 | v2_no_decoy_delta | < 5K | < 500 | 数据不足 |
| trace8 | v2_no_decoy_delta | < 5K | < 500 | 数据不足 |
| trace9 | v2_no_decoy_delta | < 5K | < 500 | 数据不足 |
| trace19 | tfold_delta_amplified | 800 | 808 | 极早期失败 |

**删除命令**:
```bash
cd /share/liuyutian/tcrppo_v2/logs
rm -f test51c_*_trace3_*.log
rm -f test51c_*_trace4_*.log
rm -f test51c_*_trace5_*.log
rm -f test51c_*_trace6_*.log
rm -f test51c_*_trace7_*.log
rm -f test51c_*_trace8_*.log
rm -f test51c_*_trace9_*.log
rm -f test53_*_trace19_*.log
```

---

## ⭐ 重点关注：长期实验 (>100K steps)

### 按 Target Affinity (Max A) 排序

| 排名 | Trace | Reward Mode | Steps | Episodes | Max A | Mean A | Last100 A | Status | 评价 |
|------|-------|-------------|-------|----------|-------|--------|-----------|--------|------|
| 🥇 1 | **trace29** | **v2_simple_target_gated_decoy** | **674K** | **12,384** | **1.30** | -1.83 | - | ⏹️ Stopped | **唯一成功！** |
| 🥈 2 | **trace11** | v2_no_decoy_delta | **577K** | **67,640** | **0.509** | -6.10 | -5.54 | 🔄 Running | Delta 失败 |
| 🥉 3 | **trace13** | v2_no_decoy | **311K** | **34,536** | **-1.79** | - | -5.88 | ⏹️ Stopped | Pure absolute 失败 |
| 4 | **trace22** | v2_no_decoy_delta | **118K** | **29,592** | **-2.59** | -6.83 | -6.53 | 🔄 Running | Delta 失败 |
| 5 | **trace16** | v2_no_decoy_delta_calibrated | **117K** | **29,344** | - | - | -6.29 | ⏹️ Stopped | Calibrated 失败 |
| 6 | **trace17** | tfold_stepwise | **104K** | **13,160** | - | - | -6.34 | 🔄 Running | Stepwise 失败 |

### 关键发现

1. **trace29 是唯一有效的方法**
   - Max A = 1.30（远超其他所有方法）
   - 但 Mean A = -1.83（极不稳定）
   - 只有 1.24% episodes 达到 A > 0

2. **所有 Delta 方法都失败了**
   - trace11: 577K steps，Max A = 0.509
   - trace22: 118K steps，Max A = -2.59
   - trace16: 117K steps，Last100 A = -6.29
   - **结论**: Delta reward 与 absolute affinity 脱钩

3. **Pure absolute 未充分测试**
   - trace13: 311K steps，Last100 A = -5.88
   - 可能需要更强的激励机制

---

## 🎯 按 Reward Mode 分类效果

### A. Pure Target (无 Decoy)

#### ✅ v2_simple_target_gated_decoy - **唯一成功**

| Trace | Steps | Episodes | Max A | Mean A | Status |
|-------|-------|----------|-------|--------|--------|
| **trace29** | **674K** | **12,384** | **1.30** ⭐ | -1.83 | ⏹️ Stopped |

**配置**:
```yaml
target_decoy_gate_logit: -2.0
target_pass_bonus: 1.0
w_affinity: 1.0
w_decoy: 0.3
```

**问题**:
- Gate 太低 (-2.0)，67% episodes "踩线"
- Decoy weight 太小 (0.3)
- 极不稳定

**改进方向**: ⭐⭐⭐⭐⭐
- 提高 gate: -2.0 → -0.5
- 增大 bonus: 1.0 → 5.0
- 增大 affinity weight: 1.0 → 3.0
- 移除或降低 decoy weight

---

#### ❌ v2_no_decoy - Pure Absolute 失败

| Trace | Steps | Episodes | Last100 A | Status |
|-------|-------|----------|-----------|--------|
| trace10 | < 10K | - | - | ⏹️ Stopped (删除) |
| **trace13** | **311K** | **34,536** | **-5.88** | ⏹️ Stopped |

**问题**: 没有足够激励机制

**潜力**: ⭐⭐ 值得重新测试，但需要：
- 更大的 w_affinity (3.0)
- 阶梯式 bonus
- 更长训练时间

---

#### ❌ v2_no_decoy_delta - Delta 完全失败

| Trace | Steps | Episodes | Max A | Last100 A | Status |
|-------|-------|----------|-------|-----------|--------|
| **trace11** | **577K** | **67,640** | **0.509** | **-5.54** | 🔄 Running |
| **trace22** | **118K** | **29,592** | **-2.59** | **-6.53** | 🔄 Running |
| trace23 | 36K | 4,949 | - | -6.90 | 🔄 Running |
| trace24 | 38K | 4,776 | - | -7.32 | 🔄 Running |

**核心问题**: 
- **Delta reward 与 absolute affinity 完全脱钩**
- trace11 训练 577K steps，R=2.4 但 A=-5.5
- 模型学会"从很差变到稍微不那么差"

**结论**: ❌ **应该立即停止所有 delta 实验，放弃此方法**

**建议操作**:
```bash
# 停止正在运行的 delta 实验
# trace11, trace22, trace23, trace24
```

---

#### ❌ v2_no_decoy_delta_calibrated - Calibrated 失败

| Trace | Steps | Episodes | Last100 A | Status |
|-------|-------|----------|-----------|--------|
| **trace16** | **117K** | **29,344** | **-6.29** | ⏹️ Stopped |

**问题**: Calibration 无法解决 delta 的根本问题

**结论**: ❌ 放弃

---

#### ❌ v2_no_decoy_sigmoid_delta - Sigmoid Delta 失败

| Trace | Steps | Episodes | Status |
|-------|-------|----------|--------|
| trace12 | < 10K | - | ⏹️ Stopped (删除) |

**结论**: ❌ 数据不足，放弃

---

### B. Stepwise

#### ❌ tfold_stepwise - Stepwise 失败

| Trace | Steps | Episodes | Last100 A | Status |
|-------|-------|----------|-----------|--------|
| **trace17** | **104K** | **13,160** | **-6.34** | 🔄 Running |
| trace18 | 46K | 6,689 | -7.35 | ⏹️ Stopped |

**问题**: 与 delta 类似，R 高但 A 差

**结论**: ❌ **应该停止 trace17**

---

#### ❌ tfold_delta_amplified - Amplified Delta 失败

| Trace | Steps | Episodes | Last100 A | Status |
|-------|-------|----------|-----------|--------|
| trace19 | 800 | 808 | -8.07 | ⏹️ Stopped (删除) |

**结论**: ❌ 极早期失败，删除

---

### C. Delta - Decoy

#### ❌ v2_delta_minus_decoy - Delta Minus Decoy 失败

| Trace | Steps | Episodes | Last100 DecDelta | Status |
|-------|-------|----------|------------------|--------|
| trace20 | 355K | 2,224 | **1.134** (正值！) | 🔄 Running |
| trace21 | 23K | 2,848 | 0.671 (正值) | 🔄 Running |

**问题**: DecDelta > 0 说明 decoy 也在增强，没有 specificity

**结论**: ❌ **应该立即停止 trace20 和 trace21**

---

### D. Curriculum

#### 🔄 v2_curriculum_climbing - 进行中

| Trace | Steps | Status |
|-------|-------|--------|
| trace50 | 进行中 | 🔄 Running |
| trace51 | 进行中 | 🔄 Running |

**状态**: 数据不足，继续观察

---

## 🔄 当前运行状态总结

### 正在运行的实验

| Trace | Reward Mode | Steps | Last100 A | 建议 |
|-------|-------------|-------|-----------|------|
| trace11 | v2_no_decoy_delta | 577K | -5.54 | ❌ **立即停止** (Delta 失败) |
| trace17 | tfold_stepwise | 104K | -6.34 | ❌ **立即停止** (Stepwise 失败) |
| trace20 | v2_delta_minus_decoy | 355K | DecDelta=1.13 | ❌ **立即停止** (Decoy 增长) |
| trace21 | v2_delta_minus_decoy | 23K | DecDelta=0.67 | ❌ **立即停止** (Decoy 增长) |
| trace22 | v2_no_decoy_delta | 118K | -6.53 | ❌ **立即停止** (Delta 失败) |
| trace23 | v2_no_decoy_delta | 36K | -6.90 | ❌ **立即停止** (Delta 失败) |
| trace24 | v2_no_decoy_delta | 38K | -7.32 | ❌ **立即停止** (Delta 失败) |
| trace50 | v2_curriculum_climbing | ? | ? | ⏳ 继续观察 |
| trace51 | v2_curriculum_climbing | ? | ? | ⏳ 继续观察 |

**建议**: 立即停止 trace11, 17, 20, 21, 22, 23, 24（7 个实验）

---

## 📋 清理操作清单

### 1. 删除短期失败实验日志 (8 个)

```bash
cd /share/liuyutian/tcrppo_v2/logs

# 删除 trace3-9 (< 5K steps)
rm -f test51c_*_trace3_*.log
rm -f test51c_*_trace4_*.log
rm -f test51c_*_trace5_*.log
rm -f test51c_*_trace6_*.log
rm -f test51c_*_trace7_*.log
rm -f test51c_*_trace8_*.log
rm -f test51c_*_trace9_*.log

# 删除 trace19 (800 steps)
rm -f test53_*_trace19_*.log

echo "✅ 已删除 8 个短期失败实验的日志"
```

### 2. 停止失败的长期实验 (7 个)

需要手动停止以下正在运行的实验：

- **trace11** (v2_no_decoy_delta) - 577K steps
- **trace17** (tfold_stepwise) - 104K steps
- **trace20** (v2_delta_minus_decoy) - 355K steps
- **trace21** (v2_delta_minus_decoy) - 23K steps
- **trace22** (v2_no_decoy_delta) - 118K steps
- **trace23** (v2_no_decoy_delta) - 36K steps
- **trace24** (v2_no_decoy_delta) - 38K steps

**原因**: 这些方法已被证明无效，继续运行浪费 GPU 资源

### 3. 归档已停止的实验日志

```bash
cd /share/liuyutian/tcrppo_v2/logs
mkdir -p archive/failed_experiments

# 归档 trace13 (pure absolute 失败)
mv test51c_*_trace13_*.log archive/failed_experiments/

# 归档 trace16 (calibrated delta 失败)
mv test51c_*_trace16_*.log archive/failed_experiments/

# 归档 trace18 (stepwise 失败)
mv test51c_*_trace18_*.log archive/failed_experiments/

echo "✅ 已归档失败实验日志"
```

---

## 🎯 V2 系列最终结论

### ✅ 唯一有效的方法

**v2_simple_target_gated_decoy (trace29)**
- Max A = 1.30 ⭐⭐⭐⭐⭐
- 但极不稳定 (Mean A = -1.83)
- **必须改进**: 提高 gate、增大 bonus、移除 decoy

### ❌ 完全失败的方法（应该放弃）

1. **所有 Delta 方法**:
   - v2_no_decoy_delta (trace11, 22, 23, 24)
   - v2_no_decoy_delta_calibrated (trace16)
   - v2_no_decoy_sigmoid_delta (trace12)
   - v2_delta_minus_decoy (trace20, 21)
   - **原因**: Delta reward 与 absolute affinity 脱钩

2. **Stepwise 方法**:
   - tfold_stepwise (trace17, 18)
   - tfold_delta_amplified (trace19)
   - **原因**: 与 delta 类似，R 高但 A 差

### ⚠️ 未充分测试

**v2_no_decoy (trace13)**
- 数据不足，可能需要更强激励
- 值得重新测试

---

## 🚀 下一步行动计划

### 立即执行（今天）

1. ✅ **删除短期失败日志** (8 个实验)
2. ❌ **停止失败的长期实验** (7 个实验)
3. 📦 **归档已停止实验** (3 个实验)

### 本周启动新实验

基于 trace29 改进：

#### 实验 1: trace29_improved_pure_target
```yaml
reward_mode: v2_no_decoy  # 或新建 v2_pure_target_amplified
R = 3.0 * A + bonus(A) + 0.05*Nat + 0.02*Div

bonus:
  A > -0.5: +5.0
  A > 0.0:  +10.0
  A > 0.5:  +20.0

# 移除所有 decoy
n_decoys: 0
```

**目标**: 500K steps 内，Mean A > -1.0，A>0 比例 > 5%

#### 实验 2: trace29_improved_high_gate
```yaml
reward_mode: v2_simple_target_gated_decoy
target_decoy_gate_logit: -0.5  # 原 -2.0
target_pass_bonus: 5.0         # 原 1.0
w_affinity: 2.0                # 原 1.0
w_decoy: 0.1                   # 原 0.3
```

**目标**: 500K steps 内，A>0 比例 > 5%

---

## 📊 资源释放估算

停止 7 个失败实验后：
- **释放 GPU**: 7 个
- **节省训练时间**: 避免浪费数百万 steps
- **专注资源**: 集中在有效方法上

---

**报告完成**: 2026-05-25  
**下一步**: 执行清理操作，启动改进实验
