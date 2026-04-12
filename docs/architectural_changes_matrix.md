# TCRPPO v2 架构改动对照表

**Last Updated:** 2026-04-12  
**Purpose:** 对比原始 TCRPPO v1 与 v2 各实验的架构改动及效果

---

## 架构改动维度说明

相较于原始 TCRPPO v1，v2 引入了以下关键改动：

| 改动ID | 改动名称 | 说明 | v1 原始方案 |
|--------|---------|------|------------|
| **A1** | Indel 动作空间 | 支持 SUB/INS/DEL/STOP 四种操作，序列长度可变 (8-27) | 仅支持定点替换 (Substitution)，长度固定 |
| **A2** | ESM-2 状态编码 | 使用 ESM-2 650M 冻结模型编码 TCR+pMHC 状态 | 使用 Autoencoder 潜在向量 (64-dim) |
| **A3** | Decoy 对比惩罚 | LogSumExp 对比惩罚，多层级 decoy 库 (A/B/C/D) | 无 decoy 惩罚，单目标优化 |
| **A4** | ESM-2 Naturalness | 使用 ESM-2 perplexity + CDR3 z-score 评估天然性 | 使用 Autoencoder 重构误差 + GMM 似然度 |
| **A4v1** | AE+GMM Naturalness Threshold | v1 使用 AE 重构 + GMM 似然度，criteria=0.5 作为硬阈值过滤 | v1 原有机制 |
| **A5** | Diversity 惩罚 | 基于 recent buffer 的序列相似度惩罚 | 无 diversity 惩罚 |
| **A6** | Per-step Delta Reward | 每步返回 delta reward (当前分数 - 初始分数) | 仅 terminal reward (episode 结束时) |
| **A7** | Z-score 归一化 | RunningNormalizer (window=10K, warmup=1K) 归一化所有 reward 分量 | 无归一化，直接使用 ERGO 原始分数 |
| **A8** | Min-steps 约束 | 强制至少 N 步才能 STOP，否则惩罚 | 无约束，可随时 STOP |
| **A9** | L0 Curriculum | 从 VDJdb 已知 binder + 3-5 随机突变开始 | 从 TCRdb 随机序列开始 |
| **A10** | Raw Reward (无归一化) | 使用原始 ERGO 分数 (0-1)，不做 z-score 归一化 | v1 本身就是 raw，v2 部分实验引入了归一化 |
| **A11** | Two-phase Training | 先训练纯 ERGO，再 fine-tune 加 decoy 惩罚 | 单阶段训练 |
| **A12** | Step-wise Absolute Reward | 每步返回绝对 ERGO 分数 (非 delta，非 terminal-only) | Terminal-only |
| **A13** | Threshold-based Penalties | 根据 affinity 阈值条件性施加惩罚 | 无条件惩罚 |

**注：** v1 使用 `reward2()` 方法，有 naturalness threshold (criteria=0.5) 硬过滤 — 只有 naturalness >= 0.5 的 TCR 才会被 ERGO 评分。这与 v2 的 A13 (条件性多目标惩罚) 不同。

---

## 实验对照矩阵

| # | 实验名称 | A1<br>Indel | A2<br>ESM-2<br>State | A3<br>Decoy<br>Penalty | A4<br>ESM-2<br>Natural | A4v1<br>AE+GMM<br>Thresh | A5<br>Diversity | A6<br>Delta<br>Reward | A7<br>Z-score<br>Norm | A8<br>Min-steps | A9<br>L0<br>Curric | A10<br>Raw<br>Reward | A11<br>Two-phase | A12<br>Stepwise<br>Abs | A13<br>Threshold<br>Penalty | Mean<br>AUROC | Avg<br>Steps | Status |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v1** | **原始 TCRPPO** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | **0.4538** | ~10 | Baseline |
| 1 | v1_ergo_only | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | **0.8075** | 8.8 | ✅ BEST |
| 2 | v2_full_run1 | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | 0.5840 | 2.0 | ✅ FAIL |
| 3 | exp1_decoy_only | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | 0.4898 | 6.6-7.8 | ✅ FAIL |
| 4 | exp2_light | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | 0.4660 | 5.4 | ✅ FAIL |
| 5 | exp3_ergo_delta | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | 0.5004 | 7.9 | ✅ FAIL |
| 6 | exp4_min_steps | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅(3) | ✅ | ❌ | ❌ | ❌ | ❌ | 0.4768 | 5.0 | ✅ FAIL |
| 7 | v2_no_decoy | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | TBD | 3.9-4.8 | 🔄 67% |
| 8 | test1_two_phase_p1 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅(P1) | ❌ | ❌ | TBD | 8.9 | 🔄 10% |
| 9 | test2_min6_raw | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅(6) | ✅ | ✅ | ❌ | ❌ | ❌ | TBD | 8.9 | 🔄 6% |
| 10 | test3_stepwise | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | TBD | 8.9 | 🔄 6% |
| 11 | test4_raw_multi | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | TBD | 8.9 | 🔄 6% |
| 12 | test5_threshold | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | TBD | 8.9 | 🔄 6% |
| **13** | **test6_pure_v2** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | TBD | TBD | 🔄 NEW |

---

## 关键发现总结

### ✅ 有效的改动组合

1. **A1 (Indel) + A2 (ESM-2 State) + A9 (L0 Curriculum) + A10 (Raw Reward)**
   - 实验 1 (v1_ergo_only): **0.8075 AUROC** — 最佳结果
   - 核心：v2 架构改进 + 保持 v1 的简单 raw reward 策略

2. **A1 (Indel) + A2 (ESM-2 State) + A10 (Raw Reward) [无 Curriculum]** ← 新实验
   - 实验 13 (test6_pure_v2): 待验证
   - 假设：测试纯架构改进的贡献，排除 L0 curriculum 的影响

### ❌ 失败的改动组合

1. **A7 (Z-score Normalization) + A3/A4/A5 (任何惩罚)**
   - 实验 2, 3, 4, 6, 7: 全部失败 (AUROC < 0.60)
   - 根本原因：RunningNormalizer 压缩了 affinity 信号
   - 即使惩罚权重极低 (d=0.2, n=0.1, v=0.05) 也无法避免

2. **A6 (Delta Reward) + A10 (Raw, 无归一化)**
   - 实验 5 (exp3_ergo_delta): 0.5004 AUROC
   - 问题：每步的 delta 信号太弱，不如 terminal 的绝对分数

3. **A8 (Min-steps) 单独使用**
   - 实验 6 (exp4_min_steps): 0.4768 AUROC
   - 问题：约束了行为但没解决底层的信号压缩问题

### 🔬 待验证的改动组合 (正在运行)

1. **A10 (Raw) + A3 (Decoy) + A8 (Min-steps=6)**
   - 实验 9 (test2_min6_raw): 当前 R=1.15 (最高)
   - 假设：Raw reward 保留强信号 + min-steps 强制探索

2. **A10 (Raw) + A3/A4/A5 (多惩罚，小权重)**
   - 实验 11 (test4_raw_multi): 当前 R=1.05
   - 假设：小的绝对惩罚不会压缩信号

3. **A11 (Two-phase Training)**
   - 实验 8 (test1_two_phase): Phase 1 进行中
   - 假设：先建立结合力，再微调特异性

4. **A12 (Stepwise Absolute)**
   - 实验 10 (test3_stepwise): 当前 R=0.86
   - 假设：每步绝对分数比 delta 更清晰

5. **A13 (Threshold-based)**
   - 实验 12 (test5_threshold): 当前 R=0.80
   - 假设：分阶段学习（先结合，后特异性）

6. **A1 + A2 + A10 (无 L0 Curriculum)** ← 新实验
   - 实验 13 (test6_pure_v2): GPU 5, PID 2809807
   - 假设：测试纯架构改进 (indel + ESM-2) 的贡献，不使用 curriculum
   - 对比 v1_ergo_only 可以隔离 L0 curriculum 的贡献

---

## 架构改动效果分析

### 单独改动效果

| 改动 | 单独效果 | 证据 |
|------|---------|------|
| A1 (Indel) | ✅ 正面 | v1_ergo_only (有 indel) 比 v1 baseline (无 indel) 提升 +0.35 AUROC |
| A2 (ESM-2 State) | ✅ 正面 | 同上，ESM-2 提供更强的生化语义理解 |
| A3 (Decoy) | ⚠️ 取决于实现 | 与 z-norm 结合失败，与 raw reward 结合待验证 |
| A4 (ESM-2 Natural) | ⚠️ 取决于实现 | 与 z-norm 结合失败，与 raw reward 结合待验证 |
| A4v1 (AE+GMM Thresh) | ⚠️ v1特有 | v1 使用 criteria=0.5 硬阈值过滤天然性差的 TCR |
| A5 (Diversity) | ⚠️ 取决于实现 | 与 z-norm 结合失败，与 raw reward 结合待验证 |
| A6 (Delta Reward) | ❌ 负面 | exp3 证明 raw delta 不如 terminal absolute |
| A7 (Z-score Norm) | ❌ 严重负面 | 所有使用 z-norm 的实验全部失败 |
| A8 (Min-steps) | ⚠️ 辅助性 | 单独无效，需配合 raw reward 使用 |
| A9 (L0 Curriculum) | 🔬 待验证 | test6 将隔离测试其贡献 (与 v1_ergo_only 对比) |
| A10 (Raw Reward) | ✅ 关键 | v1 和 v1_ergo_only 的成功核心 |

### 交互效应

| 组合 | 效果 | 说明 |
|------|------|------|
| A7 + (A3/A4/A5) | ❌ 灾难性 | Z-norm 压缩 affinity 信号，任何惩罚都会导致早停 |
| A10 + (A3/A4/A5) | 🔬 待验证 | 新实验 (test2/4/5) 正在测试这个组合 |
| A6 + A10 | ❌ 负面 | exp3 证明 delta 不如 terminal |
| A11 + A10 | 🔬 待验证 | test1 正在测试两阶段训练 |
| A12 + A10 | 🔬 待验证 | test3 正在测试 stepwise absolute |

---

## 对比 v1 的核心改进

### v1 → v1_ergo_only (实验1)

**改动:** A1 + A2 + A9 + A10  
**效果:** +0.3537 AUROC (0.4538 → 0.8075)  
**结论:** v2 架构改进 (indel + ESM-2 + curriculum) 在保持简单 reward 的前提下显著提升性能

### v1 → v2_full (实验2)

**改动:** A1 + A2 + A3 + A4 + A5 + A6 + A7 + A9  
**效果:** +0.1302 AUROC (0.4538 → 0.5840)，但远低于 v1_ergo_only  
**结论:** 过度复杂的 reward 设计 (z-norm + 多惩罚) 反而损害性能

### 教训

1. **架构改进 (A1/A2/A9) 是有效的**，但必须配合正确的 reward 设计
2. **Reward 设计的简单性至关重要** — raw terminal reward 优于复杂的 normalized multi-objective
3. **Z-score 归一化是毒药** — 在 RL 中压缩了关键信号
4. **新的惩罚机制 (A3/A4/A5) 需要与 raw reward 结合** — 这是当前 5 个新实验的核心假设

---

## 下一步实验建议

基于当前矩阵，以下组合值得测试：

1. **A10 + A3 (轻微 decoy) + A4 (轻微 natural)** — 类似 test4，但调整权重
2. **A10 + A3 + A8 (更高 min-steps=8-10)** — 如果 test2 成功，可以尝试更激进的约束
3. **A11 (Two-phase) + A3/A4/A5 (Phase 2 加多惩罚)** — 如果 test1 Phase 1 成功
4. **A1 + A2 + A9 + A10 + Ensemble Scoring** — 使用多个 affinity 模型集成，避免 ERGO overfitting

---

## 参考文档

- 完整实验配置和结果: `docs/all_experiments_tracker.md`
- v1 baseline 结果: `/share/liuyutian/TCRPPO/progress_3.md`
- v1 架构审计: `/share/liuyutian/TCRPPO/query.md`
- v2 设计规范: `docs/2026-04-09-tcrppo-v2-design.md`
