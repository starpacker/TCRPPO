# TCRPPO v2 - Lessons Learned

这份文档总结了我们在 TCRPPO v2 开发过程中得到的所有重要经验教训和实验结论。

## 📋 目录

- [核心发现](#核心发现)
- [失败的方法](#失败的方法)
- [成功的方法](#成功的方法)
- [关键配置](#关键配置)
- [最佳实践](#最佳实践)

---

## 🎯 核心发现

### 1. **Peptide-Scorer 对齐问题是关键**
- **发现**: 并非所有 peptide 都适合用 ERGO/NetTCR/tFold 评分
- **证据**: SAC test3 在 7 个 ERGO peptides 上只有 0.4127 AUROC，因为其中 3 个 peptides 有**反向奖励信号**（ERGO 给 decoys 的分数比 targets 高）
- **解决方案**: 只在 scorer AUC ≥ 0.7 的 peptides 上训练
- **参考**: `PEPTIDE_SCORER_MAPPING.md`, `docs/CRITICAL_LESSON_PEPTIDE_SCORER_ALIGNMENT.md`

### 2. **tFold 是最可靠的 Affinity Scorer**
- **性能**: tFold 在 77 个 peptides 上 AUC ≥ 0.7
- **速度**: ~1s/sample on cache miss, <1ms on cache hit
- **策略**: 预热 cache + FP32 评分（避免 AMP 数值不稳定）
- **成功案例**: trace73, trace94, trace95

### 3. **Online TCR Pool + Dynamic Band Selection 是突破关键**
- **trace61** 引入动态 pool：从训练中的优质 TCRs 采样
- **trace70-72** 引入 gate mechanism：只接受 target > gate 的 TCRs
- **trace73** 成功组合：online pool + gate=-0.5 + curated targets
- **结果**: trace73 达到 **0.5933 AUROC**（首次超过 0.5 baseline）

---

## ❌ 失败的方法

### 1. **ESM-2 Perplexity 作为 Naturalness Scorer**
- **问题**: ESM-2 perplexity 无法有效区分 natural vs unnatural TCRs
- **现象**: 模型学会生成低 perplexity 但不 natural 的序列（如 poly-C）
- **原因**: ESM-2 是通用蛋白质模型，不专注于 TCR 结构
- **替代方案**: AE+GMM scorer（来自 v1 代码）更有效

### 2. **Step-wise Delta Reward**
- **假设**: 每步给 delta reward 可以提供更密集的训练信号
- **实验**: test53 (one-step delta), trace52/80/82/86 (per-step reward)
- **结果**: **完全失败** — 模型学不到有效策略
- **原因**: 
  - 信用分配问题：很难判断哪一步的编辑真正有用
  - 中间状态的 affinity 变化噪声大
  - 破坏了 terminal reward 的清晰信号
- **结论**: **Terminal reward only** 是正确方案

### 3. **tFold + AMP (Automatic Mixed Precision)**
- **假设**: AMP 可以加速 tFold 推理
- **实验**: test51c, test53 
- **问题**: 
  - 数值不稳定：FP16 导致 affinity 分数漂移
  - 不可复现：同一 TCR 在不同 batch 得到不同分数
  - 训练不稳定：reward 信号有噪声
- **解决方案**: **FP32 only**（trace94/95）

### 4. **Cross-Attention Policy Architecture**
- **假设**: cross-attention 可以更好地建模 TCR-pMHC 交互
- **实验**: trace48
- **结果**: 没有明显优势，训练更慢，更不稳定
- **结论**: **简单的 MLP policy 已经足够**

### 5. **大量 Decoy Sampling (K=32)**
- **假设**: 更多 decoys 提供更强的 contrastive 信号
- **实验**: test43c, test47
- **问题**: 
  - 训练极慢（每步需要评分 32 个 decoys）
  - 信号过强导致过拟合
  - 没有明显的 AUROC 提升
- **最佳配置**: **K=2** decoys（trace61/94/95）

### 6. **Curriculum Learning (L0/L1/L2)**
- **假设**: 从已知 binders 开始训练更容易
- **实验**: test26, test43a/b, trace73-83
- **结果**: **大部分失败**
- **问题**: 
  - L0 (mutated known binders) 限制探索空间
  - L1 (ERGO top-500) 引入 scorer bias
  - 难以找到合适的 curriculum schedule
- **成功案例**: trace73 的 curated targets（但不是传统 curriculum）

### 7. **Imitation Learning (SFT)**
- **假设**: 先用 SFT 学习 known binders，再用 RL fine-tune
- **实验**: IL experiments (trace91-92), SFT-RL pipeline
- **结果**: **失败** — SFT 模型无法迁移到 RL
- **问题**:
  - 数据分布不匹配：SFT 数据 vs RL exploration
  - Catastrophic forgetting：RL fine-tune 后丢失 SFT 知识
  - 性能不如从头训练 RL
- **参考**: `IL_FINAL_REPORT.md`

---

## ✅ 成功的方法

### 1. **FP32 tFold + AE+GMM Naturalness**
- **配置**: trace94, trace95
- **组件**:
  - Affinity: tFold (FP32, no AMP)
  - Naturalness: AE+GMM scorer (from v1)
  - Decoy: K=2, tiers A+B
  - Diversity: buffer size 512
- **优势**:
  - 稳定的数值精度
  - 有效防止 poly-C exploit
  - 可复现的结果

### 2. **Online TCR Pool + Dynamic Band Selection**
- **引入**: trace61
- **机制**:
  - 从训练过程中收集高质量 TCRs
  - 按 affinity band 分层（-10~-4, -4~-2, -2~-1, -1~0）
  - 动态选择采样 band（基于当前性能）
- **效果**: 提供持续的训练挑战，避免过拟合

### 3. **Gate Mechanism (Simple Target Gated Decoy)**
- **引入**: trace62-63, trace70-72
- **原理**:
  ```
  if target_affinity > gate_logit:
      reward = target_bonus + affinity - decoy_penalty
  else:
      reward = affinity - decoy_penalty  # no bonus
  ```
- **最佳参数**: gate=-0.8 到 0.0, bonus=2.0-3.0
- **效果**: 强制模型学习同时满足 affinity 和 specificity

### 4. **Curated Target Selection**
- **引入**: trace73, trace79, trace83
- **策略**: 只选择 tFold AUC ≥ 0.7 的 peptides
- **结果**: trace73 在 4 个 curated targets 上达到 **0.5933 AUROC**
- **关键**: 避免在 unreliable peptides 上浪费训练时间

### 5. **Terminal Reward Only + ban_stop=True**
- **配置**: 所有成功的 traces
- **原理**:
  - 只在 episode 结束时给 reward
  - 禁止提前 STOP（强制固定长度 episodes）
- **优势**:
  - 清晰的信用分配
  - 稳定的训练信号
  - 避免 step-wise reward 的噪声

---

## ⚙️ 关键配置

### trace94: 严格 Naturalness + FP32 tFold
```yaml
# 最可靠的基线配置
affinity_model: "tfold"
naturalness_scorer_type: "ae_gmm"
naturalness_ae_threshold: 0.8
reward_mode: "v2_simple_target_gated_decoy"
w_affinity: 1.0
w_decoy: 0.3
w_naturalness: 0.15
w_diversity: 0.02
target_decoy_gate_logit: 0.0
target_pass_bonus: 2.0
ban_stop: true
max_steps: 8
terminal_reward_only: true
online_tcr_pool_enabled: true
online_tcr_pool_use_dynamic_bands: true
```

### trace95: Variable Length Episodes
```yaml
# trace94 + 允许提前 STOP
ban_stop: false
min_steps: 2
max_steps: 10
# 其他配置同 trace94
```

### trace96 (计划中): Naturalness Gating + Adaptive Curriculum
```yaml
# trace94 + 两个创新
naturalness_gate_affinity: true  # 不 natural 的序列不给 affinity reward
naturalness_gate_threshold: 0.5
online_tcr_pool_adaptive_bands: true  # 自动难度调整
w_decoy: 1.0  # 增加 decoy 权重
target_pass_bonus: 3.0  # 增加 pass bonus
```

---

## 🎓 最佳实践

### 1. **实验设计**
- ✅ **先验证 peptide-scorer 对齐** — 检查 `PEPTIDE_SCORER_MAPPING.md`
- ✅ **使用 FP32 评分** — 避免数值不稳定
- ✅ **预热 tFold cache** — 减少训练时间
- ✅ **记录完整配置** — 每个实验都要有 YAML + launch script
- ❌ 不要在未验证的 peptides 上训练
- ❌ 不要使用 AMP 加速 affinity scoring
- ❌ 不要尝试 step-wise reward（已验证无效）

### 2. **超参数选择**
- **Learning rate**: 1.5e-4（稳定）
- **Batch size**: 256（8 envs × 32 steps）
- **Entropy coef**: 0.012（鼓励探索但不过度）
- **Max steps**: 8（固定长度 episodes）
- **Decoy K**: 2（速度与效果的平衡）
- **Gate logit**: -0.8 到 0.0（根据 peptide 难度调整）
- **Pass bonus**: 2.0-3.0

### 3. **训练监控**
- 监控 **mean affinity** — 应该持续上升
- 监控 **gate pass rate** — 应该从 10% 增长到 50%+
- 监控 **naturalness score** — 应该保持在 0.5+（AE+GMM）
- 监控 **diversity** — 避免模式崩溃
- **Early stopping**: 如果 100k steps 后 affinity 没有上升，检查配置

### 4. **Debugging 技巧**
- 如果 **affinity 不上升**: 检查 peptide-scorer 对齐
- 如果 **poly-C exploit**: 增加 naturalness weight 或使用 AE+GMM
- 如果 **模式崩溃**: 增加 entropy coef 或 diversity weight
- 如果 **训练不稳定**: 检查是否使用了 AMP，切换到 FP32
- 如果 **AUROC 低**: 检查 gate 设置，可能需要更严格的 gate

### 5. **资源管理**
- **GPU 分配**: 每个 trace 需要 1 个 GPU（tFold server + training）
- **tFold cache**: 预计 4-8 GB per experiment
- **Checkpoint**: 每 20k steps 保存，最终模型 ~500 MB
- **Training time**: 2M steps ≈ 8-12 hours on A800

---

## 📊 实验历史总结

| Trace | Key Innovation | Result | Lesson |
|-------|---------------|--------|--------|
| test1-50 | Various attempts | Failed | 需要 peptide-scorer 对齐 |
| trace61 | Online pool + dynamic bands | Moderate | 引入动态 pool 机制 |
| trace70-72 | Gate mechanism | Better | Gate 有效但需要调优 |
| trace73 | Curated targets + gate=-0.5 | **0.5933 AUROC** | 首次突破 0.5！ |
| trace74-83 | Various curriculum attempts | Mixed | Curriculum 不是银弹 |
| trace94 | FP32 tFold + AE+GMM nat | **Stable baseline** | 当前最佳配置 |
| trace95 | Variable length episodes | Testing | 允许提前 STOP |
| trace96 | Nat gating + adaptive bands | Planned | 下一步创新 |

---

## 🔮 未来方向

### 短期（trace96-100）
1. **Naturalness Gating**: 不 natural 的序列直接不给 affinity reward
2. **Adaptive Curriculum Bands**: 根据性能自动调整采样难度
3. **Multi-target Training**: 同时在多个 peptides 上训练，提升泛化

### 中期
1. **Ensemble Scoring**: 结合 ERGO + NetTCR + tFold
2. **Structure-aware Reward**: 整合 AlphaFold 结构信息
3. **Active Learning**: 在不确定的区域优先探索

### 长期
1. **Foundation Model**: 预训练 TCR encoder
2. **Multi-objective RL**: 同时优化 affinity, specificity, stability, immunogenicity
3. **Wet-lab Validation**: 合成并测试 RL 生成的 TCRs

---

## 📚 重要文档索引

- **Peptide-Scorer 对齐**: `PEPTIDE_SCORER_MAPPING.md`
- **关键教训**: `docs/CRITICAL_LESSON_PEPTIDE_SCORER_ALIGNMENT.md`
- **tFold 混合训练**: `docs/TFOLD_HYBRID_TRAINING.md`
- **trace73 成功分析**: `TRACE73_SUCCESS_ANALYSIS.md`
- **trace94 配置**: `configs/trace94_strict_nat_always_on.yaml`
- **trace95 配置**: `configs/trace95_variable_length.yaml`
- **trace96 设计**: `TRACE96_IMPLEMENTATION.md`
- **IL 失败报告**: `IL_FINAL_REPORT.md`
- **实验验证追踪**: `EXPERIMENT_VALIDATION_TRACKER.md`

---

**最后更新**: 2026-06-05  
**当前最佳配置**: trace94 (FP32 tFold + AE+GMM naturalness)  
**当前最佳结果**: trace73 (0.5933 AUROC on 4 curated targets)  
**下一步计划**: trace96 (naturalness gating + adaptive bands)
