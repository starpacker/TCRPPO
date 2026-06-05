# TCRPPO v2 实验路线图

## 🎯 项目目标
设计高亲和力、高特异性的 TCR CDR3β 序列，突破 AUROC 0.65（v1 baseline: 0.45）

---

## 📍 当前状态（2026-06-05）

### ✅ 已验证成功
- **trace73**: 0.5933 AUROC（首次突破 0.5 baseline）
- **trace94**: FP32 tFold + AE+GMM naturalness（最稳定配置）
- **Key insight**: Peptide-scorer 对齐是关键

### ❌ 已验证失败
- ESM-2 perplexity naturalness
- Step-wise delta reward
- tFold + AMP (数值不稳定)
- Cross-attention policy
- 传统 curriculum learning (L0/L1/L2)
- Imitation learning (SFT → RL)

---

## 🧪 正在探索（未验证）

### trace96: Naturalness Gating + Adaptive Bands
**状态**: 配置已完成  
**配置**: `configs/trace96_nat_gated_dynamic_bands.yaml`

**创新点**:
1. **Naturalness Gating**: 不自然的序列（AE+GMM < 0.5）直接封禁 affinity reward
2. **Adaptive Curriculum Bands**: 根据最近 50 个 episodes 的平均 affinity 动态调整采样难度

**预期效果**:
- 彻底消除 poly-C exploit
- 更平滑的训练曲线
- 更高的最终 naturalness score

---

### trace97: 两阶段训练（Pre-train Naturalness → Fine-tune Specificity）
**状态**: 配置已准备  
**配置**: `configs/trace97_pretrain_then_finetune.yaml`

**方法**:
```
Phase 1 (500k steps):
  w_affinity: 0.0      # 不管 affinity
  w_decoy: 0.0         # 不管 specificity
  w_naturalness: 1.0   # 只学习生成自然的 TCR
  w_diversity: 0.1

Phase 2 (1.5M steps):
  w_affinity: 1.0
  w_decoy: 0.5
  w_naturalness: 0.15  # 保持作为软约束
  naturalness_gate_threshold: 0.6  # 硬约束
  checkpoint_resume: "phase1/final.pt"
```

**假设**:
- Phase 1 确保探索空间在自然 TCRs 范围内
- Phase 2 可以专注优化 specificity，不担心 unnaturalness
- 避免 "affinity 上升但 naturalness 崩溃" 的问题

**风险**:
- Phase 2 可能 catastrophic forgetting（类似 SFT-RL）
- 需要仔细调优 Phase 2 learning rate

---

### trace98+: 生物化学特征增强
**状态**: 代码已实现（`tcrppo_v2/utils/biochem_features.py`）

**方法**:
在 state 中加入氨基酸的电荷和疏水性信息
```python
State = [tcr_emb, pep_emb, hla_emb, 
         tcr_charge, tcr_hydro,      # 新增
         pep_charge, pep_hydro]       # 新增
```

**物理化学信息**:
- **电荷**: Asp/Glu (-1), Lys/Arg (+1), His (+0.5), 其他 (0)
- **疏水性**: Kyte-Doolittle scale (Ile: +4.5, Arg: -4.5, etc.)

**假设**:
- 显式物理化学信息加速收敛（物理先验）
- 帮助模型理解静电相互作用、疏水核心
- 更好的泛化（超越序列相似性）

**对比实验**:
- Baseline: trace94（无 biochem features）
- Enhanced: trace98（加 biochem features）
- 比较收敛速度、最终 AUROC、生成 TCRs 的物化性质分布

---

### trace99+: Peptide Embedding Centering
**状态**: 待实现

**方法**:
对 peptide 的 ESM-2 embeddings 进行 mean-centering
```python
centered_pep_emb = pep_emb - pep_emb.mean(dim=1, keepdim=True)
```

**假设**:
- 消除 peptide-specific bias
- 提升跨 peptides 的可比性
- 减少 peptide-specific overfitting

**对比实验**:
- Baseline: trace94（non-centered）
- Centered: trace99（centered embeddings）
- 比较跨 peptides 的泛化能力

---

## 🗓️ 实验优先级

### P0（立即执行）
1. **trace96**: Naturalness gating + adaptive bands
   - 理由：低风险，高可能收益
   - 预计时间：2M steps ≈ 10 hours
   - 成功标准：AUROC ≥ trace94, naturalness score > 0.6

### P1（高优先级）
2. **trace98**: Biochem features
   - 理由：代码已实现，容易测试
   - 预计时间：2M steps ≈ 10 hours
   - 成功标准：收敛速度 > baseline 或 AUROC > baseline

3. **trace97**: 两阶段训练
   - 理由：新的训练范式，可能突破性进展
   - 预计时间：Phase 1 (500k) + Phase 2 (1.5M) ≈ 12 hours
   - 成功标准：Phase 1 高 naturalness + Phase 2 高 AUROC

### P2（探索性）
4. **trace99**: Peptide centering
   - 理由：实现简单，但收益不确定
   - 预计时间：2M steps ≈ 10 hours
   - 成功标准：跨 peptides 泛化 > baseline

---

## 📊 成功指标

### 必须满足（所有实验）
- ✅ AUROC ≥ 0.55（超越 trace73）
- ✅ Mean naturalness score ≥ 0.5（AE+GMM）
- ✅ Diversity: 唯一序列数 ≥ 80%
- ✅ 无 poly-C exploit（C 出现频率 < 15%）

### 期望达到（突破性进展）
- 🎯 AUROC ≥ 0.65（项目最终目标）
- 🎯 Mean affinity score ≥ -0.5（强结合）
- 🎯 Gate pass rate ≥ 60%
- 🎯 泛化到未见过的 peptides（transfer learning 测试）

---

## 🔄 实验流程 SOP

### 1. 准备阶段
- [ ] 复制最新的 baseline config（通常是 trace94）
- [ ] 修改创新点相关的参数
- [ ] 创建 launch script（`scripts/launch_traceXX.sh`）
- [ ] 创建实验设计文档（`docs/experiments/traceXX_design.md`）
- [ ] Git commit 配置和设计文档

### 2. 启动阶段
- [ ] 启动独立的 tFold server（独立 socket 和 cache）
- [ ] 验证 GPU 可用性（`nvidia-smi`）
- [ ] 启动训练（记录 PID）
- [ ] 创建监控脚本（`scripts/monitor_traceXX.sh`）

### 3. 监控阶段（每 50k steps 检查）
- [ ] Mean affinity 趋势（应持续上升）
- [ ] Gate pass rate（应从 10% 增长）
- [ ] Naturalness score（应保持 > 0.5）
- [ ] Diversity（避免崩溃）
- [ ] 检查 poly-C exploit（C 频率）

### 4. 评估阶段（训练完成）
- [ ] 生成 50 TCRs per target
- [ ] 运行 decoy evaluation（K=50 decoys）
- [ ] 计算 per-target AUROC
- [ ] 计算 mean AUROC
- [ ] 分析失败案例（AUROC < 0.5 的 targets）
- [ ] 对比 baseline（trace94）

### 5. 总结阶段
- [ ] 更新实验文档（结果、分析、教训）
- [ ] 更新 `EXPERIMENT_VALIDATION_TRACKER.md`
- [ ] 更新 `LESSONS_LEARNED.md`（如果有新发现）
- [ ] Git commit 结果和分析
- [ ] 决定下一步（继续 / 放弃 / 改进）

---

## 💡 实验设计原则

### DO ✅
- 一次只改变 1-2 个变量（便于归因）
- 使用相同的 random seed 进行对比（42）
- 记录所有超参数和配置
- 每个实验都有明确的假设
- 失败的实验也要记录（避免重复）

### DON'T ❌
- 不要在未验证的 peptides 上训练（检查 `PEPTIDE_SCORER_MAPPING.md`）
- 不要使用 AMP（已验证会导致数值不稳定）
- 不要尝试 step-wise reward（已多次验证失败）
- 不要同时改变多个组件（难以归因）
- 不要忽略 early warning signs（100k steps 后 affinity 不上升）

---

## 📚 相关文档

- **完整经验总结**: `LESSONS_LEARNED.md`
- **Peptide 可靠性**: `PEPTIDE_SCORER_MAPPING.md`
- **trace94 配置**: `configs/trace94_strict_nat_always_on.yaml`
- **trace96 设计**: `TRACE96_IMPLEMENTATION.md`
- **trace73 成功分析**: `TRACE73_SUCCESS_ANALYSIS.md`

---

**最后更新**: 2026-06-05  
**当前最佳**: trace94 (FP32 tFold + AE+GMM)  
**历史最佳**: trace73 (0.5933 AUROC)  
**下一个实验**: trace96 (Naturalness gating + adaptive bands)
