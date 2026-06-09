# Trace73 启动计划

## 📊 背景分析

### Trace72 现状
- **最佳性能**: Step 899,840
  - Reward: -0.386
  - Advantage: -0.714
  - InitA: -2.384
  - DecViol: 1.887 ⚠️
  
- **问题诊断**:
  1. **DecViol 持续增加** (1.42 → 1.89): 模型在学习结合 decoy
  2. **InitA 偏低** (-2.38): 初始化质量不够
  3. **Online pool 无筛选**: max_decoy_violation=999 接收所有 TCR
  4. **Curriculum 是 100% L2**: 随机 TCRdb，质量低

### 根本原因
```
低质量初始化 (L2) 
  → 模型学会提升 affinity
  → 但同时也提升 decoy affinity (因为 w_decoy 太低)
  → Online pool 接收这些非特异性 TCR
  → 下一轮继续从非特异性 TCR 开始
  → 恶性循环
```

## 🎯 Trace73 改进策略

### 核心改进
1. **增强 Decoy 惩罚**: w_decoy 0.3 → 0.6 (2倍)
2. **严格筛选 Online Pool**:
   - max_decoy_violation: 999 → 0.5
   - min_affinity: -10.0 → -1.0
3. **高质量初始化**: 100% L2 → 60% L0 + 40% L1
4. **优先使用好 TCR**: online_pool_max_ratio 0.5 → 0.7

### 预期效果
| 指标 | Trace72 | Trace73 目标 | 改进 |
|------|---------|--------------|------|
| InitA | -2.38 | **> -1.5** | +0.88 |
| DecViol | 1.89 | **< 1.0** | -0.89 |
| Advantage | -0.71 | **> 0.0** | +0.71 |
| Reward | -0.39 | **> 0.0** | +0.39 |

## 🚀 启动步骤

### 1. 检查环境
```bash
# 检查 GPU 可用性
nvidia-smi

# 检查 conda 环境
conda activate tcrppo_v2
python -c "import torch; print(torch.cuda.is_available())"

# 检查 checkpoint 存在
ls -lh tcrppo_v2/output/trace72_delta_from_trace70/checkpoints/latest.pt
```

### 2. 启动训练
```bash
cd /share/liuyutian/tcrppo_v2
bash scripts/launch_trace73_improved_specificity.sh
```

### 3. 监控训练
```bash
# 实时查看日志
tail -f logs/trace73_improved_specificity_train.log

# 查看关键指标
grep "Step.*R:" logs/trace73_improved_specificity_train.log | tail -20

# 检查进程
ps aux | grep ppo_trainer
```

### 4. 关键监控指标

**前 50K steps (914K → 964K)**:
- InitA 应该从 -2.38 提升到 -2.0 左右
- DecViol 应该开始下降
- Online pool 应该开始积累高质量 TCR

**中期 (964K → 1.2M)**:
- InitA 应该达到 -1.5 以上
- DecViol 应该降到 1.5 以下
- Advantage 应该接近 -0.5

**后期 (1.2M → 1.5M)**:
- InitA 稳定在 -1.0 到 -1.5
- DecViol 稳定在 1.0 以下
- Advantage 达到 0.0 以上

## ⚠️ 风险和应对

### 风险1: Online pool 一直为空
**症状**: 日志显示 "OnlinePool: 0/20t"
**原因**: 筛选条件太严格，没有 TCR 满足
**应对**: 
```bash
# 放宽筛选条件
--online_tcr_pool_max_decoy_violation 1.0  # 0.5 → 1.0
--online_tcr_pool_min_affinity -2.0        # -1.0 → -2.0
```

### 风险2: Reward 大幅下降
**症状**: Reward 从 -0.4 降到 -1.0
**原因**: w_decoy=0.6 太高，模型无法平衡
**应对**:
```bash
# 降低 decoy 权重
--w_decoy 0.45  # 0.6 → 0.45
```

### 风险3: DecViol 仍然增加
**症状**: DecViol 继续从 1.89 增加到 2.0+
**原因**: w_decoy=0.6 仍然不够
**应对**:
```bash
# 进一步增加 decoy 权重
--w_decoy 0.8  # 0.6 → 0.8
```

### 风险4: L0/L1 seeds 不够
**症状**: 日志显示大量 "L2" 采样
**原因**: L0/L1 seeds 覆盖的 target 不够
**应对**: 检查 L0/L1 seeds 是否加载成功
```bash
# 检查 L0/L1 seeds
python -c "
from tcrppo_v2.data.tcr_pool import TCRPool
pool = TCRPool()
print(f'L0 targets: {len(pool.l0_seeds)}')
print(f'L1 targets: {len(pool.l1_seeds)}')
"
```

## 📈 成功标准

### 必须达到 (P0)
- [ ] InitA > -1.5
- [ ] DecViol < 1.5
- [ ] Advantage > -0.3

### 期望达到 (P1)
- [ ] InitA > -1.0
- [ ] DecViol < 1.0
- [ ] Advantage > 0.0

### 理想达到 (P2)
- [ ] InitA > -0.5
- [ ] DecViol < 0.8
- [ ] Advantage > 0.5

## 📝 实验记录

### 启动时间
- 计划启动: 2026-06-01
- 实际启动: _____
- 预计完成: _____ (约 12-18 小时)

### 中期检查点 (每 100K steps)
| Step | InitA | DecViol | Advantage | Reward | 备注 |
|------|-------|---------|-----------|--------|------|
| 914K | -2.38 | 1.89 | -0.71 | -0.39 | 起点 (trace72) |
| 1.0M | | | | | |
| 1.1M | | | | | |
| 1.2M | | | | | |
| 1.3M | | | | | |
| 1.4M | | | | | |
| 1.5M | | | | | 终点 |

### 最终结果
- 最佳 checkpoint: Step _____
- 最佳 Reward: _____
- 最佳 Advantage: _____
- InitA: _____
- DecViol: _____

### 结论
- [ ] 成功: 达到所有 P0 标准
- [ ] 部分成功: 达到部分标准
- [ ] 失败: 未达到 P0 标准

### 下一步
- 如果成功: _____
- 如果失败: _____
