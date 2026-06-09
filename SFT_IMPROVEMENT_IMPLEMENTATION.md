# SFT 改进实施总结

**日期**: 2026-06-01
**状态**: 训练中 (Epoch 3/50)

---

## 已完成的改进

### 1. ✅ 数据清洗
- **过滤CCC模式**: 移除所有包含3连续重复氨基酸的TCR
- **原始数据**: 268,678 TCR (96.4% 包含重复)
- **过滤后**: 9,784 TCR (0% 包含重复)
- **训练集**: 5,264 轨迹 (stratified sampling)

### 2. ✅ 轨迹重建改进
- **使用SequenceMatcher**: 基于编辑距离的智能对齐
- **防止重复插入**: 禁止连续插入相同氨基酸
- **Action分布改善**:
  - INS: 40.6% (原 77.2%)
  - SUB: 33.4% (原 22.7%)
  - DEL: 14.3% (原 0%)
  - Token分布更均匀 (C从70.8%降至6.6%)

### 3. ✅ 训练正则化
- **重复惩罚**: 检测并惩罚连续相同token的插入
- **权重**: 0.1 × repetition_penalty
- **效果**: rep loss从0.0增长到0.11 (正在学习避免重复)

### 4. ✅ 训练配置
- **Epochs**: 50 (完整训练)
- **Batch size**: 32
- **Learning rate**: 1e-4
- **Hidden dim**: 512
- **数据质量**: mean affinity -1.10 (vs 原始 -0.22)

---

## 训练进度

**当前状态** (2026-06-01 实时):
- Epoch: 3/50
- Loss: 6.20 (从7.81下降)
- GPU: 100% 使用率
- 预计完成: ~17小时

**Loss趋势**:
```
Epoch 1: 6.89
Epoch 2: 6.43
Epoch 3: 6.20 (进行中)
```

---

## 预期结果

### 最低目标 (可接受)
- ✓ 平均亲和力: > -2.0
- ✓ 无重复模式: > 80%
- ✓ 成功率 (>0.0): > 5%

### 目标 (良好)
- 平均亲和力: > -1.0
- 成功率 (>0.0): > 15%
- 成功率 (>0.6): > 3%

### 对比基准
- **原始SFT (CCC污染)**: -5.49
- **trace73 RL**: -1.172
- **训练数据质量**: -1.10
- **目标**: > -2.0

---

## 下一步

训练完成后 (预计17小时):

1. **评估模型**
   ```bash
   python scripts/eval_sft_esm.py \
       --checkpoint output/sft_filtered_training/checkpoint_best.pt \
       --n_tcrs 50
   ```

2. **对比分析**
   - vs 原始SFT (-5.49)
   - vs trace73 RL (-1.172)
   - vs 训练数据 (-1.10)

3. **如果成功 (> -2.0)**
   - 继续RL微调 (PPO)
   - 目标: > 0.0

4. **如果失败 (< -2.0)**
   - 考虑使用真实TCR数据库 (VDJdb, IEDB)
   - 或改用直接生成方法 (非轨迹重建)

---

## 监控命令

```bash
# 实时监控
bash scripts/monitor_training.sh

# 查看日志
tail -f logs/sft_filtered_training.log

# 检查checkpoints
ls -lh output/sft_filtered_training/checkpoint_*.pt
```

---

**创建时间**: 2026-06-01
**预计完成**: 2026-06-02 (17小时后)
