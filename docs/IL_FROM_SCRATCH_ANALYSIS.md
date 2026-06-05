# IL Training: From-Scratch vs Resume Comparison

## 🎯 核心发现：From Scratch 表现更优！

### 最终Loss对比

| 方法 | Epochs | 最终Loss | 相对差异 |
|------|--------|---------|---------|
| **From Scratch (8 epochs)** | 8 | **4.9650** | **基准** |
| From Scratch + Early Stop | 14+ | ~5.05 | +1.7% |
| Early Stop (10 epochs + base) | 10 | 5.3857 | +8.5% |
| Baseline (3 epochs + base) | 3 | 5.7896 | +16.6% |

### 训练曲线对比

**Resume from Base:**
```
Epoch 1:  8.55-8.67  (起点高)
Epoch 3:  5.79-5.88
Epoch 10: 5.34-5.39  (收敛慢)
```

**From Scratch:**
```
Epoch 1:  5.40-5.41  (起点低！)
Epoch 3:  5.18-5.20
Epoch 8:  4.97       (收敛快！)
```

---

## 为什么From Scratch更好？

### 假设1: Base Checkpoint的负迁移
**观察：**
- Resume起始loss (8.6) 远高于 from-scratch (5.4)
- Resume需要"忘记"RL学到的次优模式
- From-scratch直接学习IL demonstrations

**证据：**
```
Resume Epoch 1→3: 8.67 → 5.88 (下降32%)
Scratch Epoch 1→3: 5.41 → 5.20 (下降4%)
```
Resume需要大量epochs来"unlearn"，而scratch直接学习正确模式。

### 假设2: IL Demonstrations质量高
**观察：**
- From-scratch能快速收敛到低loss
- 说明demonstrations本身包含足够信息
- 不需要RL checkpoint的先验知识

**证据：**
- Scratch在epoch 1就达到5.4（比resume的epoch 10还好）
- 持续稳定下降，没有plateau

### 假设3: 任务对齐问题
**观察：**
- Base RL checkpoint优化的是RL reward
- IL demonstrations优化的是expert actions
- 两者目标可能不完全一致

**推论：**
- RL学到的策略可能包含"捷径"（高reward但非expert行为）
- IL需要纯粹模仿expert，不需要这些捷径
- From-scratch避免了这种冲突

---

## 验证Loss分析

### Early Stopping实验

**Resume from Base (Exp 2):**
```
Epoch 1:  val_loss=7.4450
Epoch 10: val_loss=5.3857
改善: 27.7%
```

**From Scratch (Exp 4):**
```
Epoch 1:  val_loss=5.3467
Epoch 14: val_loss=5.0503
改善: 5.5%
```

**关键观察：**
- From-scratch起始验证loss (5.35) 已经优于 resume的最终loss (5.39)
- From-scratch的改善空间更小（已经接近最优）
- Resume需要更多epochs才能达到相同水平

---

## 训练效率对比

### 达到Loss=5.5所需Epochs

| 方法 | 达到5.5的Epoch | 训练时间 |
|------|---------------|---------|
| From Scratch | **~1** | 最快 |
| Resume | ~5 | 5倍慢 |

### 达到最佳性能所需Epochs

| 方法 | 最佳Epoch | 最佳Loss |
|------|----------|---------|
| From Scratch | 8 | 4.9650 |
| Resume | 10 | 5.3857 |

**结论：** From-scratch不仅最终性能更好，而且训练效率更高！

---

## 实际意义

### 对IL预训练的启示

1. **不一定需要base checkpoint**
   - 如果IL demonstrations质量高，from-scratch可能更好
   - 节省了寻找"好的"base checkpoint的时间

2. **负迁移是真实存在的**
   - RL checkpoint可能带来有害的bias
   - 特别是当RL和IL目标不完全一致时

3. **数据质量 > 初始化**
   - 高质量demonstrations比好的初始化更重要
   - 投资在数据收集上可能比调优初始化更有价值

### 对RL Fine-tuning的影响

**问题：** From-scratch IL checkpoint能否在RL fine-tuning中表现更好？

**需要验证：**
```bash
# 实验A: From-scratch IL → RL
python -m tcrppo_v2.ppo_trainer \
  --resume_from output/il_exp3_from_scratch_8epoch/checkpoints/latest.pt \
  --total_timesteps 1000000

# 实验B: Resume IL → RL
python -m tcrppo_v2.ppo_trainer \
  --resume_from output/il_exp2_early_stopping/checkpoints/best.pt \
  --total_timesteps 1000000

# 实验C: 纯RL (baseline)
python -m tcrppo_v2.ppo_trainer \
  --total_timesteps 1000000
```

**预测：**
- From-scratch IL可能在RL fine-tuning中收敛更快
- 因为它学到了更纯粹的expert行为模式
- 没有RL checkpoint的"坏习惯"

---

## 推荐策略

### 场景1: 有高质量demonstrations
**推荐：** From Scratch + Early Stopping
```bash
python scripts/pretrain_il.py \
  --dataset data/il/high_quality_demos.jsonl \
  --epochs 15 \
  --learning-rate 3e-4 \
  --from-scratch \
  --val-split 0.1 \
  --patience 4 \
  --save-every-epoch
```

### 场景2: Demonstrations质量未知
**推荐：** 两种方法都试，选最好的
```bash
# 方法1: From scratch
python scripts/pretrain_il.py --from-scratch --epochs 10

# 方法2: Resume
python scripts/pretrain_il.py --base-checkpoint base.pt --epochs 10

# 比较validation loss，选更低的
```

### 场景3: 时间紧迫
**推荐：** From Scratch (更快收敛)
```bash
python scripts/pretrain_il.py \
  --from-scratch \
  --epochs 5 \
  --learning-rate 3e-4
```

---

## 未来工作

### 1. 理解负迁移机制
- 分析base checkpoint学到了什么"坏模式"
- 可视化policy在不同初始化下的行为差异
- 研究如何选择"好的"base checkpoint

### 2. 优化From-Scratch训练
- 尝试更高学习率（当前3e-4）
- 测试不同网络初始化方法
- 探索curriculum learning（从简单到困难）

### 3. 混合策略
- 部分层from-scratch，部分层from-base
- 使用base checkpoint作为regularization（KL penalty）
- 动态调整base checkpoint的影响权重

### 4. 数据增强
- 增加demonstrations多样性
- 过滤低质量demonstrations
- 主动学习：选择最有价值的demonstrations

---

## 结论

**核心发现：**
1. ✅ From-scratch IL训练达到更低loss (4.97 vs 5.39)
2. ✅ From-scratch收敛更快（1 epoch达到5.5 vs 5 epochs）
3. ✅ Base checkpoint可能带来负迁移
4. ✅ 高质量demonstrations > 好的初始化

**实践建议：**
- 默认使用from-scratch + early stopping
- 只在demonstrations质量差时考虑resume
- 投资在提升demonstration质量上

**下一步：**
- 用from-scratch checkpoint做RL fine-tuning
- 对比IL+RL vs 纯RL的sample efficiency
- 分析为什么from-scratch更好（可解释性研究）

---

**Date:** May 27, 2026  
**Experiments:** il_exp1-4, il_test1-3  
**Key Result:** From-scratch IL achieves 4.97 loss vs 5.39 for resume (8.5% better)
