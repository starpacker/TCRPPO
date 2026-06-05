# IL Training: Early Stopping & From Scratch

## 新增功能

### 1. Early Stopping（早停）
自动在验证集loss不再下降时停止训练，避免过拟合。

**参数：**
- `--val-split 0.1`: 使用10%数据作为验证集
- `--patience 3`: 验证loss连续3个epoch不下降就停止
- `--save-every-epoch`: 保存每个epoch的checkpoint

**示例：**
```bash
python scripts/pretrain_il.py \
  --config configs/trace62_multi_gates.yaml \
  --dataset data/il/highaff03_trace11_29_61_62_63_tchard_il.jsonl \
  --base-checkpoint output/test62_*/checkpoints/milestone_580000.pt \
  --out output/il_early_stop/checkpoints/latest.pt \
  --epochs 10 \
  --val-split 0.1 \
  --patience 3 \
  --save-every-epoch
```

**输出：**
```
Epoch 1/10: train_loss=8.5500 rows=10132 | val_loss=8.7234
  → New best val_loss: 8.7234 (saved to best.pt)
Epoch 2/10: train_loss=6.5306 rows=10132 | val_loss=6.8912
  → New best val_loss: 6.8912 (saved to best.pt)
Epoch 3/10: train_loss=5.7896 rows=10132 | val_loss=6.1234
  → New best val_loss: 6.1234 (saved to best.pt)
Epoch 4/10: train_loss=5.2341 rows=10132 | val_loss=6.0987
  → New best val_loss: 6.0987 (saved to best.pt)
Epoch 5/10: train_loss=4.8765 rows=10132 | val_loss=6.1523
  → Val loss did not improve (patience: 1/3)
Epoch 6/10: train_loss=4.5432 rows=10132 | val_loss=6.2341
  → Val loss did not improve (patience: 2/3)
Epoch 7/10: train_loss=4.2876 rows=10132 | val_loss=6.3456
  → Val loss did not improve (patience: 3/3)
Early stopping triggered at epoch 7. Best epoch was 4 with val_loss=6.0987
```

**生成的checkpoints：**
- `latest.pt`: 最后一个epoch（可能过拟合）
- `best.pt`: 验证loss最低的epoch（推荐使用）
- `epoch_1.pt`, `epoch_2.pt`, ...: 每个epoch的checkpoint

---

### 2. From Scratch（从头训练）
不加载base checkpoint，从随机初始化开始训练。

**参数：**
- `--from-scratch`: 从随机初始化开始
- 或者 `--base-checkpoint ""`: 设置为空字符串

**示例：**
```bash
python scripts/pretrain_il.py \
  --config configs/trace62_multi_gates.yaml \
  --dataset data/il/highaff03_trace11_29_61_62_63_tchard_il.jsonl \
  --out output/il_from_scratch/checkpoints/latest.pt \
  --epochs 8 \
  --learning-rate 3e-4 \
  --from-scratch \
  --save-every-epoch
```

**注意：**
- 从头训练需要更多epochs（建议8-15个）
- 学习率可以稍高（3e-4 vs 3e-5）
- 训练时间更长，但可能学到更纯粹的IL策略

---

## 使用场景

### 场景1：快速实验（原始方法）
```bash
# 3 epochs, 从base checkpoint开始
python scripts/pretrain_il.py \
  --dataset data/il/my_dataset.jsonl \
  --base-checkpoint output/base/checkpoints/milestone_580000.pt \
  --epochs 3
```
**适用于：** 快速验证IL是否有效

---

### 场景2：最佳性能（推荐）
```bash
# Early stopping, 从base checkpoint开始
python scripts/pretrain_il.py \
  --dataset data/il/my_dataset.jsonl \
  --base-checkpoint output/base/checkpoints/milestone_580000.pt \
  --epochs 10 \
  --val-split 0.1 \
  --patience 3 \
  --save-every-epoch
```
**适用于：** 生产环境，需要最佳checkpoint

**使用best.pt进行RL fine-tuning：**
```bash
python -m tcrppo_v2.ppo_trainer \
  --config configs/trace62_multi_gates.yaml \
  --resume_from output/il_early_stop/checkpoints/best.pt \
  --total_timesteps 1000000
```

---

### 场景3：纯IL策略
```bash
# 从头训练，不依赖RL checkpoint
python scripts/pretrain_il.py \
  --dataset data/il/my_dataset.jsonl \
  --epochs 10 \
  --learning-rate 3e-4 \
  --from-scratch \
  --val-split 0.1 \
  --patience 4 \
  --save-every-epoch
```
**适用于：** 研究IL本身的能力，或者base checkpoint质量不好

---

### 场景4：消融实验
```bash
# 运行所有4个实验
bash scripts/run_il_experiments.sh
```
**对比：**
1. Baseline (3 epochs + base)
2. Early stopping (10 epochs + base + val)
3. From scratch (8 epochs)
4. From scratch + early stopping (15 epochs + val)

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 最大训练轮数 |
| `--val-split` | 0.0 | 验证集比例（0-1） |
| `--patience` | 0 | Early stopping耐心值（0=禁用） |
| `--from-scratch` | False | 从随机初始化开始 |
| `--save-every-epoch` | False | 保存每个epoch |
| `--learning-rate` | 3e-5 | 学习率（从头训练建议3e-4） |
| `--batch-size` | 128 | 批大小 |

---

## 如何选择最佳checkpoint？

### 方法1：验证loss（推荐）
```bash
# 使用early stopping自动选择
--val-split 0.1 --patience 3

# 使用best.pt
--resume_from output/il_xxx/checkpoints/best.pt
```

### 方法2：评估生成质量
```bash
# 评估所有epoch的checkpoints
for epoch in 1 2 3 4 5; do
  python scripts/eval_checkpoint_decoy_reward_tfold.py \
    --checkpoint output/il_xxx/checkpoints/epoch_${epoch}.pt \
    --output-dir results/il_eval_epoch${epoch} \
    --n-tcrs 5
done

# 比较mean_target_reward
grep "mean_target_reward" results/il_eval_epoch*/summary_by_checkpoint.csv
```

### 方法3：RL fine-tuning效果
```bash
# 用不同epoch初始化RL，看哪个收敛最快
for epoch in 1 3 5; do
  python -m tcrppo_v2.ppo_trainer \
    --config configs/trace62_multi_gates.yaml \
    --run_name il_epoch${epoch}_then_rl \
    --resume_from output/il_xxx/checkpoints/epoch_${epoch}.pt \
    --total_timesteps 500000
done
```

---

## 预期结果

### Early Stopping
- **训练时间：** 比固定epochs少（自动停止）
- **最佳epoch：** 通常在4-6 epoch
- **过拟合风险：** 低（自动避免）

### From Scratch
- **训练时间：** 比resume长（需要更多epochs）
- **最终loss：** 可能比resume高（没有RL知识）
- **RL fine-tuning：** 可能需要更多steps才能达到相同性能

---

## 故障排除

### 问题1：验证loss一直不下降
```
Epoch 1/10: train_loss=8.5 | val_loss=8.7
Epoch 2/10: train_loss=8.3 | val_loss=8.9
Epoch 3/10: train_loss=8.1 | val_loss=9.2
```
**原因：** 过拟合或学习率太高  
**解决：** 降低学习率到1e-5，增加val_split到0.2

### 问题2：从头训练loss很高
```
Epoch 8/8: train_loss=12.3456
```
**原因：** 随机初始化需要更多训练  
**解决：** 增加epochs到15-20，或使用base checkpoint

### 问题3：Early stopping太早触发
```
Early stopping triggered at epoch 3
```
**原因：** Patience太小或val_split太小（高方差）  
**解决：** 增加patience到5，增加val_split到0.15

---

## 下一步

1. **运行实验：** `bash scripts/run_il_experiments.sh`
2. **比较结果：** 查看各个实验的训练log
3. **评估质量：** 用eval脚本测试生成的TCR
4. **RL fine-tuning：** 用最佳checkpoint初始化RL
5. **更新文档：** 记录最佳配置到项目文档

---

**更新日期：** 2026-05-26  
**相关文档：** `docs/IMITATION_LEARNING_GUIDE.md`
