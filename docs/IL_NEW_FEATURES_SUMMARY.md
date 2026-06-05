# IL Training 新功能总结

## 更新内容 (2026-05-26)

为 `scripts/pretrain_il.py` 添加了两个重要功能：

### ✅ 1. Early Stopping（早停机制）

**功能：**
- 自动划分训练集/验证集
- 监控验证集loss
- 当验证loss连续N个epoch不下降时自动停止
- 保存验证loss最低的checkpoint为 `best.pt`

**新增参数：**
```bash
--val-split 0.1      # 10%数据作为验证集
--patience 3         # 连续3个epoch验证loss不下降就停止
--save-every-epoch   # 保存每个epoch的checkpoint
```

**使用示例：**
```bash
python scripts/pretrain_il.py \
  --dataset data/il/my_dataset.jsonl \
  --base-checkpoint output/base/checkpoints/milestone_580000.pt \
  --epochs 10 \
  --val-split 0.1 \
  --patience 3 \
  --save-every-epoch
```

**输出示例：**
```
Epoch 1/10: train_loss=8.5500 | val_loss=8.7234
  → New best val_loss: 8.7234 (saved to best.pt)
Epoch 2/10: train_loss=6.5306 | val_loss=6.8912
  → New best val_loss: 6.8912 (saved to best.pt)
...
Epoch 7/10: train_loss=4.2876 | val_loss=6.3456
  → Val loss did not improve (patience: 3/3)
Early stopping triggered at epoch 7. Best epoch was 4 with val_loss=6.0987
```

**优势：**
- ✅ 自动避免过拟合
- ✅ 节省训练时间
- ✅ 自动选择最佳checkpoint
- ✅ 提供训练/验证loss曲线

---

### ✅ 2. From Scratch（从头训练）

**功能：**
- 不加载base RL checkpoint
- 从随机初始化开始训练
- 适合研究纯IL策略或base checkpoint质量不好的情况

**新增参数：**
```bash
--from-scratch       # 从随机初始化开始
```

**使用示例：**
```bash
python scripts/pretrain_il.py \
  --dataset data/il/my_dataset.jsonl \
  --epochs 8 \
  --learning-rate 3e-4 \
  --from-scratch \
  --save-every-epoch
```

**输出示例：**
```
Training from scratch (random initialization)
Epoch 1/8: train_loss=15.2341 rows=11258
Epoch 2/8: train_loss=12.5678 rows=11258
...
```

**注意事项：**
- 需要更多epochs（8-15个 vs 3-5个）
- 学习率可以更高（3e-4 vs 3e-5）
- 训练时间更长
- 最终性能可能不如从base checkpoint开始

---

## 使用场景对比

| 场景 | 配置 | 适用情况 |
|------|------|----------|
| **快速验证** | 3 epochs + base | 快速测试IL是否有效 |
| **生产环境（推荐）** | 10 epochs + base + early stopping | 需要最佳性能 |
| **纯IL研究** | 8-15 epochs + from-scratch | 研究IL本身能力 |
| **消融实验** | 多种配置对比 | 科研论文 |

---

## 快速开始

### 方案1：推荐配置（Early Stopping）
```bash
python scripts/pretrain_il.py \
  --config configs/trace62_multi_gates.yaml \
  --dataset data/il/highaff03_trace11_29_61_62_63_tchard_il.jsonl \
  --base-checkpoint output/test62_*/checkpoints/milestone_580000.pt \
  --out output/il_best/checkpoints/latest.pt \
  --epochs 10 \
  --val-split 0.1 \
  --patience 3 \
  --save-every-epoch \
  --device cuda:6
```

**然后用best.pt做RL fine-tuning：**
```bash
python -m tcrppo_v2.ppo_trainer \
  --config configs/trace62_multi_gates.yaml \
  --resume_from output/il_best/checkpoints/best.pt \
  --total_timesteps 1000000
```

### 方案2：从头训练
```bash
python scripts/pretrain_il.py \
  --config configs/trace62_multi_gates.yaml \
  --dataset data/il/highaff03_trace11_29_61_62_63_tchard_il.jsonl \
  --out output/il_scratch/checkpoints/latest.pt \
  --epochs 10 \
  --learning-rate 3e-4 \
  --from-scratch \
  --val-split 0.1 \
  --patience 4 \
  --save-every-epoch \
  --device cuda:6
```

### 方案3：运行所有实验
```bash
bash scripts/run_il_experiments.sh
```

---

## 测试新功能

```bash
# 快速测试（小规模）
bash scripts/test_il_features.sh

# 检查输出
ls output/il_test_early_stop/checkpoints/
# 应该看到: latest.pt, best.pt, epoch_1.pt, epoch_2.pt, ...

ls output/il_test_from_scratch/checkpoints/
# 应该看到: latest.pt, epoch_1.pt, epoch_2.pt, epoch_3.pt
```

---

## 参数完整列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | configs/trace62_multi_gates.yaml | 配置文件 |
| `--dataset` | str | data/il/trace29_trace61_tchard_il.jsonl | IL数据集 |
| `--base-checkpoint` | str | output/.../milestone_580000.pt | Base checkpoint（可设为空） |
| `--out` | str | output/.../latest.pt | 输出路径 |
| `--epochs` | int | 3 | 最大训练轮数 |
| `--batch-size` | int | 128 | 批大小 |
| `--learning-rate` | float | 3e-5 | 学习率 |
| `--device` | str | auto | 设备（cuda:X或cpu） |
| `--seed` | int | 42 | 随机种子 |
| `--val-split` | float | 0.0 | 验证集比例（0-1） |
| `--patience` | int | 0 | Early stopping耐心值（0=禁用） |
| `--from-scratch` | flag | False | 从随机初始化开始 |
| `--save-every-epoch` | flag | False | 保存每个epoch |

---

## 预期效果

### Early Stopping
- **自动停止：** 通常在4-7个epoch
- **最佳epoch：** 验证loss最低的epoch
- **文件输出：** `latest.pt`（最后）+ `best.pt`（最佳）+ `epoch_X.pt`（每个）

### From Scratch
- **训练时间：** 比resume长50-100%
- **Loss范围：** 初始15-20，最终5-8
- **RL fine-tuning：** 可能需要更多steps达到相同性能

---

## 故障排除

### Q1: 验证loss一直上升
```
Epoch 1: val_loss=8.7
Epoch 2: val_loss=9.2
Epoch 3: val_loss=9.8
```
**A:** 过拟合或学习率太高。降低学习率到1e-5，增加val_split到0.2。

### Q2: Early stopping太早触发
```
Early stopping triggered at epoch 3
```
**A:** Patience太小或验证集太小。增加patience到5，增加val_split到0.15。

### Q3: 从头训练loss很高
```
Epoch 8: train_loss=12.3
```
**A:** 需要更多训练。增加epochs到15-20，或使用base checkpoint。

---

## 相关文档

- **完整指南：** `docs/IMITATION_LEARNING_GUIDE.md`
- **Early Stopping详解：** `docs/IL_EARLY_STOPPING_GUIDE.md`
- **快速参考：** `docs/IL_QUICK_SUMMARY.md`

---

## 下一步

1. ✅ 运行测试：`bash scripts/test_il_features.sh`
2. ⏳ 运行完整实验：`bash scripts/run_il_experiments.sh`
3. ⏳ 评估各个checkpoint的生成质量
4. ⏳ 用最佳checkpoint做RL fine-tuning
5. ⏳ 对比IL+RL vs 纯RL的sample efficiency

---

**更新日期：** 2026-05-26  
**修改文件：** `scripts/pretrain_il.py`  
**新增脚本：** `scripts/run_il_experiments.sh`, `scripts/test_il_features.sh`
