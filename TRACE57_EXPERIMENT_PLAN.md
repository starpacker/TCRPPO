# trace57 实验计划

**创建时间**: 2026-05-25  
**状态**: 准备启动

---

## 🎯 实验目标

测试 **pMHC Embedding Centering** 对 target affinity 提升的效果。

---

## 🔬 实验设计

### 核心策略

1. **pMHC Embedding Centering** (`pmhc_embedding_transform: "center"`)
   - 减去所有训练 peptide 的 pMHC embedding 平均值
   - 目的：减少不同 peptide 之间的相似度（当前 cos-sim = 0.9685）
   - 突出 peptide 之间的差异，帮助 policy 学习 peptide-specific 策略

2. **Online TCR Pool** (复用好的 TCR init)
   - 从 502K 步开始启用
   - 保留 A > -2.0 的 TCR 作为 init seeds
   - 加速探索高 affinity 区域

3. **Large Multi-Tier Bonuses** (强激励)
   - 5 个 bonus 门槛：[-2.0, -1.0, -0.5, 0.0, 0.6]
   - 累积 bonus：[10, 15, 20, 30, 50]
   - 比 trace56 的 bonus 大 10-20 倍

### Reward Formula

```
R = 1.0 * A + bonus(A) + 0.05*Nat + 0.02*Div

Bonus Tiers (累积):
  A >= -2.0:  +10.0
  A >= -1.0:  +15.0  (10 + 5)
  A >= -0.5:  +20.0  (10 + 5 + 5)
  A >= 0.0:   +30.0  (10 + 5 + 5 + 10)
  A >= 0.6:   +50.0  (10 + 5 + 5 + 10 + 20)
```

---

## 📊 与 trace56 的对比

| 特性 | trace56 | trace57 |
|------|---------|---------|
| **pMHC Centering** | ❌ 无 | ✅ `pmhc_embedding_transform: "center"` |
| **Bonus 幅度** | 小额 [0.5, 1.5, 2.8, 4.2, 5.2, 6.3, 7.0, 8.0] | 大额 [10, 15, 20, 30, 50] |
| **Gate 数量** | 8 gates ([-8, -6, -4, -2, -1, 0, 0.4, 0.6]) | 5 gates ([-2, -1, -0.5, 0, 0.6]) |
| **Online Pool** | ✅ 启用 | ✅ 启用 |
| **Resume From** | trace11 @ 500K | trace11 @ 500K |
| **Curriculum** | L2 = 1.0 (pure random) | L2 = 1.0 (pure random) |

### 关键差异

**trace57 的独特之处：pMHC Embedding Centering**

这是 trace57 与 trace56 的**唯一本质区别**。如果 trace57 效果显著好于 trace56，说明：
- Peptide input 相似度确实是瓶颈
- Centering 能有效提升 peptide-specific 策略学习

---

## 🔧 技术细节

### pMHC Embedding Centering 实现

**代码位置**: `tcrppo_v2/ppo_trainer.py::_build_pmhc_obs_transform()`

**工作流程**:
1. 在训练开始时，计算所有训练 peptide 的 pMHC embedding
2. 计算 mean embedding (center)
3. 保存到 `output/trace57_pure_target_bonus/pmhc_embedding_center.pt`
4. 在每个 observation 构建时，从 pMHC embedding 中减去 center

**效果预期**:
- Raw pMHC embeddings: cos-sim = 0.9685 (mean), range [0.9285, 0.9876]
- Centered embeddings: cos-sim 预期降低到 0.7-0.8

**代码**:
```python
# env.py::_transform_pmhc_obs()
def _transform_pmhc_obs(self, pmhc_emb: torch.Tensor) -> torch.Tensor:
    if not self.pmhc_obs_transform:
        return pmhc_emb
    
    center = self.pmhc_obs_transform.get("center")
    if center is not None:
        pmhc_emb = pmhc_emb - center  # <--- Centering
    
    return pmhc_emb
```

### Online TCR Pool 配置

```yaml
online_tcr_pool_enabled: true
online_tcr_pool_start_step: 502000  # 从 500K checkpoint 后 2K 步开始
online_tcr_pool_warmup_steps: 20000
online_tcr_pool_max_ratio: 0.8
online_tcr_pool_max_per_target: 128
online_tcr_pool_min_affinity: -2.0
online_tcr_pool_max_decoy_violation: 999.0
online_tcr_pool_min_hamming: 2
online_tcr_pool_mutate_prob: 0.0
```

---

## 📈 预期效果

### 短期目标 (50K-200K steps)

- **50K steps**: A > -2 比例 > 50%
- **100K steps**: A > -1 比例 > 30%
- **200K steps**: A > -0.5 比例 > 20%

### 中期目标 (500K steps)

- **Mean A > -1.0**
- **A > 0 比例 > 5%**
- **Max A > 1.0**

### 成功标准

如果 trace57 达到以下任一条件，即认为 **pMHC Centering 有效**：

1. **Mean A 比 trace56 高 0.5+**
2. **A > 0 比例比 trace56 高 3%+**
3. **Max A 比 trace56 高 0.3+**

---

## 🚀 启动命令

```bash
cd /share/liuyutian/tcrppo_v2
bash launch_trace57_pure_target_bonus.sh
```

### 监控

```bash
# 查看训练日志
tail -f logs/trace57_pure_target_bonus_train.log

# 查看 tFold server 日志
tail -f logs/trace57_pure_target_bonus_tfold_amp_server.log

# 检查进程
ps aux | grep trace57 | grep python
```

### 停止

```bash
# 找到 PIDs
ps aux | grep trace57 | grep python

# 停止
kill <TRAIN_PID> <TFOLD_PID>
```

---

## 📁 文件清单

### 配置文件
- `configs/trace57_pure_target_bonus.yaml` - 完整配置

### 启动脚本
- `launch_trace57_pure_target_bonus.sh` - 启动脚本

### 输出目录
- `output/trace57_pure_target_bonus/` - checkpoints, logs, results
- `output/trace57_pure_target_bonus/pmhc_embedding_center.pt` - pMHC center

### 日志文件
- `logs/trace57_pure_target_bonus_train.log` - PPO 训练日志
- `logs/trace57_pure_target_bonus_tfold_amp_server.log` - tFold server 日志
- `logs/trace57_pure_target_bonus_tfold_completion.log` - tFold 完成日志

---

## 🔍 分析计划

### 训练过程中

1. **监控 pMHC centering 效果**
   - 检查 `pmhc_embedding_center.pt` 是否正确生成
   - 查看训练日志中的 cos-sim 统计

2. **监控 affinity 提升**
   - 每 20K steps 检查 Mean A, Max A, A>0 比例
   - 对比 trace56 的同步数据

3. **监控 online pool 使用率**
   - 检查 pool 中 TCR 数量增长
   - 检查 pool init 比例

### 训练完成后

1. **对比 trace56 vs trace57**
   - Mean A, Max A, A>0 比例
   - Per-peptide 效果分布
   - 训练曲线对比

2. **验证 centering 效果**
   - 可视化 centered vs raw pMHC embeddings
   - 计算 centered embeddings 的 cos-sim
   - 分析 policy 的 peptide sensitivity

---

## 💡 后续方向

如果 trace57 成功：
- 将 pMHC centering 作为标准配置
- 测试其他 transform 模式（`center_layernorm`）
- 探索更复杂的 peptide representation

如果 trace57 失败：
- 分析 centering 是否真的降低了 cos-sim
- 考虑其他 peptide representation 方案（见 `PEPTIDE_INPUT_PROBLEM_ANALYSIS.md`）
- 测试 multi-peptide episodes 等其他方法

---

## 📝 Notes

- trace57 使用与 trace56 相同的 resume checkpoint (trace11 @ 500K)
- 两者的主要区别是 pMHC centering 和 bonus 幅度
- 这是一个 **controlled experiment**，可以清晰评估 centering 的效果
