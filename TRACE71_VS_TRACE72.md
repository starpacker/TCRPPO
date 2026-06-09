# trace71 vs trace72: Adaptive Band Selection

## 问题诊断

### trace71 的问题（trace61 的 dynamic band 逻辑）

**当前逻辑：**
```
如果 mean final affinity = -1.5
→ 选择 band2: [-2.0, -1.0]
→ 从 -2.0 到 -1.0 的种子中采样
```

**问题：**
- 模型表现好（mean A = -1.5）时，给它**容易的种子**（-2 到 -1）
- 这些种子本身就接近目标，**改进空间小**
- 模型没有学到"如何从差的种子改进到好的结果"
- 导致**过拟合到容易的种子**，泛化能力差

### trace72 的改进（adaptive band selection）

**新逻辑（INVERSE relationship）：**
```
如果 mean final affinity = -1.5
→ 选择 adaptive band: [-3.0, -1.5]  (更难的种子)
→ 从 -3.0 到 -1.5 的种子中采样
```

**优势：**
- 模型表现好时，给它**更难的种子**
- 保持训练挑战性，避免过拟合
- 学习"如何从差的种子改进到好的结果"
- 更好的泛化能力

## Adaptive Band 策略

| Mean Final Affinity | 采样范围 | 说明 |
|---------------------|----------|------|
| **A ≥ -0.5** | [-4.0, -2.0] | 专家级表现 → 最难的种子 |
| **-1.5 ≤ A < -0.5** | [-3.0, -1.5] | 良好表现 → 中等难度种子 |
| **-2.5 ≤ A < -1.5** | [-2.5, -1.0] | 中等表现 → 较容易种子 |
| **A < -2.5** | [-2.0, 0.0] | 弱表现 → 热身种子 |

## 实验对比

### trace71 (固定 band)
- **Gate**: -0.8
- **Band 策略**: 根据 mean A 选择**相同范围**的 band
- **问题**: 表现好 → 容易种子 → 过拟合

### trace72 (adaptive band)
- **Gate**: -0.8
- **Band 策略**: 根据 mean A 选择**更难**的 band（inverse）
- **优势**: 表现好 → 困难种子 → 持续挑战

## 预期效果

### trace71 可能的问题
1. 初期快速提升（容易种子）
2. 后期停滞（过拟合）
3. 泛化能力差

### trace72 预期改进
1. 持续的训练挑战
2. 更稳定的长期提升
3. 更好的泛化能力
4. 更高的最终性能

## 启动命令

```bash
# 停止 trace71（如果需要）
pkill -f "trace71"
pkill -f "tfold_server_trace71"

# 启动 trace72
cd /share/liuyutian/tcrppo_v2
chmod +x launch_trace72_adaptive_gate_m0p8.sh
bash launch_trace72_adaptive_gate_m0p8.sh

# 监控
tail -f logs/trace72_adaptive_gate_m0p8_train.log
```

## 监控指标

关注以下指标来验证改进：

1. **DeltaA (改进幅度)**：trace72 应该有更大的 DeltaA
2. **通过 gate 的比例**：trace72 应该更稳定地提升
3. **InitA 分布**：trace72 应该采样更低的 InitA（更难的种子）
4. **长期趋势**：trace72 应该避免停滞

## 代码位置

- **Patch 文件**: `tcrppo_v2/data/tcr_pool_trace71_adaptive.py`
- **Config**: `configs/trace72_adaptive_gate_m0p8.yaml`
- **启动脚本**: `launch_trace72_adaptive_gate_m0p8.sh`
