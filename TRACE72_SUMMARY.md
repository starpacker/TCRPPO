# trace72 实验总结

## 实验设计

**trace72 = trace70 + Delta Reward**

### 与 trace70 的唯一区别
- **trace70**: `use_delta_reward = false`
  - Reward = absolute affinity + decoy + naturalness + diversity
- **trace72**: `use_delta_reward = true`
  - Reward = absolute affinity + **1.0 × delta affinity** + decoy + naturalness + diversity

### 其他设置（与 trace70 完全相同）
- Gate: -1.5
- 从 trace61 checkpoint (step 612,096) 恢复
- Dynamic band selection (trace61 patch)
- GPU: 1
- 其他所有超参数相同

## 初步结果（前 16 episodes）

### 通过 gate (-1.5) 的 episodes

| Episode | Terminal A | InitA | DeltaA | Reward | TargetSat |
|---------|-----------|-------|--------|--------|-----------|
| 1 | -0.702 | -6.520 | **+5.818** | -0.433 | 0.798 |
| 3 | -0.522 | -3.682 | **+3.159** | -0.270 | 0.978 |
| 4 | -1.056 | -3.323 | **+2.267** | -0.626 | 0.445 |
| 9 | -0.673 | -6.227 | **+5.554** | -0.391 | 0.827 |
| 10 | -1.078 | -6.729 | **+5.650** | -0.577 | 0.422 |
| 11 | -0.959 | -1.593 | **+0.634** | -0.374 | 0.541 |
| **13** | **+0.015** | -2.978 | **+2.993** | **+0.309** | **1.515** |
| 14 | -1.207 | -6.663 | **+5.456** | -0.666 | 0.293 |

### 关键发现

1. **Episode 13 达到正值亲和力**
   - Terminal A = +0.015 (突破 0！)
   - DeltaA = +2.993
   - Reward = +0.309 (正奖励)
   - TargetSat = 1.515 (远超 gate)

2. **Delta reward 的作用**
   - 即使终端 A 不够高（如 -1.2），大的 DeltaA（+5.5）仍能获得正向奖励
   - 鼓励模型学习"如何大幅改进"，而不仅仅是"达到绝对高分"

3. **通过率**
   - 16 episodes 中有 8 个通过 gate -1.5
   - 通过率 = 50%

## 理论优势

### trace70 的问题
- 只关注终端亲和力
- 可能导致模型：
  - 选择容易的种子（InitA 已经很高）
  - 做小的改进就能达标
  - 泛化能力差

### trace72 的改进
- 同时奖励终端亲和力和改进幅度
- 鼓励模型：
  - 从差的种子开始（更大的改进空间）
  - 学习有效的改进策略
  - 更好的泛化能力

## 预期效果

1. **更大的 DeltaA**
   - trace72 应该有更大的平均 DeltaA
   - 说明模型学会了更有效的改进策略

2. **更低的 InitA**
   - trace72 可能会采样更低的 InitA（更难的种子）
   - 因为大的 DeltaA 能补偿低的 InitA

3. **更好的泛化**
   - 在测试集上，trace72 应该表现更好
   - 因为它学会了"如何改进"而不是"如何选择容易的种子"

## 监控命令

```bash
# 实时监控
watch -n 30 bash monitor_trace72.sh

# 查看训练日志
tail -f logs/trace72_delta_from_trace70_train.log

# 查看 tFold server
tail -f logs/trace72_delta_from_trace70_tfold_amp_server.log
```

## 文件位置

- **Config**: `configs/trace72_adaptive_gate_m0p8.yaml`
- **启动脚本**: `launch_trace72_adaptive_gate_m0p8.sh`
- **训练日志**: `logs/trace72_delta_from_trace70_train.log`
- **监控脚本**: `monitor_trace72.sh`

## 下一步

1. 让 trace72 训练 5-10 万 steps
2. 对比 trace70 和 trace72 的学习曲线
3. 分析：
   - 平均 DeltaA 的变化
   - 平均 InitA 的变化
   - 通过 gate 的比例
   - 最终性能对比
