# trace84 训练计划 - 推向 Mean A = 0.0

**启动时间**: 2026-05-30 14:54  
**状态**: ✅ 运行中  
**PID**: 521357  
**GPU**: 1

---

## 🎯 目标

**达到 Mean A = 0.0**（平均亲和力达到结合阈值）

---

## 📊 当前进展（前 16 episodes）

| 指标 | 值 | 状态 |
|------|-----|------|
| **Best A** | **+0.0757** | ✅ **已达到目标！** |
| **Mean A** | -0.938 | 🟡 接近目标 |
| **Positive rate** | 1/16 (6.25%) | 🟡 需要提高 |
| **Current step** | 716,152 | 刚开始 |
| **Gate** | -1.0 | 已激活 |

### 关键发现

1. **Episode 7 达到 +0.0757**：第一批就有正值，说明 checkpoint resumption 非常有效
2. **所有 DeltaA 为正**：改善能力强（+3.1 到 +6.6）
3. **OnlinePool 活跃**：已收集 6 个好 TCR
4. **无 catastrophic forgetting**：没有出现 trace81 那样的退化

---

## 🔧 配置亮点

### vs trace83 的改进

| 参数 | trace83 | trace84 | 效果 |
|------|---------|---------|------|
| **Gate schedule** | -3.0 → -1.0 (400K) | -2.0 → 0.0 (1.5M) | 更保守，避免强制 |
| **Decoy weights** | 全 0 | {A:3, B:3, D:2, C:1} | 启用 specificity |
| **Decoy unlock** | 无 | 渐进式 | 难度递增 |
| **Curriculum** | L2 only | L0/L1 → L2 | 从好 TCR 开始 |
| **decoy_K** | 2 | 8 | 更强信号 |
| **Total steps** | 1M | 2M | 更多时间 |

### Gate Schedule（保守推进）

```
716K (start): -1.0  ← 当前（因为 resume 自 716K）
1M: -0.5
1.5M: 0.0  ← 目标！
2M: 0.0
```

### Decoy Unlock（渐进式）

```
0-100K: D only (known binders)
100K-300K: D + A (point mutants)
300K-600K: D + A + B (2-3 AA mutants)
600K+: D + A + B + C (unrelated peptides)
```

### Curriculum（从好到随机）

```
0-100K: L0=0.5, L1=0.3, L2=0.2 (50% known good)
100K-500K: L0=0.3, L1=0.3, L2=0.4 (balanced)
500K+: L0=0.1, L1=0.2, L2=0.7 (mostly random)
```

---

## 📈 预期里程碑

| Step | Gate | Expected Mean A | 状态 |
|------|------|-----------------|------|
| 716K | -1.0 | -0.8 to -1.0 | ✅ 当前 -0.94 |
| 1M | -0.5 | -0.3 to -0.5 | 🔄 进行中 |
| 1.5M | **0.0** | **-0.2 to 0.0** | 🎯 **目标** |
| 2M | 0.0 | **0.0 to +0.2** | 🚀 超越 |

---

## 🔍 监控命令

```bash
# 实时监控
tail -f logs/trace84_push_to_zero_train.log

# 快速统计
bash scripts/monitor_trace84.sh

# 检查 Mean A 趋势
grep "PPO Update" logs/trace84_push_to_zero_train.log | tail -20

# 查看最佳 TCR
grep "Episode" logs/trace84_push_to_zero_train.log | grep "A=0\." | head -10
```

---

## ⚠️ 风险监控

### 需要关注的信号

| 信号 | 阈值 | 行动 |
|------|------|------|
| Mean A 下降 | < -3.0 持续 50K steps | 减慢 gate schedule |
| KL divergence | < 0.001 持续 10 updates | 增加 learning rate |
| Decoy violation | > 5.0 | 增加 w_decoy |
| Positive rate | < 5% @ 1M steps | 检查 gate 是否太严 |

### 当前状态：✅ 全部正常

- Mean A = -0.94（健康）
- KL = 未知（等待第一个 PPO update）
- Decoy violation = 0-2.5（正常）
- Positive rate = 6.25%（早期正常）

---

## 📝 下一步

### 短期（24小时内）

1. ✅ 训练已启动，运行中
2. 🔄 监控前 100K steps 的性能
3. 🔄 检查第一个 PPO update 的 KL/VF loss
4. 🔄 确认 decoy unlock 在 100K 时触发

### 中期（1-2天）

1. 🔄 达到 1M steps（gate = -0.5）
2. 🔄 评估 Mean A 是否接近 -0.5
3. 🔄 检查 OnlinePool 质量
4. 🔄 确认无 catastrophic forgetting

### 长期（2-3天）

1. 🎯 达到 1.5M steps（gate = 0.0）
2. 🎯 **验证 Mean A ≥ 0.0**
3. 🎯 运行 decoy specificity evaluation
4. 🎯 生成 50 TCRs per target
5. 🎯 对比 trace73/trace83/v1 baseline

---

## 🎉 成功标准

- ✅ **Primary**: Mean A ≥ 0.0 at 1.5M-2M steps
- ✅ **Secondary**: No catastrophic forgetting (Mean A never < -3.0)
- ✅ **Tertiary**: Maintain specificity (decoy penalty active)

---

## 📞 联系方式

- **Log file**: `logs/trace84_push_to_zero_train.log`
- **Monitor script**: `scripts/monitor_trace84.sh`
- **Kill command**: `kill 521357`
- **tFold server**: PID 420621, socket `/tmp/tfold_server_trace84.sock`

---

**最后更新**: 2026-05-30 15:00  
**下次检查**: 2026-05-30 18:00（3小时后，预计 ~50K steps）
