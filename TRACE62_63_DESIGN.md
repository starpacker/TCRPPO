# Trace62 & Trace63: Reward Optimization Experiments

## 目标
将整体 reward 从当前的 -1.43 提升到 **-1.0 以上**

## 当前问题（Trace61）
- Mean reward: **-1.426**
- Mean affinity: **-1.521**
- 只有 **9.4%** 的 episodes 超过 gate (-0.5)
- Pool 被低质量 TCR 主导（44% 在 Band2 [-2,-1]）
- Reward 信号弱，改进动力不足

---

## Trace62: 多 Gate 课程学习

### 设计思路
使用 **3 个渐进式 gates**，每个 gate 给予不同的 bonus，形成阶梯式奖励：

```
Gate 1: -1.5  →  Bonus: +0.5
Gate 2: -0.5  →  Bonus: +1.0  
Gate 3: +0.6  →  Bonus: +2.0
```

### Reward 公式
```python
# 找到 affinity 超过的最高 gate
if A >= 0.6:
    bonus = 2.0
elif A >= -0.5:
    bonus = 1.0
elif A >= -1.5:
    bonus = 0.5
else:
    bonus = 0.0

reward = A + bonus + naturalness + diversity
```

### 预期效果
| Affinity | 当前 Reward | Trace62 Reward | 改进 |
|----------|-------------|----------------|------|
| -2.0 | -2.0 | -2.0 | 0.0 |
| -1.5 | -1.5 | -1.0 | **+0.5** |
| -1.0 | -1.0 | -0.5 | **+0.5** |
| -0.5 | -0.5 | +0.5 | **+1.0** |
| 0.0 | 0.0 | +1.0 | **+1.0** |
| 0.6 | 0.6 | +2.6 | **+2.0** |

**预期 mean reward**: -1.43 → **-0.8** (提升 +0.6)

### 优点
✅ 渐进式奖励，每个阶段都有明确目标  
✅ 多个里程碑，训练更稳定  
✅ 高 affinity 获得更大奖励，鼓励探索  

### 缺点
⚠️ 仍然是硬 gate，在阈值处有跳跃  
⚠️ 需要调整 gate 位置和 bonus 大小  

---

## Trace63: 平滑 Gate Reward

### 设计思路
使用 **sigmoid 函数**实现平滑过渡，消除硬 gate 的跳跃：

```python
bonus_factor = sigmoid((A - gate) / temperature)
reward = A + bonus * bonus_factor + auxiliaries
```

### 参数设置
- **Gate**: -1.0（比 trace61 的 -0.5 更容易达到）
- **Bonus**: 1.0
- **Temperature**: 0.5（控制平滑程度）

### Reward 曲线
```
A = -2.0  →  bonus_factor = 0.12  →  R = -2.0 + 0.12 = -1.88
A = -1.5  →  bonus_factor = 0.27  →  R = -1.5 + 0.27 = -1.23
A = -1.0  →  bonus_factor = 0.50  →  R = -1.0 + 0.50 = -0.50  (半 bonus)
A = -0.5  →  bonus_factor = 0.73  →  R = -0.5 + 0.73 = +0.23
A = 0.0   →  bonus_factor = 0.88  →  R = 0.0  + 0.88 = +0.88
A = 0.5   →  bonus_factor = 0.95  →  R = 0.5  + 0.95 = +1.45
```

### 预期效果
**预期 mean reward**: -1.43 → **-0.9** (提升 +0.5)

### 优点
✅ **平滑梯度**，没有跳跃，学习更稳定  
✅ **所有改进都有奖励**，即使没到 gate 也有部分 bonus  
✅ 更好的 exploration，鼓励持续改进  
✅ 理论上更优的学习信号  

### 缺点
⚠️ 需要调整 temperature 参数  
⚠️ 可能比硬 gate 慢一些（因为奖励更分散）  

---

## 实验配置对比

| 配置项 | Trace61 | Trace62 | Trace63 |
|--------|---------|---------|---------|
| **Reward Mode** | v2_simple_target_gated_decoy | v2_curriculum_climbing | v2_smooth_gate_reward |
| **Gate(s)** | -0.5 | -1.5, -0.5, 0.6 | -1.0 (smooth) |
| **Bonus** | 1.0 | 0.5, 1.0, 2.0 | 1.0 |
| **Temperature** | - | - | 0.5 |
| **Resume From** | trace29 580K | trace61 600K | trace61 600K |
| **GPU** | 1 | 2 | 3 |
| **Online Pool** | ✅ Dynamic bands | ✅ Dynamic bands | ✅ Dynamic bands |

---

## 启动实验

### Trace62 (Multi-gate)
```bash
cd /share/liuyutian/tcrppo_v2
chmod +x scripts/launch_trace62.sh
./scripts/launch_trace62.sh
```

### Trace63 (Smooth gate)
```bash
cd /share/liuyutian/tcrppo_v2
chmod +x scripts/launch_trace63.sh
./scripts/launch_trace63.sh
```

---

## 监控指标

### 关键指标
1. **Mean reward** - 目标：> -1.0
2. **Mean affinity** - 期望提升
3. **Pass rate** - 超过 gate 的比例
4. **Reward distribution** - 查看分布变化

### 监控命令
```bash
# 查看训练日志
tail -f logs/trace62_multi_gates_train.log
tail -f logs/trace63_smooth_gate_train.log

# 查看 checkpoint
ls -lh output/trace62_multi_gates/checkpoints/
ls -lh output/trace63_smooth_gate/checkpoints/

# 查看 online pool
ls -lh output/trace62_multi_gates/online_tcr_pool_events.jsonl
ls -lh output/trace63_smooth_gate/online_tcr_pool_events.jsonl
```

---

## 下一步计划

### 如果 Trace62 和 Trace63 都表现良好：
创建 **Trace64**：结合两者优点
- 平滑 reward（sigmoid）
- 多个 gates（渐进式）
- 配置示例：
  ```yaml
  reward_mode: "v2_smooth_multi_gate"  # 需要实现
  gates: [-1.5, -0.5, 0.6]
  bonuses: [0.5, 1.0, 2.0]
  temperature: 0.5
  ```

### 如果效果不理想：
1. 调整 gate 位置
2. 调整 bonus 大小
3. 调整 temperature
4. 考虑其他 reward shaping 方法

---

## 预期时间线

- **20K steps** (~2-3 hours): 初步效果
- **50K steps** (~6-8 hours): 稳定趋势
- **100K steps** (~12-16 hours): 完整评估

建议先运行 20K steps，观察趋势后决定是否继续。

---

## 成功标准

### Trace62
- Mean reward > -1.0 ✅
- 至少 30% episodes 超过 gate1 (-1.5)
- 至少 15% episodes 超过 gate2 (-0.5)

### Trace63
- Mean reward > -1.0 ✅
- Reward 曲线平滑上升
- 没有明显的训练不稳定

如果两者都达标，则进行 Trace64 的设计和实现。
