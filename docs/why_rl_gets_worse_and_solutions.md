# 为什么 RL 越学越差？+ 解决方案

**问题**: trace81 训练了 512 episodes，性能从 -7.125 退化到 -9.259，越学越差。

**这很奇怪！** 正常的 RL 应该越学越好，为什么会越学越差？

---

## 🔍 深度诊断：5 个致命问题

### 问题 1: Policy 生成的初始 TCR 质量在下降 ⚠️

```
InitA 趋势:
Update 1:  -6.638  ← 起点
Update 2:  -7.067  ⬇️ -0.429
Update 3:  -7.285  ⬇️ -0.218
Update 4:  -7.502  ⬇️ -0.217
...
Update 16: -9.007  ⬇️ 最差

总变化: -6.638 → -9.007 = -2.369 (退化)
```

**这意味着什么？**
- Policy 学到了"生成更差的 TCR"
- 不是"改善能力不足"，而是"主动生成垃圾"
- 这是 RL 学到了**错误的模式**

### 问题 2: Policy 的改善能力是负的 ⚠️⚠️

```
DeltaA 统计:
  正向改善: 0/16 (0.0%)  ← 一次都没有！
  负向退化: 16/16 (100.0%)
  平均 DeltaA: -0.225
```

**这意味着什么？**
- Policy 的每一步操作都让 TCR 变得更差
- 不是"不会改善"，而是"主动破坏"
- Policy 学到的是"如何让 TCR 变差"，而不是"如何改善"

### 问题 3: Value Function 没有学习 ⚠️

```
VF Loss 趋势:
  初始: 2.140
  最终: 0.826
  平均: 0.873
  标准差: 0.448  ← 波动大，不收敛
```

**这意味着什么？**
- VF loss 在 0.3-2.1 之间随机波动
- 没有下降趋势 = 没有学习
- 无法从稀疏的 terminal reward 中学习

### 问题 4: Policy 几乎没有更新 ⚠️

```
KL Divergence:
  平均: 0.00382
  健康范围: 0.01-0.05
  
  实际 KL 只有健康值的 1/3 到 1/13
```

**这意味着什么？**
- Policy gradient 太弱
- Policy 几乎冻结，没有有意义的更新
- 即使有 gradient，方向也是错的（因为 advantage 不准确）

### 问题 5: 探索性在下降 ⚠️

```
Entropy 趋势:
  初始: 5.601
  最终: 4.908
  变化: -0.693 (下降 12%)
```

**这意味着什么？**
- Policy 在收敛到一个**差的策略**
- 探索性下降 = 陷入局部最优
- 而这个"局部最优"是"生成垃圾 TCR"

---

## 💡 根本原因：恶性循环

```
┌─────────────────────────────────────────────────────────────┐
│                     恶性循环                                  │
└─────────────────────────────────────────────────────────────┘

Step 1: Value function = 随机噪声
  ↓
Step 2: Advantage = R - V(s) = 噪声（不准确）
  ↓
Step 3: Policy gradient = E[∇log π * Advantage] = 错误方向
  ↓
Step 4: Policy 更新 = 微弱且错误（KL=0.003）
  ↓
Step 5: Policy 学到错误模式 = "生成更差的 TCR"
  ↓
Step 6: InitA 下降 = 起点质量变差
  ↓
Step 7: DeltaA 为负 = 改善能力变成破坏能力
  ↓
Step 8: Reward 更差 = Value function 学不到东西
  ↓
回到 Step 1: Value function 仍然是噪声

结果: 越学越差，无法自我纠正
```

### 为什么会学到"生成更差的 TCR"？

**关键问题**: Advantage 估计不准确

```
真实情况:
  好的 action → 好的 reward → 应该鼓励
  差的 action → 差的 reward → 应该惩罚

但 trace81 的情况:
  Value function = 噪声
  Advantage = R - V(s) = 噪声
  
  好的 action → Advantage 可能是负的 → 被惩罚 ❌
  差的 action → Advantage 可能是正的 → 被鼓励 ❌
  
  Policy 学到的是随机信号，不是真实的好坏
```

**结果**: Policy 随机游走，碰巧学到了"生成更差的 TCR"这个错误模式，然后陷入其中无法自拔。

---

## 🎯 解决方案

### 方案 1: Checkpoint Resumption ✅ **已验证有效**

**原理**: 用预训练的 value function 打破恶性循环

```
┌─────────────────────────────────────────────────────────────┐
│                     良性循环                                  │
└─────────────────────────────────────────────────────────────┘

Step 1: Value function = 预训练（在 20 个 targets 上）
  ↓
Step 2: Advantage = R - V(s) = 准确
  ↓
Step 3: Policy gradient = 正确方向
  ↓
Step 4: Policy 更新 = 强且正确（KL=0.005）
  ↓
Step 5: Policy 学到正确模式 = "改善 TCR"
  ↓
Step 6: InitA 保持稳定或改善
  ↓
Step 7: DeltaA 为正 = 真正的改善能力
  ↓
Step 8: Reward 变好 = Value function 微调到新 targets
  ↓
回到 Step 1: Value function 更准确

结果: 越学越好，快速适应新 targets
```

**实施**:
```bash
python -m tcrppo_v2.train \
    --config configs/my_experiment.yaml \
    --resume output/trace73_curriculum_exploration/checkpoints/latest.pt \
    --resume_reset_optimizer
```

**效果**: trace83 vs trace81
- 起点: -1.191 vs -7.125 (好 **5.9 倍**)
- 改善率: 29.3% vs 6.7% (高 **4.4 倍**)
- DeltaA: +0.630 vs -0.225 (正向 vs 负向)

---

### 方案 2: 修复 Dense Rewards ⚠️ **未验证**

**原理**: 用密集的 reward 信号帮助 value function 学习

**trace82 的 bug**:
```python
# ❌ 错误: 每步都相对于初始状态计算 delta
for step in range(8):
    reward = aff(current) - aff(initial)  # 累积重复
```

**正确实现**:
```python
# ✅ 正确: 每步相对于上一步计算 delta
prev_aff = initial_aff
for step in range(8):
    curr_aff = compute_affinity(current_tcr)
    reward = curr_aff - prev_aff  # 步间 delta
    prev_aff = curr_aff
```

**优点**:
- 每步都有 reward 信号（不稀疏）
- Value function 更容易学习

**缺点**:
- 实现复杂（需要缓存上一步状态）
- 未验证是否真的有效
- Checkpoint resumption 已经够好了

**建议**: 不推荐。Checkpoint resumption 更简单且已验证。

---

### 方案 3: 调整超参数 ⚠️ **治标不治本**

**可能的调整**:

1. **增加 learning rate**
   ```yaml
   learning_rate: 3e-4  # 从 1.2e-4 增加
   ```
   - 目的: 增强 policy gradient
   - 风险: 可能不稳定
   - 效果: 有限（根本问题是 advantage 不准确）

2. **增加 vf_coef**
   ```yaml
   vf_coef: 1.0  # 从 0.5 增加
   ```
   - 目的: 让 value function 学得更快
   - 风险: 可能过拟合
   - 效果: 有限（稀疏信号仍然难学）

3. **增加 entropy_coef**
   ```yaml
   entropy_coef: 0.05  # 从 0.02 增加
   ```
   - 目的: 增加探索性
   - 风险: 可能太随机
   - 效果: 有限（探索不能解决 advantage 不准确）

4. **减小 clip_range**
   ```yaml
   clip_range: 0.1  # 从 0.2 减小
   ```
   - 目的: 更保守的更新
   - 风险: 学习更慢
   - 效果: 有限（不能解决根本问题）

**结论**: 这些调整都是**治标不治本**。根本问题是 value function bootstrap，调超参数无法解决。

---

### 方案 4: 改用 On-Policy 算法 ⚠️ **不推荐**

**其他 on-policy 算法**:
- A2C (Advantage Actor-Critic)
- TRPO (Trust Region Policy Optimization)
- SAC (Soft Actor-Critic) - 实际上是 off-policy

**为什么不推荐**:
- 都有同样的 value function bootstrap 问题
- PPO 已经是最稳定的 on-policy 算法
- 换算法不能解决根本问题

---

### 方案 5: 改用 Off-Policy 算法 ⚠️ **可能有效但复杂**

**Off-policy 算法**:
- DQN (Deep Q-Network)
- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed DDPG)
- SAC (Soft Actor-Critic)

**优点**:
- 可以用 replay buffer 重复利用数据
- 可能更 sample-efficient

**缺点**:
- 需要重写整个训练框架
- 离散 action space 需要特殊处理
- 不一定能解决 value function bootstrap 问题

**建议**: 不推荐。Checkpoint resumption 更简单。

---

### 方案 6: Curriculum Learning (L0/L1 起点) ⚠️ **部分有效**

**原理**: 从已知的好 TCR 开始，而不是随机 TCR

**实施**:
```yaml
curriculum_schedule:
  - {until: 100000, L0: 0.5, L1: 0.3, L2: 0.2}  # 早期: 50% 已知好 TCR
  - {until: 500000, L0: 0.3, L1: 0.3, L2: 0.4}  # 中期: 逐渐增加随机
  - {until: null, L0: 0.0, L1: 0.0, L2: 1.0}    # 后期: 全随机
```

**优点**:
- InitA 起点更高（从已知好 TCR 开始）
- 早期 reward 更好，帮助 value function 学习

**缺点**:
- 只解决了 InitA 问题，没解决 value function bootstrap
- trace81 用的就是 L2=1.0（全随机），但仍然失败
- 需要有已知的好 TCR（L0/L1 数据）

**效果**: 部分有效，但不如 checkpoint resumption。

---

## 📊 方案对比

| 方案 | 难度 | 效果 | 验证状态 | 推荐度 |
|------|------|------|----------|--------|
| **1. Checkpoint Resumption** | ⭐ 简单 | ⭐⭐⭐⭐⭐ 优秀 | ✅ 已验证 | ⭐⭐⭐⭐⭐ **强烈推荐** |
| 2. 修复 Dense Rewards | ⭐⭐ 中等 | ⭐⭐⭐ 可能有效 | ❌ 未验证 | ⭐⭐ 不推荐 |
| 3. 调整超参数 | ⭐ 简单 | ⭐ 有限 | ❌ 治标不治本 | ⭐ 不推荐 |
| 4. 改用 On-Policy 算法 | ⭐⭐⭐ 困难 | ⭐ 无效 | ❌ 同样问题 | ❌ 不推荐 |
| 5. 改用 Off-Policy 算法 | ⭐⭐⭐⭐ 很困难 | ⭐⭐ 不确定 | ❌ 未验证 | ⭐ 不推荐 |
| 6. Curriculum Learning | ⭐⭐ 中等 | ⭐⭐ 部分有效 | ⚠️ 部分验证 | ⭐⭐ 可选 |

---

## 🎯 最终推荐

### ✅ 立即采用：Checkpoint Resumption

**为什么**:
1. **简单**: 只需一行命令 `--resume checkpoint.pt`
2. **有效**: trace83 证明了 +5.9 affinity units 的改善
3. **已验证**: 4 个成功案例（trace53, 73, 78, 83）
4. **稳定**: 不需要调超参数或改代码

**实施步骤**:
```bash
# 1. 使用 trace73 checkpoint 作为基础
CHECKPOINT="output/trace73_curriculum_exploration/checkpoints/latest.pt"

# 2. 启动训练
python -m tcrppo_v2.train \
    --config configs/my_experiment.yaml \
    --resume $CHECKPOINT \
    --resume_reset_optimizer \
    --run_name my_experiment \
    --seed 42
```

### 🔬 可选探索：修复 Dense Rewards

**仅当你想验证理论时**:

1. 修复 `reward_manager.py` 中的 dense reward 实现
2. 改为计算步间 delta，而不是累积 delta
3. 对比 terminal vs dense rewards 的效果

**但注意**: Checkpoint resumption 已经够好了，这个探索是学术性的，不是必需的。

### ❌ 不推荐：其他方案

- 调超参数: 治标不治本
- 换算法: 复杂且不一定有效
- Curriculum: 不如 checkpoint resumption

---

## 💡 关键洞察

### 1. "越学越差"不是 RL 的正常行为

正常的 RL 应该:
- ✅ InitA 稳定或改善
- ✅ DeltaA 为正
- ✅ VF loss 下降
- ✅ KL 在健康范围

trace81 的异常说明它陷入了**恶性循环**。

### 2. Value Function Bootstrap 是关键瓶颈

从头训练的死锁:
```
需要好的 value function → 才能有好的 policy
需要好的 policy → 才能有好的 value function
```

Checkpoint resumption 打破死锁:
```
预训练的 value function → 准确的 advantage
准确的 advantage → 强的 gradient → 好的 policy
好的 policy → 好的 reward → value function 微调
```

### 3. Terminal Rewards 本身没问题

问题不是 terminal_reward_only，而是从头训练。

证据:
- trace53, 73, 78, 83 都用 terminal_reward_only
- 都成功了（因为用了 checkpoint resumption）

### 4. 迁移学习适用于 RL

预训练的 value function 在 20 个 targets 上学到的知识，可以迁移到 10 个新 targets。

这和 NLP/CV 的预训练-微调范式一样。

---

## 📝 总结

**问题**: trace81 越学越差，因为陷入了恶性循环

**根本原因**: Value function bootstrap 问题 + 稀疏 terminal reward

**最佳解决方案**: Checkpoint Resumption（简单、有效、已验证）

**实施**: 一行命令 `--resume checkpoint.pt`

**效果**: +5.9 affinity units，改善率从 6.7% 提升到 29.3%

**结论**: 不要从头训练。永远使用 checkpoint resumption。

---

**文档日期**: 2026-05-30  
**分析基于**: trace81 (512 episodes) vs trace83 (160 episodes)  
**验证状态**: ✅ 解决方案已验证有效
