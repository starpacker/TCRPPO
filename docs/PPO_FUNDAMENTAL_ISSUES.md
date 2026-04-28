# PPO架构的根本性问题分析

**Date**: 2026-04-25  
**Context**: 深度审视PPO实现的结构性缺陷

---

## 核心问题：PPO不适合这个任务的本质特征

### 问题1: **On-Policy算法 vs 昂贵的Reward函数**

**现状**:
- PPO是on-policy算法：每次更新后，旧的rollout数据就被丢弃
- 每个样本只被使用 `n_epochs=4` 次（16个gradient steps）
- 然后必须重新采样，重新调用ERGO/NetTCR scorer

**问题**:
- ERGO/NetTCR scoring非常昂贵（每次forward pass ~10-50ms）
- 每个rollout收集1024个样本，需要1024次scorer调用
- 这些昂贵的样本只被用4次就扔掉了
- **样本效率极低**

**证据**:
```python
# ppo_trainer.py line 800-840
for epoch in range(self.n_epochs):  # 只用4次
    for batch in self.buffer.get_batches(self.batch_size):
        # ... PPO update
        
# 然后buffer被清空，重新rollout
self.buffer.ptr = 0  # 所有数据丢弃
```

**为什么这是致命的**:
- 2M steps训练需要 2M次ERGO调用
- 如果用off-policy算法（SAC/TD3），同样的2M次调用可以训练10M+ steps
- **当前方法浪费了80%+的计算资源**

---

### 问题2: **Terminal Reward + 固定Episode长度 = 信用分配崩溃**

**现状**:
- `ban_stop=True` + `max_steps=8` → 所有episode都是恰好8步
- 奖励只在第8步给出（terminal reward）
- 前7步的reward都是0

**问题**:
```
Step 0: SUB at pos 5 → reward = 0
Step 1: INS at pos 3 → reward = 0
Step 2: DEL at pos 7 → reward = 0
...
Step 7: STOP → reward = 2.1 (ERGO score)
```

**PPO如何分配信用**:
- 使用GAE (Generalized Advantage Estimation)
- Advantage = Q(s,a) - V(s)
- 但Q(s,a)是通过TD估计的：Q(s,a) ≈ r + γV(s')

**问题在于**:
- 前7步的r=0，所以Q(s,a) ≈ γ^k * final_reward
- 即使gamma=0.99，第0步只能看到 0.99^8 = 0.923 的信号
- **但更严重的是**：Value function V(s)必须学会预测"8步后的奖励"
- 这对于一个2层MLP来说几乎不可能，因为：
  - 状态空间巨大（ESM-2 embedding 2560维）
  - 动作空间巨大（4×20×20 = 1600种组合）
  - 轨迹依赖性强（第0步的SUB会影响第7步的结果）

**证据**:
```
# 训练日志显示value loss一直很高
VF: 0.1-0.2  # Value function无法准确预测
PG: 0.015-0.025  # Policy gradient很小，因为advantage估计不准
```

**为什么这是致命的**:
- Value function学不好 → Advantage估计不准 → Policy gradient方向错误
- 前面的动作得不到正确的信用 → 只有最后几步被优化
- **这就是为什么训练这么慢的根本原因**

---

### 问题3: **Autoregressive Action Space + PPO Clipping = 梯度消失**

**现状**:
- 动作是3个头的乘积：P(a) = P(op) × P(pos|op) × P(tok|op,pos)
- PPO clipping作用在总的ratio上：ratio = P_new(a) / P_old(a)

**问题**:
```python
# policy.py line 193
total_log_prob = op_log_prob + pos_log_prob + tok_log_prob

# ppo_trainer.py line 817
ratio = torch.exp(log_probs - batch["old_log_probs"])
# ratio = exp(log P_new(op,pos,tok) - log P_old(op,pos,tok))
#       = P_new(op) * P_new(pos|op) * P_new(tok|op,pos) / [P_old(...)]
```

**当一个头变化很大时**:
- 假设P_new(op) / P_old(op) = 2.0（op头学到了更好的策略）
- 但P_new(pos|op) / P_old(pos|op) = 0.6（pos头还没学好）
- 总ratio = 2.0 × 0.6 = 1.2（在clip range内）
- **看起来没问题，但实际上op头的更新被pos头拖累了**

**更严重的情况**:
- 如果三个头的ratio分别是 [2.5, 0.8, 0.7]
- 总ratio = 2.5 × 0.8 × 0.7 = 1.4（在clip range内）
- 但op头的2.5已经超出了clip range，应该被clip
- **PPO无法正确处理autoregressive action space的clipping**

**为什么这是致命的**:
- 三个头的学习速度不同步
- 快的头被慢的头拖累
- 慢的头的错误被快的头掩盖
- **整体学习效率极低**

---

### 问题4: **Shared Backbone + 多任务学习 = 梯度冲突**

**现状**:
```python
# policy.py line 40-45
self.backbone = nn.Sequential(
    nn.Linear(obs_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
)
# 然后backbone的输出被4个head共享：
# - op_head (actor)
# - pos_head (actor, conditioned)
# - token_head (actor, conditioned)
# - value_head (critic)
```

**问题**:
- Actor和Critic共享backbone
- Actor的3个头也共享backbone
- 4个不同的loss通过同一个backbone反向传播

**梯度冲突**:
```
Loss = PG_loss + vf_coef * VF_loss + ent_coef * Ent_loss

∂Loss/∂backbone = ∂PG/∂backbone + 0.5 * ∂VF/∂backbone + 0.05 * ∂Ent/∂backbone
```

- Policy gradient希望backbone提取"哪个动作好"的特征
- Value function希望backbone提取"这个状态值多少"的特征
- 这两个目标经常冲突

**证据**:
- 标准PPO实现（OpenAI Baselines, Stable-Baselines3）都使用**分离的backbone**
- Actor和Critic各自有独立的特征提取器
- 只在最简单的任务（CartPole）才共享

**为什么这是致命的**:
- Backbone被迫学习一个"妥协"的表示
- 既不利于policy，也不利于value
- **这就是为什么value loss一直很高的原因之一**

---

### 问题5: **固定的Episode长度 = 浪费了RL的核心优势**

**现状**:
- `max_steps=8` + `ban_stop=True`
- 所有episode都是8步，不能提前停止

**问题**:
- RL的核心优势是**学习何时停止**
- 当前设置下，agent无法学习：
  - "这个TCR已经够好了，不需要再编辑"
  - "这个target很难，需要更多步骤"
  - "这个初始TCR很差，应该大幅修改"

**对比**:
- AlphaGo可以学习何时pass
- 文本生成可以学习何时输出EOS
- 机器人控制可以学习何时停止动作

**当前方法**:
- 强制所有TCR都编辑8步
- 即使初始TCR已经很好（L1 seeds），也要编辑8步
- 即使前3步已经达到最优，也要继续编辑5步（可能变差）

**为什么这是致命的**:
- 浪费了计算（不必要的编辑）
- 破坏了好的TCR（过度编辑）
- **无法学习"适可而止"的策略**

---

## 根本性的解决方案

### 方案A: **切换到Off-Policy算法 (SAC/TD3)**

**优势**:
1. **样本效率提升5-10倍**
   - Replay buffer可以存储100K+样本
   - 每个样本可以被使用100+次
   - 同样的ERGO调用次数，训练步数增加10倍

2. **更适合昂贵的reward**
   - Off-policy天然适合expensive reward
   - 可以用小batch频繁更新

3. **更好的exploration**
   - SAC有entropy maximization
   - 比PPO的entropy bonus更principled

**挑战**:
- 需要重写整个训练循环
- Autoregressive action space需要特殊处理
- 但收益巨大

---

### 方案B: **引入Hindsight Experience Replay (HER)**

**核心思想**:
- 即使episode失败（低reward），也能学到东西
- 通过"假装"目标是实际达到的结果

**应用到TCR设计**:
```python
# 原始episode:
initial_TCR = "CASSLGQAYEQYF"
actions = [SUB(5,A), INS(3,G), ...]
final_TCR = "CASSLAQGAYEQYF"
reward = ERGO(final_TCR, target) = 0.3  # 很低

# HER: 假装目标就是生成这个TCR
# 重新计算reward，假设final_TCR就是"正确答案"
hindsight_reward = 1.0  # 因为我们"成功"生成了final_TCR

# 这样即使低reward的episode也能学到：
# "如果目标是生成CASSLAQGAYEQYF，这些动作是对的"
```

**优势**:
- 大幅提升样本效率
- 特别适合sparse reward
- 已在机器人控制中证明有效

---

### 方案C: **分层强化学习 (Hierarchical RL)**

**核心思想**:
- High-level policy: 决定"编辑策略"（保守/激进/targeted）
- Low-level policy: 执行具体的编辑动作

**应用到TCR设计**:
```python
# High-level (每个episode开始时决定一次):
strategy = high_level_policy(initial_TCR, target)
# strategy ∈ {conservative, moderate, aggressive}

# Low-level (每步):
action = low_level_policy(state, strategy)
```

**优势**:
- 简化信用分配（high-level只需要预测episode-level reward）
- Low-level可以用imitation learning预训练
- 更符合人类设计TCR的思维

---

### 方案D: **Model-Based RL + Planning**

**核心思想**:
- 学习一个forward model: s' = f(s, a)
- 学习一个reward model: r = g(s, a)
- 用learned model做planning，而不是直接学policy

**应用到TCR设计**:
```python
# 学习TCR编辑的dynamics
forward_model(TCR, action) → next_TCR

# 学习ERGO的近似
reward_model(TCR, peptide) → predicted_affinity

# Planning: 用learned models搜索最优动作序列
best_actions = MCTS(forward_model, reward_model, initial_TCR, target)
```

**优势**:
- 样本效率极高（learned model可以无限rollout）
- 可以结合搜索算法（MCTS, CEM）
- Forward model比policy更容易学

---

### 方案E: **Imitation Learning + RL Fine-tuning**

**核心思想**:
- 先用已知的好TCR做behavioral cloning
- 然后用RL fine-tune

**数据来源**:
- VDJdb/IEDB的已知binder
- 用简单的启发式规则生成"编辑轨迹"
- 例如：从random TCR → known binder的最短编辑路径

**优势**:
- 快速bootstrap（不需要从零探索）
- RL只需要fine-tune，不需要从头学
- 已在机器人、游戏AI中证明有效

---

## 我的建议

**短期（1周内）**:
1. **修复advantage normalization**（per rollout, not per minibatch）
2. **增加gamma到0.99**
3. **分离actor和critic的backbone**
4. **加入KL monitoring**

**中期（2-4周）**:
5. **实现SAC/TD3 off-policy版本**
   - 预期样本效率提升5-10倍
   - 可以用更小的batch size
   - 更适合expensive reward

**长期（1-2月）**:
6. **探索Hierarchical RL或Model-Based RL**
   - 根本性解决信用分配问题
   - 可能带来10-100倍的效率提升

---

## 关键问题

**你觉得哪个方向最值得探索？**

1. Off-policy (SAC/TD3) - 工程量中等，收益确定
2. Hierarchical RL - 工程量大，收益可能很大
3. Model-Based RL - 工程量大，风险高但潜力巨大
4. Imitation Learning - 工程量小，但需要expert data

**或者你有其他想法？**
