# trace73 实验总结

## 🎯 实验目标

基于 trace70（目前表现最好的实验），通过 **curriculum learning** 和 **increased exploration** 来进一步提升性能。

## 📊 问题分析

### 当前实验性能对比（最近 30 步）

| 排名 | 实验 | Gate | Delta | Mean R | Mean A | 状态 |
|------|------|------|-------|--------|--------|------|
| 🥇 | **trace70** | **-1.5** | ❌ | **-0.874** | **-1.177** | ✅ 最佳 |
| 🥈 | trace61 | -0.5 | ❌ | -0.878 | -1.243 | ✅ 稳定 |
| 🥉 | trace72 | -1.5 | ✅ | -0.939 | -1.236 | ➡️ 平稳 |
| 💥 | trace71 | -0.8 | ❌ | -5.375 | -5.371 | ❌ **崩溃** |

### 关键发现

1. **trace70 是最好的**，但趋势略微下降（-0.001443）
2. **trace72 的 delta reward 没有帮助**（比 trace70 差 0.065）
3. **trace71 完全崩溃**（gate -0.8 太高）
4. **所有实验都遇到了平台期**

## 🚀 trace73 设计

### 核心策略

#### 1. **Curriculum Learning（渐进式学习）**

使用动态 gate schedule，从易到难逐步提升标准：

```yaml
gate_schedule:
  644096: -2.0   # Phase 1: 起点，容易通过，快速积累高质量 TCR
  670000: -1.5   # Phase 2: 提高难度（26K steps 后）
  700000: -1.0   # Phase 3: 更难（56K steps 后）
  750000: -0.5   # Phase 4: 最终目标（106K steps 后）
```

**预期效果**:
- Phase 1 (-2.0): 快速建立高质量 online pool
- Phase 2 (-1.5): 保持 trace70 的水平
- Phase 3 (-1.0): 突破当前瓶颈
- Phase 4 (-0.5): 达到最终目标

#### 2. **Increased Exploration（增加探索）**

```yaml
entropy_coef: 0.020  # 从 0.012 提高到 0.020 (+67%)
```

**预期效果**:
- 更多样化的动作选择
- 避免过早收敛到局部最优
- 发现新的高亲和力 TCR 序列

#### 3. **Slightly Lower Learning Rate（略微降低学习率）**

```yaml
learning_rate: 1.2e-4  # 从 1.5e-4 降到 1.2e-4 (-20%)
```

**原因**: Curriculum learning 需要更稳定的训练，避免在 gate 切换时出现剧烈波动。

### 技术实现

#### 新增代码

在 `ppo_trainer.py` 中添加了 `_update_gate_schedule()` 方法：

```python
def _update_gate_schedule(self, global_step: int) -> None:
    """Apply curriculum gate schedule transitions based on step count."""
    gate_schedule = self.config.get("gate_schedule")
    if not gate_schedule:
        return
    
    # Find the latest gate that should be active
    active_steps = sorted([int(s) for s in gate_schedule.keys() if int(s) <= global_step])
    if not active_steps:
        return
    
    latest_step = active_steps[-1]
    new_gate = float(gate_schedule[latest_step])
    
    # Check if we already applied this gate
    if getattr(self, "_last_gate_schedule_step", None) == latest_step:
        return
    
    # Update the gate in reward_manager
    old_gate = getattr(self.reward_manager, 'target_decoy_gate_logit', 
                      getattr(self.reward_manager, 'target_affinity_gate', None))
    
    # Update both possible gate attributes (for compatibility)
    if hasattr(self.reward_manager, 'target_decoy_gate_logit'):
        self.reward_manager.target_decoy_gate_logit = new_gate
    if hasattr(self.reward_manager, 'target_affinity_gate'):
        self.reward_manager.target_affinity_gate = new_gate
    
    self._last_gate_schedule_step = latest_step
    print(f"\n[Gate Schedule] Step {global_step:,}: gate {old_gate} -> {new_gate}")
```

## 📁 文件位置

- **配置**: `configs/trace73_curriculum_exploration.yaml`
- **启动脚本**: `scripts/launch_trace73.sh`
- **训练日志**: `logs/trace73_curriculum_exploration_train.log`
- **tFold 日志**: `logs/trace73_curriculum_exploration_tfold_amp_server.log`
- **输出目录**: `output/trace73_curriculum_exploration/`

## 🔧 运行状态

### 启动信息

- **起点**: trace70 checkpoint @ step 644,096
- **GPU**: GPU 3 (CUDA_VISIBLE_DEVICES=3)
- **tFold Server**: `/tmp/tfold_server_trace73_curriculum_exploration.sock`
- **初始 Gate**: -2.0 ✅

### 首批 Episodes

```
Episode 1 | Step 644152 | R=-0.341 | A=-0.6653 | InitA=-6.8148 | DeltaA=6.1495
Episode 2 | Step 644152 | R=-0.905 | A=-1.4370 | InitA=-6.5852 | DeltaA=5.1481
Episode 3 | Step 644152 | R=-0.017 | A=-0.3104 | InitA=-4.0819 | DeltaA=3.7715
...
```

**观察**:
- ✅ Gate schedule 已生效（初始 gate = -2.0）
- ✅ 训练正常运行
- ✅ 大部分 episodes 都通过了 gate（TargetSat > 0）
- ✅ DeltaA 很大（3-6），说明从低质量种子有很大提升

## 📈 监控命令

```bash
# 查看训练日志
tail -f logs/trace73_curriculum_exploration_train.log

# 查看最近的 episodes
grep "^Episode" logs/trace73_curriculum_exploration_train.log | tail -20

# 查看 gate 切换
grep "Gate Schedule" logs/trace73_curriculum_exploration_train.log

# 查看 step summaries
grep "^Step" logs/trace73_curriculum_exploration_train.log | tail -10

# Attach 到训练 session
tmux attach -t trace73_train

# Attach 到 tFold server
tmux attach -t tfold_trace73
```

## 🎯 预期结果

### 短期（670K steps，Phase 1 结束）
- Mean R 应该接近或超过 trace70 的 -0.874
- Online pool 应该积累大量高质量 TCR（gate -2.0 很容易通过）

### 中期（700K steps，Phase 2 结束）
- Gate 切换到 -1.5，应该保持 trace70 的水平
- Mean A 应该在 -1.2 左右

### 长期（750K+ steps，Phase 3-4）
- Gate 逐步提高到 -1.0 和 -0.5
- 如果 curriculum 有效，Mean R 应该持续上升
- 目标：Mean R > -0.8，Mean A > -1.0

## 🔍 关键指标

监控以下指标来判断实验是否成功：

1. **Mean R 趋势**: 应该持续上升，不应该平稳或下降
2. **Gate 切换时的稳定性**: 切换 gate 时不应该出现剧烈波动
3. **Online Pool 质量**: Pool 中的 TCR 平均亲和力应该逐步提升
4. **Exploration 效果**: Entropy 应该保持在较高水平（~1.3-1.5）

## 📝 备注

- 如果 Phase 1 效果不好，可以考虑延长 Phase 1 的时间
- 如果 gate 切换时出现剧烈波动，可以考虑更平滑的 schedule
- 如果 exploration 太高导致不稳定，可以降低 entropy_coef

---

**创建时间**: 2026-05-28 14:25  
**状态**: ✅ 运行中  
**预计完成**: Phase 1 @ 670K steps (~26K steps, ~8-10 小时)
