# Dynamic Band Selection 机制详解

## 📊 Band 定义

系统定义了 4 个 affinity bands：

```python
band1: [-4.0, -2.0)  # 最低档
band2: [-2.0, -1.0)  # 中低档
band3: [-1.0,  0.0)  # 中高档
band4: [ 0.0,  0.6]  # 最高档
```

## 🔄 工作流程

### 1. 记录 Final Affinity

每个 episode 结束后：
```python
record_episode_affinity(target="YLQPRTFLL", final_affinity=-1.2)
```

- 记录到该 peptide 的 recent_affinities 队列（最多保留 50 个）
- 计数器 +1

### 2. 更新 Band（每 50 个 episodes）

当计数器达到 50 时：
```python
# 计算最近 50 个 episodes 的平均 affinity
mean_affinity = np.mean(recent_affinities)  # 例如：-1.0

# 找到包含这个 mean 的 band
if -1.0 <= mean_affinity < 0.0:
    cached_band = band3  # [-1.0, 0.0)
```

### 3. 采样时过滤

下次采样时，只从当前 band 中选择：
```python
# 假设 pool 中有这些 TCR：
pool = [
    {"tcr": "A", "affinity": -3.5},  # band1
    {"tcr": "B", "affinity": -2.3},  # band2
    {"tcr": "C", "affinity": -1.5},  # band3 ✓
    {"tcr": "D", "affinity": -0.8},  # band3 ✓
    {"tcr": "E", "affinity": -0.2},  # band3 ✓
    {"tcr": "F", "affinity":  0.3},  # band4
]

# 当前 band = band3 [-1.0, 0.0)
candidates = [C, D, E]  # 只返回 band3 中的 TCR
```

## 🎯 你的问题：如果过去 50 个 episode 的 final affinity 平均是 -1.0

### 场景分析

**假设**：
- 某个 peptide（例如 YLQPRTFLL）
- 最近 50 个 episodes 的 final affinity 平均值 = -1.0

### Step 1: 确定 Band

```python
mean_affinity = -1.0

# 检查每个 band：
# band1: [-4.0, -2.0) → -1.0 不在这里
# band2: [-2.0, -1.0) → -1.0 不在这里（注意是左闭右开）
# band3: [-1.0,  0.0) → -1.0 在这里！✓
# band4: [ 0.0,  0.6] → -1.0 不在这里

selected_band = band3  # [-1.0, 0.0)
```

**注意**：-1.0 正好在 band3 的下界，属于 band3。

### Step 2: 从 Pool 中过滤候选

假设 online pool 中有这些 TCR：

```python
pool = [
    {"tcr": "TCR_A", "affinity": -3.8},  # band1
    {"tcr": "TCR_B", "affinity": -2.5},  # band2
    {"tcr": "TCR_C", "affinity": -1.8},  # band3 ✓
    {"tcr": "TCR_D", "affinity": -1.2},  # band3 ✓
    {"tcr": "TCR_E", "affinity": -0.5},  # band3 ✓
    {"tcr": "TCR_F", "affinity": -0.1},  # band3 ✓
    {"tcr": "TCR_G", "affinity":  0.2},  # band4
    {"tcr": "TCR_H", "affinity":  0.5},  # band4
]

# 过滤：只保留 affinity 在 [-1.0, 0.0) 范围内的
candidates = [
    {"tcr": "TCR_C", "affinity": -1.8},  # ❌ -1.8 < -1.0，不在 band3
    {"tcr": "TCR_D", "affinity": -1.2},  # ❌ -1.2 < -1.0，不在 band3
    {"tcr": "TCR_E", "affinity": -0.5},  # ✓ -1.0 <= -0.5 < 0.0
    {"tcr": "TCR_F", "affinity": -0.1},  # ✓ -1.0 <= -0.1 < 0.0
]
```

**等等！这里有问题！**

如果 mean = -1.0，但 pool 中大部分 TCR 的 affinity 都 < -1.0（比如 -1.8, -1.2），那么 candidates 会很少或为空！

### Step 3: Fallback 机制

如果当前 band 为空，代码会 fallback：

```python
if candidates:
    return candidates  # 如果有候选，直接返回
    
# 如果当前 band 为空，找更低的 band
below = [item for item in pool if item["affinity"] < min_aff]
if below:
    return below  # 返回所有低于当前 band 的 TCR
    
return pool  # 最后 fallback：返回所有
```

**在我们的例子中**：
```python
# band3 的候选只有 2 个：[-0.5, -0.1]
# 如果这 2 个不够，fallback 到 below：
below = [
    {"tcr": "TCR_A", "affinity": -3.8},
    {"tcr": "TCR_B", "affinity": -2.5},
    {"tcr": "TCR_C", "affinity": -1.8},
    {"tcr": "TCR_D", "affinity": -1.2},
]
# 返回所有 affinity < -1.0 的 TCR
```

## 🤔 问题：这个机制合理吗？

### 潜在问题

**问题 1：Band 边界不匹配**

如果 mean = -1.0，选择 band3 [-1.0, 0.0)，但：
- Pool 中大部分 TCR 是 -1.5 到 -1.0（刚被添加的）
- 这些 TCR 都不在 band3 中（因为 < -1.0）
- 结果：band3 为空，fallback 到更低的 band

**问题 2：Fallback 可能采样到很差的 TCR**

如果 fallback 到 below，可能会采样到 affinity = -3.8 的 TCR，这比当前性能（-1.0）差很多！

### 更合理的设计

**方案 1：使用重叠的 bands**
```python
bands = [
    {"min": -4.0, "max": -1.5},  # 包含 -1.8, -1.2
    {"min": -2.0, "max": -0.5},  # 包含 -1.5, -0.8
    {"min": -1.0, "max":  0.5},  # 包含 -0.8, 0.2
    {"min": -0.5, "max":  0.6},  # 包含 0.0, 0.5
]
```

**方案 2：动态调整 band 范围**
```python
# 以 mean 为中心，设置一个窗口
mean = -1.0
window = 1.0
band = {"min": mean - window/2, "max": mean + window/2}
# 结果：[-1.5, -0.5)
```

**方案 3：使用百分位数**
```python
# 采样 pool 中 affinity 在 25%-75% 百分位的 TCR
p25 = np.percentile(pool_affinities, 25)
p75 = np.percentile(pool_affinities, 75)
band = {"min": p25, "max": p75}
```

## 📊 实际例子：Trace62 当前状态

让我检查一下 trace62 的实际情况...

假设：
- 最近 50 个 episodes 的 mean affinity = -1.0
- Pool 中有 100 个 TCR，分布如下：
  - band1 [-4.0, -2.0): 10 个
  - band2 [-2.0, -1.0): 40 个
  - band3 [-1.0,  0.0): 30 个
  - band4 [ 0.0,  0.6]: 20 个

**采样行为**：
1. 选择 band3 [-1.0, 0.0)
2. 候选：30 个 TCR
3. 从这 30 个中随机选择（偏向 top-k）

**如果 band3 为空**：
1. Fallback 到 below（band1 + band2）：50 个 TCR
2. 从这 50 个中选择
3. **问题**：可能选到 affinity = -3.5 的 TCR，比当前性能差！

## 💡 建议

### 当前机制的优点
- ✓ 自动适应性能提升
- ✓ 避免过早使用太好的 TCR
- ✓ 保持 curriculum 压力

### 当前机制的缺点
- ❌ Fallback 可能选到比当前性能差的 TCR
- ❌ Band 边界固定，可能不匹配实际分布
- ❌ 如果 mean 正好在边界（-1.0），可能导致候选很少

### 改进建议

**短期**：修改 fallback 逻辑
```python
# 不要 fallback 到所有 below，而是 fallback 到相邻的 band
if not candidates:
    # 尝试相邻的更低 band
    adjacent_band = get_adjacent_lower_band(current_band)
    candidates = filter_by_band(pool, adjacent_band)
    
    # 如果还是空，再扩大范围
    if not candidates:
        candidates = pool  # 最后才返回所有
```

**长期**：使用动态 band
```python
# 根据 pool 的实际分布动态调整 band
mean = np.mean(recent_affinities)
std = np.std(recent_affinities)
band = {
    "min": mean - std,
    "max": mean + std
}
```

## 🎯 回答你的问题

**如果过去 50 个 episode 的 final affinity 平均是 -1.0，接下来会怎么采样？**

1. **选择 band3** [-1.0, 0.0)
2. **从 pool 中过滤**：只保留 affinity 在 [-1.0, 0.0) 的 TCR
3. **如果有候选**：从中随机选择（偏向 top-k）
4. **如果 band3 为空**：
   - Fallback 到所有 affinity < -1.0 的 TCR
   - 可能选到 -3.5 的 TCR（比当前性能差）
5. **如果 pool 完全为空**：从 L0/L1/L2 采样

**关键点**：
- ✓ 正常情况下，会采样 affinity 在 [-1.0, 0.0) 的 TCR
- ⚠️ 如果这个范围为空，可能会采样到更差的 TCR
- ⚠️ 这可能导致性能波动
