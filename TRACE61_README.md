# trace61: Dynamic Affinity-Based Online Pool

**Created**: 2026-05-25  
**Base**: trace29 (v2_simple_target_gated_decoy) from 580K checkpoint  
**Innovation**: Online pool with dynamic affinity band selection

---

## 🎯 核心思想

在 trace29 的基础上，添加一个**动态的、基于每个 peptide 最近表现的 online pool**：

1. **4 个 affinity bands**:
   - Band 1: `[-4, -2]` - 很差
   - Band 2: `[-2, -1]` - 较差
   - Band 3: `[-1, 0]` - 接近成功
   - Band 4: `[0, 0.6]` - 成功

2. **动态 band 选择**:
   - 跟踪每个 peptide 最近 50 个 episode 的 final affinity
   - 每 50 个 episode 更新一次 band（避免频繁计算）
   - 根据最近 50 个 episode 的**平均 affinity** 选择对应的 band
   - 从选中的 band 中采样 TCR 作为初始种子

3. **50-50 混合策略**:
   - 从 resume 开始就 50% L2 + 50% online pool
   - 即使 pool 为空也保持这个比例（pool 为空时自动 fallback 到 L2）

---

## 📁 文件结构

### 核心文件
- `configs/trace61_dynamic_pool.yaml` - 配置文件
- `tcrppo_v2/data/tcr_pool_trace61_patch.py` - TCRPool 动态 band 选择 patch
- `launch_trace61_dynamic_pool.py` - 启动脚本（应用 patch + 记录 affinity）
- `launch_trace61.sh` - Bash 启动脚本

### 测试文件
- `test_trace61_patch.py` - 测试 patch 是否正常工作

---

## 🚀 启动方法

```bash
cd /share/liuyutian/tcrppo_v2
./launch_trace61.sh
```

或者手动启动：

```bash
# 1. 启动 tFold server
nohup conda run -n tfold --no-capture-output \
    python scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_trace61_dynamic_pool.sock \
    --gpu 1 \
    --use-amp-wrapper \
    --chunk-size 64 \
    --completion-log logs/trace61_dynamic_pool_tfold_completion.log \
    > logs/trace61_dynamic_pool_tfold_amp_server.log 2>&1 &

# 2. 启动训练
conda run -n tcrppo_v2 --no-capture-output \
    python launch_trace61_dynamic_pool.py \
    --config configs/trace61_dynamic_pool.yaml \
    --run_name trace61_dynamic_pool \
    --seed 42 \
    --resume_from output/test62_simple_target_gated_decoy_trace29_simple_target_gated_decoy/checkpoints/milestone_580000.pt \
    --resume_reset_optimizer \
    2>&1 | tee logs/trace61_dynamic_pool_train.log
```

---

## ⚙️ 关键配置

```yaml
# Online pool 配置
online_tcr_pool_enabled: true
online_tcr_pool_start_step: 580000  # 从 resume 立即开始
online_tcr_pool_warmup_steps: 1     # 立即 50-50 混合
online_tcr_pool_max_ratio: 0.5      # 50% online pool
online_tcr_pool_max_per_target: 256 # 每个 peptide 最多存 256 个 TCR
online_tcr_pool_min_affinity: -10.0 # 接受所有（band 会过滤）
online_tcr_pool_min_hamming: 2      # 最小 Hamming 距离

# 动态 band 选择（由 patch 实现）
online_tcr_pool_use_dynamic_bands: true
online_tcr_pool_recent_window: 50   # 最近 50 个 episode
```

---

## 🔧 实现细节

### 1. TCRPool Patch

`tcr_pool_trace61_patch.py` 通过 monkey-patching 添加：

- `recent_affinities`: 每个 peptide 的最近 50 个 final affinity（deque）
- `cached_bands`: 缓存的 band 选择（每 50 个 episode 更新一次）
- `band_update_counters`: 更新计数器
- `record_episode_affinity()`: 记录 episode 结束时的 final affinity
- `get_dynamic_band()`: 获取当前 peptide 的 band（使用缓存）
- `_compute_band()`: 计算 band（基于最近 50 个 episode 的平均值）
- `_online_candidates()`: 重写，使用动态 band 过滤候选 TCR

### 2. 启动脚本

`launch_trace61_dynamic_pool.py` 做了两件事：

1. **启动时**: 导入 patch，应用到 TCRPool
2. **训练时**: Hook `vec_env.step()`，在每个 episode 结束时调用 `record_episode_affinity()`

### 3. Band 选择逻辑

```python
# 每 50 个 episode 更新一次
if episode_count % 50 == 0:
    mean_affinity = np.mean(recent_50_episodes)
    
    # 选择包含 mean_affinity 的 band
    if -4 <= mean_affinity < -2:
        band = band1  # [-4, -2]
    elif -2 <= mean_affinity < -1:
        band = band2  # [-2, -1]
    elif -1 <= mean_affinity < 0:
        band = band3  # [-1, 0]
    elif 0 <= mean_affinity < 0.6:
        band = band4  # [0, 0.6]
    
    # 从 band 中采样 TCR
    candidates = [tcr for tcr in pool if band.min <= tcr.affinity < band.max]
    
    # 如果 band 为空，fallback 到更低的 band
    if not candidates:
        candidates = [tcr for tcr in pool if tcr.affinity < band.min]
```

---

## 📊 预期效果

1. **自适应难度**: 
   - 初期（affinity 很低）：从 band1 [-4,-2] 采样，学习基础改进
   - 中期（affinity 提升）：逐步切换到 band2, band3
   - 后期（affinity 接近 0）：从 band4 [0,0.6] 采样，学习突破

2. **避免 mode collapse**:
   - 50-50 L2 混合保证探索性
   - 每个 peptide 独立跟踪，避免全局平均掩盖个体差异

3. **高效更新**:
   - 每 50 个 episode 更新一次 band，避免频繁计算
   - 使用缓存，采样时直接查表

---

## ✅ 测试结果

运行 `test_trace61_patch.py` 验证：

```
✓ TCRPool patched with dynamic band selection for trace61
  Bands: [-4,-2], [-2,-1], [-1,0], [0,0.6]
  Window: last 50 episodes per peptide

✓ Band selection works correctly:
  - mean=-3.50 → band1 [-4,-2] ✓
  - mean=-1.63 → band2 [-2,-1] ✓
  - mean=-0.63 → band3 [-1,0] ✓
  - mean=0.30 → band4 [0,0.6] ✓
```

---

## 🆚 与 trace29 的区别

| 特性 | trace29 | trace61 |
|------|---------|---------|
| 初始种子 | 纯 L2 | 50% L2 + 50% online pool |
| Online pool | ❌ 无 | ✅ 有（动态 band） |
| Band 选择 | - | 基于最近 50 个 episode 的平均 affinity |
| 更新频率 | - | 每 50 个 episode 更新一次 |
| Resume from | - | trace29 580K checkpoint |

---

## 📝 注意事项

1. **GPU 使用**: tFold server 使用 GPU 1
2. **Checkpoint**: 从 trace29 的 580K checkpoint 开始
3. **Optimizer**: Resume 时重置 optimizer
4. **Cache**: 使用独立的 tfold cache 文件
5. **Socket**: 使用独立的 socket 路径

---

## 🔍 监控指标

训练时关注：

1. **Online pool 大小**: 每个 peptide 的 pool 中有多少 TCR
2. **Band 分布**: 各个 peptide 当前在哪个 band
3. **采样比例**: L2 vs online pool 的实际采样比例
4. **Affinity 进展**: 每个 peptide 的最近 50 个 episode 的平均 affinity

日志中会输出：
```
Online pool stats: {peptide1: 128 TCRs, peptide2: 64 TCRs, ...}
```

---

## 🎯 成功标准

如果 trace61 成功，应该看到：

1. **更快的 affinity 提升**: 相比 trace29，更快达到正值
2. **更高的 A>0 比例**: 超过 trace29 的 1.36%
3. **更稳定的训练**: 减少 affinity 的波动
4. **更好的 max affinity**: 超过 trace29 的 1.30

---

**Good luck! 🚀**
