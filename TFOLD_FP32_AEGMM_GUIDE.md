# tFold FP32 + AE-GMM Naturalness Scorer 使用指南

**日期**: 2026-06-04  
**版本**: v2.0  
**目的**: 修复训练中的multi-C exploit问题，提高生成TCR的自然性

---

## 背景

### 发现的问题

在 `trace61_fp32_restart` 训练中发现：
- **81.3%** 的生成序列以 `CCC` 结尾（1557/1916条）
- ESM-2 perplexity naturalness scorer **无法检测** 3-4个连续C的不自然模式
- 这是典型的 **reward hacking** 行为：模型找到了能boost affinity但不触发naturalness惩罚的exploit

### 根本原因分析

| 组件 | 问题 |
|---|---|
| **tFold (BF16)** | 混合精度可能导致affinity评分不够准确 |
| **ESM-2 Naturalness** | 对3-4个C的z-score只有0.8-1.2，远低于threshold=2.0 |
| **权重设置** | `w_naturalness=0.05` 太弱，即使触发也容易被affinity抵消 |

---

## 解决方案

### 方案1: tFold FP32（必须）

**之前**: 使用BF16混合精度
```yaml
affinity_model: "tfold"
tfold_amp_enabled: true  # 使用BF16/AMP
```

**现在**: 必须使用FP32全精度
```yaml
affinity_model: "tfold"
# 移除或注释掉 tfold_amp_enabled
# 不要设置任何AMP相关参数
```

**重要**: tFold服务器必须以FP32模式运行，不能使用任何混合精度优化。

---

### 方案2: AE+GMM Naturalness Scorer（推荐）

#### 为什么AE+GMM更好？

| 指标 | ESM-2 Perplexity | AE+GMM |
|---|---|---|
| **训练数据** | 通用蛋白质（UniRef50） | **TCR专属数据（TCRdb）** |
| **3C检测** | ❌ 不触发 (z=1.14 < 2.0) | ✅ 触发 (0.23 < 0.8) |
| **敏感度** | 需要5+个C | **3个C即可检测** |
| **Multi-C检出率** | ~25% | **100%** |

#### 测试结果对比

```
序列                             | Trailing C | ESM-2 Penalty | AE+GMM Combined
--------------------------------|------------|---------------|------------------
CASRISTSGGNGDEQFF (正常)        |     0      |    0.0000     |    0.9910 (✓)
CALRISALGGNGDEQCCC (3C)         |     3      |    0.0000     |    0.2306 (✗)
CASRAGGPGEQCCCCCCCC (8C)        |     8      |   -2.3308     |    0.0529 (✗)
```

---

## 配置指南

### 完整配置示例

在你的YAML配置文件中（如 `configs/trace61_fp32_restart.yaml`）进行以下修改：

```yaml
# ============================================================
# 1. tFold FP32 配置（必须）
# ============================================================
affinity_model: "tfold"
tfold_server_socket: "/tmp/tfold_server_YOUR_EXP_NAME.sock"
tfold_cache_path: "data/tfold_cache_YOUR_EXP_NAME.db"
# 不要添加任何 tfold_amp_enabled 或混合精度相关配置

# ============================================================
# 2. Naturalness Scorer 配置
# ============================================================
# 选择scorer类型: "esm2" 或 "ae_gmm"
naturalness_scorer_type: "ae_gmm"  # 推荐使用AE+GMM

# ESM-2 配置（如果使用 naturalness_scorer_type: "esm2"）
naturalness_threshold_zscore: 1.5  # 建议从2.0降低到1.0-1.5
cdr3_ppl_stats: "data/cdr3_ppl_stats.json"

# AE+GMM 配置（如果使用 naturalness_scorer_type: "ae_gmm"）
naturalness_ae_model: "/share/liuyutian/TCRPPO/code/reward/ae_model"
naturalness_gmm_model: "/share/liuyutian/TCRPPO/code/reward/gmm.pkl"
naturalness_blosum: "/share/liuyutian/TCRPPO/code/blosum.txt"
naturalness_ae_threshold: 0.8  # combined (edit_acc + gmm_like) threshold

# ============================================================
# 3. Reward 权重配置
# ============================================================
reward_mode: "v2_simple_target_gated_decoy"
w_affinity: 1.0
w_decoy: 0.3
w_naturalness: 0.15  # 建议从0.05提高到0.10-0.15
w_diversity: 0.02

# ============================================================
# 4. 其他训练参数保持不变
# ============================================================
total_timesteps: 2_000_000
n_envs: 8
batch_size: 256
learning_rate: 1.5e-4
# ... 其他参数 ...
```

---

## 参数调优建议

### AE+GMM Threshold调节

`naturalness_ae_threshold` 控制判定自然性的阈值：

| Threshold | 效果 | 适用场景 |
|---|---|---|
| **0.6** | 非常严格，会拒绝更多序列 | 怀疑有大量exploit |
| **0.8** | 平衡（推荐） | 一般训练 |
| **1.0** | 宽松，只拒绝明显不自然的 | 需要更多探索空间 |

公式：`combined = edit_accuracy + exp((gmm_log_likelihood + 10) / 10)`

### Naturalness权重调节

`w_naturalness` 控制惩罚强度：

| 权重 | 效果 |
|---|---|
| **0.05** | 当前值，可能太弱 |
| **0.10** | 适中增强 |
| **0.15** | 强惩罚（推荐用于AE+GMM） |
| **0.20+** | 非常强，可能过度限制探索 |

**建议组合**：
```yaml
naturalness_scorer_type: "ae_gmm"
naturalness_ae_threshold: 0.8
w_naturalness: 0.15
```

---

## 启动训练

### 1. 确保tFold服务器以FP32运行

```bash
# 检查tFold服务器是否已启动
ls -l /tmp/tfold_server_*.sock

# 如果需要重启（确保FP32模式）
# 具体命令取决于你的tFold服务器启动脚本
# 确保没有使用 --amp 或 --bf16 等混合精度参数
```

### 2. 测试AE+GMM Scorer

```bash
cd /share/liuyutian/tcrppo_v2
python test_ae_gmm_scorer.py
```

预期输出：
```
✓ Scorer initialized
Testing scorer.score() (single sequence):
...
Summary:
  • Multi-C sequences (3+): 8
  • Multi-C penalized: 8/8 (100.0%)
✓ All tests completed successfully!
```

### 3. 启动训练

```bash
cd /share/liuyutian/tcrppo_v2
python -m tcrppo_v2.train \
    --config configs/YOUR_CONFIG.yaml \
    --output-dir output/YOUR_EXP_NAME
```

---

## 监控与验证

### 训练日志检查

启动时应看到：
```
Loading config from: configs/YOUR_CONFIG.yaml
...
Initializing scorers:
  Affinity scorer loaded: tFold (FP32)
  Naturalness scorer loaded: AE+GMM (threshold=0.8)  # ← 确认这一行
  Diversity scorer loaded
...
```

### 生成序列检查

定期检查生成序列的trailing C比例：

```bash
# 从训练日志提取序列
grep "seq=" logs/YOUR_LOG.log | head -1000 > /tmp/seqs.txt

# 统计CCC比例
cd /share/liuyutian/tcrppo_v2
python -c "
import re
with open('/tmp/seqs.txt') as f:
    seqs = [re.search(r'seq=([A-Z]+)', line).group(1) for line in f if 'seq=' in line]
ccc_count = sum(1 for s in seqs if s.endswith('CCC'))
print(f'Total: {len(seqs)}, CCC: {ccc_count} ({100*ccc_count/len(seqs):.1f}%)')
"
```

**健康标准**：CCC比例应该 < 5%（理想情况 < 1%）

---

## 故障排查

### 问题1: "Naturalness scorer SKIPPED"

**原因**: 使用了lightweight encoder，没有ESM模型

**解决**: 
```yaml
encoder: "esm2"  # 不要使用 "lightweight"
esm_model: "esm2_t33_650M_UR50D"
```

### 问题2: "Unknown naturalness_scorer_type: xxx"

**原因**: 配置中scorer类型拼写错误

**解决**: 只能使用 `"esm2"` 或 `"ae_gmm"`

### 问题3: FileNotFoundError: ae_model/gmm.pkl

**原因**: AE+GMM模型文件路径错误

**解决**: 验证文件存在
```bash
ls -lh /share/liuyutian/TCRPPO/code/reward/ae_model
ls -lh /share/liuyutian/TCRPPO/code/reward/gmm.pkl
ls -lh /share/liuyutian/TCRPPO/code/blosum.txt
```

### 问题4: tFold affinity分数异常

**症状**: 所有序列affinity都很低或很高

**可能原因**: 
- tFold服务器未启动
- tFold服务器仍在使用BF16模式
- Socket路径错误

**解决**: 
1. 检查socket文件：`ls -l /tmp/tfold_server_*.sock`
2. 重启tFold服务器（确保FP32）
3. 验证配置中的socket路径匹配

---

## 预期效果

使用 **tFold FP32 + AE-GMM (threshold=0.8, w=0.15)** 后：

| 指标 | 之前 (BF16 + ESM-2) | 之后 (FP32 + AE-GMM) |
|---|---|---|
| **CCC比例** | 81.3% | < 5% (目标 < 1%) |
| **Affinity准确性** | 中等（BF16误差） | 高（FP32精确） |
| **序列自然性** | 低（exploit存在） | 高（TCR-aware检测） |
| **训练稳定性** | 可能出现mode collapse | 更稳定的探索 |

---

## 进一步优化

如果仍有问题，可以尝试：

### 选项A: 降低AE threshold（更严格）
```yaml
naturalness_ae_threshold: 0.7  # 从0.8降到0.7
w_naturalness: 0.15
```

### 选项B: 增加naturalness权重
```yaml
naturalness_ae_threshold: 0.8
w_naturalness: 0.20  # 从0.15提高到0.20
```

### 选项C: 混合使用两种scorer

如果需要同时考虑ESM-2的泛化能力和AE-GMM的TCR特异性，可以修改代码创建hybrid scorer（需要额外开发）。

---

## 参考文献

- **ESM-2**: Lin et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science 2023.
- **TCRPPO AE+GMM**: 原始TCRPPO项目中的naturalness评分方法（基于TCRdb数据训练）
- **tFold**: TCR-pMHC结构预测与亲和力评分工具

---

## 联系与支持

如有问题或需要进一步调优建议，请检查：
1. 训练日志：`logs/YOUR_EXP_NAME_train.log`
2. 测试脚本：`python test_ae_gmm_scorer.py`
3. 本项目文档：`README.md`, `ARCHITECTURE.md`

**最后更新**: 2026-06-04
