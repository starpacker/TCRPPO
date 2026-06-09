# SFT 模型改进方案

## 问题诊断

### 1. 根本原因
- **原始 RL 数据污染**: 90.3% 的 TCR 包含 CCC 模式，89.8% 以 CCC 结尾
- **轨迹重建放大偏差**: 训练数据中 97.5% 包含 CCC
- **模型学到错误模式**: 生成的 TCR 中 56% 包含重复模式
- **插入操作偏差**: 训练数据中 70.8% 的插入是 C，模型生成时 51.7% 插入 C

### 2. 性能表现
- **当前 SFT 模型**: 平均亲和力 -5.49
- **原始 RL baseline**: 平均亲和力 -1.172
- **训练数据质量**: 平均亲和力 -0.22
- **差距**: 模型比训练数据差 5.27 个单位

### 3. CCC 模式分析
```
原始 RL 数据:     90.3% 包含 CCC
训练轨迹数据:     97.5% 包含 CCC
SFT 生成数据:     56.0% 包含重复模式
高亲和力样本:     96.7% 包含 CCC
```

**结论**: CCC 是 RL 训练的伪相关模式，不是真实的 TCR-peptide 结合特征。

---

## 改进方案

### 方案 A: 清洗训练数据 + 重新训练 (推荐)

#### A1. 过滤 CCC 模式
```python
# 移除包含 CCC/YYY 等重复模式的 TCR
filtered_tcrs = [
    r for r in records 
    if not any(aa*3 in r['cdr3b'] for aa in 'ACDEFGHIKLMNPQRSTVWY')
]
```

**预期效果**:
- 训练数据从 268,678 → ~27,000 (10%)
- 平均亲和力可能下降 (因为高亲和力样本也多是 CCC)
- 但模型会学到更真实的 TCR 模式

#### A2. 使用真实 TCR 数据库
从公开数据库 (VDJdb, IEDB, McPAS-TCR) 获取真实的 TCR-peptide 配对:
- 这些数据来自实验验证，不含 RL 伪相关
- 数据量可能较小 (几千到几万)
- 需要用 tFold 重新打分

#### A3. 混合策略
- 50% 真实 TCR 数据 (从数据库)
- 50% 过滤后的 RL 数据 (移除 CCC)
- 平衡真实性和数据量

---

### 方案 B: 改进轨迹重建算法

当前问题: 重建算法为了达到目标长度，大量插入 C

#### B1. 限制重复插入
```python
# 在轨迹重建时，禁止连续插入相同氨基酸
if len(actions) > 0 and actions[-1]['op'] == 'INS':
    if actions[-1]['token'] == token:
        # 选择不同的氨基酸
        token = random.choice([aa for aa in AA_LIST if aa != token])
```

#### B2. 使用编辑距离算法
用 Levenshtein 距离或序列比对算法重建轨迹:
- 更自然的编辑序列
- 减少插入操作
- 增加替换和删除

#### B3. 反向轨迹 (从目标到初始)
当前: 初始 → 目标 (容易插入填充)
改进: 目标 → 初始 (更多删除和替换)

---

### 方案 C: 直接使用真实 TCR 作为训练目标

不重建轨迹，直接用真实 TCR:

#### C1. 单步生成
```python
# 输入: peptide embedding
# 输出: 完整 TCR 序列 (一次性生成)
# 模型: Transformer decoder
```

#### C2. 自回归生成
```python
# 输入: peptide + 已生成的 TCR 前缀
# 输出: 下一个氨基酸
# 模型: GPT-style autoregressive
```

#### C3. 扩散模型
```python
# 从噪声开始，逐步去噪生成 TCR
# 类似 ProteinMPNN, RFdiffusion
```

---

### 方案 D: 增强当前 SFT 模型

在现有基础上改进:

#### D1. 添加正则化
```python
# 惩罚重复模式
repetition_penalty = sum(1 for i in range(len(tcr)-2) if tcr[i] == tcr[i+1] == tcr[i+2])
loss += lambda_rep * repetition_penalty

# 惩罚过度插入
insertion_penalty = sum(1 for action in actions if action['op'] == 'INS')
loss += lambda_ins * insertion_penalty
```

#### D2. 对抗训练
```python
# 判别器: 区分真实 TCR vs 生成 TCR
# 生成器: 欺骗判别器
# 目标: 生成更真实的 TCR
```

#### D3. 强化学习微调
```python
# 用 tFold 作为 reward
# 在 SFT 基础上用 PPO 微调
# 目标: 最大化亲和力
```

---

## 推荐实施路径

### 阶段 1: 快速验证 (1-2 天)

**目标**: 验证数据清洗是否有效

1. **过滤 CCC 模式**
   ```bash
   python scripts/filter_ccc_patterns.py \
       --input data/high_quality_tcrs.json \
       --output data/filtered_tcrs.json \
       --remove_repeats 3
   ```

2. **重新准备 SFT 数据**
   ```bash
   python scripts/prepare_high_quality_sft_data.py \
       --input data/filtered_tcrs.json \
       --output data/filtered_sft_trajectories.json \
       --n_samples 5000
   ```

3. **快速训练 (10 epochs)**
   ```bash
   python scripts/train_sft_esm.py \
       --data data/filtered_sft_trajectories.json \
       --epochs 10 \
       --output output/sft_filtered
   ```

4. **评估**
   ```bash
   python scripts/eval_sft_esm.py \
       --checkpoint output/sft_filtered/checkpoint_best.pt \
       --n_tcrs 50
   ```

**预期结果**:
- 如果平均亲和力 > -3.0: 数据清洗有效，继续
- 如果平均亲和力 < -5.0: 数据清洗无效，考虑方案 C

---

### 阶段 2: 扩大数据规模 (3-5 天)

**目标**: 增加训练数据量和多样性

1. **从公开数据库获取真实 TCR**
   - VDJdb: https://vdjdb.cdr3.net/
   - IEDB: https://www.iedb.org/
   - McPAS-TCR: http://friedmanlab.weizmann.ac.il/McPAS-TCR/

2. **用 tFold 打分**
   ```bash
   python scripts/score_public_tcrs.py \
       --input data/vdjdb_tcrs.csv \
       --output data/vdjdb_scored.json \
       --tfold_socket /tmp/tfold_server.sock
   ```

3. **混合数据集**
   ```bash
   python scripts/merge_datasets.py \
       --rl_data data/filtered_tcrs.json \
       --public_data data/vdjdb_scored.json \
       --output data/merged_tcrs.json \
       --ratio 0.5
   ```

4. **重新训练 (50 epochs)**
   ```bash
   python scripts/train_sft_esm.py \
       --data data/merged_sft_trajectories.json \
       --epochs 50 \
       --batch_size 64 \
       --output output/sft_merged
   ```

**预期结果**:
- 平均亲和力 > -1.0
- 成功率 (>0.0) > 10%

---

### 阶段 3: RL 微调 (5-7 天)

**目标**: 在 SFT 基础上用 RL 优化

1. **初始化 RL 策略**
   ```python
   policy = ActorCritic.from_sft_checkpoint(
       'output/sft_merged/checkpoint_best.pt'
   )
   ```

2. **PPO 训练**
   ```bash
   python scripts/train_ppo.py \
       --init_policy output/sft_merged/checkpoint_best.pt \
       --reward tfold \
       --epochs 100 \
       --output output/ppo_from_sft
   ```

3. **评估**
   ```bash
   python scripts/eval_ppo.py \
       --checkpoint output/ppo_from_sft/checkpoint_best.pt \
       --n_tcrs 100
   ```

**预期结果**:
- 平均亲和力 > 0.0
- 成功率 (>0.0) > 30%
- 成功率 (>0.6) > 10%

---

## 备选方案: 如果 SFT 路线失败

### 直接 RL 训练 (无 SFT)
- 从随机策略开始
- 用 tFold 作为 reward
- 用 curriculum learning (简单 peptide → 困难 peptide)
- 预期时间: 2-3 周

### 基于结构的设计
- 用 AlphaFold/tFold 预测 TCR-peptide-MHC 复合物结构
- 用 ProteinMPNN 设计 TCR 序列
- 用 tFold 验证亲和力
- 预期时间: 1-2 周

### 检索增强生成
- 从数据库检索相似 peptide 的 TCR
- 用 LLM 修改 TCR 序列
- 用 tFold 评分和筛选
- 预期时间: 1 周

---

## 资源需求

### 计算资源
- GPU: 1x A100 (40GB) 或 2x V100 (32GB)
- 训练时间: 
  - 阶段 1: 2-3 小时
  - 阶段 2: 12-15 小时
  - 阶段 3: 24-48 小时
- tFold 服务器: 持续运行

### 数据资源
- 过滤后的 RL 数据: ~27,000 TCR
- 公开数据库: ~50,000 TCR (VDJdb + IEDB)
- 总训练数据: ~70,000 TCR

### 人力资源
- 数据清洗和准备: 1-2 天
- 模型训练和调试: 3-5 天
- 评估和分析: 1-2 天
- 总计: 1-2 周

---

## 成功指标

### 最低目标 (可接受)
- 平均亲和力: > -2.0
- 成功率 (>0.0): > 5%
- 无重复模式: > 80%

### 目标 (良好)
- 平均亲和力: > -1.0
- 成功率 (>0.0): > 15%
- 成功率 (>0.6): > 3%

### 理想目标 (优秀)
- 平均亲和力: > 0.0
- 成功率 (>0.0): > 30%
- 成功率 (>0.6): > 10%

---

## 下一步行动

### 立即执行 (今天)
1. ✅ 分析当前模型问题 (已完成)
2. ⏭️ 编写数据过滤脚本
3. ⏭️ 过滤 CCC 模式，生成清洗后的数据集
4. ⏭️ 快速训练验证 (10 epochs)

### 明天
5. 评估过滤后模型的性能
6. 如果有效，扩大数据规模
7. 如果无效，考虑方案 C (直接生成)

### 本周内
8. 完成阶段 1 和阶段 2
9. 开始 RL 微调 (如果 SFT 有效)
10. 准备中期报告

---

## 参考文献

1. **TCR 数据库**
   - VDJdb: Bagaev et al., NAR 2020
   - IEDB: Vita et al., NAR 2019
   - McPAS-TCR: Tickotsky et al., Immunology 2017

2. **序列生成方法**
   - ProteinMPNN: Dauparas et al., Science 2022
   - RFdiffusion: Watson et al., Nature 2023
   - ESM-2: Lin et al., Science 2023

3. **TCR-peptide 预测**
   - tFold: (当前使用的模型)
   - NetTCR: Montemurro et al., eLife 2021
   - TITAN: Weber et al., Nat Commun 2021

---

**创建时间**: 2026-05-31
**状态**: 待实施
**优先级**: 高
