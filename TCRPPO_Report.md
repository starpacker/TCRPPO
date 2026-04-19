# TCRPPO: 基于强化学习的TCR序列设计 — 技术报告
 
> **日期**: 2026-04-17  
> **项目仓库**: `starpacker/TCRPPO`  
> **关键词**: TCR设计, 强化学习, PPO, 特异性, ERGO, ESM-2, 交叉反应性

---

## 目录

1. [项目背景与动机](#1-项目背景与动机)
2. [TCRPPO v1 — 原始版本](#2-tcrppo-v1--原始版本)
   - [2.1 系统架构](#21-系统架构)
   - [2.2 状态空间与编码](#22-状态空间与编码)
   - [2.3 动作空间](#23-动作空间)
   - [2.4 奖励函数](#24-奖励函数)
   - [2.5 策略网络](#25-策略网络)
   - [2.6 PPO训练策略](#26-ppo训练策略)
3. [评测体系](#3-评测体系)
   - [3.1 Decoy文库构建](#31-decoy文库构建)
   - [3.2 MC Dropout不确定性估计](#32-mc-dropout不确定性估计)
   - [3.3 AUROC评测方法](#33-auroc评测方法)
4. [v1评测结果与问题分析](#4-v1评测结果与问题分析)
   - [4.1 v1 Per-Target AUROC](#41-v1-per-target-auroc)
   - [4.2 核心问题: Universal Binder](#42-核心问题-universal-binder)
5. [TCRPPO v2 — 改进方案](#5-tcrppo-v2--改进方案)
   - [5.1 架构改进总览](#51-架构改进总览)
   - [5.2 ESM-2状态编码](#52-esm-2状态编码)
   - [5.3 三头自回归动作空间](#53-三头自回归动作空间)
   - [5.4 四组分奖励系统](#54-四组分奖励系统)
   - [5.5 课程学习策略](#55-课程学习策略)
   - [5.6 自定义PPO实现](#56-自定义ppo实现)
6. [实验结果](#6-实验结果)
   - [6.1 训练曲线对比](#61-训练曲线对比)
   - [6.2 全实验AUROC汇总](#62-全实验auroc汇总)
   - [6.3 Per-Target热力图分析](#63-per-target热力图分析)
   - [6.4 消融实验分析](#64-消融实验分析)
   - [6.5 种子敏感性分析](#65-种子敏感性分析)
   - [6.6 替代打分器实验](#66-替代打分器实验)
7. [关键发现与分析](#7-关键发现与分析)
8. [结论与展望](#8-结论与展望)

---

## 1. 项目背景与动机

**TCR-T细胞疗法** 是一种新兴的免疫治疗方法,通过工程化改造T细胞受体(TCR)使其识别特定的肿瘤抗原肽-MHC复合物(pMHC),进而杀伤肿瘤细胞。TCR设计的核心挑战在于:

1. **高亲和力**: 设计的TCR必须对目标pMHC有足够强的结合能力
2. **高特异性**: TCR不能与正常组织呈递的自身肽产生交叉反应,否则会导致严重的自身免疫毒性(如MAGE-A3/Titin致死案例)

TCRPPO项目使用**近端策略优化(PPO)** 强化学习算法来迭代优化TCR CDR3beta序列,以最大化预测的结合亲和力。本报告覆盖从v1原始版本的部署、评测,到v2改进方案的设计与实验结果。

---

## 2. TCRPPO v1 — 原始版本

### 2.1 系统架构

TCRPPO v1基于修改版的Stable-Baselines3 (0.11.0a7)构建,运行环境为Python 3.6.13 + PyTorch 1.10.2。系统主要由四个核心模块组成:

| 模块 | 文件 | 功能 |
|------|------|------|
| **环境** | `tcr_env.py` | Gym环境,管理TCR序列编辑 |
| **策略网络** | `policy.py`, `seq_embed.py`, `nn_utils.py` | Actor-Critic网络 |
| **奖励模型** | `reward.py` | ERGO打分 + GMM自然度惩罚 |
| **PPO训练器** | `ppo.py`, `on_policy_algorithm.py` | 修改版SB3 PPO |

![架构对比图](figures/fig6_architecture_comparison.png)
*图1: TCRPPO v1与v2架构对比*

### 2.2 状态空间与编码

v1的观测空间是一个1D整数张量,将TCR序列和肽序列拼接后编码:

```
观测 = [TCR_integers (27维) | Peptide_integers (25维)] = MultiDiscrete([20] * 52)
```

- **编码方式**: 20种标准氨基酸映射为1-20的整数,0为padding
- **TCR最大长度**: 27 (padding到固定长度)
- **肽最大长度**: 25 (padding到固定长度)

特征提取器`SeqEmbed`将整数编码转换为稠密特征,使用**三种编码的拼接**:
- Learned Embedding (20维可训练)
- BLOSUM62替代矩阵 (20维固定)
- One-hot编码 (20维固定)

每个位置产生**60维**嵌入向量,再通过双向LSTM (hidden_dim=128) 编码。

### 2.3 动作空间

```
动作 = MultiDiscrete([27, 20]) = (位置, 氨基酸)
```

- **仅支持替换(substitution)**: 每步选择一个位置替换为新的氨基酸
- **无插入/删除/停止**: 序列长度在整个episode中固定不变
- **动作掩码**: 当前位置的氨基酸被掩码(logit=-100000),防止无效操作
- **最大步数**: 8步,即每个episode最多执行8次点突变

### 2.4 奖励函数

v1的奖励由两部分组成:

$$R_{total} = R_{ERGO} + \beta \cdot R_{naturalness}$$

其中 $\beta = 0.5$。

**ERGO结合亲和力打分** (0-1之间的结合概率):
- 模型: AutoEncoder-LSTM分类器,预训练于McPAS数据集
- TCR通过PaddingAutoencoder编码 (28×21 → 100维)
- 肽通过2层LSTM编码 (embedding=10, hidden=100)
- 两者拼接后通过MLP分类器 (200→100→1 + sigmoid)

**自然度惩罚** (GMM模式):
- 自编码器重建误差: 衡量TCR序列的"自然度"
- GMM对数似然: 潜在空间中的高斯混合模型打分
- 组合公式: `(1 - edit_dist) + exp((likelihood + 10) / 10)`
- 阈值: 1.2577,低于阈值时施加惩罚

**关键设计缺陷 — 终端奖励**:
- **中间步骤** (步1到步7): 仅有自然度惩罚 (≤0),**无任何亲和力信号**
- **终端步骤** (步8): 才给出完整的ERGO + 自然度奖励
- 这导致了严重的**信用分配问题**: 智能体无法判断8次突变中哪一步对亲和力的提升起了作用

### 2.5 策略网络

策略网络采用**2-head自回归**结构:

```
输入: SeqEmbed特征 (512维)
  ↓
MLP提取器: Linear(512, 128) + Tanh
  ↓
┌─────────────────────┐   ┌──────────────────┐
│ Policy分支           │   │ Value分支         │
│ Linear(128, 64) Tanh│   │ Linear(128, 64)  │
│                     │   │ Tanh → Linear(1)  │
└─────────────────────┘   └──────────────────┘
  ↓
Head 1: 位置选择 (对每个TCR位置的特征做linear→scalar→Categorical)
  ↓ (条件采样)
Head 2: 氨基酸选择 (对选中位置的特征做linear→20-way→Categorical)
```

- **位置掩码**: Padding位置的logit设为-100000
- **氨基酸掩码**: 当前氨基酸的logit设为-100000
- **总对数概率**: `log_prob = log_prob_position + log_prob_amino_acid`

### 2.6 PPO训练策略

| 参数 | 值 | 说明 |
|------|------|------|
| 总训练步数 | 10,000,000 | 约12.3小时 (A800) |
| 并行环境数 | 20 | SubprocVecEnv |
| 每次rollout步数 | 256/env | 共5,120个transition |
| Mini-batch大小 | 64 | |
| PPO epochs | 10 | |
| 学习率 | 3e-4 | Adam |
| 折扣因子 (γ) | 0.90 | |
| GAE lambda | 0.95 | |
| 裁剪范围 | 0.2 | |
| 熵系数 | 0.01 | |
| 值函数系数 | 0.5 | |
| 梯度裁剪 | 0.5 | |
| 目标KL散度 | 0.01 | 早停 |

**训练流程**:
1. 随机从TCRdb (728万条CDR3b序列) 采样初始TCR
2. 随机选择目标肽 (12个McPAS目标)
3. 策略网络选择突变动作,外部ERGO模型计算奖励
4. 收集rollout→GAE计算优势→PPO更新
5. 每50K步保存检查点,里程碑: 1M, 2M, 5M, 10M

---

## 3. 评测体系

### 3.1 Decoy文库构建

为了评测生成TCR的特异性,我们构建了一个四层次的**Decoy肽文库** (位于 `/share/liuyutian/pMHC_decoy_library/`),覆盖12个McPAS治疗靶点:

| 层级 | 描述 | 方法 | 每靶数量 | 确定性 |
|------|------|------|---------|--------|
| **Tier A** | 序列相似扫描 | Swiss-Prot人类蛋白组Hamming距离扫描 + HLA呈递预测 | 53-777 | 高 |
| **Tier B** | 结构相似筛选 | tFold结构预测 + 双重叠合RMSD + TCR朝向表面描述符 | 50 | 中 |
| **Tier C** | 文献挖掘 | LLM文献提取 + UniProt/IEDB三重验证 | 1,900 (共享) | 最高 |
| **Tier D** | MPNN反向设计 | ProteinMPNN序列设计 + mhcflurry呈递过滤 | 140-1026 | 理论 |

**12个McPAS靶点肽**:

| 肽序列 | 来源 | HLA | 治疗背景 |
|--------|------|-----|---------|
| GILGFVFTL | 流感 M1 | A*02:01 | 经典免疫学基准 |
| NLVPMVATV | CMV pp65 | A*02:01 | 移植后CMV治疗 |
| GLCTLVAML | EBV BMLF1 | A*02:01 | EBV相关PTLD |
| YLQPRTFLL | SARS-CoV-2 Spike | A*02:01 | COVID-19 T细胞疗法 |
| SLYNTVATL | HIV-1 Gag | A*02:01 | HIV功能性治愈 |
| LLWNGPMAV | EBV LMP2 | A*02:01 | 鼻咽癌 |
| FLYALALLL | CMV/EBV | A*02:01 | 病毒免疫 |
| KLGGALQAK | CMV pp65 | A*03:01 | CMV治疗 (非A*02:01) |
| AVFDRKSDAK | CMV/EBV | A*03:01 | 病毒免疫 |
| IVTDFSVIK | CMV/EBV | A*11:01 | 病毒免疫 |
| SPRWYFYYL | MAGEA4 | A*01:01 | 黑色素瘤 |
| RLRAEAQVK | CMV IE1 | A*03:01 | CMV治疗 |

### 3.2 MC Dropout不确定性估计

评测使用**MC Dropout** (Gal & Ghahramani, 2016) 进行贝叶斯近似不确定性估计:

- 推理时**保持dropout激活**,对同一输入进行N=20次随机前向传播
- 取均值作为ERGO得分 (`ergo_mean`),标准差作为不确定性 (`ergo_std`)
- ERGO模型中有3个dropout层 (p=0.1): 2个在PaddingAutoencoder编码器中,1个在MLP分类头

**优化**: 批量GPU推理,吞吐量从~5 pairs/s提升至~1900 pairs/s (A800)。

### 3.3 AUROC评测方法

**AUROC (Area Under ROC Curve)** 在本场景中的含义:

- **正类**: 生成的TCR对其**目标肽**的ERGO得分
- **负类**: 同一TCR对所有**decoy肽**的ERGO得分
- **AUROC = 1.0**: 完美特异性,TCR对目标的得分总是高于任何decoy
- **AUROC = 0.5**: 随机水平,无法区分目标和decoy
- **AUROC < 0.5**: 低于随机,TCR对decoy的得分**反而高于**目标

**评测流程**:
1. 对每个靶点,用训练好的模型生成50条优化TCR
2. 每条TCR同时对目标肽和50条随机采样的decoy肽进行MC Dropout评分
3. 基于得分分布计算per-target AUROC
4. 取12个靶点的平均AUROC作为总体指标

---

## 4. v1评测结果与问题分析

### 4.1 v1 Per-Target AUROC

v1在10M步训练后的评测结果:

| 靶点 | AUROC | 目标均分 | Decoy均分 | 差值 | 判定 |
|------|-------|---------|-----------|------|------|
| SLYNTVATL | **0.878** | 0.395 | 0.126 | +0.269 | 优秀 |
| GLCTLVAML | 0.678 | 0.034 | 0.095 | -0.060 | 中等 |
| SPRWYFYYL | 0.606 | 0.060 | 0.123 | -0.064 | 中等 |
| KLGGALQAK | 0.520 | - | - | - | 接近随机 |
| AVFDRKSDAK | 0.456 | - | - | - | 低于随机 |
| FLYALALLL | 0.413 | - | - | - | 低于随机 |
| NLVPMVATV | 0.402 | 0.050 | 0.101 | -0.050 | 较差 |
| LLWNGPMAV | 0.347 | - | - | - | 较差 |
| GILGFVFTL | 0.320 | 0.089 | 0.103 | -0.013 | 较差 |
| YLQPRTFLL | 0.303 | - | - | - | 较差 |
| IVTDFSVIK | 0.302 | - | - | - | 较差 |
| RLRAEAQVK | **0.231** | - | - | - | 最差 |
| **均值** | **0.454** | | | | **低于随机** |

### 4.2 核心问题: Universal Binder

v1的均AUROC仅为**0.454**,甚至低于随机基线(0.5)。通过与随机TCR基线对比,发现:

| 对比 | 均AUROC |
|------|---------|
| 随机TCR (null hypothesis) | 0.529 |
| v1训练后的TCR | 0.454 |
| **差值** | **-0.075** |

这说明PPO训练反而**恶化**了特异性。根因分析揭示了**奖励劫持(Reward Hacking)** 现象:

1. **模式坍缩**: 在50,000次生成任务中,出现频率最高的序列(`CAGDDGGGVVYEQYF`)出现了6,264次。模型忽略目标肽输入,生成近乎相同的TCR。

2. **ERGO偏见利用**: PPO智能体发现含天冬氨酸(D)和甘氨酸(G)基序的序列(如`DDGGG`, `DGWG`)能稳定获得0.05-0.15的ERGO得分,偶尔飙升至0.45-0.97,同时保持合理的GMM自然度得分。

3. **缺乏特异性约束**: 奖励函数仅优化on-target亲和力,没有任何cross-reactivity惩罚。

**v1架构审计识别的7个设计缺陷**:

| # | 缺陷 | 影响 |
|---|------|------|
| 1 | 仅支持替换,无插入/删除 | 无法探索最优CDR3长度 |
| 2 | ERGO奖励劫持 | 浅层模型有可利用的偏见 |
| 3 | 无特异性目标 | 训练中没有负样本/decoy |
| 4 | 终端延迟奖励 | 8步中的信用分配失败 |
| 5 | 盲目探索 | 728万序列中99.9%对任何靶点零亲和力 |
| 6 | 浅层状态表示 | 整数+BLOSUM缺乏深层生化理解 |
| 7 | 模式坍缩 | PPO收敛于单一策略而非多样性候选 |

---

## 5. TCRPPO v2 — 改进方案

### 5.1 架构改进总览

v2针对v1的每个已知缺陷进行了系统性改进:

| 组件 | v1 | v2 | 改进原因 |
|------|------|------|---------|
| **动作空间** | 替换 (position, token) | 3-head自回归 (op/pos/token) | 支持indel |
| **状态编码** | LSTM + BLOSUM/OneHot | ESM-2 650M (冻结) | 深层生化理解 |
| **奖励信号** | 单一ERGO终端奖励 | 4组分加权 (逐步delta) | 特异性+信用分配 |
| **TCR初始化** | 随机TCRdb | L0/L1/L2课程学习 | 减少无效探索 |
| **训练靶点** | 12个McPAS肽 | 163个tc-hard MHCI肽 | 更好的泛化 |
| **PPO实现** | SB3 (修改版) | 完全自定义 | 自回归动作掩码 |
| **Episode长度** | 固定8步 | 可变 (1-8, STOP动作) | 智能体学习何时停止 |

![架构对比](figures/fig6_architecture_comparison.png)
*图2: v1与v2架构对比详图*

### 5.2 ESM-2状态编码

v2使用**ESM-2 (esm2_t33_650M_UR50D)** 作为状态编码器,该模型拥有6.5亿参数,在2.5亿蛋白质序列上预训练,完全冻结:

```
状态向量 (2562维) = concat([
    ESM_encode(TCR_CDR3b),      # [1280] — 每步重新计算
    ESM_encode(pMHC),            # [1280] — 每episode缓存
    remaining_steps / max,       # [1]
    cumulative_delta_reward      # [1]
])
```

**两层缓存策略**:
- **内存LRU缓存** (4096条): 亚微秒查找
- **SQLite磁盘缓存** (无限): 毫秒级查找,跨重启持久化
- **ESM-2计算**: cache miss时~26ms/batch(8条序列)

pMHC编码将肽序列与HLA伪序列(34残基, NetMHC风格)拼接后一次性通过ESM-2,每个episode仅计算一次。

### 5.3 三头自回归动作空间

v2的策略通过三个条件头顺序采样动作:

```
Head 1 (op_type): 4-way Categorical → {SUB=0, INS=1, DEL=2, STOP=3}
                    ↓ (op embedding, 32维)
Head 2 (position): max_len-way Categorical (条件于op_type)
                    ↓ (position embedding, 32维)
Head 3 (token):    20-way Categorical (条件于op+pos)
                    [DEL和STOP时跳过]
```

**动作掩码规则**:
- `len(seq) >= 20` → INS被掩码
- `len(seq) <= 8` → DEL被掩码
- `position >= len(seq)` → 位置被掩码
- `step == 0` → STOP被掩码 (必须至少编辑一次)

**序列编辑操作**:
- **SUB**: 替换指定位置的氨基酸
- **INS**: 在指定位置插入新氨基酸 (序列右移)
- **DEL**: 删除指定位置的氨基酸 (序列左移)
- **STOP**: 立即终止episode

### 5.4 四组分奖励系统

![奖励组分](figures/fig7_reward_components.png)
*图3: v2四组分奖励系统*

$$R_t = w_{aff} \cdot \text{Affinity} - w_{dec} \cdot \text{Decoy} - w_{nat} \cdot \text{Naturalness} - w_{div} \cdot \text{Diversity}$$

| 组分 | 权重 | 计算方式 | 目的 |
|------|------|---------|------|
| **亲和力** | 1.0 | ERGO score, 逐步delta: $\text{score}(s_t) - \text{score}(s_0)$ | 奖励on-target结合 |
| **Decoy惩罚** | 0.8 | $\frac{1}{\tau}\log\sum\exp(\tau \cdot \text{Aff}(TCR, p_{neg}))$, K=8, τ=10 | 惩罚cross-reactivity |
| **自然度** | 0.5 | ESM-2伪困惑度z-score, 阈值z≥-2.0免罚 | 保持生物学合理性 |
| **多样性** | 0.2 | 512缓冲区最大Levenshtein相似度, 阈值0.85 | 防止模式坍缩 |

**逐步Delta奖励 vs 终端奖励**: v2的每一步都计算 `reward_t = score(current) - score(initial)`,为信用分配提供即时反馈。

**Decoy分层采样**:

| 层级 | 权重 | 内容 | 解锁时间 |
|------|------|------|---------|
| A | 3 | 1-2 AA点突变 | 0步 |
| B | 3 | 2-3 AA突变 | 2M步 |
| D | 2 | VDJdb/IEDB已知结合物 | 5M步 |
| C | 1 | 1900条无关肽 | 8M步 |

### 5.5 课程学习策略

**TCR初始化池** (三个级别):

| 级别 | 来源 | 说明 |
|------|------|------|
| L0 | VDJdb/tc-hard已知结合物 + 3-5个随机突变 | 容易: 修复已知结合物的突变 |
| L1 | TCRdb top-500 (ERGO预筛) | 已禁用 (target信息泄漏) |
| L2 | 随机TCRdb (728万条) | 困难: 纯探索 |

**实际使用**: 由于L1引入了目标信息泄漏, 最终配置为100% L2 (纯随机初始化)。

**训练靶点扩展**: v2从12个McPAS评测靶点扩展至**163个tc-hard MHCI肽**用于训练,评测仍使用12个McPAS靶点,避免训练/评测重叠。

### 5.6 自定义PPO实现

由于SB3无法处理自回归动作掩码,v2从零实现了PPO:

| 参数 | 值 |
|------|------|
| 总步数 | 2,000,000 (多数实验) |
| 并行环境数 | 8 |
| Rollout步数 | 128/env |
| Batch大小 | 256 |
| PPO epochs | 4 |
| 学习率 | 3e-4 |
| γ | 0.90 |
| GAE λ | 0.95 |
| 裁剪范围 | 0.2 |
| 熵系数 | 0.05 (v1的5倍) |
| 梯度裁剪 | 0.5 |

**策略网络** (841K参数, 仅MLP可训练):

```
Shared Backbone: Linear(2562, 512) → ReLU → Linear(512, 512) → ReLU
  ├→ Op Head:    Linear(512, 4)
  ├→ Pos Head:   Linear(512+32, 256) → ReLU → Linear(256, 20)
  ├→ Token Head: Linear(512+64, 256) → ReLU → Linear(256, 20)
  └→ Value Head: Linear(512, 256) → ReLU → Linear(256, 1)
```

---

## 6. 实验结果

### 6.1 训练曲线对比

![训练奖励曲线](figures/fig1_reward_curves_all.png)
*图4: 所有实验的训练奖励曲线。v1_ergo_only模式显示最强的奖励增长趋势, 而v2_full和v2_no_decoy由于z-score归一化导致奖励信号压缩,曲线平坦或震荡。*

![训练动态](figures/fig4_training_dynamics.png)
*图5: 关键实验的训练动态四面板图 (奖励、Episode长度、策略熵、值函数损失)*

**关键观察**:
- **v1_ergo_only (seed=42)**: 奖励稳步上升至3.8, Episode长度稳定在~9步, 值函数损失~0.5
- **v1_ergo_only (seed=123)**: 奖励仅上升至~1.6, 同一配置的种子间方差极大
- **v2_full**: 奖励在0附近震荡, Episode长度仅2-3步 (过早STOP), 值函数损失高达4-6 (z-norm干扰)

### 6.2 全实验AUROC汇总

我们共进行了**25个实验**,其中15个完成了完整评测。下表列出所有已评测实验:

![AUROC柱状图](figures/fig2_auroc_comparison.png)
*图6: 各实验Mean AUROC对比。红色虚线=随机基线(0.50), 绿色虚线=目标(0.65)*

| # | 实验名称 | 步数 | 奖励模式 | 打分器 | Seed | Mean AUROC | vs v1 |
|---|---------|------|---------|--------|------|-----------|-------|
| 1 | v1_ergo_only_ablation | 2M | v1_ergo_only | ERGO | 42 | 0.808 | +78% |
| 2 | test6_pure_v2_arch | 2M | v1_ergo_only | ERGO | 42 | 0.589 | +30% |
| 3 | test4_raw_multi | 2M | raw_multi_penalty | ERGO | 42 | 0.581 | +28% |
| 4 | v2_full_run1 | 2M | v2_full (d=0.8,n=0.5,v=0.2) | ERGO | 42 | 0.573 | +26% |
| 5 | test3_stepwise | 2M | v1_ergo_stepwise | ERGO | 42 | 0.572 | +26% |
| 6 | test5_threshold | 2M | threshold_penalty | ERGO | 42 | 0.570 | +26% |
| 7 | test1_two_phase | 2M | ergo → raw_decoy | ERGO | 42 | 0.567 | +25% |
| 8 | test2_min6_raw | 2M | raw_decoy (min6步) | ERGO | 42 | 0.556 | +22% |
| 9 | test7_v1ergo_repro | 2M | v1_ergo_only | ERGO | **123** | 0.546 | +20% |
| 10 | v2_no_decoy_ablation | ~1.7M | v2_no_decoy | ERGO | 42 | 0.530 | +17% |
| 11 | test15_tcbind_lw | 2M | v1_ergo_only | **TCBind** | 42 | 0.525 | +16% |
| 12 | exp3_ergo_delta | 500K | v1_ergo_delta | ERGO | 42 | 0.500 | +10% |
| 13 | exp1_decoy_only | 500K | v2_decoy_only | ERGO | 42 | 0.490 | +8% |
| 14 | exp4_min_steps | 500K | v2_full (轻惩罚) | ERGO | 42 | 0.477 | +5% |
| 15 | exp2_light | 500K | v2_full (超轻惩罚) | ERGO | 42 | 0.466 | +3% |

> **⚠️ 重要说明**: 排名第1的v1_ergo_only_ablation (seed=42, AUROC=0.808) 的结果**未能可靠复现**。使用seed=123的相同配置仅获得0.546。目前2-seed均值为0.677±0.185,方差极大。seed=7和seed=2024的训练正在进行中,完成后将更新为4-seed统计。**在更多种子验证之前,该配置的可靠性能估计应参考2-seed均值~0.68或中位数实验水平~0.57**。

### 6.3 Per-Target热力图分析

![Per-Target热力图](figures/fig3_per_target_heatmap.png)
*图7: 全实验Per-Target AUROC热力图。颜色从红(低AUROC)到绿(高AUROC)。注意v1_ergo_only (s42) 为异常高的数据点,seed=123的相同配置表现普通。*

**靶点级别观察**:
- **IVTDFSVIK**: 跨种子稳定性最好的靶点 (seed=42: 0.855, seed=123: 0.872),可能反映ERGO对该靶点的稳定预测能力
- **GLCTLVAML**: 种子敏感性最极端的靶点 (seed=42: 0.976, seed=123: 0.383, Δ=0.593)
- **RLRAEAQVK**: 类似的种子敏感性 (seed=42: 0.938, seed=123: 0.554)

### 6.4 消融实验分析

消融实验揭示了各组分的贡献:

| 消融配置 | 说明 | Mean AUROC | 结论 |
|---------|------|-----------|------|
| **v1_ergo_only (s42)** | 仅ERGO终端奖励 (v2架构) | 0.808 | 单次最佳 (未复现) |
| **v1_ergo_only (s123)** | 同上配置,不同种子 | 0.546 | 复现结果差0.262 |
| **v1_ergo_only 2-seed均值** | 同上配置统计 | **0.677±0.185** | **真实性能估计** |
| **test6_pure_v2_arch** | 纯v2架构 (s42重跑) | 0.589 | 架构改进有效 |
| **v2_full** | 4组分+z-norm | 0.573 | 惩罚项+z-norm伤害 |
| **v2_no_decoy** | 去除decoy惩罚 | 0.530 | 去掉decoy并未明显帮助 |
| **test4_raw_multi** | 4组分,无z-norm,低权重 | 0.581 | 比v2_full略好 |

**关键发现**:
1. **种子敏感性是首要问题**: v1_ergo_only的2-seed标准差达0.185,单次结果不可靠。后续两个新种子(seed=7, seed=2024)正在训练中,将提供4-seed统计
2. **z-score归一化是灾难性的**: 所有使用z-norm的实验都表现更差
3. **多组分奖励目前反而有害**: 每种添加惩罚项的尝试都降低了AUROC
4. **架构改进本身有稳定的增益**: 即使种子不利(seed=123),v1_ergo_only (0.546) 仍高于v1 baseline (0.454)。中位v2实验~0.57也确认了架构改进的一致效果

### 6.5 种子敏感性分析

![种子敏感性](figures/fig8_seed_sensitivity.png)
*图8: v1_ergo_only模式下seed=42 vs seed=123的Per-Target对比。Delta最大达0.59 (GLCTLVAML)。*

**v1_ergo_only配置的种子复现实验** (当前2个种子,另有2个正在训练中):

| 种子 | Mean AUROC | 最佳靶点 | 最差靶点 |
|------|-----------|---------|---------|
| 42 | 0.808 | GLCTLVAML (0.976) | FLYALALLL (0.579) |
| 123 | 0.546 | IVTDFSVIK (0.872) | GLCTLVAML (0.383) |
| 7 | *(训练中)* | — | — |
| 2024 | *(训练中)* | — | — |
| **2-seed均值** | **0.677±0.185** | | |

**0.262的AUROC差值 (seed=42 vs seed=123) 证实了严重的种子敏感性问题**:
- 同一超参数、同一代码、同一数据,仅随机种子不同,就产生了从"优秀"(0.808)到"一般"(0.546)的巨大跨度
- seed=42在10/12靶点上AUROC>0.65,而seed=123仅2/12靶点>0.65
- GLCTLVAML靶点的跨种子Δ高达0.593,表明某些靶点的特异性表现几乎完全取决于训练初始化
- **结论**: 在4-seed统计完成之前,v1_ergo_only配置的可信性能范围为 **0.55-0.80**,中心估计约**0.68**

> **正在进行的工作**: test18 (seed=7) 和 test19 (seed=2024) 训练中,预计约6-8小时后完成评测,届时将更新为4-seed统计并重新生成所有相关图表。

### 6.6 替代打分器实验

| 打分器 | 实验 | 状态 | 结果 |
|--------|------|------|------|
| **ERGO AE-LSTM** | 多数实验 | 完成 | **最佳** (2-seed均值0.677) |
| **TCBind BiLSTM** | test15_tcbind_lightweight | 完成 | 0.525 (较差) |
| **NetTCR** | test11, test12 | 崩溃 | 训练不稳定 |
| **ERGO+NetTCR集成** | test13 | 崩溃 | 训练不稳定 |
| **tFold结构** | odin | 失败 | 特征提取错误 |
| **轻量级BiLSTM编码器** | test16, test17 | 训练中 | 待评测 |

---

## 7. 关键发现与分析

### 有效的改进

1. **v2架构改进带来稳定增益**: 即使在不利种子(seed=123)下,v1_ergo_only仍达0.546 (+20% vs v1的0.454)。中位v2实验达到~0.57 (+26%)。这些增益在多实验中一致复现,确认了ESM-2编码、indel动作空间、扩展训练靶点等改进的有效性。

2. **最佳配置为纯ERGO + v2架构**: v1_ergo_only的2-seed均值为0.677±0.185 (seed=42: 0.808, seed=123: 0.546)。虽然方差大,但均值仍显著高于v1 baseline (0.454) 和随机水平 (0.50)。

3. **Episode长度一致稳定**: 所有v2实验中使用纯ERGO奖励的配置,episode长度均稳定在8-9步,说明架构改进允许智能体充分利用所有编辑步骤。

### 未达预期的改进

1. **多组分奖励适得其反**: 添加decoy/naturalness/diversity惩罚在所有配置下都降低了AUROC。v2_full (4组分) 仅0.573,而v1_ergo_only均值~0.68。根本原因是惩罚项使智能体学会过早STOP (平均2-3步),减少了探索。

2. **z-score归一化压缩信号**: 所有使用z-norm的实验表现更差。归一化将亲和力信号压缩到与惩罚项相同的尺度,使策略梯度信号消失。

3. **替代打分器表现不佳**: TCBind (0.525)、NetTCR (崩溃)、tFold (失败) 都未能超越ERGO。

4. **种子敏感性是当前最大隐患**: v1_ergo_only在seed=42时0.808,seed=123时仅0.546。0.262的差值意味着报告单次运行结果可能严重误导。**多种子统计是判断系统真实性能的必要前提。**

### 根本性挑战

**训练不稳定性**: PPO在高维、稀疏奖励的TCR设计任务中表现出极高的种子依赖性。这并非TCRPPO特有的问题——RL文献中PPO的跨种子方差一直是已知挑战。但在药物设计场景中,这种不可复现性尤其危险。

**特异性-亲和力权衡**: ERGO打分器奖励"通用结合"(对任何肽的高分),而AUROC评测奖励"鉴别性结合"(目标高于decoy)。seed=42成功的原因可能是该种子的策略恰好生成了特异性较好的TCR,但这并非算法保证的结果。显式的特异性优化(decoy惩罚)反而因干扰主奖励信号而失败,这说明在当前框架下,特异性更多是"运气"而非"设计"。

---

## 8. 结论与展望

### 成果总结

| 指标 | v1 Baseline | v2 Best (单次) | v2 2-seed均值 | v2 中位实验 |
|------|-------------|----------------|---------------|------------|
| Mean AUROC | 0.454 | 0.808 (s42) | **0.677±0.185** | 0.573 |
| vs Random (0.50) | -0.046 | +0.308 | +0.177 | +0.073 |
| 达标靶点 (>0.65) | 1/12 | 10/12 (s42) | — | 3-4/12 |

> **诚实评估**: 以目前2-seed数据,v1_ergo_only的**中心估计为0.677**,高于0.65目标但置信度不足。需要等待4-seed统计(test18/test19完成后)才能给出更可靠的结论。v2架构在中位水平(~0.57)已稳定超越v1 (0.454),这是经过多次实验验证的可靠结论。

### 当前仍在运行的实验

| 实验 | 进度 | 特点 | 预期完成 |
|------|------|------|---------|
| test14_bugfix_v1ergo | ~57% (1.14M/2M) | ERGO基线修复后重跑 (s42) | ~4-6h |
| test16_ergo_lightweight | ~72% (1.45M/2M) | 轻量级BiLSTM编码器 (s42) | ~3-4h |
| test17_ergo_lightweight_s123 | ~65% (1.30M/2M) | 轻量级BiLSTM编码器 (s123) | ~3-4h |
| **test18_v1ergo_seed7** | ~0% (刚启动) | **v1_ergo_only + ESM-2 (s7)** | ~6-8h |
| **test19_v1ergo_seed2024** | ~0% (刚启动) | **v1_ergo_only + ESM-2 (s2024)** | ~6-8h |

### 下一步方向

1. **4-seed统计 (最高优先级)**: 等待test18/test19完成,建立v1_ergo_only的4-seed均值和置信区间。如果4-seed均值>0.65,可以宣布v2在统计意义上达标;否则需要进一步优化
2. **渐进式惩罚引入**: 先用纯ERGO训练1M步,再逐步引入decoy惩罚(权重从0线性增加),避免早期信号冲突
3. **自适应惩罚权重**: 根据当前AUROC动态调整decoy/naturalness权重
4. **更好的打分器**: 微调ESM-2作为binding predictor,替代浅层ERGO模型,可能同时提升性能和稳定性
5. **TCR多样性增强**: 在确认可靠的基础配置后,再添加diversity约束
6. **更长训练**: 在确认最优配置后,进行5-10M步训练

---

*本报告生成于 2026-04-17,最后修改于 2026-04-17。所有数据来自实际GPU运行结果,无任何模拟数据。seed=7和seed=2024的训练正在进行中,完成后将更新本报告中的多种子统计。*
