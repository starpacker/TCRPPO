# TCRPPO v2 Code Guide

**目标读者**: 熟悉 Python 和 TCR/pMHC 领域的开发者  
**Last Updated**: 2026-04-15

---

## 目录

1. [整体架构与数据流](#1-整体架构与数据流)
2. [常量与编码 — `utils/constants.py`](#2-常量与编码)
3. [ESM-2 缓存层 — `utils/esm_cache.py`](#3-esm-2-缓存层)
4. [数据管道 — `data/pmhc_loader.py` + `data/tcr_pool.py`](#4-数据管道)
5. [打分器 — `scorers/`](#5-打分器)
6. [奖励管理器 — `reward_manager.py`](#6-奖励管理器)
7. [环境 — `env.py`](#7-环境)
8. [策略网络 — `policy.py`](#8-策略网络)
9. [PPO 训练器 — `ppo_trainer.py`](#9-ppo-训练器)
10. [评估 — `test_tcrs.py`](#10-评估)
11. [配置系统 — `configs/default.yaml`](#11-配置系统)
12. [关键设计决策与已知问题](#12-关键设计决策与已知问题)

---

## 1. 整体架构与数据流

### 1.1 系统总览

```
┌─────────────────────────────────────────────────────────┐
│                    ppo_trainer.py                        │
│  main loop: collect rollout → compute GAE → PPO update  │
└────────────┬────────────────────────┬───────────────────┘
             │                        │
     ┌───────▼───────┐       ┌───────▼───────┐
     │  VecTCREditEnv │       │  ActorCritic  │
     │  (n_envs=8)    │       │  (policy.py)  │
     └───┬───┬───┬───┘       └───────────────┘
         │   │   │
    ┌────┘   │   └────┐
    ▼        ▼        ▼
ESMCache  RewardMgr  TCRPool
 (state)  (reward)   (init TCR)
    │        │
    │   ┌────┴────┐
    │   ▼    ▼    ▼
    │  ERGO Decoy Nat/Div
    │  Scorer Scorer Scorer
    │
    ▼
 PMHCLoader
 (target peptides)
```

### 1.2 单个 timestep 的完整数据流

以 `VecTCREditEnv.step()` (env.py:406) 为入口，一个 timestep 经过以下阶段：

```
1. PPO 收集阶段 (ppo_trainer.py:462-511)
   │
   ├─ obs → GPU tensor (ppo_trainer.py:471)
   ├─ policy.forward(obs, masks) → sample (op, pos, tok) (policy.py:109-144)
   ├─ policy.forward(obs, masks, actions) → log_prob (policy.py:146-194)
   │
   └─ VecEnv.step(actions) — 三阶段批处理 (env.py:406-472)
       │
       ├─ Phase 1: 逐 env 执行 action (env.py:424-435)
       │   对 done 的 env: _reset_internal() 自动重置
       │   对 active 的 env: _step_action_only() 修改 TCR 序列
       │
       ├─ Phase 2: 批量奖励计算 (env.py:438-452)
       │   reward_manager.compute_reward_batch()
       │   └─ 内部: 批量 ERGO forward → 逐样本 decoy/nat/div
       │
       └─ Phase 3: 批量 ESM 编码 (env.py:454-472)
           esm_cache.encode_tcr_batch() → 新的 obs

2. GAE 计算 (ppo_trainer.py:514-519)

3. PPO 更新 (ppo_trainer.py:522-567)
   n_epochs 轮 × minibatch:
   policy._evaluate() → log_prob, entropy, value
   clipped ratio objective + MSE value loss + entropy bonus
```

### 1.3 Observation 构成

观测向量维度 = 2562，构成如下（env.py:66）：

```
obs = [ tcr_emb(1280) | pmhc_emb(1280) | remaining_steps(1) | cumulative_delta(1) ]
        ─────┬─────     ──────┬──────     ────────┬────────     ─────────┬──────────
        ESM-2对当前      ESM-2对目标        (max_steps - step_count)   累计奖励
        CDR3β的均值      peptide+HLA         / max_steps
        池化嵌入         伪序列的嵌入        归一化到[0,1]
```

- `tcr_emb`: 每步重新计算（序列变了），有 LRU 缓存
- `pmhc_emb`: 整个 episode 不变，永久缓存
- `remaining_steps`: 线性衰减，让 policy 知道剩余预算
- `cumulative_delta`: 当前 episode 累积奖励，帮助 value estimation

---

## 2. 常量与编码

**文件**: `tcrppo_v2/utils/constants.py` (53 行)

### 2.1 氨基酸字母表

```python
AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")  # 标准20种, line 7
AA_TO_IDX / IDX_TO_AA  # 双向映射, line 10-11
```

ERGO 模型使用不同的编码方案（line 14-19）：
- `ERGO_TCR_ATOX`: 21-class (20 AA + "X" 终止符), 0-indexed
- `ERGO_PEP_ATOX`: 21-class ("PAD" + 20 AA), "PAD"=0

**注意**: policy 输出的 token ∈ [0,19] 使用 `IDX_TO_AA` 映射，而 ERGO 评分时内部使用 `ERGO_TCR_ATOX`，两套编码互不干扰。

### 2.2 核心常量

| 常量 | 值 | 用途 | 行号 |
|------|-----|------|------|
| `MAX_TCR_LEN` | 27 | CDR3β 最长长度，action masking 上界 | 22 |
| `MIN_TCR_LEN` | 8 | CDR3β 最短长度，DEL masking 下界 | 23 |
| `MAX_STEPS_PER_EPISODE` | 8 | 每个 episode 最多编辑步数 | 29 |
| `OP_SUB/INS/DEL/STOP` | 0/1/2/3 | 操作类型编码 | 32-35 |
| `NUM_OPS` | 4 | 操作类型数量 | 36 |
| `ERGO_MAX_LEN` | 28 | ERGO 模型填充长度 | 25 |

### 2.3 路径推导逻辑

路径推导（line 41-52）：

```
constants.py 位于: tcrppo_v2/tcrppo_v2/utils/constants.py
_PACKAGE_DIR = tcrppo_v2/tcrppo_v2/   (Python 包目录)
PROJECT_ROOT = tcrppo_v2/              (项目根目录)
ERGO_DIR     = tcrppo_v2/tcrppo_v2/ERGO/
ERGO_AE_FILE = tcrppo_v2/tcrppo_v2/ERGO/TCR_Autoencoder/tcr_ae_dim_100.pt
ERGO_MODEL_DIR = tcrppo_v2/tcrppo_v2/ERGO/models/
```

外部数据路径硬编码指向共享存储（line 49-52）：
- `DECOY_LIBRARY_PATH` = `/share/liuyutian/pMHC_decoy_library`
- `TCRDB_PATH` = `/share/liuyutian/TCRPPO/data/tcrdb`

---

## 3. ESM-2 缓存层

**文件**: `tcrppo_v2/utils/esm_cache.py` (206 行)

### 3.1 初始化

`ESMCache.__init__()` (line 18-56):
- 加载 `esm2_t33_650M_UR50D` — 650M 参数，33 层 Transformer
- 输出维度: `embed_dim = 1280`
- 默认冻结所有参数（`frozen=True`）
- 两个独立缓存：
  - `_pmhc_cache: Dict[str, Tensor]` — 永久缓存，key=序列字符串
  - `_tcr_cache: OrderedDict` — LRU 缓存，默认 4096 条目

### 3.2 编码流程

**单序列编码** `encode_sequence()` (line 63-84):

```python
# 1. batch_converter 将序列转为 token tensor
data = [("seq", sequence)]
_, _, tokens = self.batch_converter(data)  # shape: [1, seq_len+2]
                                            # +2 是 BOS 和 EOS token

# 2. ESM-2 前向传播，取第33层表示
results = self.model(tokens, repr_layers=[33])
token_repr = results["representations"][33]  # [1, seq_len+2, 1280]

# 3. 均值池化（排除 BOS 和 EOS）
embedding = token_repr[0, 1:seq_len+1, :].mean(dim=0)  # [1280]
```

**关键细节**: ESM-2 的 batch_converter 会在序列前后加 BOS/EOS special token，所以 `token_repr` 的 shape 是 `[B, seq_len+2, 1280]`。取 `[1:seq_len+1]` 才是实际序列位置的表示。

### 3.3 缓存策略

**TCR 缓存** `encode_tcr()` (line 114-134):
- 命中: `move_to_end()` 更新 LRU 位置，返回缓存结果
- 未命中: 编码 → LRU 满则淘汰最老（`popitem(last=False)`）→ 存入缓存
- 为什么用 LRU: TCR 序列每步都在变，但同一 episode 内的编辑通常只改 1 个位置。大量 TCR 会被反复查询（特别是 VecEnv 中同时活跃的多条序列）

**pMHC 缓存** `encode_pmhc()` (line 136-150):
- 永久缓存，没有淘汰策略
- 因为训练时只有 163 个 target peptide（train mode），总共也就 163 条 pMHC
- 每条 pMHC 字符串 = peptide + HLA pseudosequence（如 `"GILGFVFTL" + "YFAMYQENAAHTLRWEPYSEGAEYLERTCEW"`)

### 3.4 批量编码与部分缓存命中

`encode_tcr_batch()` (line 152-186) — **VecEnv 的性能关键**:

```python
# 1. 分离已缓存和未缓存的序列
for i, seq in enumerate(tcr_seqs):
    if seq in self._tcr_cache:
        results.append((i, self._tcr_cache[seq]))  # 命中
    else:
        uncached_indices.append(i)                  # 未命中

# 2. 只对未命中的序列做一次批量 ESM 前向
if uncached_seqs:
    new_embeddings = self.encode_sequences_batch(uncached_seqs)

# 3. 按原始顺序重新排列
results.sort(key=lambda x: x[0])
return torch.stack([r[1] for r in results])
```

这意味着如果 8 个 env 中有 5 个的 TCR 已缓存，ESM-2 只需对 3 个序列做前向传播。对 SUB 操作特别有效——如果两个 env 碰巧编辑出相同序列，只算一次。

---

## 4. 数据管道

### 4.1 PMHCLoader — 目标管理

**文件**: `tcrppo_v2/data/pmhc_loader.py` (195 行)

#### 核心数据结构

12 个 McPAS 评估目标 (line 31-44):

```python
EVAL_TARGETS = {
    "GILGFVFTL":  "HLA-A*02:01",   # Influenza M1
    "NLVPMVATV":  "HLA-A*02:01",   # CMV pp65
    "GLCTLVAML":  "HLA-A*02:01",   # EBV BMLF1
    ...
    "SPRWYFYYL":  "HLA-B*07:02",   # CMV pp65 (唯一的 B*07:02)
    "RLRAEAQVK":  "HLA-A*03:01",   # CMV IE1
}
```

HLA 伪序列（line 20-28）—— 来自 NetMHCpan 4.1，是 HLA 分子与肽段接触位点的 ~30 氨基酸摘要表示：

```python
HLA_PSEUDOSEQUENCES = {
    "HLA-A*02:01": "YFAMYQENAAHTLRWEPYSEGAEYLERTCEW",  # 30字符
    "HLA-A*03:01": "YFAMYQENDAHTLRWEAYSEGAEYLERTCEW",
    ...
}
```

#### 两种模式

`PMHCLoader.__init__()` (line 70-119):

- **eval 模式**: 只加载 12 个 `EVAL_TARGETS`
- **train 模式**: 加载 163 个 tc-hard MHCI targets（从 `data/tc_hard_targets.json`），包含 eval targets

train 模式的 163 个 target 是从 tc-hard 数据集提取的——每个 target 至少有 10 个已知 CDR3β 序列。这比 eval 的 12 个大很多，让 policy 能学到更通用的编辑策略。

#### pMHC 字符串构建

`get_pmhc_string()` (line 142-154):

```python
def get_pmhc_string(self, peptide: str) -> str:
    return peptide + pseudoseq
    # 例: "GILGFVFTL" + "YFAMYQENAAHTLRWEPYSEGAEYLERTCEW"
    # = "GILGFVFTLYFAMYQENAAHTLRWEPYSEGAEYLERTCEW" (39字符)
```

这个拼接字符串直接送入 ESM-2 编码。ESM-2 对任意氨基酸序列都能产生表示，不要求格式——它只是逐 token 编码后均值池化。

### 4.2 TCRPool — 课程采样

**文件**: `tcrppo_v2/data/tcr_pool.py` (234 行)

#### 三级课程

```
L0: 已知 binder + 3-5 个随机突变
    来源: decoy_d (VDJdb/IEDB) + tc-hard 已知 CDR3β
    为什么要突变: 给 policy 留出优化空间，不是直接给答案

L1: 预计算的 ERGO top-K (已禁用 — 计算量太大)
    实际 weight 始终为 0

L2: 随机 TCRdb 序列 (728万条)
    来源: /share/liuyutian/TCRPPO/data/tcrdb/train_uniq_tcr_seqs.txt
```

#### 课程调度

`get_curriculum_weights()` (line 147-155) 根据 `global_step` 选择权重：

```
步数          L0    L1    L2
< 1M         70%   0%    30%    ← 大量用已知 binder 的变体启动
1M - 3M      40%   0%    60%
3M - 6M      20%   0%    80%
> 6M         10%   0%    90%    ← 逐渐过渡到随机初始化
```

实际运行中因为 L1 被禁用，权重会重新分配（line 183-187）：
```python
if not has_l1:
    w_l2 += w_l1  # L1 的权重转给 L2
    w_l1 = 0.0
```

#### L0 采样细节

`_sample_l0()` (line 201-208):

```python
def _sample_l0(self, target: str) -> str:
    binder = self.rng.choice(self.l0_seeds[target])  # 随机选一个已知 binder
    n_mutations = self.rng.integers(3, 6)             # 3-5 个突变
    n_mutations = min(n_mutations, len(binder) - 1)   # 不能全改
    return mutate_sequence(binder, n_mutations, self.rng)
```

`mutate_sequence` 在 `utils/encoding.py` 中实现，随机选 `n_mutations` 个不同位置，每个位置替换为随机氨基酸。

#### L0 数据加载

`load_l0_from_decoy_d()` (line 87-113): 从 decoy library 的 tier D 目录中加载已知 binder。tier D 是 VDJdb/IEDB 中的已知 TCR-pMHC binding pair。

```
/share/liuyutian/pMHC_decoy_library/data/decoy_d/<TARGET>/decoy_d_results.csv
```

CSV 中有 `sequence` 列，取出后过滤：只保留纯氨基酸、长度 5-15 的序列。

`load_l0_from_dir()` (line 115-145): 从 `data/l0_seeds_tchard/` 加载额外的 L0 种子。每个 target 一个 `.txt` 文件，一行一个 CDR3β。和 decoy_d 来源合并去重。

---

## 5. 打分器

### 5.1 BaseScorer

**文件**: `tcrppo_v2/scorers/base.py` (27 行)

```python
class BaseScorer(ABC):
    @abstractmethod
    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """返回 (score, confidence)"""

    def score_batch(self, tcrs, peptides, **kwargs) -> Tuple[list, list]:
        """默认实现: 逐个调 score()"""
```

所有打分器必须实现 `score()` 返回 `(分数, 置信度)`。`score_batch()` 有默认的循环实现，子类可以覆盖做批量优化。

### 5.2 AffinityERGOScorer — ERGO 结合亲和力打分

**文件**: `tcrppo_v2/scorers/affinity_ergo.py` (151 行)

#### 模型加载

`_load_model()` (line 37-45):

```python
model = AutoencoderLSTMClassifier(
    10,         # embedding_dim (AE 编码后)
    device,
    ERGO_MAX_LEN,  # 28
    21,         # vocab_size (20 AA + X)
    100,        # ae_dim (autoencoder latent dim)
    1,          # num_layers
    ae_file,    # tcr_ae_dim_100.pt
    False       # use_cuda flag (由 device 参数控制)
)
```

ERGO 是 AE-LSTM 架构：
1. TCR CDR3β → Autoencoder 编码为 100-dim latent vector → LSTM
2. Peptide → padding 到 28 → Embedding → LSTM
3. 两个 LSTM 输出 concat → FC → sigmoid → binding probability [0,1]

模型权重: `tcrppo_v2/ERGO/models/ae_mcpas1.pt`  
AE 权重: `tcrppo_v2/ERGO/TCR_Autoencoder/tcr_ae_dim_100.pt`

#### 两种推理模式

**训练时 — 快速模式** `score_batch_fast()` (line 148-150):
```python
def score_batch_fast(self, tcrs, peptides) -> List[float]:
    return self._get_predictions(tcrs, peptides)  # 单次前向，eval模式，无dropout
```
一次 forward pass，确定性输出。这是 `RewardManager` 在训练时调用的方法。

**评估时 — MC Dropout** `mc_dropout_score()` (line 111-134):
```python
def mc_dropout_score(self, tcrs, peps) -> Tuple[np.ndarray, np.ndarray]:
    self._enable_dropout()       # 把 Dropout 层设为 train mode
    gpu_batches = self._build_gpu_batches(tcrs, peps)  # 数据上 GPU（只传一次）
    samples = []
    for _ in range(self.mc_samples):  # 默认 10 次
        preds = self._predict_mc(gpu_batches, expected_n)  # 前向（共享 GPU 数据）
        samples.append(preds)
    self._disable_dropout()      # 恢复 eval mode
    return stacked.mean(axis=0), stacked.std(axis=0)  # 均值=预测, 标准差=不确定性
```

MC Dropout 的关键优化 (line 59-83): `_build_gpu_batches()` 先把 tensor 传到 GPU 一次，10 次 MC 采样共享同一份 GPU 数据，避免重复 CPU→GPU 传输。

`_enable_dropout()` (line 96-103) 只把 `nn.Dropout` 层切到 `train()` 模式，模型其余部分保持 `eval()`。这是标准 MC Dropout 做法。

评估时的打分（`score()` 和 `score_batch()`）使用 MC Dropout，训练时的打分（`score_batch_fast()`）不使用——速度差约 10 倍。

#### ERGO 数据预处理

`_get_predictions()` (line 47-57):

```python
# ae_utils.get_full_batches() 做以下事情：
# 1. TCR 通过 ERGO_TCR_ATOX 映射为整数序列，append 'X' 终止符
# 2. Peptide 通过 ERGO_PEP_ATOX 映射，pad 到 ERGO_MAX_LEN=28
# 3. 按 batch_size=4096 分批
# 4. 返回 (tcr_tensor, pep_tensor, pep_length, signs) 元组列表
```

`preds = ae.predict(self.model, batches, self.device)` 对每个 batch 做前向传播，返回 sigmoid 输出的列表。`preds[:len(tcrs)]` 是因为 `get_full_batches` 可能会 pad 最后一个 batch。

### 5.3 AffinityNetTCRScorer

**文件**: `tcrppo_v2/scorers/affinity_nettcr.py`

关键点：
- 在 import 阶段强制 TF 只用 CPU: `tf.config.set_visible_devices([], 'GPU')`
- 避免 TF 和 PyTorch 争抢 GPU 显存
- NetTCR 模型很小 (~50K params CNN)，CPU 推理够快
- 使用 BLOSUM50 编码（不是 one-hot）
- 模型权重: `data/nettcr_model.weights.h5`

### 5.4 EnsembleAffinityScorer

**文件**: `tcrppo_v2/scorers/affinity_ensemble.py`

对多个 scorer 的输出做加权平均：
```python
# 默认 ERGO:NetTCR = 50:50
score = 0.5 * ergo_score + 0.5 * nettcr_score
```

---

## 6. 奖励管理器

**文件**: `tcrppo_v2/reward_manager.py` (342 行)

### 6.1 RunningNormalizer

`RunningNormalizer` (line 9-27):

```python
class RunningNormalizer:
    def __init__(self, window=10000, warmup=1000):
        self.buffer = deque(maxlen=window)  # 滑动窗口
        self.warmup = warmup

    def normalize(self, value):
        self.buffer.append(value)
        if len(self.buffer) < self.warmup:
            return value                    # 热身期：不归一化，直接返回原值
        mean = np.mean(self.buffer)
        std = np.std(self.buffer) + 1e-8
        return (value - mean) / std         # z-score 归一化
```

**只在 v2_full / v2_decoy_only 等 z-norm 模式下真正生效**。v1_ergo_only 等 raw 模式虽然也调用 `normalize()`，但结果不使用（取的是 `aff_score` 而非 `norm_aff`）。

### 6.2 奖励模式分支

`compute_reward()` (line 80-217) 和 `compute_reward_batch()` (line 219-342) 有完全相同的分支逻辑。以下是所有模式（以 `compute_reward` 为例）：

| 模式 | 行号 | 公式 | 备注 |
|------|------|------|------|
| `v1_ergo_only` | 170-171 | `total = aff_score` | 原始 ERGO 分数，不归一化，terminal |
| `v1_ergo_squared` | 172-173 | `total = aff_score²` | 放大高分差异 |
| `v1_ergo_delta` | 174-175 | `total = aff_delta` | 每步改善量（当前-初始）|
| `v2_full` | 203-209 | `w_a·z(Δ) - w_d·z(decoy) - w_n·z(nat) - w_v·z(div)` | 全 z-norm |
| `v2_decoy_only` | 176-180 | `w_a·z(Δ) - w_d·z(decoy)` | 只有亲和力和 decoy |
| `raw_decoy` | 182-184 | `aff - w_d·decoy` | 无 z-norm |
| `v1_ergo_stepwise` | 185-187 | `aff_score` | 与 v1_ergo_only 相同但每步给 |
| `raw_multi_penalty` | 188-193 | `aff - 0.05d - 0.02n - 0.01v` | 使用 CLI 权重 |
| `threshold_penalty` | 194-202 | `if aff<0.5: aff; else: aff-penalties` | 条件惩罚 |

### 6.3 Terminal vs Per-Step 奖励

**关键理解**: reward_mode 字符串决定了公式，但 **terminal/per-step 的区分由 env 控制，不在 RewardManager 里**。

`RewardManager.compute_reward()` 每一步都会被调用（env.py:179）。env 并没有只在 `done=True` 时才计算奖励的逻辑。所以：

- `v1_ergo_only` 实际上是 **per-step** 给的，只不过每步给的都是当前 ERGO 分数（绝对值）。从 RL 角度看，policy 在第 1 步就收到了奖励信号。
- 真正的 "terminal only" 需要在中间步给 reward=0——当前实现没有这个区分。

这并不影响 v1_ergo_only 的效果，因为 `gamma=0.90` + GAE 会自然折扣未来奖励。但如果你想实现严格的 terminal reward，需要修改 env 的 step 逻辑。

### 6.4 打分频率优化

为节省计算量，部分打分器不是每步都调用（line 50-51, 126-130, 141-146）：

```python
naturalness_eval_freq: int = 4   # 每 4 次调用才真正计算一次
decoy_eval_freq: int = 2         # 每 2 次调用才真正计算一次
```

中间步使用上次缓存的分数（`_last_nat_score` / `_last_decoy_score`）。

**注意**: `_call_count` 是全局计数器，不区分 env。在 `compute_reward_batch()` 中（line 275），`_call_count` 在循环内递增，所以同一 batch 内不同 env 的频率采样是交错的。

### 6.5 Batch 与 Single 的关系

`compute_reward_batch()` (line 219-342):
1. **ERGO 推理是真正批量的** (line 237-244): 用 `score_batch_fast()` 一次性对所有 env 做 forward
2. **Decoy/Naturalness/Diversity 仍然逐个计算** (line 249-): 因为 decoy sampling 是 per-target 的，naturalness 要单独调 ESM，diversity 需要访问历史 buffer

这是一个 partial batching 设计——批量化的瓶颈在 ERGO 推理（占训练时间 ~60%），其余打分器计算量较小。

---

## 7. 环境

**文件**: `tcrppo_v2/env.py` (481 行)

### 7.1 TCREditEnv — 单环境

#### Action Space

3-head 自回归动作空间:
```
Head 1: op_type ∈ {SUB=0, INS=1, DEL=2, STOP=3}
Head 2: position ∈ [0, MAX_TCR_LEN-1]     即 [0, 26]
Head 3: token ∈ [0, NUM_AMINO_ACIDS-1]     即 [0, 19]
```

总动作空间大小 = 4 × 27 × 20 = 2160，但大部分被 masking 掉。

#### Action Masking

`get_action_mask()` (line 313-334):

```python
# Op mask (4维 bool)
op_mask = ones(4)
if len(tcr) >= MAX_TCR_LEN:  op_mask[INS] = False   # 太长不能插入
if len(tcr) <= MIN_TCR_LEN:  op_mask[DEL] = False   # 太短不能删除
if step_count == 0:          op_mask[STOP] = False   # 第一步不能停

# Pos mask (27维 bool)
pos_mask = zeros(MAX_TCR_LEN)
pos_mask[:len(tcr)] = True   # 只有序列实际长度内的位置有效
```

Token 没有 mask（所有 20 种氨基酸都合法）。对于 DEL 和 STOP 操作，token 虽然会被采样，但其 log_prob 在 policy._evaluate() 中被 mask 为 0（policy.py:183-185），不影响梯度。

#### 序列编辑操作

`_apply_sub()` (line 197-203):
```python
# 替换: seq[position] = amino_acid
pos = min(position, len(seq) - 1)  # 安全截断
seq[pos] = IDX_TO_AA[token]
```

`_apply_ins()` (line 205-213):
```python
# 插入: 在 position 处插入一个氨基酸
if len(seq) >= max_tcr_len: return seq  # 安全检查（理论上 masking 已阻止）
pos = min(position, len(seq))  # 可以在末尾插入
seq.insert(pos, IDX_TO_AA[token])
```

`_apply_del()` (line 215-222):
```python
# 删除: 移除 position 处的氨基酸
if len(seq) <= min_tcr_len: return seq  # 安全检查
pos = min(position, len(seq) - 1)
del seq[pos]
```

**注意 `min()` 截断**: 如果 position 超出实际序列长度（虽然 pos_mask 应该已经阻止），会被截断到最后一个合法位置。这是一道防御性编程。

#### Min-Steps 惩罚

`step()` (line 186-189):
```python
if op_type == OP_STOP and self.min_steps > 0 and self.step_count < self.min_steps:
    reward += self.min_steps_penalty  # 通常是 -2.0 或 -3.0
```

这是**加到**奖励上的，不是替代。如果 min_steps=6 且 policy 在第 3 步 STOP，它会收到正常奖励 + 一个大负数惩罚。

#### reset 流程

`reset()` (line 84-139):
1. 选择 target peptide（随机或指定）
2. 编码 pMHC（永久缓存）
3. 选择初始 TCR（课程采样或指定）
4. **计算初始亲和力** `initial_affinity` — 用于 delta reward 模式
5. 重置 step_count、cumulative_delta、done 标志
6. 返回 observation

初始亲和力计算（line 124-133）使用 `score_batch_fast()`——快速模式，无 MC Dropout。这个值在整个 episode 中不变，作为 baseline 计算 `aff_delta = current_aff - initial_affinity`。

### 7.2 VecTCREditEnv — 向量化环境

`VecTCREditEnv` (line 342-481) 是对 `n_envs` 个 `TCREditEnv` 的包装，**核心优化是三阶段批处理**。

#### reset()

`VecTCREditEnv.reset()` (line 381-400):
```python
# 1. 逐 env 做 _reset_internal()（不计算 obs）
for env in self.envs:
    env._reset_internal()

# 2. 批量 ESM 编码所有 TCR
tcr_seqs = [env.tcr_seq for env in self.envs]
tcr_embs = esm_cache.encode_tcr_batch(tcr_seqs)

# 3. 逐 env 拼接 obs
for i, env in enumerate(self.envs):
    obs = cat([tcr_embs[i], env.pmhc_emb, [remaining, cumulative_delta]])
```

关键是 `_reset_internal()` (env.py:224-259) 做了 reset 的所有工作**除了**计算 observation。这样 VecEnv 可以先收集所有 TCR，然后批量编码，而不是逐个编码。

#### step() — 三阶段

`VecTCREditEnv.step()` (line 406-472):

**Phase 1: Apply actions** (line 424-435)
```python
for i, (env, action) in enumerate(zip(self.envs, actions)):
    if env.done:
        env._reset_internal()          # 自动重置已结束的 env
        rewards[i] = 0.0               # 重置步不给奖励
        reset_indices.append(i)
    else:
        done, info = env._step_action_only(action)  # 只改序列，不算奖励
        stepped_indices.append(i)
```

`_step_action_only()` (env.py:261-293) 是 `step()` 的"轻量版"——只执行序列编辑和状态更新，**不计算奖励，不计算 observation**。

**Phase 2: Batch reward** (line 438-452)
```python
batch_tcrs = [envs[i].tcr_seq for i in stepped_indices]
batch_peps = [envs[i].peptide for i in stepped_indices]
batch_rewards, batch_components = reward_manager.compute_reward_batch(...)
```

只对 `stepped_indices`（真正做了动作的 env）计算奖励。reset 的 env 奖励为 0。

**Phase 3: Batch ESM encoding** (line 454-472)
```python
all_indices = reset_indices + stepped_indices
tcr_seqs = [envs[i].tcr_seq for i in all_indices]
tcr_embs = esm_cache.encode_tcr_batch(tcr_seqs)
```

对**所有** env（包括刚 reset 的）做批量 ESM 编码。这是因为 reset 的 env 有了新的 TCR 也需要新的 obs。

#### 自动重置机制

env 的 auto-reset 发生在 **下一次 step 被调用时**，不是在 done=True 的那一步：

```
step N:   action → done=True, reward=R      ← 正常返回最后一步的奖励
step N+1: env.done → _reset_internal()       ← 新 episode 开始，reward=0
```

这意味着 PPO buffer 中 done=True 那一步有正常的奖励，下一步（新 episode 的第一步）reward=0、done=False。GAE 在 `next_non_terminal = 1.0 - dones[t]` 处正确处理了 episode 边界。

---

## 8. 策略网络

**文件**: `tcrppo_v2/policy.py` (199 行)

### 8.1 网络结构

```
                    obs [B, 2562]
                         │
                    ┌────▼────┐
                    │ backbone │  Linear(2562→512) → ReLU → Linear(512→512) → ReLU
                    └────┬────┘
                         │
            features [B, 512]
           ┌─────┬───┴───┬──────┐
           │     │       │      │
      ┌────▼───┐ │  ┌────▼───┐  │
      │op_head │ │  │pos_head│  │
      │Lin(4)  │ │  │MLP     │  │
      └────┬───┘ │  └────┬───┘  │
           │     │       │      │
      op [B,4]   │  pos [B,27]  │
           │     │       │      │
      ┌────▼───┐ │  ┌────▼──────▼───┐
      │op_embed│ │  │  token_head   │
      │Emb(32) │ │  │  MLP(20)      │
      └────┬───┘ │  └──────┬────────┘
           │     │         │
           │     │    tok [B,20]
           │     │
           │  ┌──▼──────┐
           │  │value_head│
           │  │MLP → [1] │
           │  └──────────┘
```

参数量（hidden_dim=512 时）：
- backbone: 2562×512 + 512×512 ≈ 1.57M
- op_head: 512×4 ≈ 2K
- pos_head: (512+32)×256 + 256×27 ≈ 146K
- token_head: (512+32+32)×256 + 256×20 ≈ 152K
- value_head: 512×256 + 256×1 ≈ 131K
- embeddings: 4×32 + 27×32 ≈ 1K
- **总计: ~2M 参数**

### 8.2 自回归采样

`_sample()` (line 109-144) 按顺序采样三个 head：

```
Step 1: op_logits = op_head(features)           # [B, 4]
        op_logits = mask(op_logits, op_mask)     # 无效操作设为 -inf
        op = Categorical(op_logits).sample()     # [B]

Step 2: op_emb = op_embed(op)                   # [B, 32]
        pos_input = cat([features, op_emb])      # [B, 544]
        pos_logits = pos_head(pos_input)         # [B, 27]
        pos_logits = mask(pos_logits, pos_mask)  # 超出序列长度的位置设为 -inf
        pos = Categorical(pos_logits).sample()   # [B]

Step 3: pos_emb = pos_embed(pos)                # [B, 32]
        tok_input = cat([features, op_emb, pos_emb])  # [B, 576]
        tok_logits = token_head(tok_input)       # [B, 20]
        tok = Categorical(tok_logits).sample()   # [B]
```

**自回归的意义**: position 的分布取决于选择了哪种 op（比如 STOP 时 position 无意义），token 的分布取决于 op 和 position（比如 SUB 在位置 5 替换 vs INS 在位置 5 插入，可能有不同的 token 偏好）。

### 8.3 PPO 评估（log-prob 计算）

`_evaluate()` (line 146-194) 在 PPO 更新时调用，给定已采样的 actions 计算 log-prob：

```python
total_log_prob = op_log_prob + pos_log_prob + tok_log_prob
total_entropy  = op_entropy  + pos_entropy  + tok_entropy
```

**Token masking** (line 183-185):
```python
needs_token = (op_actions == OP_SUB) | (op_actions == OP_INS)
tok_log_prob = tok_log_prob * needs_token.float()  # DEL/STOP: tok_log_prob = 0
tok_entropy  = tok_entropy  * needs_token.float()  # DEL/STOP: tok_entropy = 0
```

对于 DEL 和 STOP，token 是无意义的采样——虽然采了但不应影响 policy gradient。乘以 0 确保了这一点。

### 8.4 权重初始化

`_init_weights()` (line 75-83):

```python
# 所有 Linear 层: orthogonal init, gain=sqrt(2)
nn.init.orthogonal_(module.weight, gain=np.sqrt(2))

# 特例: policy head 用小 gain
nn.init.orthogonal_(self.op_head.weight, gain=0.01)   # ← 关键: 初始分布接近均匀

# 特例: value head 用 gain=1.0
nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
```

`gain=0.01` 让 op_head 的初始输出接近 0，softmax 后接近均匀分布 (1/4, 1/4, 1/4, 1/4)。这是标准的 RL policy init 技巧——初始时 exploration 最大化。

---

## 9. PPO 训练器

**文件**: `tcrppo_v2/ppo_trainer.py` (729 行)

### 9.1 RolloutBuffer

`RolloutBuffer` (line 38-139):

预分配 numpy 数组，shape 都是 `(n_steps, n_envs, ...)`：

```python
self.obs        = zeros((n_steps, n_envs, obs_dim))   # 观测
self.ops        = zeros((n_steps, n_envs))             # 操作类型
self.positions  = zeros((n_steps, n_envs))             # 位置
self.tokens     = zeros((n_steps, n_envs))             # token
self.log_probs  = zeros((n_steps, n_envs))             # 采样时的 log-prob
self.rewards    = zeros((n_steps, n_envs))             # 奖励
self.dones      = zeros((n_steps, n_envs))             # episode 结束标志
self.values     = zeros((n_steps, n_envs))             # value 估计
self.op_masks   = zeros((n_steps, n_envs, 4))          # op action mask
self.pos_masks  = zeros((n_steps, n_envs, 27))         # position action mask
```

#### GAE 计算

`compute_gae()` (line 92-104):

```python
# 从后往前遍历
for t in reversed(range(n_steps)):
    if t == n_steps - 1:
        next_values = last_value           # 用 policy 估计的最后一步 value
    else:
        next_values = self.values[t + 1]
    
    next_non_terminal = 1.0 - self.dones[t]  # done=True 时截断
    
    # TD 误差
    delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
    
    # GAE 递推
    last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    advantages[t] = last_gae

returns = advantages + values  # V_target = A + V
```

`next_non_terminal` 在 done=True 时为 0，切断了 episode 之间的 value bootstrap。这确保一个 episode 的最后一步不会 bootstrap 到下一个 episode 的 value。

#### Minibatch 生成

`get_batches()` (line 106-135):

1. Flatten: `(n_steps, n_envs)` → `(n_steps × n_envs,)`
2. 随机打乱索引
3. 按 `batch_size` 切分
4. 每个 batch 转为 GPU tensor dict

### 9.2 PPOTrainer.setup()

`setup()` (line 188-374) 按以下顺序初始化所有组件：

```
1. affinity_scorer     (ergo/nettcr/ensemble，由 config["affinity_model"] 选择)
2. ESMCache           (esm2_t33_650M_UR50D, 冻结)
3. PMHCLoader          (train 模式: 163 targets)
4. TCRPool             (L0 种子从 decoy_d + tchard 加载)
5. DecoyScorer         (条件加载: 只在用 decoy 的 reward_mode 下)
6. NaturalnessScorer   (条件加载: 只在用 naturalness 的 reward_mode 下)
7. DiversityScorer     (条件加载: 同上)
8. RewardManager       (组合所有 scorers)
9. VecTCREditEnv       (n_envs 个并行 env)
10. ActorCritic policy  (obs_dim=2562, hidden_dim=512)
11. Adam optimizer      (lr=3e-4, eps=1e-5)
12. RolloutBuffer       (n_steps=128, n_envs=8)
13. TensorBoard writer
```

条件加载 scorers 的逻辑（line 276-314）:

```python
# 只在需要 decoy 的模式下加载 DecoyScorer
if reward_mode in ("v2_full", "v2_decoy_only", "raw_decoy", "raw_multi_penalty", "threshold_penalty"):
    decoy_scorer = DecoyScorer(...)

# 只在需要 naturalness 的模式下加载 NaturalnessScorer
if reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty"):
    naturalness_scorer = NaturalnessScorer(...)
```

这意味着 `v1_ergo_only` 模式不加载 DecoyScorer、NaturalnessScorer、DiversityScorer——节省了 ESM perplexity 计算和 decoy sampling 的开销。

### 9.3 训练主循环

`train()` (line 424-613) 结构如下:

```python
obs = vec_env.reset()
global_step = resume_step

while global_step < total_timesteps:

    # ① 更新 decoy tier 解锁
    _update_decoy_schedule(global_step)

    # ② 收集 rollout (n_steps × n_envs 个 timestep)
    buffer.reset()
    policy.eval()
    for step in range(n_steps):           # n_steps=128
        vec_env.set_global_step(global_step)
        masks = vec_env.get_action_masks()
        
        # 采样动作
        with torch.no_grad():
            ops, pos, tok, values = policy(obs_tensor, mask_dict)
        
        # 计算 log-prob（第二次 forward，但在 no_grad 下）
        with torch.no_grad():
            log_probs, _, _, _ = policy(obs_tensor, mask_dict, actions=...)
        
        # 环境 step
        next_obs, rewards, dones, infos = vec_env.step(actions)
        
        buffer.add(...)
        global_step += n_envs  # 每步推进 n_envs 个 timestep

    # ③ 计算 GAE
    with torch.no_grad():
        last_values = policy.get_value(obs)
    buffer.compute_gae(last_values, gamma, gae_lambda)

    # ④ PPO 更新
    policy.train()
    for epoch in range(n_epochs):         # n_epochs=4
        for batch in buffer.get_batches(batch_size):  # batch_size=256
            # 当前 policy 评估
            log_probs, entropy, values, _ = policy(batch["obs"], masks, actions)
            
            # PPO loss
            ratio = exp(log_probs - old_log_probs)
            pg_loss = max(-adv * ratio, -adv * clip(ratio, 1±ε))
            vf_loss = MSE(values, returns)
            loss = pg_loss + 0.5 * vf_loss + 0.05 * (-entropy)
            
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(params, 0.5)
            optimizer.step()

    # ⑤ 日志 + 检查点
    if n_updates % 10 == 0: print(stats)
    if global_step >= milestone: save_checkpoint()
```

#### 每步两次 forward pass

注意 line 477-491，每个 step 有两次 policy forward：

```python
# 第一次: 采样动作
ops, pos, tok, values = self.policy(obs_tensor, mask_dict)

# 第二次: 计算采样动作的 log-prob
log_probs, _, _, _ = self.policy(obs_tensor, mask_dict, actions=(ops, pos, tok))
```

为什么不在第一次 forward 中同时拿到 log-prob？因为 `_sample()` 返回的是 actions + values，`_evaluate()` 返回的是 log_probs + entropy + values。两种模式的返回类型不同（见 policy.py:90-101 的 docstring）。

一个可能的优化点：让 `_sample()` 同时返回 log_prob，避免第二次 forward。

#### Timestep 计数

```python
global_step += self.n_envs  # line 512
```

每调用一次 `vec_env.step()`，global_step 增加 `n_envs`（而非 1）。所以 `total_timesteps=2_000_000` 且 `n_envs=8` 时，实际的 step 调用次数是 250,000。

每个 rollout 收集 `n_steps=128` 个 step，所以每次 rollout 贡献 `128 × 8 = 1024` 个 timestep。

### 9.4 Decoy Tier 解锁

`_update_decoy_schedule()` (line 615-627):

```
< 2M steps:   只有 Tier A (1-2 AA 点突变)
2M - 5M:      + Tier B (2-3 AA 突变)
5M - 8M:      + Tier D (VDJdb/IEDB 已知 binder)
> 8M:         + Tier C (1900 个不相关肽段)
```

这是一个从简单到困难的解锁策略。但在实践中，大部分实验只跑 2M steps，所以基本只用到了 Tier A。

### 9.5 检查点

`save_checkpoint()` (line 629-636):
```python
torch.save({
    "policy_state_dict": policy.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config,
}, path)
```

保存内容: policy 权重 + optimizer 状态 + config dict。不保存 `global_step`——恢复时从 checkpoint 文件名推断步数（line 648: `milestone_1000000.pt` → step=1000000）。

Checkpoint 保存策略:
- `milestones` 列表中的步数: 500K, 1M, 2M, 5M, 10M
- 每 `checkpoint_interval`=100K 步: 覆盖 `latest.pt`
- 训练结束: `final.pt`

### 9.6 两阶段训练

支持从 checkpoint 恢复并切换 reward_mode (line 430-443):

```python
# 用法:
# Phase 1: --reward_mode v1_ergo_only --total_timesteps 1000000
# Phase 2: --resume_from output/.../milestone_1000000.pt \
#           --resume_change_reward_mode raw_decoy \
#           --resume_reset_optimizer
```

`_resume_from` 恢复 policy + optimizer 权重，`_resume_change_reward_mode` 修改 RewardManager 的 reward_mode，`_resume_reset_optimizer` 重新创建 Adam（清除动量）。

### 9.7 CLI 参数

`main()` (line 659-729) 支持以下 CLI 覆盖：

| 参数 | 类型 | 说明 |
|------|------|------|
| `--config` | str | YAML 配置文件路径 |
| `--run_name` | str | 实验名 (决定输出目录) |
| `--seed` | int | 随机种子 |
| `--reward_mode` | str | 奖励模式 |
| `--total_timesteps` | int | 总训练步数 |
| `--n_envs` | int | 并行环境数 |
| `--w_affinity/decoy/naturalness/diversity` | float | 奖励权重 |
| `--min_steps` | int | 最少编辑步数 |
| `--min_steps_penalty` | float | 早停惩罚值 |
| `--hidden_dim` | int | 策略网络隐藏层维度 |
| `--learning_rate` | float | 学习率 |
| `--affinity_scorer` | str | 打分器选择: ergo/nettcr/ensemble |
| `--resume_from` | str | 恢复 checkpoint 路径 |
| `--resume_change_reward_mode` | str | 恢复时切换奖励模式 |
| `--resume_reset_optimizer` | flag | 恢复时重置优化器 |

所有 CLI 参数覆盖 YAML 中的同名配置。

---

## 10. 评估

**文件**: `tcrppo_v2/test_tcrs.py` (336 行)

### 10.1 TCR 生成

`generate_tcrs()` (line 50-90):

对每个 target peptide，运行 `n_tcrs` 次 episode（默认 50）：
1. `env.reset(peptide=target)` — 课程采样初始 TCR
2. 循环直到 `env.done`:
   - 获取 action mask
   - policy forward（采样模式）
   - env.step(action)
   - 记录轨迹
3. 收集 `final_tcr` 和步数

评估时用的是 **单个 TCREditEnv**（不是 VecEnv），因为逐 target 生成，不需要向量化。

### 10.2 特异性评估

`evaluate_specificity()` (line 93-158):

AUROC 计算逻辑:
```
对每个生成的 TCR:
  1. target_score = ERGO_MC_Dropout(TCR, target_peptide)     # 正样本分数
  2. decoy_scores = ERGO_MC_Dropout(TCR, decoy_peptides×50)  # 负样本分数

labels = [1]*n_pos + [0]*n_neg
scores = target_scores + decoy_scores
AUROC = roc_auc_score(labels, scores)
```

**关键**: 评估**始终使用 ERGO MC Dropout**，无论训练时用的是什么 scorer（line 188-193）。这是因为 ERGO 是 Tier 1 评估标准。即使训练时用 NetTCR 打分，评估时也用 ERGO 打分。

`score()` vs `score_batch()` 的区别：
- `score()` (line 117): 单个 TCR-target 对，使用 MC Dropout (N=10)
- `score_batch()` (line 125-126): 批量 TCR-decoy 对，也使用 MC Dropout

### 10.3 输出格式

`eval_results.json` 结构:
```json
{
  "GILGFVFTL": {
    "specificity": {
      "auroc": 0.5894,
      "mean_target_score": 0.445,
      "mean_decoy_score": 0.134,
      "n_tcrs": 50,
      "n_decoys_per_tcr": 50
    },
    "n_unique": 48,
    "mean_steps": 8.0,
    "mean_reward": 0.204,
    "generated_tcrs": ["CASSIRSSYEQYF", ...]
  },
  ...
  "_summary": {
    "mean_auroc": 0.5894,
    "v1_baseline": 0.4538,
    "delta": 0.1356,
    "checkpoint": "output/.../final.pt"
  }
}
```

`_summary` key 以下划线开头，作为元数据标记。

---

## 11. 配置系统

**文件**: `configs/default.yaml` (113 行)

### 关键参数组

**PPO 超参数** (line 1-13):
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma` | 0.90 | 折扣因子——偏低，鼓励短期回报 |
| `gae_lambda` | 0.95 | GAE 平滑参数 |
| `clip_range` | 0.2 | PPO ratio clip |
| `entropy_coef` | 0.05 | 熵正则系数——鼓励探索 |
| `n_steps` | 128 | 每次 rollout 步数 |
| `batch_size` | 256 | minibatch 大小 |
| `n_epochs` | 4 | 每次 rollout 的 PPO 更新轮数 |

**奖励权重** (line 32-35):
```yaml
w_affinity: 1.0      # 基准权重
w_decoy: 0.8         # 相对较大——强调 decoy 惩罚
w_naturalness: 0.5
w_diversity: 0.2     # 最小——多样性是软约束
```

这些权重只在 z-norm 模式（v2_full 等）下有意义。Raw 模式下 CLI 覆盖为更小的值（如 0.05）。

**课程调度** (line 74-78):
```yaml
curriculum_schedule:
  - {until: 1000000,  L0: 0.7, L1: 0.2, L2: 0.1}
  - {until: 3000000,  L0: 0.4, L1: 0.4, L2: 0.2}
  - {until: 6000000,  L0: 0.2, L1: 0.4, L2: 0.4}
  - {until: null,      L0: 0.1, L1: 0.3, L2: 0.6}
```

**注意**: L1 在代码中被禁用（权重会被转移到 L2），所以实际调度是:
```
< 1M:   L0=70%, L2=30%
1M-3M:  L0=40%, L2=60%
3M-6M:  L0=20%, L2=80%
> 6M:   L0=10%, L2=90%
```

---

## 12. 关键设计决策与已知问题

### 12.1 为什么 Z-Norm 会导致训练失败

**现象**: 所有使用 z-score 归一化的奖励模式（v2_full, v2_decoy_only 等）AUROC ≈ 0.50（随机水平），而 v1_ergo_only (raw) 达到 0.80+。

**根因**: `RunningNormalizer` 用 10000 窗口的均值和标准差归一化。当 affinity delta 的分布和 decoy/nat/div 的分布量级差异大时，归一化后它们被压到相同的尺度，affinity 信号被淹没。

实验证据:
- exp1_decoy_only (z-norm, w_decoy=0.3) → AUROC 0.49
- exp2_light (z-norm, w_decoy=0.2) → AUROC 0.47
- 即使把惩罚权重降到原来的 1/10，z-norm 仍然压垮 affinity

### 12.2 ERGO 打分器的局限性

**训练-评估耦合**: ERGO 同时作为训练奖励信号和评估指标。Policy 可能学会 "取悦 ERGO" 而非真正提高结合亲和力。

证据:
- v1_ergo_only seed=42 → AUROC 0.8075
- v1_ergo_only seed=123 → AUROC 0.5462 (32% 下降)
- NetTCR 交叉验证: seed=42 结果从 0.8075 降到 0.5754

所有排除 seed=42 outlier 后的实验收敛到 ~0.55 AUROC，提示 0.55 可能是 ERGO 打分器下的实际性能上限。

### 12.3 seed 依赖性

seed=42 可能碰到了一个特别好的 TCR 初始化 + target 采样序列，使得 policy 学到了对 12 个 eval target 特别有效的编辑策略。这种 seed 敏感性在 RL 中很常见（ERGO 0.8075 vs 0.5462）。

### 12.4 Per-step vs Terminal reward

当前实现中，`v1_ergo_only` 模式在**每一步**都返回完整的 ERGO 分数（不是只在 terminal step）。这与文档中 "terminal reward" 的描述有出入。

实际效果: 由于 `gamma=0.90`，8 步 episode 的最后一步的未折扣奖励在第一步被折扣为 `0.90^7 ≈ 0.48`。如果每步都给相同量级的奖励，早期步的奖励权重更高——这可能促使 policy 在早期就选择高分 TCR。

### 12.5 VecEnv 不支持 min_steps 的 op_mask

min_steps 惩罚 (env.py:186-189) 是在 step 之后加到 reward 上的，但 **action mask 并没有阻止 STOP**。也就是说，policy 仍然可以选择在 step 2 STOP——它只是会收到一个大负惩罚。

如果想要严格禁止早停，应该在 `get_action_mask()` 中加入:
```python
if self.step_count < self.min_steps:
    op_mask[OP_STOP] = False
```

目前的 "软惩罚" 设计让 policy 自己学习何时 STOP 比较好。

### 12.6 observation 中的 cumulative_delta

`cumulative_delta` 作为 obs 的一部分（env.py:309），让 policy 知道当前 episode 的累计奖励。这是一个有争议的设计：
- **优点**: 帮助 value function 估计 — 如果 episode 已经积累了高奖励，剩余步数的预期回报可能更高
- **缺点**: 可能导致 policy 对历史奖励过度反应，比如前几步奖励很高后就倾向于 STOP

---

*本文档基于 2026-04-15 版本的代码生成。后续代码修改（特别是替换打分器）后请更新对应章节。*
