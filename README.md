# TCRPPO v2 - TCR Design via Reinforcement Learning

深度强化学习驱动的 T 细胞受体（TCR）序列设计系统，专注于生成**高亲和力**且**高特异性**的 TCR 序列。

## 🎯 项目目标

设计能够特异性识别目标 pMHC（peptide-MHC complex）的 TCR CDR3β 序列，同时满足：
1. **高亲和力** (High Affinity): 与目标肽段强结合
2. **高特异性** (High Specificity): 不与 decoy/self 肽段交叉反应
3. **自然性** (Naturalness): 序列符合真实 TCR 分布
4. **多样性** (Diversity): 避免模式崩溃

## 📊 主要成果

- **trace73**: 在 4 个精选 targets 上达到 **0.5933 AUROC**（首次突破 random baseline 0.5）
- **trace94**: 最稳定的基线配置（FP32 tFold + AE+GMM naturalness）
- **trace95**: 支持可变长度 episodes 的改进版本
- **核心发现**: Peptide-Scorer 对齐问题是关键 — 只在可靠的 peptides 上训练

## 🏗️ 系统架构

```
TCR Design Environment (Gym)
├── State: ESM-2 embeddings (TCR + pMHC)
├── Action: Autoregressive editing (SUB/INS/DEL/STOP)
└── Reward: Multi-component
    ├── Affinity (tFold): 与目标肽段的结合分数
    ├── Decoy Penalty: 与 decoy 肽段的交叉反应惩罚
    ├── Naturalness (AE+GMM): TCR 序列的自然度
    └── Diversity: 避免重复序列

Policy Network (PPO)
├── Encoder: ESM-2 (frozen, 650M params)
├── Policy: MLP with 3 autoregressive heads
└── Critic: Value head for advantage estimation

Affinity Scoring
├── tFold: 结构预测 + 亲和力打分 (最可靠)
├── ERGO: 基于序列的快速打分
└── NetTCR: 另一个基于序列的打分器
```

## 📁 项目结构

```
tcrppo_v2/
├── tcrppo_v2/              # 核心代码
│   ├── env.py              # RL 环境（Gym interface）
│   ├── policy.py           # Actor-Critic 策略网络
│   ├── ppo_trainer.py      # PPO 训练主程序
│   ├── reward_manager.py   # 多组件奖励管理
│   ├── sac_trainer.py      # SAC 训练（实验性）
│   ├── scorers/            # 评分模块
│   │   ├── affinity_tfold.py      # tFold 亲和力打分
│   │   ├── affinity_ergo.py       # ERGO 亲和力打分
│   │   ├── decoy.py               # Decoy 特异性打分
│   │   ├── naturalness_ae_gmm.py  # AE+GMM 自然度打分
│   │   └── diversity.py           # 多样性打分
│   ├── data/               # 数据加载与管理
│   │   ├── tcr_pool.py            # TCR 初始化池
│   │   ├── pmhc_loader.py         # pMHC 数据加载
│   │   └── decoy_sampler.py       # Decoy 采样策略
│   └── utils/              # 工具函数
│       ├── esm_cache.py           # ESM-2 嵌入缓存
│       └── constants.py           # 常量定义
│
├── configs/                # 实验配置文件
│   ├── trace94_strict_nat_always_on.yaml  # 当前最佳基线
│   ├── trace95_variable_length.yaml       # 可变长度版本
│   └── trace96_nat_gated_dynamic_bands.yaml  # 计划中的改进
│
├── scripts/                # 实验脚本
│   ├── launch_trace*.sh    # 各实验的启动脚本
│   ├── monitor_trace*.sh   # 训练监控脚本
│   └── train_*.py          # 训练入口脚本
│
├── docs/                   # 文档
│   ├── CRITICAL_LESSON_PEPTIDE_SCORER_ALIGNMENT.md
│   ├── TFOLD_HYBRID_TRAINING.md
│   └── 2026-04-09-tcrppo-v2-design.md
│
├── tests/                  # 单元测试
│   ├── test_env.py
│   ├── test_policy.py
│   └── test_scorers.py
│
├── LESSONS_LEARNED.md      # **关键经验总结**
├── PEPTIDE_SCORER_MAPPING.md  # Peptide-Scorer 对齐表
└── README.md               # 本文件
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 创建 conda 环境
conda create -n tcrppo_v2 python=3.10 -y
conda activate tcrppo_v2

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install fair-esm transformers numpy scipy scikit-learn pandas matplotlib \
    pyyaml tensorboard wandb tqdm pytest
```

### 2. 启动 tFold 服务器（必需）

```bash
# 在单独的终端启动
python scripts/start_tfold_server.py \
    --socket /tmp/tfold_server_trace94.sock \
    --cache data/tfold_cache_trace94.db \
    --gpu 0
```

### 3. 运行训练

```bash
# 使用 trace94 配置（推荐）
CUDA_VISIBLE_DEVICES=1 python tcrppo_v2/ppo_trainer.py \
    --config configs/trace94_strict_nat_always_on.yaml \
    --run_name my_trace94_run \
    --seed 42
```

### 4. 监控训练

```bash
# TensorBoard
tensorboard --logdir output/

# 或使用 Weights & Biases
wandb login
# 训练时会自动上传
```

## 📈 关键配置说明

### trace94: 当前最佳基线

```yaml
# Affinity Scorer
affinity_model: "tfold"         # 使用 tFold（最可靠）
# NO AMP!                        # 必须用 FP32，不要用 AMP

# Naturalness Scorer
naturalness_scorer_type: "ae_gmm"  # AE+GMM（防止 poly-C exploit）
w_naturalness: 0.15                # 自然度权重

# Reward Mode
reward_mode: "v2_simple_target_gated_decoy"
target_decoy_gate_logit: 0.0    # Gate 阈值
target_pass_bonus: 2.0          # 通过 gate 的奖励

# Episode 设置
ban_stop: true                  # 禁止提前结束
max_steps: 8                    # 固定 8 步
terminal_reward_only: true      # 只在结束时给奖励

# Online TCR Pool
online_tcr_pool_enabled: true
online_tcr_pool_use_dynamic_bands: true  # 动态难度调整
```

## 🎓 核心经验教训

### ✅ 成功的方法

1. **FP32 tFold + AE+GMM Naturalness** — 最稳定的组合
2. **Terminal Reward Only** — 清晰的信用分配
3. **Online TCR Pool + Dynamic Bands** — 持续的训练挑战
4. **Gate Mechanism** — 强制同时满足 affinity 和 specificity
5. **Peptide-Scorer 对齐** — 只在可靠的 peptides 上训练

### ❌ 失败的方法（避免重复）

1. **ESM-2 Perplexity** — 不能有效评估 TCR naturalness
2. **Step-wise Delta Reward** — 信用分配混乱，训练失败
3. **tFold + AMP** — 数值不稳定，结果不可复现
4. **大量 Decoy Sampling (K=32)** — 太慢且无明显提升
5. **传统 Curriculum Learning** — 限制探索，效果不佳
6. **Imitation Learning (SFT)** — 分布不匹配，catastrophic forgetting

**详细说明请参考 `LESSONS_LEARNED.md`**

## 📊 主要实验结果

| Trace | Configuration | AUROC | Key Innovation |
|-------|--------------|-------|----------------|
| trace73 | Curated targets + gate=-0.5 | **0.5933** | 首次突破 baseline！ |
| trace94 | FP32 tFold + AE+GMM | Stable | 当前最佳基线 |
| trace95 | trace94 + variable length | Testing | 允许提前 STOP |
| trace96 | Nat gating + adaptive bands | Planned | 下一步创新 |

## 🔍 评估指标

### AUROC (Area Under ROC Curve)
- **目标**: > 0.65（v1 baseline: 0.45）
- **计算**: 用 MC Dropout (N=10) 对 targets 和 decoys 打分，计算 ROC AUC
- **意义**: 衡量模型区分 target 和 decoy 的能力

### 其他指标
- **Mean Affinity**: 平均亲和力分数（应持续上升）
- **Gate Pass Rate**: 通过 gate 的比例（50%+ 为佳）
- **Naturalness Score**: AE+GMM 分数（> 0.5 为佳）
- **Diversity**: 唯一序列数量（避免模式崩溃）

## 📚 重要文档

### 必读
- **`LESSONS_LEARNED.md`** — 所有实验经验总结（最重要！）
- **`PEPTIDE_SCORER_MAPPING.md`** — Peptide-Scorer 对齐表（训练前必看）
- **`docs/CRITICAL_LESSON_PEPTIDE_SCORER_ALIGNMENT.md`** — 对齐问题详解

### 配置参考
- **`configs/trace94_strict_nat_always_on.yaml`** — 推荐的基线配置
- **`TRACE96_IMPLEMENTATION.md`** — 下一步创新设计

### 技术文档
- **`docs/TFOLD_HYBRID_TRAINING.md`** — tFold 混合训练策略
- **`docs/2026-04-09-tcrppo-v2-design.md`** — 完整系统设计文档

## 🛠️ 开发指南

### 添加新的 Scorer

1. 在 `tcrppo_v2/scorers/` 创建新文件
2. 继承 `BaseScorer` 类
3. 实现 `score()` 方法返回 `(score, uncertainty)`
4. 在 `reward_manager.py` 中注册

### 添加新的实验配置

1. 复制 `configs/trace94_strict_nat_always_on.yaml`
2. 修改你想测试的参数
3. 创建 `scripts/launch_traceXX.sh`
4. 记录实验假设和预期结果

### 调试技巧

- **Affinity 不上升**: 检查 peptide-scorer 对齐
- **Poly-C exploit**: 增加 naturalness weight 或用 AE+GMM
- **训练不稳定**: 检查是否使用了 AMP，切换到 FP32
- **AUROC 低**: 调整 gate 参数

## 🤝 贡献指南

1. Fork 本仓库
2. 创建 feature branch (`git checkout -b feature/amazing-feature`)
3. Commit 你的改动 (`git commit -m 'Add amazing feature'`)
4. Push 到 branch (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

## 📝 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@software{tcrppo_v2_2026,
  title={TCRPPO v2: TCR Design via Reinforcement Learning},
  author={Liu, Yutian and Contributors},
  year={2026},
  url={https://github.com/starpacker/TCRPPO}
}
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- **ERGO**: TCR-pMHC binding prediction model
- **tFold**: TCR structure and affinity prediction
- **ESM-2**: Protein language model for sequence embeddings
- **OpenAI Gym**: Reinforcement learning framework

---

**Last Updated**: 2026-06-05  
**Current Best**: trace94 (FP32 tFold + AE+GMM)  
**Next Step**: trace96 (Naturalness Gating + Adaptive Bands)
