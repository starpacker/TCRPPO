# 启动前检查清单

## 文件准备 ✓

- [x] `configs/trace91_ultimate_fresh_start.yaml` - Fresh start配置
- [x] `configs/trace92_ultimate_sft_rl.yaml` - SFT+RL配置
- [x] `configs/ULTIMATE_CONFIG_README.md` - 设计文档
- [x] `scripts/launch_ultimate_experiments.sh` - 启动脚本
- [x] `scripts/compare_ultimate_configs.py` - 配置对比工具
- [x] `logs/experiment_analysis_report.md` - 165个实验分析报告

## 启动前检查

### 1. 检查依赖
```bash
# tFold server 是否可用
ls /tmp/tfold_server_trace91*.sock 2>/dev/null || echo "Will be created on start"

# ESM cache
ls data/cdr3_ppl_stats.json

# Training targets
ls data/tfold_excellent_peptides.txt
```

### 2. 选择resume策略

#### trace91 - 三个选项:
```yaml
# 选项A: 真正从零开始 (最慢，3M steps)
resume_from: null

# 选项B: 从trace72开始 (推荐，节省时间)
resume_from: "output/trace72_delta_from_trace70/checkpoints/latest.pt"

# 选项C: 从其他好checkpoint开始
resume_from: "output/trace70_gate_m1p5_from_trace61/checkpoints/latest.pt"
```

#### trace92 - 两个选项:
```yaml
# 选项A: 从SFT checkpoint开始 (如果有的话)
resume_from: "output/sft_filtered_training/checkpoint_final.pt"

# 选项B: 从trace72开始 (备选)
resume_from: "output/trace72_delta_from_trace70/checkpoints/latest.pt"
```

### 3. 修改配置文件

根据上面的选择，编辑配置文件中的 `resume_from` 字段。

### 4. 检查checkpoint是否存在
```bash
# 检查trace72 checkpoint
ls -lh output/trace72_delta_from_trace70/checkpoints/latest.pt

# 检查SFT checkpoint (如果用trace92)
ls -lh output/sft_filtered_training/checkpoint_final.pt
```

## 启动实验

### 方式1: 使用启动脚本 (推荐)
```bash
cd /share/liuyutian/tcrppo_v2

# 只启动trace91 (Fresh start)
./scripts/launch_ultimate_experiments.sh fresh

# 只启动trace92 (SFT+RL)
./scripts/launch_ultimate_experiments.sh sft

# 同时启动两个
./scripts/launch_ultimate_experiments.sh both
```

### 方式2: 手动启动
```bash
# trace91
nohup python tcrppo_v2/train.py \
    --config configs/trace91_ultimate_fresh_start.yaml \
    > logs/trace91_ultimate_fresh_start_train.log 2>&1 &

# trace92
nohup python tcrppo_v2/train.py \
    --config configs/trace92_ultimate_sft_rl.yaml \
    > logs/trace92_ultimate_sft_rl_train.log 2>&1 &
```

## 监控训练

### 实时查看日志
```bash
tail -f logs/trace91_ultimate_fresh_start_train.log
tail -f logs/trace92_ultimate_sft_rl_train.log
```

### 关键指标监控
```bash
# Mean Affinity (每个episode后打印)
grep "Mean A" logs/trace91_ultimate_fresh_start_train.log | tail -20

# Episode详情
grep "Episode.*A=" logs/trace91_ultimate_fresh_start_train.log | tail -20

# Naturalness penalty触发情况
grep "Nat=" logs/trace91_ultimate_fresh_start_train.log | grep -v "Nat=0.0000" | wc -l

# CCC pattern检查
grep "CCC" logs/trace91_ultimate_fresh_start_train.log | wc -l
```

### 检查训练是否正常
```bash
# 每隔10分钟检查一次affinity
watch -n 600 'grep "Mean A" logs/trace91_ultimate_fresh_start_train.log | tail -5'

# 检查进程是否还在运行
ps aux | grep train.py
```

## 成功标准

### trace91 (Fresh Start)
- **100K steps**: Mean A > -5.0 (走出初始混沌)
- **300K steps**: Mean A > -3.0 (online pool开始工作)
- **600K steps**: Mean A > -2.0 (进入良好区间)
- **1M steps**: Mean A > -1.0 (接近trace72水平)
- **2M steps**: Mean A > -0.5 (超越trace72)
- **3M steps**: Mean A > 0.0 ✓✓✓ (目标达成!)

### trace92 (SFT+RL)
- **50K steps**: Mean A > -1.5 (保持SFT质量)
- **200K steps**: Mean A > -0.8 (快速改善)
- **500K steps**: Mean A > -0.3 (接近目标)
- **1M steps**: Mean A > 0.0 ✓✓✓ (目标达成!)
- **2M steps**: Mean A > 0.2 (冲击更高)

## 如果遇到问题

### 问题1: 训练卡住/affinity不动
```bash
# 检查是否有NaN
grep -i "nan\|inf" logs/trace91_ultimate_fresh_start_train.log

# 检查learning rate是否需要调低
# 编辑配置文件，降低learning_rate: 1.5e-4 → 1.0e-4
```

### 问题2: 大量CCC出现
```bash
# 统计CCC pattern
grep -o "C\{3,\}" logs/trace91_ultimate_fresh_start_train.log | wc -l

# 如果>1000，说明naturalness还不够强
# 编辑配置，进一步提高:
# naturalness_threshold_zscore: 1.0 → 0.5
# w_naturalness: 0.15 → 0.25
```

### 问题3: 训练崩溃/affinity突然下降
```bash
# 检查最近的checkpoints
ls -lht output/trace91_ultimate_fresh_start/checkpoints/ | head -10

# 从更早的stable checkpoint恢复
# 编辑配置:
# resume_from: "output/trace91_ultimate_fresh_start/checkpoints/step_800000.pt"
```

## 预期timeline

### trace91 (3M steps, 8 envs)
- 每step: ~32 env-steps
- 每小时: ~15K-20K steps (取决于tFold速度)
- **预计总时间: 150-200 小时 (6-8天)**

### trace92 (2M steps)
- **预计总时间: 100-130 小时 (4-5天)**

## 最终验证

训练完成后:
```bash
# 生成最终报告
python scripts/analyze_all_experiments_v2.py

# 检查trace91/92在排名中的位置
head -20 logs/experiment_analysis_summary.csv

# 如果Mean A > 0.0，恭喜！🎉
# 如果Mean A = 0.2+，这是历史性突破！🚀
```

---

## 立即开始

**推荐配置** (如果想快速看到结果):
1. trace91: `resume_from: trace72`
2. trace92: `resume_from: trace72` (如果没有SFT checkpoint)

```bash
# 1. 检查并修改配置文件
vim configs/trace91_ultimate_fresh_start.yaml  # 确认resume_from
vim configs/trace92_ultimate_sft_rl.yaml       # 确认resume_from

# 2. 启动
./scripts/launch_ultimate_experiments.sh both

# 3. 监控
tail -f logs/trace91_ultimate_fresh_start_train.log

# 让我们冲击 Mean Affinity 0.0! 🎯
```
