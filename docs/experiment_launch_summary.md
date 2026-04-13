# 实验启动总结

**时间:** 2026-04-12 06:34  
**状态:** ✅ 所有5个新实验已成功启动

## 已启动的实验

| # | 实验名称 | GPU | PID | 奖励模式 | 关键配置 | 日志文件 |
|---|---------|-----|-----|---------|---------|---------|
| 1 | test1_two_phase_p1 | 0 | 1830808 | v1_ergo_only | Phase 1: 1M步纯ERGO | test1_two_phase_p1_train.log |
| 2 | test2_min6_raw | 1 | 1831087 | raw_decoy | min_steps=6, d=0.05 | test2_min6_raw_train.log |
| 3 | test3_stepwise | 2 | 1831409 | v1_ergo_stepwise | 每步绝对ERGO分数 | test3_stepwise_train.log |
| 4 | test4_raw_multi | 3 | 1831700 | raw_multi_penalty | d=0.05, n=0.02, v=0.01 | test4_raw_multi_train.log |
| 5 | test5_threshold | 4 | 1832408 | threshold_penalty | 阈值0.5条件惩罚 | test5_threshold_train.log |
| 6 | test6_pure_v2 | 5 | 2809807 | v1_ergo_only | **A1+A2+A10 only, NO L0** | test6_pure_v2_arch_train.log |
| 7 | test7_v1ergo_repro | 2 | 4096146 | v1_ergo_only | **复现验证 seed=123** | test7_v1ergo_repro_train.log |
| 8 | test8_longer_5M | 0 | 2113269 | v1_ergo_only | **5M步长训练** | test8_longer_5M_train.log |
| 9 | test9_squared | TBD | TBD | v1_ergo_squared | **reward=ergo^2** | test9_squared_train.log |
| 10 | test10_big_slow | TBD | TBD | v1_ergo_only | **lr=1e-4, hidden=768, 3M** | test10_big_slow_train.log |

## 代码修改

### 1. reward_manager.py - 新增4个奖励模式

```python
# raw_decoy: 原始亲和力 - 轻微decoy惩罚
elif self.reward_mode == "raw_decoy":
    total = aff_score - self.weights["decoy"] * decoy_score

# v1_ergo_stepwise: 每步返回绝对ERGO分数
elif self.reward_mode == "v1_ergo_stepwise":
    total = aff_score

# raw_multi_penalty: 原始亲和力 - 多个轻微惩罚
elif self.reward_mode == "raw_multi_penalty":
    total = (aff_score
            - self.weights["decoy"] * decoy_score
            - self.weights["naturalness"] * nat_score
            - self.weights["diversity"] * div_score)

# threshold_penalty: 基于阈值的条件惩罚
elif self.reward_mode == "threshold_penalty":
    if aff_score < 0.5:
        total = aff_score  # 纯亲和力信号
    else:
        total = (aff_score - penalties)  # 加上惩罚
```

### 2. ppo_trainer.py - 两阶段训练支持

```python
parser.add_argument("--resume_from", help="Checkpoint to resume from")
parser.add_argument("--resume_change_reward_mode", help="Change reward mode on resume")
parser.add_argument("--resume_reset_optimizer", action="store_true")

# 在main()中:
if args.resume_from:
    trainer.load_checkpoint(args.resume_from)
    if args.resume_change_reward_mode:
        trainer.reward_manager.reward_mode = args.resume_change_reward_mode
```

## 监控命令

```bash
# 查看所有实验进度
./monitor_new_tests.sh

# 查看单个实验日志
tail -f output/test1_two_phase_p1_train.log
tail -f output/test2_min6_raw_train.log
tail -f output/test3_stepwise_train.log
tail -f output/test4_raw_multi_train.log
tail -f output/test5_threshold_train.log

# 检查GPU使用
nvidia-smi

# 检查进程
ps aux | grep ppo_trainer
```

## Test 1 Phase 2 启动命令（Phase 1完成后执行）

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python -u \
    tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test1_two_phase_p2 \
    --resume_from output/test1_two_phase_p1/checkpoints/milestone_1000000.pt \
    --resume_change_reward_mode raw_decoy \
    --reward_mode raw_decoy \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --w_decoy 0.05 \
    --seed 42 \
    > output/test1_two_phase_p2_train.log 2>&1 &
```

## 预计完成时间

- **Test 1 Phase 1:** ~12小时 (1M步, v1_ergo_only速度~58 s/s)
- **Test 2-5:** ~24小时 (2M步, 有ESM/decoy开销, 速度~22 s/s)

**最早结果:** 2026-04-13 上午 (Test 1 Phase 1)  
**全部完成:** 2026-04-13 晚上

## Git提交

```
commit 68a6f75
v2(phase8): implement 4 new raw reward modes and launch 5 parallel experiments (tests 1-5)
```

## 下一步

1. 等待Test 1 Phase 1完成 (~12小时)
2. 启动Test 1 Phase 2 (使用上面的命令)
3. 等待所有实验完成 (~24小时)
4. 评估所有实验结果
5. 更新 docs/all_experiments_tracker.md
6. 与v1_ergo_only基线(0.8075 AUROC)对比
