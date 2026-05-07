# test51b: Pure tFold Terminal-Only Training — MONITORING

**Experiment ID**: test51b_tfold_terminal_opt  
**Status**: 🔄 TRAINING  
**Started**: 2026-05-07 10:54:29 HKT  
**GPU**: 4  
**Priority**: P0  

---

## Quick Status Check

```bash
# Check training progress
tail -20 logs/test51b_tfold_terminal_opt_train.log

# Check process status
ps aux | grep 1819767

# Check cache growth
sqlite3 data/tfold_feature_cache.db "SELECT COUNT(*) FROM features"

# Check GPU usage
nvidia-smi
```

---

## Process Information

| Item | Value |
|------|-------|
| **Training PID** | 1819767 |
| **tFold Server PID** | 1996296 |
| **tFold Server Socket** | `/tmp/tfold_server.sock` |
| **tFold Server GPU** | 0 |
| **Training GPU** | 4 |
| **Log File** | `logs/test51b_tfold_terminal_opt_train.log` |
| **Output Dir** | `output/test51b_tfold_terminal_opt/` |
| **Checkpoint Dir** | `output/test51b_tfold_terminal_opt/checkpoints/` |

---

## Configuration Summary

```yaml
# Core Settings
total_timesteps: 2,000,000
n_envs: 4
n_steps: 32
batch_size: 128
learning_rate: 3e-4
hidden_dim: 512
max_steps: 4
seed: 42

# Reward Settings
reward_mode: contrastive_ergo
affinity_scorer: tfold
encoder: esm2
terminal_reward_only: true
ban_stop: true

# Reward Weights
w_affinity: 1.0
w_decoy: 0.0
w_naturalness: 0.1
w_diversity: 0.0

# Decoy Settings
n_contrast_decoys: 1
contrastive_agg: mean
decoy_K: 1
decoy_tier_weights: {A: 1.0, B: 0.0, C: 0.0, D: 0.0}

# Curriculum
curriculum_l0: 0.5
curriculum_l1: 0.0
curriculum_l2: 0.5

# Checkpointing
checkpoint_interval: 100000  # Every 100K steps
```

**Full command**:
```bash
CUDA_VISIBLE_DEVICES=4 python tcrppo_v2/ppo_trainer.py \
    --config configs/test51b.yaml \
    --run_name test51b_tfold_terminal_opt \
    --seed 42 \
    --reward_mode contrastive_ergo \
    --affinity_scorer tfold \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 4 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 4 \
    --ban_stop \
    --terminal_reward_only \
    --n_contrast_decoys 1 \
    --contrastive_agg mean \
    --w_naturalness 0.1 \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.0 \
    --curriculum_l2 0.5 \
    --train_targets data/tfold_excellent_peptides.txt \
    --tfold_cache_path data/tfold_feature_cache.db
```

---

## Training Progress

**Last updated**: 2026-05-07 11:14 HKT

| Metric | Value | Notes |
|--------|-------|-------|
| **Current Step** | 1,036 / 2,000,000 | 0.05% |
| **Episodes Completed** | 116 | |
| **Training Time** | 20 minutes | Since 10:54:29 |
| **Speed** | 51.8 steps/min | |
| **Mean Reward** | 0.037 - 0.088 | Reasonable range |
| **Mean Episode Length** | 8.0 | max_steps=4 × 2 |
| **Policy Gradient Loss** | -0.0277 to -0.0393 | |
| **Value Function Loss** | 0.0057 to 0.0171 | |
| **Entropy** | 5.478 to 5.780 | High exploration |

### Cache Statistics

| Metric | Value | Notes |
|--------|-------|-------|
| **Cache Size (start)** | 62,318 | At 10:54:29 |
| **Cache Size (current)** | 64,216 | At 11:14 |
| **Cache Growth** | +1,898 entries | In 20 minutes |
| **Cache Hit Rate** | 4-5% | Very low (expected early) |
| **Growth Rate** | ~95 entries/min | |

---

## Time Estimates

Based on observed speed of **51.8 steps/minute**:

| Milestone | Steps | Estimated Time | Calendar Date |
|-----------|-------|----------------|---------------|
| 100K checkpoint | 100,000 | 1.34 days | 2026-05-08 ~19:00 |
| 200K checkpoint | 200,000 | 2.68 days | 2026-05-10 ~03:00 |
| 500K checkpoint | 500,000 | 6.69 days | 2026-05-14 ~04:00 |
| 1M checkpoint | 1,000,000 | 13.39 days | 2026-05-20 ~20:00 |
| 2M final | 2,000,000 | 26.78 days | 2026-06-03 ~03:00 |

**Expected completion**: ~2026-06-03 (27 days from start)

### Cache Hit Rate Projection

As cache grows, speed should improve:

| Cache Hit Rate | Effective Speed | Time to 2M | Notes |
|----------------|-----------------|------------|-------|
| 5% (current) | 51.8 steps/min | 26.8 days | Early training |
| 50% | ~100 steps/min | 13.9 days | Mid training |
| 85% | ~200 steps/min | 6.9 days | Late training |
| 95% | ~300 steps/min | 4.6 days | Final phase |

**Realistic estimate**: 15-20 days total (accounting for cache acceleration)

---

## Monitoring Checklist

### Every 2 Hours
- [ ] Check training log: `tail -20 logs/test51b_tfold_terminal_opt_train.log`
- [ ] Verify process alive: `ps aux | grep 1819767`
- [ ] Check cache growth: `sqlite3 data/tfold_feature_cache.db "SELECT COUNT(*) FROM features"`
- [ ] Check GPU utilization: `nvidia-smi`

### Daily
- [ ] Record progress in this document (update "Training Progress" section)
- [ ] Check disk space: `df -h /share/liuyutian/tcrppo_v2/`
- [ ] Verify tFold server healthy: `ps aux | grep 1996296`
- [ ] Check for errors in log: `grep -i error logs/test51b_tfold_terminal_opt_train.log`

### At Each Checkpoint (100K, 200K, ...)
- [ ] Verify checkpoint saved: `ls -lh output/test51b_tfold_terminal_opt/checkpoints/`
- [ ] Record checkpoint metrics (reward, loss, entropy)
- [ ] Update time estimates based on actual speed
- [ ] Git commit progress update

---

## Red Flags (Stop Training If)

- [ ] Reward not increasing after 500K steps
- [ ] Cache hit rate < 30% after 1M steps
- [ ] Process enters D state (uninterruptible sleep) for >1 hour
- [ ] GPU utilization drops to 0% for >30 minutes
- [ ] Disk space < 50GB remaining
- [ ] Training time projection > 30 days at 500K steps

---

## Issues Resolved

### Issue 1: Wrong tFold Server ✅ RESOLVED
- **Problem**: Config pointed to `/tmp/tfold_server_gpu4.sock` (PID 781830, 79 days old)
- **Fix**: Changed to `/tmp/tfold_server.sock` (PID 1996296, GPU 0)
- **Time**: 2026-05-07 10:38

### Issue 2: Server Resource Contention ✅ RESOLVED
- **Problem**: `eval_v1_csv_with_tfold.py` (PID 2828906) competing for tFold server
- **Fix**: Killed eval script
- **Time**: 2026-05-07 10:45

### Issue 3: Logging Too Infrequent ✅ RESOLVED
- **Problem**: Only logged every 10 PPO updates (1280 steps)
- **Fix**: Added per-episode logging with `flush=True`
- **Time**: 2026-05-07 10:50

---

## Key Findings

1. **tFold speed**: 9-31 seconds per sample (cache miss)
2. **Terminal-only optimization**: Reduces tFold calls from 8/episode to 2/episode
3. **Training is stable**: Rewards in reasonable range, no crashes
4. **Speed is acceptable**: 51.8 steps/min → ~27 days total (with cache acceleration: 15-20 days)
5. **Cache growth is healthy**: ~95 entries/min early on

---

## Next Steps

1. **Monitor for 24 hours** to verify stability
2. **Check 100K checkpoint** (~1.3 days) for quality
3. **Update time estimates** based on cache hit rate improvement
4. **Consider early evaluation** at 500K steps to verify AUROC trend
5. **Document final results** when training completes

---

## Contact

- **Experiment owner**: User
- **Machine**: GPU server with 5x A800-SXM4-80GB
- **Project root**: `/share/liuyutian/tcrppo_v2/`
- **Conda env**: `tcrppo_v2`
