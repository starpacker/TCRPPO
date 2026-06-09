# 🚀 Quick Start: SFT + RL Fine-tuning

## Pre-flight Checklist

✅ SFT checkpoint ready: `output/sft_filtered_training/checkpoint_final.pt`
✅ Config file: `configs/sft_rl_finetune.yaml`
✅ Launch script: `scripts/launch_sft_rl_finetune.sh`
✅ Training targets: `data/tfold_excellent_peptides.txt`
✅ CDR3 stats: `data/cdr3_ppl_stats.json`
✅ tFold cache: `data/tfold_feature_cache.db`

## Launch

```bash
cd /share/liuyutian/tcrppo_v2
bash scripts/launch_sft_rl_finetune.sh
```

## Monitor

```bash
# Real-time training log
tail -f logs/sft_rl_finetune_train.log

# Attach to training session
tmux attach -t sft_rl_train

# Check GPU
nvidia-smi
```

## Expected Progress

| Steps | Time | Affinity | Gate | Status |
|-------|------|----------|------|--------|
| 0 | 0h | -5.49 | -4.0 | Starting (SFT baseline) |
| 100K | 3h | -3.5 | -3.0 | Warm-up complete |
| 200K | 6h | -2.5 | -2.0 | Approaching target |
| 400K | 12h | **-2.0** | -1.5 | **Minimum target achieved** |
| 600K | 18h | -1.5 | -1.0 | Optimization |
| 1M | 30h | -1.2 | -1.0 | Approaching trace73 |
| 2M | 60h | **-1.0** | -1.0 | **Final target** |

## Key Files

- **Config**: `configs/sft_rl_finetune.yaml`
- **Checkpoints**: `output/sft_rl_finetune/checkpoints/`
- **Logs**: `logs/sft_rl_finetune_train.log`
- **Plan**: `SFT_RL_FINETUNE_PLAN.md`

## Success Criteria

- ✅ **Minimum**: Mean affinity > -2.0 by 400K steps
- 🎯 **Target**: Mean affinity > -1.5 by 600K steps  
- 🌟 **Ideal**: Mean affinity > -1.0 by 2M steps

## Troubleshooting

### Training not starting
```bash
# Check if tFold server is running
ls -la /tmp/tfold_server_sft_rl_finetune.sock

# Restart if needed
tmux kill-session -t tfold_sft_rl
tmux kill-session -t sft_rl_train
bash scripts/launch_sft_rl_finetune.sh
```

### GPU out of memory
```bash
# Reduce batch size in config
# Change: batch_size: 256 -> 128
# Or reduce n_envs: 8 -> 4
```

### Slow progress
```bash
# Check tFold cache hit rate in log
grep "Cache hit" logs/sft_rl_finetune_train.log | tail -20

# Should be >90% after warm-up
```

---

**Ready to launch!** 🚀
