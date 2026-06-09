# 🎯 Breaking Mean Affinity -0.5 Today - Quick Start

## 📊 Current Situation

Based on analysis of all traces:
- **Best mean affinity so far:** -1.059 (trace75, but only 13 episodes)
- **Best sustained performance:** -1.182 (trace73, 4288 episodes, RUNNING)
- **trace73 recent episodes:** Some reaching -0.4 to -0.7 range
- **Key insight:** We have good TCRs, just need to filter out the bad ones

## 🚀 Solution: trace78 - Aggressive Push

I've created **trace78** with an aggressive strategy to break -0.5:

### Key Changes from trace73:
1. ✅ **Skip curriculum** - Go straight to gate -0.5 (no warm-up)
2. ✅ **Increase pool ratio** - 0.85 (from 0.5) - use more proven TCRs
3. ✅ **Strict filtering** - Only keep TCRs with affinity > -0.8
4. ✅ **Higher rewards** - Target bonus 2.0 (from 1.0)
5. ✅ **Less exploration** - Entropy 0.008 (from 0.020) - we know what works
6. ✅ **Faster learning** - LR 2.0e-4 (from 1.2e-4)

### Expected Timeline:
- **0-10K steps:** Mean -1.1 → -0.9 (pool builds)
- **10K-30K steps:** Mean -0.9 → -0.7 (filtering works)
- **30K-50K steps:** Mean -0.7 → -0.5 ✅ **TARGET!**
- **Total time:** ~2-3 hours on GPU 2

## 🎬 Launch Instructions

### Step 1: Launch trace78
```bash
cd /share/liuyutian/tcrppo_v2
./scripts/launch_trace78_aggressive_push.sh
```

This will:
- Start tFold server on GPU 2
- Launch training from trace73's latest checkpoint
- Save logs to `logs/trace78_aggressive_push_train.log`

### Step 2: Monitor Progress
```bash
# Real-time monitor (updates every 10 seconds)
watch -n 10 ./scripts/monitor_trace78.sh

# Or just run once
./scripts/monitor_trace78.sh

# Or watch raw log
tail -f logs/trace78_aggressive_push_train.log
```

### Step 3: Check Results
```bash
# Analyze affinity distribution
python analyze_affinity_distribution.py

# Visualize trace78
python visualize_single_trace.py 78
```

## 📁 Files Created

### Configs:
- `configs/trace78_aggressive_push.yaml` - Training configuration

### Scripts:
- `scripts/launch_trace78_aggressive_push.sh` - Launch script
- `scripts/monitor_trace78.sh` - Progress monitor

### Documentation:
- `STRATEGY_BREAK_MINUS_0_5.md` - Detailed strategy
- `QUICK_START_TRACE78.md` - This file

## 🎯 Success Criteria

We'll know we've succeeded when:
- ✅ Mean affinity > -0.5
- ✅ Recent mean (last 500 eps) > -0.4
- ✅ % episodes > 0: at least 5%
- ✅ % episodes > 0.6: at least 1%

## 📊 What to Watch

### Good Signs ✅
- Mean affinity trending upward
- More episodes with affinity > 0
- Online pool size growing
- High cache hit rate (>80%)

### Warning Signs ⚠️
- Mean affinity stuck or decreasing
- Very few episodes > -0.5
- Pool not growing

## 🔧 Troubleshooting

### If training doesn't start:
```bash
# Check GPU 2
nvidia-smi -i 2

# Check if trace73 checkpoint exists
ls -lh output/trace73_curriculum_exploration/checkpoints/latest.pt

# Check logs
tail -50 logs/trace78_aggressive_push_train.log
```

### If progress is too slow:
After 30K steps, if mean affinity is still < -0.7, we can:
1. Increase pool ratio to 0.95
2. Increase min_affinity filter to -0.6
3. Increase target bonus to 3.0

### If it's working well:
Continue to 100K steps and beyond to stabilize!

## 📈 Alternative Approaches (if needed)

### Plan B: Elite-only training
If trace78 doesn't work, we can:
1. Extract all TCRs with affinity > -0.5 from trace73's pool
2. Use them as L1 seeds
3. Train with strict filtering from the start

### Plan C: Ensemble approach
Train multiple traces in parallel with different strategies:
- trace78: Aggressive filtering (current)
- trace79: Elite-only initialization
- trace80: Delta reward + filtering

## 💡 Tips

1. **Be patient** - First 10K steps are building the pool
2. **Watch the pool** - OnlinePool stats show how many good TCRs we have
3. **Check cache** - High cache hit rate means we're reusing good TCRs
4. **Monitor recent mean** - More important than overall mean

## 🎉 When We Succeed

Once mean affinity > -0.5:
1. Let it run to 100K steps to stabilize
2. Save the checkpoint
3. Analyze what worked
4. Use as base for future experiments

---

## 🚀 Ready to Launch?

```bash
cd /share/liuyutian/tcrppo_v2
./scripts/launch_trace78_aggressive_push.sh
```

Then monitor with:
```bash
watch -n 10 ./scripts/monitor_trace78.sh
```

**Let's break -0.5 today! 🎯**
