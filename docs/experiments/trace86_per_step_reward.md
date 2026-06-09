# trace86: Per-step Reward Test

**Date**: 2026-05-30  
**Status**: 🔄 PLANNED  
**GPU**: 2  
**Priority**: P0 (Critical test)

---

## 🎯 Hypothesis

**Terminal reward only causes credit assignment problem, leading to low learning efficiency.**

Per-step delta reward should significantly improve learning efficiency by providing immediate feedback at every step.

---

## 📊 Problem Analysis (from trace53)

### trace53 Results (32K episodes, terminal reward only):
- **Mean A**: -2.28 → -1.72 (+0.56 improvement)
- **Positive rate**: <1.5% (almost no positive TCRs)
- **Best A**: 1.26 (unstable, rare)
- **Learning efficiency**: Very low, no clear upward trend

### Root Causes Identified:

1. **Terminal reward only** → Credit assignment problem
   - 8 steps, only final step gets reward
   - Policy doesn't know which intermediate action was good/bad
   
2. **Random initialization (L2=100%)** → Bad starting point
   - Mean InitA = -5.13 (very poor)
   - 55% episodes start from InitA < -5
   
3. **Limited improvement capacity** → Can't reach target
   - DeltaA = +3.34 (not enough)
   - InitA -5.13 + DeltaA +3.34 = Final A -1.79 (still negative)

---

## 🔧 Configuration

### Key Changes vs trace84:

| Parameter | trace84 | trace86 | Rationale |
|-----------|---------|---------|-----------|
| **terminal_reward_only** | true | **false** | Enable per-step reward! |
| **max_steps** | 8 | **12** | More room to improve |
| **curriculum** | ✅ | ✅ | Keep (start from good TCRs) |
| **gate_schedule** | ✅ | ✅ | Keep (conservative) |

### Full Config:

```yaml
terminal_reward_only: false  # KEY: Per-step reward
use_delta_reward: true
max_steps: 12  # Increased from 8

curriculum_schedule:
  - {until: 100000, L0: 0.5, L1: 0.3, L2: 0.2}   # 50% known binders
  - {until: 500000, L0: 0.3, L1: 0.3, L2: 0.4}   # balanced
  - {until: null, L0: 0.1, L1: 0.2, L2: 0.7}     # mostly random

gate_schedule:
  0: -2.0
  200000: -1.5
  500000: -1.0
  1000000: -0.5
  1500000: 0.0
```

---

## 🎯 Expected Outcome

### If hypothesis is correct:

| Metric | trace53 (terminal) | trace86 (per-step) | Improvement |
|--------|-------------------|-------------------|-------------|
| **Episodes to see improvement** | 32K (no clear trend) | 5K-10K | 3-6× faster |
| **Positive rate @ 10K** | <1.5% | >5% | 3× better |
| **Mean A @ 50K** | -1.8 | -1.0 | 0.8 improvement |
| **Mean A @ 100K** | -1.7 | 0.0 | Target reached |

### Success Criteria:

- ✅ **Primary**: Mean A reaches -1.0 within 50K episodes (vs trace53: -1.8 @ 32K)
- ✅ **Secondary**: Positive rate >5% within 10K episodes (vs trace53: <1.5%)
- ✅ **Tertiary**: Clear upward trend in learning curve (vs trace53: flat)

---

## 🔬 What This Test Proves

### If successful (Mean A reaches -1.0 within 50K):
- **Per-step reward solves credit assignment problem**
- Terminal reward only is the main bottleneck
- RL algorithm (PPO) is fine, just needs better reward signal

### If failed (no improvement over trace53):
- Credit assignment is not the main problem
- Need to investigate:
  - Value function learning (VF loss trends)
  - Policy gradient strength (KL divergence)
  - Architecture bottleneck (hidden_dim, layers)
  - Exploration (entropy trends)

---

## 📝 Launch Command

```bash
bash scripts/launch_trace86_per_step_reward.sh
```

---

## 🔍 Monitoring

### Quick check:
```bash
tail -f logs/trace86_per_step_reward_train.log
```

### Statistics (every 1K episodes):
```bash
tail -1000 logs/trace86_per_step_reward_train.log | grep "Episode" | \
  awk '{sum+=$10; count++} END {print "Mean A:", sum/count}'
```

### Compare with trace53:
```bash
python scripts/plot_active_traces.py  # Add trace86 to the list
```

---

## ⚠️ Risk Monitoring

| Signal | Threshold | Action |
|--------|-----------|--------|
| Mean A not improving | No change after 10K episodes | Check VF loss, KL divergence |
| Positive rate still low | <2% @ 10K episodes | Hypothesis rejected, investigate further |
| Training crashes | OOM, NaN loss | Reduce batch_size or learning_rate |

---

## 📊 Comparison Table (To be filled)

| Metric | trace53 (terminal) | trace86 (per-step) | Delta |
|--------|-------------------|-------------------|-------|
| Episodes | 32,312 | TBD | - |
| Mean A (final) | -1.72 | TBD | - |
| Best A | 1.26 | TBD | - |
| Positive rate | 0.9% | TBD | - |
| Episodes to first positive | 99 | TBD | - |

---

**Next update**: After 10K episodes (~12 hours)
