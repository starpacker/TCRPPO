# Implementation Summary: test44 & test45

**Date**: 2026-05-03
**Status**: Both experiments launched successfully

## Completed Tasks

### 1. ✅ Stopped Running Processes
- Killed 4 warmup_tfold_parallel processes (PIDs: 41334, 42551, 43040, 43288)
- Kept tFold feature server running (PID: 1996296)

### 2. ✅ Created Peptide Lists
- `data/tfold_good_peptides.txt`: 29 tFold-trainable peptides (AUC ≥ 0.7)
- `data/ergo_positive_peptides.txt`: 4 ERGO positive-aligned peptides

### 3. ✅ Implemented OOD Penalty in reward_manager.py
- Added `v1_ergo_ood_penalty` reward mode
- Supports soft/hard penalty modes
- Soft penalty: `penalty = (uncertainty - threshold) * weight` when `uncertainty > threshold`
- Hard penalty: `penalty = uncertainty * weight` when `uncertainty > threshold`
- Added OOD stats tracking: `get_ood_stats()`, `reset_ood_stats()`

### 4. ✅ Added OOD Support to ppo_trainer.py
- Added CLI args: `--ood_threshold`, `--ood_penalty_weight`, `--ood_penalty_mode`
- Passed OOD params to RewardManager initialization
- Added OOD stats logging (trigger rate, triggered count)
- Added tFold cache stats logging (hit rate, cache size)
- Updated status line to show OOD trigger rate and cache hit rate

### 5. ✅ Launched Both Experiments

**test44_pure_tfold_nocache** (PID: 1257366)
- GPU: 0
- Scorer: tFold V3.4 (cache_only=False)
- Targets: 29 peptides
- Cache: 18,990 entries at startup
- Expected: 1.4-22 days depending on cache hit rate

**test45_ergo_ood_penalty** (PID: 1264557)
- GPU: 4
- Scorer: ERGO with MC Dropout OOD penalty
- Targets: 4 positive-aligned peptides
- OOD: threshold=0.15, weight=1.0, mode=soft
- Expected: 12-24 hours

## Code Changes

### reward_manager.py
- Added `ood_threshold`, `ood_penalty_weight`, `ood_penalty_mode` parameters
- Added `_ood_triggered`, `_ood_total` counters
- Modified `compute_reward()` to apply OOD penalty when `reward_mode == "v1_ergo_ood_penalty"`
- Modified `compute_reward_batch()` to handle OOD penalty in batch mode
- Added `get_ood_stats()` and `reset_ood_stats()` methods

### ppo_trainer.py
- Added CLI args for OOD parameters
- Passed OOD params to RewardManager
- Added OOD stats logging every 10 updates
- Added tFold cache stats logging
- Updated status line to show OOD trigger rate and cache hit rate

### Launch Scripts
- `scripts/launch_test44_pure_tfold.sh`
- `scripts/launch_test45_ergo_ood_penalty.sh`

### Experiment Docs
- `docs/experiments/test44_pure_tfold.md`
- `docs/experiments/test45_ergo_ood_penalty.md`

## Monitoring

```bash
# test44 (tFold)
tail -f logs/test44_pure_tfold_train.log
# Watch for: Cache hit rate, reward trends

# test45 (ERGO OOD)
tail -f logs/test45_ergo_ood_penalty_train.log
# Watch for: OOD trigger rate (should decrease over time)

# Check processes
ps aux | grep ppo_trainer | grep -v grep

# GPU usage
nvidia-smi
```

## Risk Mitigation Applied

### test44 Risks
1. **Training too slow**: Reduced n_envs to 4, using tFold server
2. **Cache hit rate monitoring**: Will log cache stats every 10 updates
3. **Early abort**: Can manually stop if cache hit rate < 50% at 100K steps

### test45 Risks
1. **OOD penalty too aggressive**: Using soft mode (only penalizes excess)
2. **Conservative weight**: Started with weight=1.0 (can reduce to 0.5 if needed)
3. **Threshold tuning**: Set to 0.15 (75th percentile), can adjust based on trigger rate
4. **Limited peptides**: Only 4 peptides, but all positive-aligned

## Expected Outcomes

### test44 (Pure tFold)
- **Hypothesis**: tFold structure-aware scoring achieves >0.65 AUROC
- **Success**: Mean AUROC 0.65-0.70 on 29 peptides
- **Failure**: Training too slow (>1 month) or tFold scores RL-generated TCRs poorly

### test45 (ERGO OOD)
- **Hypothesis**: OOD penalty prevents ERGO exploitation, improves robustness
- **Success**: Mean AUROC 0.60-0.65, OOD trigger rate decreases over time
- **Failure**: OOD penalty too aggressive (limits exploration) or ERGO uncertainty not reliable

## Next Steps

1. Monitor both experiments for first 100K steps (~6-12 hours)
2. Check test44 cache hit rate at 100K steps
3. Check test45 OOD trigger rate at 100K steps
4. Adjust parameters if needed (threshold, weight, or abort if too slow)
5. Full evaluation at 2M steps completion
