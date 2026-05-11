# Two-Phase Training Implementation Summary

**Date**: 2026-05-11  
**Status**: ✅ Ready to launch (waiting for test51c 100K checkpoint)

---

## Overview

Successfully implemented infrastructure for two-phase training experiments that resume from test51c checkpoints:

1. **test51c_amp**: AMP-accelerated tFold (4× faster training)
2. **test51c_decoy**: Late-stage decoy penalty (improved specificity)

Both experiments will resume from test51c's 100K checkpoint and continue to 2M steps.

---

## Implementation Completed

### 1. AMP-Accelerated tFold Scorer ✅

**File**: `tcrppo_v2/scorers/affinity_tfold_amp.py`

**Key features**:
- Uses `TFoldAMPWrapper` for 3.97× faster inference (6.44s → 1.62s)
- Keeps V3.4 classifier and SQLite cache (compatible with test51c)
- Falls back to subprocess if AMP fails (robustness)
- Shares cache with test51c (no redundant feature extraction)

**Integration**: Added `tfold_amp` option to `ppo_trainer.py`

### 2. Experiment Configurations ✅

**Files**:
- `configs/test51c_amp.yaml` - AMP experiment config
- `configs/test51c_decoy.yaml` - Decoy experiment config

**Key settings**:
- test51c_amp: `affinity_model=tfold_amp`, all other settings identical to test51c
- test51c_decoy: `w_decoy=0.4` (conservative), `n_contrast_decoys=4`, `reward_mode=v2_full`

### 3. Launch Scripts ✅

**Files**:
- `scripts/launch_test51c_amp.sh` - Auto-launch test51c_amp on GPU 2
- `scripts/launch_test51c_decoy.sh` - Auto-launch test51c_decoy on GPU 3
- `scripts/monitor_test51c_checkpoint.sh` - Monitor test51c and auto-launch at 100K

**Features**:
- Wait for `milestone_100000.pt` to exist
- Auto-launch both experiments when checkpoint is ready
- Proper GPU allocation (GPU 1: test51c, GPU 2: amp, GPU 3: decoy)

### 4. Documentation ✅

**Files**:
- `docs/experiments/test51c_amp.md` - AMP experiment documentation
- `docs/experiments/test51c_decoy.md` - Decoy experiment documentation
- `docs/TFOLD_AMP_INTEGRATION_REPORT.md` - AMP integration report (from earlier)

---

## Experiment Design

### test51c (baseline)
- **Status**: Running on GPU 1
- **Progress**: 15,464 / 2,000,000 steps (0.77%)
- **ETA to 100K**: ~28 hours
- **Will continue**: Yes, to 2M steps (full baseline)

### test51c_amp (AMP acceleration)
- **GPU**: 2
- **Resume from**: milestone_100000.pt
- **Key change**: `affinity_model=tfold_amp`
- **Expected**: 4× faster training, same quality
- **Hypothesis**: AMP does not affect learning dynamics

### test51c_decoy (late-stage decoy)
- **GPU**: 3
- **Resume from**: milestone_100000.pt
- **Key changes**: `w_decoy=0.4`, `n_contrast_decoys=4`, `reward_mode=v2_full`
- **Expected**: Improved specificity (AUROC +0.10)
- **Hypothesis**: Late-stage decoy avoids cold-start instability

---

## Timeline

| Time | Event | Action |
|------|-------|--------|
| Now | test51c at 15K steps | Wait for 100K checkpoint |
| +28h | test51c reaches 100K | Auto-launch test51c_amp + test51c_decoy |
| +28h → +12d | All 3 experiments run | Monitor progress, compare metrics |
| +12d | All reach 2M steps | Final evaluation and comparison |

**Key milestones**:
- 100K: Launch new experiments
- 200K: First comparison (100K after resume)
- 500K: Mid-point eval
- 1M: Major checkpoint eval
- 2M: Final eval

---

## Monitoring

### Auto-launch (recommended)
```bash
# Start monitor in background (will auto-launch at 100K)
nohup bash scripts/monitor_test51c_checkpoint.sh > logs/checkpoint_monitor.log 2>&1 &

# Check monitor status
tail -f logs/checkpoint_monitor.log
```

### Manual launch (if needed)
```bash
# Wait for checkpoint
ls -lh output/test51c_no_decoy_long_ep/checkpoints/milestone_100000.pt

# Launch manually
bash scripts/launch_test51c_amp.sh
bash scripts/launch_test51c_decoy.sh
```

### Monitor training
```bash
# Check all experiments
tail -f logs/test51c_no_decoy_long_ep_train.log  # Baseline
tail -f logs/test51c_amp_train.log               # AMP
tail -f logs/test51c_decoy_train.log             # Decoy

# Check GPU usage
nvidia-smi

# Check processes
ps aux | grep ppo_trainer
```

---

## Success Criteria

### test51c_amp
1. ✅ Training speed: ~200 steps/min (vs ~50 for test51c)
2. ✅ Reward curve: Similar to test51c (no regression)
3. ✅ AUROC: Similar to test51c (AMP doesn't hurt quality)
4. ✅ No crashes or OOM errors

### test51c_decoy
1. ✅ AUROC: > test51c (target: +0.10)
2. ✅ Training stability: No reward collapse
3. ✅ Decoy penalty: Negative values (penalizing cross-reactivity)
4. ✅ Convergence: Stabilizes within 200K steps

---

## Files Created

```
tcrppo_v2/
├── tcrppo_v2/
│   ├── scorers/
│   │   └── affinity_tfold_amp.py          # NEW: AMP scorer
│   └── ppo_trainer.py                     # MODIFIED: Added tfold_amp option
├── configs/
│   ├── test51c_amp.yaml                   # NEW: AMP config
│   └── test51c_decoy.yaml                 # NEW: Decoy config
├── scripts/
│   ├── launch_test51c_amp.sh              # NEW: AMP launch script
│   ├── launch_test51c_decoy.sh            # NEW: Decoy launch script
│   └── monitor_test51c_checkpoint.sh      # NEW: Auto-launch monitor
└── docs/
    ├── experiments/
    │   ├── test51c_amp.md                 # NEW: AMP documentation
    │   └── test51c_decoy.md               # NEW: Decoy documentation
    ├── TFOLD_AMP_INTEGRATION_REPORT.md    # EXISTING: AMP integration report
    └── TWO_PHASE_TRAINING_SUMMARY.md      # NEW: This file
```

---

## Next Steps

1. **Now**: Start checkpoint monitor
   ```bash
   nohup bash scripts/monitor_test51c_checkpoint.sh > logs/checkpoint_monitor.log 2>&1 &
   ```

2. **+28h**: Verify experiments launched successfully
   ```bash
   ps aux | grep ppo_trainer  # Should see 3 processes
   nvidia-smi                 # Should see GPU 1, 2, 3 in use
   ```

3. **+28h → +12d**: Monitor training progress
   - Check reward curves in TensorBoard
   - Compare training speeds (test51c_amp should be 4× faster)
   - Watch for any crashes or anomalies

4. **+12d**: Run final evaluation
   - Generate TCRs from all 3 experiments
   - Run comprehensive decoy eval
   - Compare AUROC, diversity, naturalness
   - Document findings

---

## Risk Mitigation

### Risk 1: AMP causes OOM
- **Mitigation**: Fallback to subprocess scorer (already implemented)
- **Monitor**: Check `n_subprocess_fallback` in logs

### Risk 2: Decoy penalty destabilizes training
- **Mitigation**: Conservative w=0.4 (user choice)
- **Monitor**: Watch reward curve for collapse
- **Fallback**: Reduce to w=0.2 if needed

### Risk 3: Checkpoint incompatibility
- **Mitigation**: Verified policy architecture matches
- **Test**: Dry-run resume before full launch (optional)

---

## Questions?

- **How to check progress?** `tail -f logs/checkpoint_monitor.log`
- **How to stop monitor?** `pkill -f monitor_test51c_checkpoint`
- **How to manually launch?** `bash scripts/launch_test51c_amp.sh`
- **How to check GPU usage?** `nvidia-smi`
- **Where are checkpoints?** `output/<run_name>/checkpoints/`

---

**Status**: ✅ All code implemented, waiting for test51c 100K checkpoint (~28 hours)
