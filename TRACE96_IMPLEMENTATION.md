# Trace96 Implementation Summary

## Overview
trace96 implements two major innovations on top of trace94:
1. **Naturalness Gating**: Ban affinity reward for unnatural sequences
2. **Adaptive Curriculum Bands**: Dynamic difficulty adjustment based on performance

## Configuration Changes

### From trace94 to trace96:
```yaml
# Increased decoy weight
w_decoy: 1.0  # was 0.3

# Increased pass bonus
target_pass_bonus: 3.0  # was 2.0

# NEW: Naturalness gating
naturalness_gate_affinity: true
naturalness_gate_threshold: 0.5  # Ban affinity if combined score < 0.5

# NEW: Adaptive curriculum bands
online_tcr_pool_adaptive_bands: true
online_tcr_pool_recent_window: 50

# Independent tFold server
tfold_server_socket: "/tmp/tfold_server_trace96.sock"
tfold_cache_path: "data/tfold_cache_trace96.db"
```

## Feature 1: Naturalness Gating

### Implementation
- **File**: `tcrppo_v2/reward_manager.py`
- **Method**: `_apply_naturalness_gate()`
- **Logic**: 
  ```python
  if naturalness_gate_affinity and nat_score < threshold:
      affinity_reward = 0.0  # Ban affinity reward
  ```

### How it works:
1. AE+GMM scorer computes combined score (edit_acc + gmm_like)
2. If combined score < 0.5 (threshold), affinity reward is zeroed out
3. Model MUST satisfy naturalness first before getting affinity reward
4. Prevents poly-C exploit and other unnatural shortcuts

## Feature 2: Adaptive Curriculum Bands

### Implementation
- **File**: `tcrppo_v2/data/tcr_pool_trace96_adaptive.py`
- **Strategy**: DIRECT curriculum (easier → harder as performance improves)

### Band Mapping:
| Mean Affinity Range | Sample From Band | Description |
|---|---|---|
| [-10, -4) | [-10, -2) | Very weak → very easy seeds |
| [-4, -2) | [-10, -4) | Weak → easy seeds |
| [-2, -1) | [-4, -2) | Medium → medium seeds |
| [-1, 0) | [-2, -1) | Good → hard seeds |
| [0, +∞) | [-1, 0) | Expert → hardest seeds |

### How it works:
1. Track last 50 episodes' final affinity per peptide
2. Every 20 episodes, compute mean affinity
3. Select sampling band based on mean (curriculum principle)
4. Early training: sample easier seeds (high affinity)
5. Late training: sample harder seeds (low affinity)
6. Automatic difficulty adjustment without manual schedule

### Example Progression:
```
Step 1k:   Mean A = -3.5  → Sample from [-10, -4) (easy warmup)
Step 10k:  Mean A = -1.8  → Sample from [-4, -2) (medium)
Step 20k:  Mean A = -0.7  → Sample from [-2, -1) (hard)
Step 30k:  Mean A = 0.3   → Sample from [-1, 0) (hardest)
```

## Code Changes Summary

### 1. reward_manager.py
- Added `naturalness_gate_threshold` parameter to `__init__`
- Modified `_apply_naturalness_gate()` to use configurable threshold
- Integrated gating into `_simple_target_gated_decoy_reward()`

### 2. ppo_trainer.py
- Pass `naturalness_gate_affinity` and `naturalness_gate_threshold` to RewardManager
- Load `tcr_pool_trace96_adaptive.py` patch when `online_tcr_pool_adaptive_bands: true`
- Hook already exists for `record_episode_affinity()` at line 1493

### 3. tcr_pool_trace96_adaptive.py (NEW)
- Monkey-patch TCRPool with adaptive band selection
- Track recent affinities per peptide
- Compute appropriate sampling bands
- Override `sample_from_online_pool()` to use bands

## Expected Benefits

### Naturalness Gating:
- ✅ Eliminates poly-C and other unnatural exploits
- ✅ Forces model to learn natural sequences first
- ✅ Cleaner training signal (no reward for cheating)
- ✅ Better generalization to real TCRs

### Adaptive Curriculum:
- ✅ Smooth learning progression (easy → hard)
- ✅ No manual curriculum schedule needed
- ✅ Per-peptide adaptation (different peptides at different stages)
- ✅ Prevents overfitting to easy seeds
- ✅ Maintains training challenge throughout

## Comparison with trace94

| Feature | trace94 | trace96 |
|---|---|---|
| Naturalness Scorer | AE+GMM (penalty) | AE+GMM (gating) |
| Decoy Weight | 0.3 | 1.0 |
| Pass Bonus | 2.0 | 3.0 |
| Pool Sampling | Static bands | Adaptive curriculum |
| Poly-C Prevention | Weak | Strong |
| Training Difficulty | Fixed | Adaptive |

## Testing Checklist

- [ ] Start trace96 tFold server
- [ ] Launch trace96 training
- [ ] Verify naturalness gating in logs (affinity=0 when nat<0.5)
- [ ] Verify adaptive bands messages ("Band Update" logs)
- [ ] Monitor affinity progression (should be smoother than trace94)
- [ ] Check for poly-C sequences (should be near zero)
- [ ] Compare convergence speed with trace94

## Expected Log Messages

```
[Trace96 Adaptive Bands] Initialized with 5 bands, window=50
[Trace96 Band Update] GILGFVFTL: mean_A=-2.3 | weak→easy → medium→medium
Naturalness gating: seq=CASSLGGTEAFFCCCCC nat=0.32 < 0.5 → affinity reward banned
```

## Notes

- trace96 is more conservative than trace94 (stronger naturalness enforcement)
- May have slower initial progress but better final quality
- Adaptive bands should reduce training variance
- Gating threshold (0.5) may need tuning based on results
