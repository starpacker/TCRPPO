# BF16 to FP32 Migration Guide for trace93

## Problem Summary

Investigation showed that BF16 (bfloat16) precision in tFold feature extraction causes:
- **Non-systematic bias**: CV=0.77 (>> 0.30 threshold)
- **Poor ranking correlation**: Spearman ρ=0.17 (p=0.37, not significant)
- **Unpredictable per-sample error**: bias ranges from -6.28 to +1.19
- **Unreliable for RL**: Rankings are essentially random compared to FP32 ground truth

While BF16 is 2.28x faster (7.9s vs 18.0s per sample), it produces unreliable rankings that would cause RL to optimize in the wrong direction.

## What Was Changed

### 1. Launch Script Updated
**File**: `launch_trace93_fresh_random_mutated_init.sh`
- **Removed**: `--use-amp-wrapper` flag from tfold_feature_server.py invocation
- **Effect**: Server now uses FP32 precision instead of BF16

### 2. BF16 Caches Identified
The following caches were generated with BF16 and need to be cleared:
- `tfold_feature_cache_trace43_20pep.db` (13GB) - trace93's cache
- `tfold_feature_cache_baseline_amp.db` (21GB) - baseline cache
- Total: ~34GB to backup/clear

## Migration Steps

### Step 1: Stop Running trace93 Processes
```bash
# Check for running processes
ps aux | grep trace93 | grep -v grep

# If found, kill them
pkill -f trace93
# Or use specific PIDs from logs
```

### Step 2: Backup BF16 Caches (SAFE - Does Not Delete)
```bash
cd /share/liuyutian/tcrppo_v2
./scripts/cleanup_bf16_cache.sh
# This will:
# - Create timestamped backup directory
# - Move (not delete) BF16 caches to backup
# - Preserve data in case of issues
```

### Step 3: Restart Training with FP32
```bash
cd /share/liuyutian/tcrppo_v2
./launch_trace93_fresh_random_mutated_init.sh
```

The training will now:
- Start tFold server in FP32 mode (no --use-amp-wrapper)
- Cache misses will regenerate features with FP32 precision
- Scoring will be slower (18s vs 7.9s per sample) but accurate

### Step 4: Monitor FP32 Performance
```bash
# Watch training log
tail -f logs/trace93_fresh_random_mutated_init_train.log

# Watch tFold server log (should show "precision=fp32" not "precision=amp_bf16")
tail -f logs/trace93_fresh_random_mutated_init_tfold_server.log

# Check cache growth
watch -n 10 'ls -lh data/tfold_feature_cache_trace43_20pep.db*'
```

### Step 5: Verify FP32 Mode
Check the tfold_server.log for these lines:
```
# FP32 mode (correct):
Extract request: X samples
... precision=fp32 ...

# BF16 mode (if you see this, something went wrong):
AMP enabled (automatic mixed precision, dtype=bf16)
... precision=amp_bf16 ...
```

### Step 6: After Confirming FP32 Works
Once you've verified FP32 training runs smoothly for a few hours:
```bash
# Delete the backup (optional, to free 34GB disk space)
rm -rf data/cache_backup_bf16_*
```

## Performance Expectations

### FP32 vs BF16 Trade-offs
| Metric | BF16 | FP32 | Change |
|--------|------|------|--------|
| Speed per sample | 7.9s | 18.0s | 2.28x slower |
| Ranking correlation | 0.17 (random) | 1.00 (perfect) | Fixed! |
| Bias CV | 0.77 (non-systematic) | 0.00 (none) | Fixed! |
| RL optimization | Wrong direction | Correct | Fixed! |

### Expected Impact on Training
- **InitA scoring**: 2.28x slower per batch
- **Total training time**: Depends on scoring frequency
  - If scoring every episode: ~2x slower overall
  - If scoring every N episodes: impact is amortized
- **Quality**: Rankings are now reliable for RL optimization

## Disk Space

### Before Cleanup
- BF16 caches: ~34GB
- Other caches: ~1.4TB
- Total: ~1.43TB

### After Cleanup + FP32 Regeneration
- Backup (if kept): ~34GB
- FP32 cache (will grow): starts empty, grows to ~13GB+
- Net change: minimal if backup is deleted

## Troubleshooting

### If training still uses BF16:
1. Check launch script was actually updated (grep for --use-amp-wrapper)
2. Kill all trace93 processes completely
3. Remove socket file: `rm /tmp/tfold_server_trace93.sock`
4. Restart

### If scoring is too slow:
Options to speed up FP32:
1. **Reduce scoring frequency** - score every N episodes instead of every episode
2. **Batch scoring** - increase chunk size (already 64, limited by GPU memory)
3. **Add GPU** - run multiple scoring servers in parallel
4. **Accept the slowdown** - 2.28x is the cost of accurate rankings

### If cache doesn't regenerate:
Check that old cache was actually moved:
```bash
ls -lh data/tfold_feature_cache_trace43_20pep.db*
# Should show "No such file or directory" if cleaned successfully
```

## References

Test results documenting the BF16 precision issue:
- `logs/tfold_bf16_bias_systematic.json` - statistical analysis
- `figures/tfold_bf16_bias_systematic.png` - visualization
- `logs/tfold_restart_test.json` - drift elimination test
- `logs/tfold_three_way_comparison.json` - FP32 vs BF16 comparison

## Summary

**What you gained**: Accurate, reliable TCR rankings for RL optimization
**What you lost**: 2.28x speed on tFold scoring
**Net result**: RL can now actually learn to improve TCRs instead of optimizing random noise
