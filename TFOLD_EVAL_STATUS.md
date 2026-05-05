# tFold Re-evaluation Status

## Current Task
Re-evaluating test41 (and later test14, test33, test39) using tFold scorer instead of ERGO.

## Progress

### test41
- **Status**: In progress (PID 803680)
- **Started**: 2026-04-28 ~23:30
- **Log**: `logs/test41_tfold_eval_cached.log`
- **Strategy**: Cache-first (score cached pairs instantly, score cache misses one-by-one)
- **Current progress**: 
  - Target GILGFVFTL: Scoring 50 cache misses (20/50 done after ~7 min)
  - Estimated time per target: ~15-20 minutes
  - Estimated total time: ~3-4 hours for all 12 targets

### Cache Statistics
- **Target cache hit rate**: ~75% (150/200 for GILGFVFTL)
- **Decoy cache hit rate**: Unknown (being measured during run)
- **Cache location**: `/share/liuyutian/tcrppo_v2/data/tfold_feature_cache.db`
- **Cache size**: ~4.4GB, 8K+ entries

## Technical Details

### Why So Slow?
1. **tFold server bottleneck**: Single-threaded feature extraction server
2. **Multiple competing processes**: 
   - SAC test3 (PID 203487) - using ERGO
   - SAC test5 (PID 2695316) - paused, was using tFold hybrid
   - PPO test43a/b/c (PIDs 3212516, 3261514, 3261617) - using ERGO
3. **Cache misses**: Each cache miss requires ~12-17 seconds for feature extraction
4. **Batch scoring fails**: Server times out (600s) when processing large batches with many cache misses

### Solution Implemented
- **Cache-first strategy**: 
  1. Score all pairs with cache-only mode (instant for cache hits, -0.5 for misses)
  2. Identify cache misses (score == -0.5)
  3. Score cache misses one-by-one with retries
- **Script**: `scripts/reevaluate_with_tfold_cached.py`

### Previous Attempts
1. **Batch scoring** (`reevaluate_with_tfold.py`): Failed - server timeout with 50-TCR batches
2. **One-by-one with delays** (`reevaluate_with_tfold_simple.py`): Too slow - 351 hours estimated
3. **Parallel batching** (`reevaluate_with_tfold_parallel.py`): Failed - server overwhelmed

## Expected Output

Results will be saved to:
- `results/test41_eval/eval_results_with_tfold.json`
- `results/test14_eval/eval_results_with_tfold.json`
- `results/test33_eval/eval_results_with_tfold.json`
- `results/test39_eval/eval_results_with_tfold.json`

Format:
```json
{
  "GILGFVFTL": {
    "specificity": {
      "ergo": { "auroc": 0.3886, ... },
      "tfold": { "auroc": 0.xxxx, ... }
    }
  }
}
```

## Next Steps

1. **Monitor test41**: Let it run overnight (~3-4 hours)
2. **Resume SAC test5**: After evaluation completes, resume with `kill -CONT 2695316`
3. **Evaluate remaining models**: Run same script for test14, test33, test39
4. **Update tracker**: Add tFold AUROC column to `docs/all_experiments_tracker.md`

## Commands

### Check progress
```bash
tail -f logs/test41_tfold_eval_cached.log
ps -p 803680 -o pid,etime,%cpu,stat
```

### Stop evaluation
```bash
kill 803680
```

### Resume SAC training
```bash
kill -CONT 2695316
```
