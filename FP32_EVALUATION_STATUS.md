# FP32 Evaluation Status

**Date**: 2026-06-03 10:49
**Status**: ✅ RUNNING

## Overview

Evaluating three RL training checkpoints (trace61, trace72, trace73) using **pure FP32 tFold scorer** to select the best performing trace for continued training.

## Critical Fix Applied

**Problem identified**: Previous evaluation was using contaminated cache (`data/tfold_feature_cache.db`, 665GB) containing BF16/AMP results, leading to inaccurate affinity scores.

**Solution**: 
- Backed up old cache to `data/tfold_feature_cache_bf16_backup.db`
- Created separate fresh FP32 caches per trace in `data/fp32_eval_cache/`
- Running evaluations **sequentially** (not parallel) to avoid server crashes

## Current Progress

**Started**: 2026-06-03 10:47
**Expected completion**: ~13:47 (3 hours total)

| Trace | Status | Progress | ETA |
|-------|--------|----------|-----|
| trace61 | 🔄 RUNNING | 1/20 targets | ~11:47 |
| trace72 | ⏳ PENDING | 0/20 targets | ~12:47 |
| trace73 | ⏳ PENDING | 0/20 targets | ~13:47 |

## Evaluation Configuration

- **Script**: `scripts/eval_three_traces_fp32_sequential.sh`
- **Output**: `results/fp32_eval_sequential_20260603_104721/`
- **Logs**: `logs/eval_sequential.log`
- **Per-trace logs**: `results/fp32_eval_sequential_20260603_104721/{trace61,trace72,trace73}.log`

### Checkpoints & Configs

| Trace | Checkpoint | Config |
|-------|-----------|--------|
| trace61 | `output/trace61_dynamic_pool/checkpoints/latest.pt` | `configs/trace61_dynamic_pool.yaml` |
| trace72 | `output/trace72_delta_from_trace70/checkpoints/latest.pt` | `configs/trace72_adaptive_gate_m0p8.yaml` |
| trace73 | `output/trace73_curriculum_exploration/checkpoints/latest.pt` | `configs/trace73_curriculum_exploration.yaml` |

### Evaluation Parameters

- **Target peptides**: 20 from `data/tfold_excellent_peptides.txt`
- **TCRs per peptide**: 20
- **Decoys per TCR**: 0 (affinity-only evaluation)
- **Scorer**: tFold V3.4 (FP32 only)
- **Cache**: Separate per-trace FP32 caches (fresh extraction)

### tFold Server Configuration

- **Precision**: FP32 (NO `--use-amp-wrapper` flag)
- **Chunk size**: 64
- **Socket**: `/tmp/tfold_server_fp32_{trace_name}.sock`
- **Cache**: `data/fp32_eval_cache/tfold_cache_{trace_name}_fp32.db`

## Performance Metrics

- **Feature extraction time**: ~9 seconds per TCR (FP32 full precision)
- **Cache hit time**: <1ms (V3.4 classifier only)
- **Expected cache misses**: 400 per trace (20 TCRs × 20 peptides)
- **Expected wall time**: ~1 hour per trace

## Monitoring

**Check progress**:
```bash
bash scripts/monitor_fp32_eval.sh
```

**View live logs**:
```bash
# Main script log
tail -f logs/eval_sequential.log

# Per-trace logs
tail -f results/fp32_eval_sequential_20260603_104721/trace61.log
tail -f results/fp32_eval_sequential_20260603_104721/trace72.log
tail -f results/fp32_eval_sequential_20260603_104721/trace73.log
```

**Check tFold server**:
```bash
ps aux | grep tfold_feature_server.py
tail -f logs/tfold_fp32_trace61.log  # Current active server
```

## Next Steps (After Completion)

1. **Aggregate results** (automatic via `scripts/aggregate_fp32_eval_results.py`)
   - Compute mean/max affinity per trace
   - Generate comparison table
   - Identify best performing trace

2. **Select best trace** based on:
   - Mean affinity across all peptides
   - Max affinity per peptide
   - Consistency across targets

3. **Prepare FP32 restart configuration**:
   - Create `configs/{best_trace}_fp32_restart.yaml`
   - Set `affinity_scorer: tfold` (replace ERGO)
   - Configure tFold server socket path
   - Set FP32 cache path

4. **Launch FP32 training**:
   - Start tFold FP32 server (persistent)
   - Resume training from best checkpoint
   - Monitor for improved specificity

## Verification Checklist

✅ Old BF16 cache backed up
✅ Fresh FP32 caches created
✅ tFold servers running in FP32 mode (no AMP wrapper)
✅ Sequential execution (avoids server conflicts)
✅ Correct config files mapped (trace72 uses `trace72_adaptive_gate_m0p8.yaml`)
✅ Cache isolation per trace (no cross-contamination)

## Expected Outcomes

After evaluation completes, we expect:

- **Quantitative comparison**: Mean affinity scores for each trace under pure FP32 tFold scoring
- **Best trace identification**: The trace with highest mean affinity becomes the FP32 restart candidate
- **Clean FP32 baseline**: All results based on fresh FP32 features, no BF16 contamination

## Files Generated

```
results/fp32_eval_sequential_20260603_104721/
├── trace61.log                      # Evaluation log
├── trace61/
│   ├── evaluation_results.json      # Structured results
│   └── generated_tcrs.json          # All generated TCRs
├── trace72.log
├── trace72/
│   ├── evaluation_results.json
│   └── generated_tcrs.json
├── trace73.log
├── trace73/
│   ├── evaluation_results.json
│   └── generated_tcrs.json
└── summary.json                     # Aggregated comparison

data/fp32_eval_cache/
├── tfold_cache_trace61_fp32.db      # FP32 feature cache
├── tfold_cache_trace72_fp32.db
└── tfold_cache_trace73_fp32.db
```
