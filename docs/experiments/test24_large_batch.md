# test24: Large Batch (n_envs=32) with seed=123

**Date**: 2026-04-21
**Status**: planned
**GPU**: 1
**Priority**: P0

## Hypothesis

Seed instability (seed=42: AUROC=0.8075, seed=123: AUROC=0.5462) may be caused by high gradient variance with small batch (n_envs=8, batch=2048). Larger batch (n_envs=32, batch=8192) reduces gradient noise and may stabilize training across seeds.

Note: seed=123 was previously tested as test7_v1ergo_repro (lightweight encoder, no ban_stop). This is the first test of seed=123 with ESM-2 + ban_stop + large batch.

## Configuration

```bash
CUDA_VISIBLE_DEVICES=1 nohup /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python \
    -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test24_large_batch \
    --seed 123 \
    --reward_mode v1_ergo_only \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --total_timesteps 2000000 \
    --n_envs 32 \
    --device cuda \
    > logs/test24_large_batch_train.log 2>&1 &
```

## Key Differences from Previous Experiments

- vs test7_v1ergo_repro (seed=123): ESM-2 encoder (was lightweight), ban_stop=True (was False), n_envs=32 (was 8)
- vs test22b_ergo_only (seed=42, n_envs=8): seed=123, n_envs=32
- NEW: first ESM-2 + ban_stop + seed=123 + large batch combination

## Expected Outcome

- If successful: AUROC > 0.75 with seed=123 (closes gap with seed=42)
- If failed: seed instability is fundamental, not batch-size related

## Code Changes Required

None — all args already supported.

## Dependencies

- Code changes needed: none
- Requires completion of: none
