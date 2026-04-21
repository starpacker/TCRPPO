# test26: L0-Heavy Curriculum

**Date**: 2026-04-21
**Status**: planned
**GPU**: 4
**Priority**: P1

## Hypothesis

Current config uses pure L2 (random TCRdb): `L0: 0.0, L1: 0.0, L2: 1.0`. L0 seeds are known binders with 3-5 mutations — they start with non-zero ERGO scores, providing a better learning signal from step 1. Higher L0 ratio should reduce wasted exploration on zero-affinity sequences.

## Configuration

```bash
CUDA_VISIBLE_DEVICES=4 nohup /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python \
    -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test26_curriculum_l0 \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.2 \
    --curriculum_l2 0.3 \
    --device cuda \
    > logs/test26_curriculum_l0_train.log 2>&1 &
```

## Key Differences from Previous Experiments

- vs test22b_ergo_only: curriculum L0=0.5 (was 0.0), L1=0.2 (was 0.0), L2=0.3 (was 1.0)
- vs test6_pure_v2_arch: ESM-2 encoder (was lightweight), ban_stop=True, L0 curriculum (test6 had NO curriculum)
- NEW: first ESM-2 + ban_stop + L0-heavy curriculum combination

## Expected Outcome

- Faster reward convergence in early training (steps 0-500K)
- If successful: AUROC > 0.80 with more stable early training curve
- If failed: L0 seeds are too narrow, policy overfits to known binder patterns

## Code Changes Required

- `tcrppo_v2/ppo_trainer.py`: add `--curriculum_l0/l1/l2` args to override config
- OR: modify `configs/default.yaml` curriculum_schedule for this run

## Dependencies

- Code changes needed: ppo_trainer.py (add curriculum override args)
- Requires completion of: none
