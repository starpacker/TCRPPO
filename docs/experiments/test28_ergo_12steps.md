# test28: ERGO with 12 Steps

**Date**: 2026-04-24
**Status**: blocked
**GPU**: 1
**Priority**: P1

## Result: BLOCKED - Process Hung After Initialization

**Issue**: Training process initialized successfully but hung before first rollout.

**Last log output**: `Saved experiment.json to output/test28_ergo_12steps/experiment.json`

**Observations**:
- Process ran for 12+ minutes with CPU usage but no log output
- ESM cache not growing (no new embeddings being computed)
- TensorFlow warnings appeared (ERGO may use TF internally)
- Likely hung during first `vec_env.reset()` call

**Possible causes**:
1. ERGO scorer also has TF/PyTorch conflict (ERGO uses TF for autoencoder)
2. Deadlock in curriculum sampling or ERGO MC Dropout
3. Memory issue with max_steps=12 (larger state space)

**Next steps**:
- Test with lightweight encoder (no PyTorch GPU) + ERGO
- Or test max_steps=12 with a pure PyTorch scorer

## Hypothesis

Increasing max_steps from 8 to 12 with ERGO scorer will:
1. Allow more complex sequence edits with longer trajectories
2. Potentially improve both binding affinity and specificity
3. Test whether the current performance is limited by trajectory length

This isolates the effect of max_steps without introducing NetTCR complications.

## Configuration

```bash
CUDA_VISIBLE_DEVICES=1 /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python -u \
    tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test28_ergo_12steps \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --affinity_scorer ergo \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 12 \
    --ban_stop \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.2 \
    --curriculum_l2 0.3
```

## Key Differences from Previous Experiments

- vs test26_curriculum_l0: Increased max_steps from 8 to 12
- vs test27_nettcr_12steps: Using ERGO instead of NetTCR (avoids TF/PyTorch conflict)
- NEW: First experiment testing max_steps=12 with ERGO + curriculum

## Expected Outcome

- Mean AUROC: > 0.50 (match or exceed test26's 0.5027)
- Avg steps per episode: 7-11 (longer trajectories due to max_steps=12)
- Final reward: 1.5-2.5 (similar to test26)
- If successful: Proves longer trajectories help
- If failed: Suggests max_steps=8 is already sufficient

## Dependencies

- Code changes needed: None (--max_steps already implemented)
- Requires completion of: None (can run immediately)
- Blocks: None
