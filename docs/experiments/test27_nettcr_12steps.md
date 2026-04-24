# test27: NetTCR with 12 Steps

**Date**: 2026-04-24
**Status**: failed
**GPU**: 1,5
**Priority**: P1

## Result: FAILED - TensorFlow/PyTorch GPU Conflict

**Issue**: NetTCR (TensorFlow) and ESM-2 (PyTorch) cannot coexist in the same process with GPU acceleration.

**Error**: `tensorflow.python.framework.errors_impl.FailedPreconditionError: DNN library initialization failed`

**Attempted solutions**:
1. NetTCR on CPU → Too slow (10.6 seqs/s vs needed ~100+ seqs/s)
2. Dual-GPU setup (PyTorch on GPU 1, TensorFlow on GPU 5) → Still conflicts
3. TensorFlow `set_visible_devices` isolation → Insufficient

**Root cause**: TensorFlow and PyTorch both initialize cuDNN in the same process, causing library conflicts even when using different physical GPUs.

**Possible solutions** (not implemented):
- Run NetTCR as separate process with RPC/IPC communication
- Use lightweight encoder (CPU-based) to free GPU for NetTCR
- Reimplement NetTCR in pure PyTorch

## Hypothesis

Switching from ERGO to NetTCR as the affinity scorer and increasing max_steps from 8 to 12 will:
1. Break the train-eval coupling issue (NetTCR is independent of ERGO)
2. Allow more complex sequence edits with longer trajectories
3. Potentially improve both binding affinity and specificity

NetTCR uses a different architecture (CNN with BLOSUM50 encoding) compared to ERGO (LSTM with learned embeddings), which may provide complementary signal.

## Configuration

```bash
CUDA_VISIBLE_DEVICES=<GPU> /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python -u \
    tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test27_nettcr_12steps \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --affinity_scorer nettcr \
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

- vs test26_curriculum_l0: Changed affinity_scorer from `ergo` to `nettcr`, increased max_steps from 8 to 12
- vs test11_nettcr: Added curriculum (L0/L1/L2), increased max_steps from 8 to 12, using ESM-2 encoder
- NEW: First experiment combining NetTCR + 12 steps + curriculum + ESM-2

## Expected Outcome

- Mean AUROC: > 0.50 (better than test26's 0.5027)
- Avg steps per episode: 6-10 (longer trajectories due to max_steps=12)
- Final reward: 1.5-2.5 (NetTCR scores may differ from ERGO)
- If successful: Proves NetTCR can match or exceed ERGO performance
- If failed: Suggests ERGO's learned embeddings are critical for this task

## Dependencies

- Code changes needed: None (NetTCR scorer already implemented)
- Requires completion of: None (can run immediately)
- Blocks: None
