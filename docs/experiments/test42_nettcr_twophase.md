# test42: NetTCR Two-Phase Training (Replicate test41 with NetTCR)

**Date**: 2026-04-25
**Status**: planned
**GPU**: 2
**Priority**: P0

## Hypothesis

Replicate test41's successful two-phase training strategy (0.6243 AUROC) but replace ERGO with NetTCR throughout the entire pipeline. This will:

1. **Break train-eval coupling**: NetTCR is architecturally independent from ERGO
2. **Test scorer generalization**: Verify that two-phase contrastive training works with different affinity scorers
3. **Potentially improve performance**: NetTCR's CNN architecture may capture different binding patterns than ERGO's LSTM

**Key Question**: Can NetTCR match or exceed ERGO's performance in the proven two-phase framework?

## Three-Phase Training Strategy

### Phase 1: Pure NetTCR Warm-Start (2M steps)
- **Goal**: Achieve strong binding affinity (R≈2.0-2.5, matching test22b)
- **Reward**: Pure NetTCR affinity (no contrastive penalty)
- **Expected**: Policy learns to generate high-affinity TCRs

### Phase 2: Contrastive Fine-Tuning with 8 Decoys (1.5M steps)
- **Goal**: Introduce specificity signal (matching test33)
- **Reward**: NetTCR(target) - mean(NetTCR(8 decoys))
- **Expected**: AUROC 0.55-0.60

### Phase 3: Increase to 16 Decoys (1M steps)
- **Goal**: Strengthen specificity signal (matching test41)
- **Reward**: NetTCR(target) - mean(NetTCR(16 decoys))
- **Expected**: AUROC > 0.62 (match or exceed test41's 0.6243)

## Configuration

### Phase 1: launch_test42_nettcr_phase1.sh
```bash
CUDA_VISIBLE_DEVICES=2 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test42_nettcr_phase1 \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --affinity_scorer nettcr \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.2 \
    --curriculum_l2 0.3
```

### Phase 2: launch_test42_nettcr_phase2.sh
```bash
CUDA_VISIBLE_DEVICES=2 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test42_nettcr_phase2 \
    --seed 42 \
    --resume_from output/test42_nettcr_phase1/checkpoints/final.pt \
    --resume_change_reward_mode contrastive_ergo \
    --reward_mode contrastive_ergo \
    --affinity_scorer nettcr \
    --encoder esm2 \
    --n_contrast_decoys 8 \
    --contrastive_agg mean \
    --total_timesteps 1500000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --entropy_coef_final 0.01 \
    --entropy_decay_start 100000
```

### Phase 3: launch_test42_nettcr_phase3.sh
```bash
CUDA_VISIBLE_DEVICES=2 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test42_nettcr_phase3 \
    --seed 42 \
    --resume_from output/test42_nettcr_phase2/checkpoints/milestone_1000000.pt \
    --resume_change_reward_mode contrastive_ergo \
    --reward_mode contrastive_ergo \
    --affinity_scorer nettcr \
    --encoder esm2 \
    --n_contrast_decoys 16 \
    --contrastive_agg mean \
    --total_timesteps 1000000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --entropy_coef_final 0.01 \
    --entropy_decay_start 100000
```

## Key Differences from test41

| Aspect | test41 (ERGO) | test42 (NetTCR) |
|--------|---------------|-----------------|
| Affinity scorer | ERGO (LSTM) | NetTCR (CNN) |
| Architecture | Learned embeddings | BLOSUM50 + Conv1D |
| Train-eval coupling | Same model | Independent |
| Phase 1 warm-start | test22b (ERGO) | test42_phase1 (NetTCR) |
| Phase 2 contrastive | test33 (ERGO, 8 decoys) | test42_phase2 (NetTCR, 8 decoys) |
| Phase 3 final | test41 (ERGO, 16 decoys) | test42_phase3 (NetTCR, 16 decoys) |

## Expected Outcomes

### Success Criteria
- **Phase 1**: Final reward R > 2.0 (comparable to test22b's R≈2.05)
- **Phase 2**: Mean AUROC > 0.55 (comparable to test33's 0.5983)
- **Phase 3**: Mean AUROC > 0.62 (match or exceed test41's 0.6243)

### If Successful
- Proves two-phase contrastive training is scorer-agnostic
- NetTCR is a viable alternative to ERGO for TCR design
- Opens door to ensemble approaches (ERGO + NetTCR)

### If Failed (AUROC < 0.55)
- NetTCR may lack the expressiveness of ERGO's learned embeddings
- CNN architecture may not capture sequential dependencies as well as LSTM
- Consider hybrid: ERGO warm-start → NetTCR contrastive fine-tuning

## Dependencies

- **Code**: NetTCR-PyTorch scorer already implemented (`affinity_nettcr.py`)
- **Weights**: NetTCR weights at `/share/liuyutian/tcrppo_v2/data/nettcr_pytorch.pt`
- **Blocks**: None (can start immediately)
- **Requires**: None

## Timeline

- **Phase 1**: ~8-10 hours (2M steps)
- **Phase 2**: ~6-8 hours (1.5M steps)
- **Phase 3**: ~4-5 hours (1M steps)
- **Evaluation**: ~1 hour
- **Total**: ~20-24 hours wall time

## Notes

- Use same seed (42) as test41 for fair comparison
- Monitor Phase 1 reward curve - should reach R≈2.0 by 2M steps
- If Phase 1 reward plateaus below 1.5, may need to adjust learning rate or curriculum
- Save milestone checkpoints at 500K, 1M, 1.5M, 2M for Phase 1
- Phase 2 resumes from Phase 1 final checkpoint (2M)
- Phase 3 resumes from Phase 2 milestone at 1M (not final)
