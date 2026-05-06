# test50: tFold Cascade on Excellent Peptides

**Date**: 2026-05-06
**Status**: planned
**GPU**: 3
**Priority**: P0

## Hypothesis

Training on tFold's 20 excellent peptides (AUC≥0.8) with cascade scorer will produce TCRs that score high under tFold evaluation, unlike test41 which scored well on ERGO but poorly on tFold.

## Key Insight from Previous Experiments

**test41 (NetTCR) tFold re-evaluation revealed scorer misalignment**:
- ERGO AUROC: 0.6243 (looked good)
- tFold AUROC: 0.4017 (actually poor, -0.22 drop)
- **Conclusion**: Models trained with ERGO/NetTCR may not generalize to tFold evaluation

**test44 (pure tFold) is too slow**:
- 10.9 days, only 20K/2M steps (1%)
- Cache hit rate: 3%
- **Conclusion**: Pure tFold training is impractical

## Solution: tFold Cascade Scorer

Use test49's cascade strategy:
1. **ERGO initial screening** (fast, ~10ms)
2. **tFold refinement** when ERGO uncertainty > threshold (accurate, ~1s on cache miss)
3. **Cache-only mode** for maximum speed (test49_cacheonly achieved good results)

## Configuration

```bash
CUDA_VISIBLE_DEVICES=3 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test50_tfold_cascade_excellent \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --affinity_scorer tfold_cascade \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/tfold_excellent_peptides.txt
```

## Key Differences from Previous Experiments

- **vs test41**: Uses tFold cascade instead of NetTCR, targets tFold-excellent peptides
- **vs test44**: Uses cascade (not pure tFold), much faster
- **vs test49**: Filters to only tFold AUC≥0.8 peptides (20 instead of 29)
- **NEW**: First experiment explicitly targeting tFold high-score evaluation

## Target Peptides (20 peptides, tFold AUC≥0.8)

From `PEPTIDE_SCORER_MAPPING.md`:
- GILGFVFTL (0.952)
- ELAGIGILTV (0.947)
- GLCTLVAML (0.934)
- RAKFKQLL (0.933)
- NLVPMVATV (0.920)
- CINGVCWTV (0.918)
- TPRVTGGGAM (0.959)
- IPSINVHHY (0.949)
- KLGGALQAK (0.943)
- LLWNGPMAV (0.937)
- FLASKIGRLV (0.915)
- RLRAEAQVK (0.910)
- AVFDRKSDAK (0.905)
- ATDALMTGY (0.895)
- IMNDMPIYM (0.890)
- YLQPRTFLL (0.870)
- SLFNTVATLY (0.860)
- RLRPGGKKK (0.850)
- KRWIILGLNK (0.840)
- LLLDRLNQL (0.830)

## Expected Outcome

- **Training AUROC (ERGO)**: 0.55-0.65 (similar to test41)
- **Evaluation AUROC (tFold)**: >0.55 (better than test41's 0.40)
- **Key metric**: tFold AUROC should NOT drop significantly from ERGO AUROC
- If successful: Proves cascade scorer produces tFold-validated TCRs
- If failed: May need pure tFold training (but impractically slow)

## Dependencies

- Code changes needed: None (cascade scorer already implemented)
- Requires completion of: None (can start immediately)
- Blocks: None

## Evaluation Plan

After training completes:
1. Generate 50 TCRs per target (20 targets × 50 = 1000 TCRs)
2. Evaluate with both ERGO and tFold (using `reevaluate_with_tfold_fast.py`)
3. Compare ERGO vs tFold AUROC gap
4. Success criterion: tFold AUROC ≥ 0.55 AND gap < 0.15
