# test23: Contrastive ERGO Reward

**Date**: 2026-04-21
**Status**: planned
**GPU**: 0
**Priority**: P0

## Hypothesis

ERGO overfitting is the core bottleneck: training optimizes ERGO score → eval uses same ERGO → inflated AUROC that doesn't generalize (NetTCR cross-val drops to 0.5754).

Solution: reward = ERGO(TCR, target) - mean(ERGO(TCR, decoy_i)) forces the policy to learn *relative* specificity. ERGO's bias cancels out between target and decoy terms, so the policy must learn genuine target-specific features.

## Configuration

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python \
    -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test23_contrastive_ergo \
    --seed 42 \
    --reward_mode contrastive_ergo \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --n_contrast_decoys 4 \
    --device cuda \
    > logs/test23_contrastive_ergo_train.log 2>&1 &
```

## Key Differences from Previous Experiments

- vs test22b (v1_ergo_only): reward is contrastive (target - decoy), not absolute
- vs test1_two_phase (raw_decoy): decoy penalty uses ERGO scorer (same model), not independent decoy scorer — ERGO bias cancels
- vs exp1_decoy_only: no z-norm, no LogSumExp, direct ERGO score difference
- NEW: first experiment using ERGO itself to score both target and decoy in same reward

## Reward Formula

```python
# reward_mode = "contrastive_ergo"
target_score = ergo(tcr, target_peptide)
decoy_scores = [ergo(tcr, d) for d in sample_decoys(n=4)]
reward = target_score - mean(decoy_scores)
```

## Expected Outcome

- Reward range: [-1, 1] (vs [0,1] for v1_ergo_only)
- If successful: AUROC > 0.80 AND NetTCR cross-val > 0.65
- If failed: reward collapses to 0 (policy learns to score everything equally)

## Code Changes Required

- `tcrppo_v2/reward_manager.py`: add `contrastive_ergo` mode
- `tcrppo_v2/ppo_trainer.py`: add `--n_contrast_decoys` arg

## Dependencies

- Code changes needed: reward_manager.py, ppo_trainer.py
- Requires completion of: none
- Blocks: test25_finetune_contrastive
