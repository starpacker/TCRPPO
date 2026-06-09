# trace73: Improved Specificity Training

**Date**: 2026-06-01
**Status**: planned
**GPU**: 0
**Priority**: P0

## Hypothesis

Trace72 achieved good target affinity (Advantage -0.71) but suffered from increasing decoy violation (DecViol 1.42 → 1.89). The root cause is:

1. **Weak online pool filtering**: `max_decoy_violation=999` accepts any TCR, including those that bind decoys
2. **Low decoy penalty weight**: `w_decoy=0.3` is insufficient to enforce specificity
3. **Poor initialization quality**: 100% L2 (random TCRdb) leads to low InitA (-2.38)

**Hypothesis**: By strengthening decoy penalty, strictly filtering online pool, and using high-quality L0/L1 initialization, we can achieve:
- **InitA > -1.5** (better starting point)
- **DecViol < 1.0** (reduced off-target binding)
- **Final Advantage > 0.0** (positive net benefit)

## Configuration

```bash
CUDA_VISIBLE_DEVICES=0 python tcrppo_v2/ppo_trainer.py \
    --run_name trace73_improved_specificity \
    --seed 43 \
    --resume_from output/trace72_delta_from_trace70/checkpoints/latest.pt \
    --total_timesteps 1500000 \
    --n_envs 8 \
    --learning_rate 0.00015 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --terminal_reward_only \
    --affinity_scorer tfold \
    --reward_mode v2_simple_target_gated_decoy \
    --w_affinity 1.0 \
    --w_decoy 0.6 \
    --w_naturalness 0.05 \
    --w_diversity 0.02 \
    --entropy_coef 0.012 \
    --train_targets data/tfold_excellent_peptides.txt \
    --curriculum_schedule '[{"until": 1200000, "L0": 0.6, "L1": 0.4, "L2": 0.0}, {"until": null, "L0": 0.5, "L1": 0.3, "L2": 0.2}]' \
    --online_tcr_pool_enabled \
    --online_tcr_pool_start_step 914176 \
    --online_tcr_pool_warmup_steps 100000 \
    --online_tcr_pool_max_ratio 0.7 \
    --online_tcr_pool_max_per_target 512 \
    --online_tcr_pool_min_affinity -1.0 \
    --online_tcr_pool_max_decoy_violation 0.5 \
    --online_tcr_pool_min_hamming 2 \
    --online_tcr_pool_use_dynamic_bands \
    --tfold_use_cache \
    --checkpoint_interval 50000
```

## Key Differences from trace72

| Parameter | trace72 | trace73 | Rationale |
|-----------|---------|---------|-----------|
| **w_decoy** | 0.3 | **0.6** | 2x stronger specificity enforcement |
| **online_pool_max_decoy_violation** | 999.0 | **0.5** | Only accept specific TCRs |
| **online_pool_min_affinity** | -10.0 | **-1.0** | Only accept high-affinity TCRs |
| **online_pool_max_ratio** | 0.5 | **0.7** | Prioritize discovered good TCRs |
| **curriculum** | 100% L2 | **60% L0 + 40% L1** | High-quality initialization |
| **resume_from** | - | **trace72 step 914K** | Continue from best model |

## Expected Outcome

### Success Criteria
- **InitA > -1.5**: Better starting TCRs
- **DecViol < 1.0**: Reduced off-target binding
- **Final Advantage > 0.0**: Positive net reward
- **Reward trend**: Monotonic increase without DecViol increase

### Monitoring Metrics
- **InitA**: Should improve from -2.38 to > -1.5
- **DeltaA**: Should maintain ~1.5-2.0
- **DecViol**: Should decrease from 1.89 to < 1.0
- **DecA**: Should improve from -0.76 to > -0.5
- **Advantage (A)**: Should improve from -0.71 to > 0.0

### If Successful
This proves that:
1. Strict online pool filtering is critical for specificity
2. Higher w_decoy (0.6) is necessary for balanced optimization
3. L0/L1 curriculum provides better initialization than random L2

### If Failed
Possible reasons:
1. w_decoy=0.6 too high → model can't find any good TCRs
2. online_pool filters too strict → pool stays empty
3. L0/L1 seeds themselves have poor specificity

Next steps:
- If pool empty: relax filters (max_decoy_violation 0.5 → 1.0)
- If reward drops: reduce w_decoy (0.6 → 0.45)
- If DecViol still high: increase w_decoy (0.6 → 0.8)

## Dependencies
- Code changes needed: None (uses existing ppo_trainer.py)
- Requires completion of: trace72 (completed)
- Blocks: None

## Notes

**Why resume from trace72?**
- trace72 already learned good editing strategies (DeltaA ~1.7)
- We only need to fix the specificity problem, not relearn editing
- Saves ~600K training steps

**Why L0/L1 instead of L2?**
- L0: Known binders (from VDJdb/IEDB)
- L1: Top-500 TCRdb by ERGO score
- Both are pre-screened for target affinity
- L2 (random TCRdb) has InitA ~ -3.0, too low

**Why strict online pool filters?**
- trace72's online pool accepted everything (max_decoy_violation=999)
- This polluted the pool with non-specific TCRs
- Strict filters ensure only high-quality TCRs are reused
