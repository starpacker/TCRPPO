# test51b: Pure tFold with Terminal-Only Reward (Optimized)

**Date**: 2026-05-07
**Status**: running
**GPU**: 4
**Priority**: P0
**Started**: 2026-05-07 08:20
**PID**: 2474882 (training), 781830 (tFold server)

## Hypothesis

Optimized configuration enables practical pure tFold training in ~4 days by:
1. **Reducing decoys from 2 to 1** → 33% fewer tFold calls
2. **Reducing n_steps from 128 to 32** → faster rollout cycles
3. **Reducing n_envs from 8 to 4** → lower GPU memory pressure
4. **Terminal-only reward** → 3x fewer tFold calls vs step-wise

**Key insight**: Single decoy (tier A only) still provides contrastive signal while dramatically reducing computational cost.

## Configuration

```bash
CUDA_VISIBLE_DEVICES=4 python tcrppo_v2/ppo_trainer.py \
    --config configs/test51b.yaml \
    --run_name test51b_tfold_terminal_opt \
    --seed 42 \
    --reward_mode contrastive_ergo \
    --affinity_scorer tfold \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 4 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 4 \
    --ban_stop \
    --terminal_reward_only \
    --n_contrast_decoys 1 \
    --contrastive_agg mean \
    --w_naturalness 0.1 \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.0 \
    --curriculum_l2 0.5 \
    --train_targets data/tfold_excellent_peptides.txt \
    --tfold_cache_path data/tfold_feature_cache.db
```

**Config file**: `configs/test51b.yaml`
- n_steps: 32 (from 128)
- n_envs: 4 (from 8)
- batch_size: 128 (adjusted to n_steps × n_envs)
- checkpoint_interval: 100000 (from 200000)
- decoy_K: 1 (from 2)
- decoy_tier_weights: A=1, B=0 (only tier A)

## Key Differences from test51

| Parameter | test51 | test51b | Impact |
|-----------|--------|---------|--------|
| n_contrast_decoys | 2 | 1 | 33% fewer tFold calls |
| n_steps | 128 | 32 | 4x faster rollout |
| n_envs | 8 | 4 | 50% lower GPU memory |
| checkpoint_interval | 200K | 100K | More frequent saves |
| decoy tiers | A+B | A only | Simpler, hardest negatives |

**tFold calls per episode**: 2 (1 target + 1 decoy) vs test51's 3

## Expected Outcome

**Training time**: ~3.5 days (target: <4 days)
- 2M steps / 4 envs / 4 steps = 500,000 episodes
- 500,000 episodes × 2 tFold calls = 1,000,000 total calls
- At 4 sec/call with 85% cache hit rate: 1,000,000 × 0.15 × 4s = 600,000s = 167h = 6.9 days
- **Correction**: At 85% cache hit rate: 1,000,000 × 0.15 × 4s / 3600 / 24 = 6.9 days
- **Need 90% cache hit rate**: 1,000,000 × 0.10 × 4s / 3600 / 24 = 4.6 days ✓

**First rollout**: 
- 32 steps × 4 envs = 128 steps
- 128 / 4 = 32 episodes
- 32 episodes × 2 tFold calls = 64 calls
- At 4 sec/call: 64 × 4s = 256s = 4.3 minutes (0% cache hit)

**Metrics**:
- Mean AUROC > 0.65 (target)
- Mean target affinity ≥ v1 level
- Naturalness score > test41

**If successful**: Proves pure tFold training is viable with optimized configuration
**If failed**: May need cascade approach (test49) or further reduction in decoys/steps

## Dependencies

**Code changes**: None (all parameters supported)

**Data requirements**:
- ✅ `data/tfold_excellent_peptides.txt` (20 peptides)
- ✅ tFold feature cache at `data/tfold_feature_cache.db` (54,425 entries at start)
- ✅ TCRdb data for L0/L2 curriculum

**Blocks**: None

## Reward Formula

```
reward = affinity(TCR, target) - affinity(TCR, decoy_A) + 0.1 * naturalness(TCR)
```

Where:
- `affinity`: tFold binding score (structure-aware)
- `decoy_A`: 1 peptide from tier A (1-2 AA point mutants, hardest negatives)
- `naturalness`: ESM perplexity z-score

## Checkpoint Strategy

- Every 100K steps: `milestone_100000.pt`, `milestone_200000.pt`, ..., `milestone_2000000.pt`
- Latest: `latest.pt` (every 50K steps, overwritten)
- Final: `final.pt` (at 2M steps)

Total: 20 milestone checkpoints + 1 final

## Evaluation Protocol

**Training evaluation**: Disabled (`eval_interval=999999999`)

**Final evaluation** (after training completes):
```bash
python tcrppo_v2/test_tcrs.py \
    --checkpoint output/test51b_tfold_terminal_opt/checkpoints/final.pt \
    --n_tcrs 10 \
    --n_decoys 32 \
    --affinity_scorer tfold \
    --output_dir results/test51b_tfold_terminal_opt/
```

Metrics:
- Per-target AUROC (10 TCRs vs 32 decoys per target)
- Mean AUROC across 20 targets
- Mean target affinity
- Mean decoy affinity
- Naturalness score distribution

## Risk Assessment

**Low risk**:
- Configuration is well-tested (only parameter changes from test51)
- tFold cache saves every prediction (no data loss)
- More frequent checkpoints (100K) allow early stopping if needed

**Medium risk**:
- Only 1 decoy may provide weaker specificity signal than 2 decoys
  - Mitigation: Tier A decoys are hardest negatives (1-2 AA mutations)
- 90% cache hit rate assumption may be optimistic
  - Mitigation: Can monitor and adjust if needed

**High risk**: None

## Monitoring Plan

**During training** (via `tail -f logs/test51b_tfold_terminal_opt_train.log`):
- First rollout completion time (should be ~4 minutes)
- Episode reward trend (should increase)
- tFold cache hit rate (should reach 85-90% after 500K steps)
- Episode length (should stabilize around 4 steps)
- Naturalness component (should be positive)

**Red flags** (consider stopping if observed):
- Reward not increasing after 500K steps
- Cache hit rate < 70% after 1M steps
- Training time projection > 5 days at 500K steps

## Time Estimates

Based on observed tFold speed (4 sec/call on GPU):

| Checkpoint | Steps | Episodes | tFold calls | Time (85% hit) | Time (90% hit) |
|------------|-------|----------|-------------|----------------|----------------|
| 100K | 100K | 25K | 50K | 8.3h | 5.6h |
| 500K | 500K | 125K | 250K | 41.7h | 27.8h |
| 1M | 1M | 250K | 500K | 83.3h | 55.6h |
| 2M | 2M | 500K | 1M | 166.7h | 111.1h |

**Target**: Complete 2M steps in <4 days (96 hours) requires **>90% cache hit rate**

## Next Steps

**If successful** (AUROC > 0.65, time < 4.5 days):
- test52: Increase to 2 decoys to test if specificity improves
- test53: Extend to 5M steps to test convergence
- test54: Try different decoy tiers (A+D instead of A only)

**If too slow** (projected time > 5 days at 500K steps):
- Stop and switch to test49 cascade approach
- Or reduce to 1M steps for faster validation

**If AUROC < 0.55**:
- Diagnose: Check reward-AUROC alignment for tFold
- May need 2 decoys for sufficient specificity signal
