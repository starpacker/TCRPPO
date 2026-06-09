# trace84: Push to Mean A = 0.0

**Date**: 2026-05-30  
**Status**: planned  
**GPU**: 0  
**Priority**: P0 (critical goal)

## Hypothesis

trace83 achieved Mean A = -1.191 (excellent start) but degraded to -6.123 due to:
1. **Too aggressive gate schedule**: -3.0 → -1.0 in 400K steps
2. **No decoy penalty**: decoy_tier_weights all zero
3. **No curriculum**: L2 only (random starts)

**Solution**: Conservative gate schedule + gradual decoy unlock + curriculum learning should push Mean A to 0.0 without catastrophic forgetting.

## Configuration

```bash
CUDA_VISIBLE_DEVICES=0 python tcrppo_v2/ppo_trainer.py \
    --config configs/trace84_push_to_zero.yaml \
    --resume output/trace73_curriculum_exploration/checkpoints/latest.pt \
    --resume_reset_optimizer \
    --run_name trace84_push_to_zero \
    --seed 42
```

### Key Differences from trace83

| Parameter | trace83 | trace84 | Rationale |
|-----------|---------|---------|-----------|
| **Gate schedule** | -3.0 → -1.0 (400K) | -2.0 → 0.0 (1.5M) | Conservative, avoid forcing bad TCRs |
| **Decoy weights** | {A:0, B:0, D:0, C:0} | {A:3, B:3, D:2, C:1} | Enable specificity pressure |
| **Decoy unlock** | None | D → D+A → D+A+B → D+A+B+C | Gradual difficulty increase |
| **Curriculum** | L2 only | L0/L1 → balanced → L2 | Start from good TCRs |
| **decoy_K** | 2 | 8 | More decoys for better signal |
| **Total steps** | 1M | 2M | More time to reach 0.0 |
| **Online pool min_aff** | -10.0 | -1.0 | Only keep good TCRs |

### Gate Schedule (Conservative)

```yaml
0: -2.0          # Start where trace73 was
200000: -1.5     # Gentle push
500000: -1.0     # Half way
1000000: -0.5    # Almost there
1500000: 0.0     # TARGET!
```

### Decoy Unlock Schedule (Gradual)

```yaml
0: ["D"]                    # Known binders only (easiest)
100000: ["D", "A"]          # + point mutants
300000: ["D", "A", "B"]     # + 2-3 AA mutants
600000: ["D", "A", "B", "C"] # + unrelated peptides (hardest)
```

### Curriculum Schedule

```yaml
0-100K: L0=0.5, L1=0.3, L2=0.2    # Start from known good TCRs
100K-500K: L0=0.3, L1=0.3, L2=0.4  # Balanced
500K+: L0=0.1, L1=0.2, L2=0.7      # Mostly random (exploration)
```

## Expected Outcome

### Performance Milestones

| Step | Gate | Mean A (expected) | Decoy Tiers | Curriculum |
|------|------|-------------------|-------------|------------|
| 0 | -2.0 | -1.5 to -2.0 | D only | L0/L1 heavy |
| 100K | -2.0 | -1.5 to -1.8 | D + A | L0/L1 heavy |
| 200K | -1.5 | -1.2 to -1.5 | D + A | Balanced |
| 500K | -1.0 | -0.8 to -1.0 | D + A + B | Balanced |
| 1M | -0.5 | -0.3 to -0.5 | D + A + B + C | L2 heavy |
| 1.5M | **0.0** | **-0.2 to 0.0** | D + A + B + C | L2 heavy |
| 2M | 0.0 | **0.0 to +0.2** | D + A + B + C | L2 heavy |

### Success Criteria

- ✅ **Primary**: Mean A ≥ 0.0 at 1.5M-2M steps
- ✅ **Secondary**: No catastrophic forgetting (Mean A never drops below -3.0)
- ✅ **Tertiary**: Maintain specificity (decoy penalty active)

### Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Mean A stuck at -1.5 | Gate too conservative | Accelerate gate schedule |
| Mean A drops to -5.0 | Catastrophic forgetting | Slow down gate, increase L0/L1 |
| High decoy violation | Decoy penalty too weak | Increase w_decoy |
| Policy frozen (KL < 0.001) | Learning rate too low | Increase LR |

## Dependencies

- **Checkpoint**: `output/trace73_curriculum_exploration/checkpoints/latest.pt` (step 710K)
- **Targets**: `data/trace79_curated_targets.txt` (10 curated peptides)
- **tFold cache**: `data/tfold_feature_cache.db`
- **tFold server**: Must be running on `/tmp/tfold_server_trace84.sock`

## Monitoring

```bash
# Watch training progress
tail -f logs/trace84_push_to_zero_train.log

# Check Mean A trend
grep "Mean A" logs/trace84_push_to_zero_train.log | tail -20

# Check gate transitions
grep "Gate Schedule" logs/trace84_push_to_zero_train.log

# Check decoy unlock
grep "Decoy unlock" logs/trace84_push_to_zero_train.log
```

## Risk Assessment

**Low risk**:
- Resume from proven checkpoint (trace73)
- Conservative gate schedule (no forcing)
- Gradual decoy unlock (no shock)
- Curriculum provides safety net (L0/L1 fallback)

**Mitigation**:
- If Mean A drops below -3.0 for >50K steps → abort and adjust gate
- If KL < 0.001 for >10 updates → increase LR
- If decoy violation > 5.0 → increase w_decoy

## Timeline

- **Launch**: 2026-05-30
- **Expected completion**: ~24-36 hours (2M steps @ 8 envs)
- **First checkpoint**: 100K steps (~1-2 hours)
- **Target milestone**: 1.5M steps (~18-24 hours)

## Next Steps After Success

If trace84 reaches Mean A = 0.0:

1. **Evaluate specificity**: Run full decoy eval on final checkpoint
2. **Compare to baseline**: vs trace73, trace83, v1 baseline
3. **Generate TCRs**: 50 TCRs per target for analysis
4. **Document**: Update experiment tracker, write success report
5. **Push further**: Try trace85 with gate = +0.5 (super-binders)
