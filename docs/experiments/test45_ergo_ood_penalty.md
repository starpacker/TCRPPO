# test45: ERGO with OOD Penalty

**Date**: 2026-05-03
**Status**: completed (FAILED)
**GPU**: 4
**Priority**: P0
**Completed**: 2026-05-03 23:06
**Training time**: ~4 hours

## Hypothesis

Constraining ERGO to its reliable domain prevents exploitation of OOD blind spots, resulting in more robust TCR designs with better real-world specificity.

When ERGO encounters TCRs outside its training distribution (high MC Dropout uncertainty), the reward is penalized. This forces the agent to design TCRs within ERGO's reliable domain, avoiding "OOD exploits" where sequences score high but are not truly binding.

## Configuration

```bash
CUDA_VISIBLE_DEVICES=4 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test45_ergo_ood_penalty \
    --seed 42 \
    --reward_mode v1_ergo_ood_penalty \
    --affinity_scorer ergo \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/ergo_positive_peptides.txt \
    --ood_threshold 0.15 \
    --ood_penalty_weight 1.0 \
    --ood_penalty_mode soft \
    --checkpoint_freq 50000
```

## Key Differences from Previous Experiments

- vs test41: Uses v1_ergo_ood_penalty reward mode instead of v1_ergo_only
- vs test41: Uses MC Dropout uncertainty estimation (10 forward passes) instead of single pass
- vs test41: Only 4 positive-aligned ERGO peptides instead of 12 McPAS
- vs test14: Same base settings but with OOD penalty to constrain exploration
- NEW: OOD detection via ERGO MC Dropout uncertainty
- NEW: Soft penalty = (uncertainty - threshold) * weight when uncertainty > threshold

## OOD Penalty Mechanism

```python
# MC Dropout: 10 forward passes with dropout enabled
means, stds = ergo_scorer.mc_dropout_score([tcr], [peptide])
uncertainty = stds[0]  # High std = likely OOD

# Soft penalty (only penalizes excess beyond threshold)
if uncertainty > 0.15:
    penalty = (uncertainty - 0.15) * 1.0
    reward = mean_score - penalty
else:
    reward = mean_score  # No penalty when in-distribution
```

## Target Peptides (4 ERGO Positive-Aligned)

From CRITICAL_LESSON_PEPTIDE_SCORER_ALIGNMENT.md:
- KLWASPLHV (gap +0.027)
- FPRPWLHGL (gap +0.050)
- KAFSPEVIPMF (gap +0.046)
- HSKKKCDEL (gap +0.081)

Excluded 3 reversed peptides: RFYKTLRAEQASQ, DRFYKTLRAEQASQEV, FRCPRRFCF

## Expected Speed

- Same as standard ERGO training BUT with MC Dropout overhead
- MC Dropout: 10 forward passes instead of 1 → ~10x slower scoring
- ERGO single pass: ~10ms, MC Dropout: ~100ms per sample
- With n_envs=8, n_steps=128: 1024 samples per rollout × 100ms = ~102s per rollout (~2 min)
- Total training time: ~12-24 hours

## Risk Analysis and Mitigations

### Risk 1: OOD penalty too aggressive → limits exploration
- **Symptom**: Agent stops improving early (reward plateau)
- **Detection**: OOD trigger rate > 90% at 100K steps
- **Mitigation**: Reduce ood_penalty_weight from 1.0 to 0.5, or raise threshold from 0.15 to 0.20
- **Design**: Using "soft" mode (only penalizes excess) is inherently more conservative than "hard"

### Risk 2: Threshold set incorrectly
- **Symptom**: Either no samples penalized (threshold too high) or all samples penalized (threshold too low)
- **Detection**: Monitor OOD trigger rate in logs
- **Mitigation**: If trigger rate < 5%, lower threshold to 0.10; if > 80%, raise to 0.20
- **Note**: 0.15 is based on typical ERGO MC Dropout std distribution (~25th percentile)

### Risk 3: ERGO uncertainty is not a reliable OOD indicator
- **Symptom**: High uncertainty on known-good TCRs, or low uncertainty on adversarial TCRs
- **Detection**: Compare OOD penalty rate with actual binding (tFold validation)
- **Mitigation**: If correlation < 0.3, switch to TFoldCascadeScorer approach

### Risk 4: Only 4 peptides → insufficient training signal diversity
- **Symptom**: Overfitting, poor generalization to other peptides
- **Detection**: Eval AUROC drops for non-training peptides
- **Mitigation**: Expand to 7 peptides including reversed ones (OOD penalty should handle the reversal)

## Expected Outcome

- **Mean AUROC**: 0.60-0.65 (comparable to test41, more robust)
- **OOD trigger rate**: Should start at ~30-50% and decrease to ~10-20% as agent learns domain
- **Training time**: ~12-24 hours
- **If successful**: Proves OOD constraint prevents ERGO exploitation
- **If failed**: ERGO MC Dropout is not a reliable enough OOD signal

## Monitoring

```bash
# Watch training progress + OOD stats
tail -f logs/test45_ergo_ood_penalty_train.log

# Check OOD trigger rate
grep "OOD:" logs/test45_ergo_ood_penalty_train.log | tail -20

# Check reward trends
grep "R:" logs/test45_ergo_ood_penalty_train.log | tail -20
```

## Success Criteria

1. Mean AUROC ≥ 0.60 on 12 McPAS targets (eval uses all targets, not just training 4)
2. OOD trigger rate decreases over time (agent learns to stay in-domain)
3. No worse than test14 (0.6091) on the 4 training peptides
4. tFold validation on generated TCRs shows higher scores than test41 (structure plausibility)

---

## Results

**Training completed**: 2026-05-03 23:06
**Total time**: ~4 hours (19:08-23:06)
**Final reward**: R=4.25
**OOD trigger rate**: 36.5% (stable throughout training, did NOT decrease)

### AUROC Results (12 McPAS targets)

| Scorer | Mean AUROC | vs test41 (0.6243) | vs test14 (0.6091) |
|--------|-----------|-------------------|-------------------|
| **ERGO** | **0.4031** | **-0.2212** | **-0.2060** |
| NetTCR | 0.5436 | -0.0807 | -0.0655 |
| tFold | 0.5000 | N/A (all cache miss) | N/A |

### Per-Target AUROC (ERGO)

| Target | AUROC | vs test41 | Status |
|--------|-------|-----------|--------|
| IVTDFSVIK | 0.8744 | +0.0630 | ✅ Good |
| AVFDRKSDAK | 0.6562 | -0.0772 | ⚠️ Acceptable |
| KLGGALQAK | 0.5256 | -0.1022 | ❌ Poor |
| RLRAEAQVK | 0.4781 | -0.1933 | ❌ Poor |
| SPRWYFYYL | 0.3456 | -0.1816 | ❌ Poor |
| NLVPMVATV | 0.3187 | -0.1697 | ❌ Poor |
| YLQPRTFLL | 0.2996 | -0.5982 | ❌ Catastrophic |
| FLYALALLL | 0.2754 | -0.1998 | ❌ Poor |
| SLYNTVATL | 0.2558 | -0.3587 | ❌ Catastrophic |
| GLCTLVAML | 0.2544 | -0.0994 | ❌ Poor |
| GILGFVFTL | 0.1754 | -0.2132 | ❌ Poor |
| LLWNGPMAV | 0.1408 | -0.6615 | ❌ Catastrophic |
| **Mean** | **0.4031** | **-0.2212** | **FAILED** |

### Training Curve

- Step 10K: R=1.22, OOD=13.2%
- Step 100K: R=2.50, OOD=23.5%
- Step 500K: R=3.45, OOD=33.8%
- Step 1M: R=3.85, OOD=35.5%
- Step 2M: R=4.25, OOD=36.5%

### Key Observations

1. **OOD trigger rate increased** from 13% to 36.5% during training — agent did NOT learn to stay in-domain
2. **ERGO AUROC worse than v1 baseline** (0.4031 vs 0.4538) — OOD penalty harmed learning
3. **Only 2/12 targets acceptable** (AUROC > 0.60) — much worse than test41 (5/12)
4. **tFold cache miss on all generated TCRs** — sequences are outside tFold's training distribution
5. **Reward increased normally** (1.22 → 4.25) — agent optimized ERGO score but lost specificity

## Analysis

**Hypothesis rejected**: OOD penalty did NOT constrain agent to ERGO's reliable domain.

**Root causes**:

1. **OOD threshold too low (0.15)**: 36.5% trigger rate means penalty was applied too frequently, interfering with normal learning
2. **ERGO MC Dropout uncertainty is not a reliable OOD indicator**: High uncertainty does not correlate with poor specificity
3. **Soft penalty insufficient**: (uncertainty - 0.15) * 1.0 was too weak to guide behavior
4. **Only 4 training peptides**: Insufficient diversity, agent overfitted to these 4 targets

**Why OOD rate increased**:
- Agent learned to generate TCRs with high ERGO score AND high uncertainty
- High uncertainty TCRs are likely novel/unusual sequences that ERGO scores unreliably
- Penalty discouraged exploration of reliable in-domain space

**Comparison with test14 (pure ERGO, no OOD penalty)**:
- test14: 0.6091 AUROC (4/12 targets > 0.65)
- test45: 0.4031 AUROC (2/12 targets > 0.65)
- **Conclusion**: Adding OOD penalty made results WORSE

## Lessons Learned

1. **MC Dropout uncertainty ≠ OOD detection**: ERGO's uncertainty does not predict specificity failure
2. **Penalty-based constraints are fragile**: Easy to set wrong threshold/weight and harm learning
3. **Need better OOD signal**: Consider tFold cascade (ERGO + tFold validation) instead of uncertainty
4. **More training peptides needed**: 4 peptides insufficient for robust learning

## Next Steps

1. ❌ **Do NOT retry OOD penalty** with different hyperparameters — fundamental approach is flawed
2. ✅ **Use tFold cascade approach** (test44) — structure-based validation is more reliable
3. ✅ **Expand training peptides** to 29 tFold-good peptides (test44 already doing this)
4. ✅ **Two-phase training** remains best approach (test41: 0.6243 AUROC)
