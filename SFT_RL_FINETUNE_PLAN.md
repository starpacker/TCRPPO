# SFT + RL Fine-tuning Plan

**Date**: 2026-06-02
**Status**: Ready to launch
**Goal**: Improve SFT model from -5.49 to > -2.0 using RL fine-tuning

---

## Background

### SFT Model Performance (Baseline)
- **Mean Affinity**: -5.49 (failed to meet -2.0 target)
- **Training Data Quality**: -1.10
- **Degradation**: 4.39 units from training data to generated TCRs
- **Root Cause**: SFT learns trajectory imitation, not affinity optimization

### Why RL Fine-tuning?
1. **Direct Optimization**: RL optimizes affinity directly, not trajectory imitation
2. **Proven Success**: trace73 achieved -1.172 using RL
3. **Good Initialization**: SFT provides better starting point than random (-5.49 vs -10+)
4. **Faster Convergence**: Pre-trained model understands sequence editing operations

---

## Configuration

### Base Model
- **Checkpoint**: `output/sft_filtered_training/checkpoint_final.pt`
- **Architecture**: Actor-Critic with 3-head autoregressive policy
- **Hidden Dim**: 512
- **Training**: 50 epochs SFT on filtered data (no CCC patterns)

### RL Settings (Based on trace73)
- **Algorithm**: PPO
- **Total Steps**: 2M
- **Environments**: 8 parallel
- **Learning Rate**: 1.2e-4
- **Entropy Coef**: 0.020 (increased exploration)
- **Optimizer**: Reset (fresh start for RL)

### Reward Configuration
```yaml
reward_mode: v2_simple_target_gated_decoy
weights:
  affinity: 1.0
  decoy: 0.3
  naturalness: 0.05
  diversity: 0.02
```

### Curriculum Gate Schedule
Progressive difficulty to guide learning:

| Steps | Gate | Rationale |
|-------|------|-----------|
| 0-100K | -4.0 | Very easy (SFT baseline is -5.49) |
| 100K-200K | -3.0 | Gradual increase |
| 200K-400K | -2.0 | **Minimum target** |
| 400K-600K | -1.5 | Push higher |
| 600K+ | -1.0 | **Final target** |

**Note**: Starts easier than trace73 (-4.0 vs -2.0) because SFT baseline is worse.

### Training Targets
- **Peptides**: `data/tfold_excellent_peptides.txt` (20 high-quality peptides)
- **Affinity Scorer**: tFold (same as trace73)
- **Cache**: Reuse existing `data/tfold_feature_cache.db`

---

## Expected Timeline

### Phase 1: Warm-up (0-100K steps, ~2-3 hours)
- **Goal**: Adapt SFT policy to RL reward signal
- **Expected**: Affinity improves from -5.49 to -3.5
- **Gate**: -4.0 (easy to pass)

### Phase 2: Target Approach (100K-200K steps, ~2-3 hours)
- **Goal**: Reach minimum target
- **Expected**: Affinity improves to -2.5 to -2.0
- **Gate**: -3.0 → -2.0

### Phase 3: Target Achievement (200K-400K steps, ~4-6 hours)
- **Goal**: Consistently beat -2.0
- **Expected**: Affinity -2.0 to -1.5
- **Gate**: -2.0 (minimum target)

### Phase 4: Optimization (400K-600K steps, ~4-6 hours)
- **Goal**: Approach trace73 performance
- **Expected**: Affinity -1.5 to -1.2
- **Gate**: -1.5

### Phase 5: Final Push (600K-2M steps, ~12-16 hours)
- **Goal**: Match or exceed trace73 (-1.172)
- **Expected**: Affinity -1.2 to -1.0
- **Gate**: -1.0

**Total Estimated Time**: 24-34 hours

---

## Success Criteria

### Minimum (Must Achieve)
- ✅ Mean affinity > -2.0 by 400K steps
- ✅ No regression in sequence quality (no CCC patterns)
- ✅ Stable training (no collapse)

### Target (Desired)
- 🎯 Mean affinity > -1.5 by 600K steps
- 🎯 Mean affinity > -1.2 by 1M steps
- 🎯 At least 5% success rate (affinity > 0.0)

### Ideal (Stretch Goal)
- 🌟 Mean affinity > -1.0 by 2M steps
- 🌟 Match or exceed trace73 (-1.172)
- 🌟 10%+ success rate (affinity > 0.0)

---

## Monitoring

### Key Metrics
1. **Mean Affinity**: Primary metric, track per checkpoint
2. **Gate Pass Rate**: % of episodes passing current gate
3. **Reward Components**: Affinity, decoy, naturalness, diversity
4. **TCR Pool Size**: Online pool growth (should reach ~256/target)
5. **Sequence Quality**: Check for CCC patterns, length distribution

### Checkpoints
- **Interval**: Every 20K steps
- **Milestones**: 100K, 200K, 300K, 400K, 500K, 600K, 800K, 1M, 1.2M, 1.5M, 2M
- **Evaluation**: Generate 50 TCRs per milestone, score with tFold

### Logs
- **Training**: `logs/sft_rl_finetune_train.log`
- **tFold Server**: `logs/sft_rl_finetune_tfold_server.log`
- **Completion**: `logs/sft_rl_finetune_tfold_completion.log`

---

## Launch Commands

### Start Training
```bash
bash scripts/launch_sft_rl_finetune.sh
```

### Monitor Progress
```bash
# Training log
tail -f logs/sft_rl_finetune_train.log

# Attach to training session
tmux attach -t sft_rl_train

# Attach to tFold server
tmux attach -t tfold_sft_rl
```

### Check Status
```bash
# GPU usage
nvidia-smi

# Training process
ps aux | grep ppo_trainer

# Latest checkpoint
ls -lht output/sft_rl_finetune/checkpoints/ | head -5
```

---

## Comparison with Baselines

| Model | Mean Affinity | Method | Notes |
|-------|---------------|--------|-------|
| **Target** | **> -2.0** | - | Minimum acceptable |
| SFT (filtered) | -5.49 | Supervised | Failed, trajectory imitation insufficient |
| SFT (original) | -7.10 | Supervised | Worse, dummy observations |
| trace73 RL | -1.172 | PPO | Success, direct optimization |
| Training Data | -1.10 | - | Upper bound for SFT |
| **SFT+RL (this)** | **TBD** | PPO | Expected: -2.0 to -1.0 |

---

## Risk Mitigation

### Risk 1: Policy Collapse
- **Symptom**: Reward drops suddenly, all TCRs become identical
- **Mitigation**: Entropy bonus (0.020), diversity penalty, online pool
- **Action**: If detected, reduce learning rate, increase entropy

### Risk 2: Slow Convergence
- **Symptom**: Affinity stuck at -4.0 after 200K steps
- **Mitigation**: Curriculum gates, online TCR pool
- **Action**: Lower gate threshold, increase exploration

### Risk 3: CCC Pattern Re-emergence
- **Symptom**: Generated TCRs contain repetitive patterns
- **Mitigation**: Naturalness penalty, ESM perplexity
- **Action**: Increase naturalness weight, add explicit CCC penalty

### Risk 4: tFold Server Crash
- **Symptom**: Training hangs, socket errors
- **Mitigation**: AMP wrapper, completion log, auto-restart
- **Action**: Restart server, resume training from last checkpoint

---

## Next Steps After Completion

### If Successful (> -2.0)
1. **Evaluate on full test set**: 10 peptides × 50 TCRs
2. **Compare with trace73**: Same evaluation protocol
3. **Analyze improvements**: What did RL fix that SFT couldn't?
4. **Document findings**: Update progress_v2.md

### If Partially Successful (-3.0 to -2.0)
1. **Extend training**: Run to 3M or 5M steps
2. **Adjust curriculum**: Slower gate progression
3. **Increase exploration**: Higher entropy coefficient

### If Failed (< -3.0)
1. **Diagnose root cause**: Check reward components, policy gradients
2. **Try alternative**: Direct RL from scratch (no SFT)
3. **Consider hybrid**: SFT on real TCR databases (VDJdb, IEDB)

---

**Created**: 2026-06-02
**Ready to Launch**: ✅
