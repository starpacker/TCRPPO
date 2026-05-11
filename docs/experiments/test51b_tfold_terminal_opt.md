# test51b: tFold Terminal Reward with Single Decoy Contrastive Learning

**Date**: 2026-05-08 to 2026-05-09
**Status**: ❌ FAILED - Stopped due to severe reward instability
**GPU**: 0
**Priority**: P0
**Duration**: 214 hours (9 days)

## Configuration

```bash
CUDA_VISIBLE_DEVICES=0 python tcrppo_v2/ppo_trainer.py \
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

## Key Configuration
- **Reward mode**: contrastive_ergo (R = target_affinity - decoy_affinity + 0.1*naturalness)
- **n_contrast_decoys**: 1 (single random decoy per episode)
- **max_steps**: 4 (very short episodes)
- **gamma**: 0.90 (low discount factor)
- **use_znorm**: false (no reward normalization)
- **n_envs**: 4, **batch_size**: 128

## Results

**Training Progress:**
- Total steps: 5,632 / 2,000,000 (0.28% complete)
- Training time: 214 hours (9 days)
- Speed: ~26 steps/hour (extremely slow)

**Reward Statistics:**
- Mean reward: 0.109
- Std reward: 0.375 (coefficient of variation = 3.4x - extremely high!)
- Median reward: -0.002
- Max reward: 0.179 @ step 4,352
- Final reward: 0.050 @ step 5,632
- Negative episodes: 40.4%

**Reward Trajectory (last 20 checkpoints):**
```
Step 3200: 0.099
Step 3328: 0.072
Step 3456: 0.078
Step 3584: 0.116
Step 3712: 0.147
Step 3840: 0.131
Step 3968: 0.111
Step 4096: 0.128
Step 4224: 0.169
Step 4352: 0.179  ← Peak
Step 4480: 0.170
Step 4608: 0.163
Step 4736: 0.151
Step 4864: 0.132
Step 4992: 0.128
Step 5120: 0.088
Step 5248: 0.063
Step 5376: 0.058
Step 5504: 0.054
Step 5632: 0.050  ← Final (declining trend)
```

**Cache Performance:**
- tFold cache size: 82,867 entries
- Cache hit rate: 3-4% (extremely low)
- Most episodes require fresh tFold computation

## Analysis

### Root Causes of Failure

**1. Single Decoy High Variance (Critical)**
- Using only 1 random decoy per episode creates extreme reward variance
- Reward = target_score - single_decoy_score is highly sensitive to which decoy is sampled
- Easy decoy → large margin → high reward
- Hard decoy → small/negative margin → low/negative reward
- This random noise prevents stable learning

**2. No Reward Normalization**
- tFold scores have varying absolute ranges across peptides
- Contrastive margin uses raw score differences without z-score normalization
- Different peptides have incomparable reward scales

**3. Very Short Episodes (max_steps=4)**
- Only 4 editing steps per episode
- Terminal-only reward provides sparse learning signal
- Policy has insufficient opportunity to learn meaningful editing patterns

**4. Low Gamma for Terminal Reward**
- gamma=0.90 means step-0 action only receives 0.90^4 = 0.66 of terminal reward
- Severely weakens credit assignment for early actions
- Incompatible with terminal-only reward structure

**5. Low Cache Hit Rate**
- 96% of episodes compute fresh tFold features
- tFold structure prediction has inherent stochasticity
- Amplifies reward noise from computation variance

### Evidence of Instability

- **Reward peaked at step 4,352 then declined 72%** (0.179 → 0.050)
- **40% of episodes have negative rewards** (decoy scored higher than target)
- **Coefficient of variation = 3.4** (std/mean) - indicates extreme instability
- **No convergence after 9 days** - policy oscillating rather than learning

## Hypothesis Confirmed/Rejected

**Hypothesis**: Terminal-only tFold reward with single decoy contrastive learning will provide stable training signal.

**Result**: ❌ **REJECTED**

**Why**: Single decoy sampling introduces unacceptable variance. The reward signal is dominated by random decoy selection rather than actual TCR quality. Combined with short episodes, low gamma, and no normalization, the policy cannot extract meaningful learning signal.

## Key Findings

1. **Single decoy is insufficient** - Need 4-8 decoys with mean aggregation to reduce variance
2. **Terminal-only reward needs high gamma** - Should use gamma ≥ 0.98 for 4-8 step episodes
3. **tFold cache hit rate is low** - Pre-warming cache or using faster scorer (ERGO) for majority of episodes would help
4. **Reward normalization is critical** - Raw tFold scores need z-score normalization across peptides

## Next Steps

**test51c** will address these issues:
- ✅ Remove decoy entirely (n_contrast_decoys=0) - eliminate variance source
- ✅ Longer episodes (max_steps=8) - more editing opportunities
- ✅ Higher gamma (0.98) - better credit assignment
- ✅ V2 full reward (affinity + naturalness + diversity) - richer signal
- ✅ Larger batch (n_envs=8, batch_size=256) - more stable gradients

## Files

- Config: `configs/test51b.yaml`
- Launch script: `scripts/launch_test51b_tfold_terminal_opt.sh`
- Log: `logs/test51b_tfold_terminal_opt_train.log`
- Output: `output/test51b_tfold_terminal_opt/`
- Reward data: `/tmp/test51b_rewards.txt`
- Reward curve: `figures/test51b_reward_curve.png`
