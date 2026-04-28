# PPO Implementation Analysis and Potential Issues

**Date**: 2026-04-25  
**Context**: User asked to review PPO algorithm implementation for potential improvements

---

## Current PPO Configuration

```yaml
n_envs: 8                    # Parallel environments
n_steps: 128                 # Steps per rollout
batch_size: 256              # Minibatch size for updates
n_epochs: 4                  # Epochs per rollout
learning_rate: 3e-4          # Adam learning rate
gamma: 0.90                  # Discount factor
gae_lambda: 0.95             # GAE lambda
clip_range: 0.2              # PPO clip epsilon
entropy_coef: 0.05           # Entropy bonus
vf_coef: 0.5                 # Value loss coefficient
max_grad_norm: 0.5           # Gradient clipping
```

**Effective batch size per update**: 8 envs × 128 steps = 1024 samples  
**Minibatches per epoch**: 1024 / 256 = 4 minibatches  
**Total gradient steps per rollout**: 4 epochs × 4 minibatches = 16 gradient steps

---

## Identified Issues and Proposed Fixes

### Issue 1: **Gamma Too Low for Long-Horizon Credit Assignment**

**Problem**:
- `gamma = 0.90` means rewards decay to 35% after 10 steps
- TCR editing episodes are 8 steps long, but the reward is terminal (only at STOP)
- With gamma=0.90, step 0 only sees 0.90^8 = 0.43 of the final reward
- This severely weakens credit assignment for early actions

**Evidence**:
- Episodes are exactly 8 steps (max_steps=8, ban_stop=True forces full trajectory)
- Reward is terminal (only given at episode end)
- Early actions (CDR3 region edits) are critical but get 57% discounted signal

**Fix**:
```yaml
gamma: 0.99  # Standard for episodic tasks
# OR
gamma: 0.95  # Moderate compromise
```

**Expected Impact**: Stronger credit assignment to early actions → better CDR3 region optimization

---

### Issue 2: **Clip Range May Be Too Conservative**

**Problem**:
- `clip_range = 0.2` is standard for continuous control (MuJoCo)
- But TCR editing has:
  - Discrete action space (not continuous)
  - Sparse terminal reward (not dense)
  - Autoregressive dependencies (op → pos → token)
- Conservative clipping may slow learning when policy needs large updates

**Evidence from training logs**:
- Policy gradient loss (PG) is consistently small (-0.015 to -0.025)
- Suggests policy updates are being clipped frequently
- Reward improvements are slow (R: 1.69 → 2.49 over 235K steps in test27)

**Fix Options**:

**Option A: Adaptive clipping** (recommended)
```python
# Start with larger clip_range, decay over time
def get_clip_range(global_step, total_steps):
    # Linear decay from 0.3 to 0.1
    progress = global_step / total_steps
    return 0.3 - 0.2 * progress
```

**Option B: Increase static clip_range**
```yaml
clip_range: 0.3  # Allow larger policy updates
```

**Expected Impact**: Faster convergence in early training, more aggressive exploration

---

### Issue 3: **Batch Size Too Small Relative to Rollout Size**

**Problem**:
- Rollout size: 1024 samples (8 envs × 128 steps)
- Minibatch size: 256 samples
- Only 4 minibatches per epoch
- With 4 epochs, each sample is seen 4 times with only 4 different gradient estimates

**Standard PPO practice**:
- Minibatch size should be 1/8 to 1/16 of rollout size
- Current: 256/1024 = 1/4 (too large)
- Leads to high variance in gradient estimates

**Fix**:
```yaml
batch_size: 128  # 1/8 of rollout size
# This gives 8 minibatches per epoch
# Total: 4 epochs × 8 minibatches = 32 gradient steps per rollout
```

**Expected Impact**: Lower variance gradients, more stable learning

---

### Issue 4: **Value Function Coefficient Too High**

**Problem**:
- `vf_coef = 0.5` means value loss has equal weight to policy loss
- Standard PPO uses 0.5, but this is for continuous control with dense rewards
- For sparse terminal rewards, value function is harder to learn
- High vf_coef can cause value function to dominate training

**Evidence**:
- Value loss (VF) is consistently higher than policy loss (0.10-0.20 vs 0.015-0.025)
- Value function may be overfitting to noisy terminal rewards

**Fix**:
```yaml
vf_coef: 0.25  # Reduce value loss weight
# OR
vf_coef: 0.1   # Even lower for very sparse rewards
```

**Expected Impact**: Policy updates less constrained by value function errors

---

### Issue 5: **No KL Divergence Monitoring or Early Stopping**

**Problem**:
- PPO clips ratio but doesn't monitor KL divergence between old and new policy
- Without KL monitoring, we don't know if clipping is too tight or too loose
- No early stopping within epochs if KL exceeds threshold

**Standard PPO practice**:
- Compute KL divergence after each minibatch
- Stop epoch early if mean KL > target_kl (e.g., 0.01-0.03)
- Prevents policy from changing too much per update

**Fix**: Add KL monitoring and early stopping
```python
# In PPO update loop
target_kl = 0.015  # Standard value
approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

if approx_kl > target_kl:
    print(f"Early stopping at epoch {epoch} due to KL={approx_kl:.4f}")
    break
```

**Expected Impact**: More stable training, prevents policy collapse

---

### Issue 6: **Advantage Normalization Per Minibatch (Not Per Rollout)**

**Current code** (line 803-804):
```python
adv = batch["advantages"]
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
```

**Problem**:
- Advantages are normalized per minibatch (256 samples)
- Different minibatches have different normalization scales
- This breaks the relative importance of different samples

**Standard practice**:
- Normalize advantages once per rollout (all 1024 samples together)
- Then split into minibatches

**Fix**:
```python
# In buffer.compute_gae(), after computing advantages:
self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

# In PPO update loop:
adv = batch["advantages"]  # Already normalized, don't normalize again
```

**Expected Impact**: More consistent gradient magnitudes across minibatches

---

### Issue 7: **Entropy Coefficient Too Low for Exploration**

**Problem**:
- `entropy_coef = 0.05` is standard for continuous control
- But TCR editing has:
  - Large discrete action space (4 ops × 20 positions × 20 tokens = 1600 actions)
  - Sparse rewards (only at episode end)
  - Needs strong exploration early in training

**Evidence**:
- Entropy decays quickly (5.35 → 4.71 in test27 logs)
- Policy may be collapsing to local optima too early

**Fix**:
```yaml
entropy_coef: 0.1           # Start higher
entropy_coef_final: 0.01    # Decay to lower value
entropy_decay_start: 500000 # Start decay after warmup
```

**Expected Impact**: Better exploration in early training, prevents premature convergence

---

### Issue 8: **No Reward Scaling or Normalization**

**Problem**:
- Raw rewards range from ~1.5 to ~2.5 (ERGO affinity scores)
- No reward normalization or scaling
- PPO is sensitive to reward scale - affects advantage magnitude

**Standard practice**:
- Normalize rewards using running mean/std
- OR scale rewards to [-1, 1] or [0, 1] range

**Fix**: Add reward normalization
```python
class RunningMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
    
    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count
    
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

# In trainer:
self.reward_normalizer = RunningMeanStd()

# In rollout collection:
normalized_rewards = self.reward_normalizer.normalize(rewards)
self.reward_normalizer.update(rewards)
```

**Expected Impact**: More stable learning, consistent advantage magnitudes

---

## Recommended Action Plan

### Priority 1: Critical Fixes (Implement Immediately)

1. **Increase gamma to 0.99** - Most impactful for terminal reward tasks
2. **Normalize advantages per rollout, not per minibatch** - Fixes gradient inconsistency
3. **Add KL monitoring and early stopping** - Prevents policy collapse

### Priority 2: High-Impact Improvements

4. **Reduce batch_size to 128** - Better gradient estimates
5. **Increase entropy_coef to 0.1 with decay** - Better exploration
6. **Reduce vf_coef to 0.25** - Less value function interference

### Priority 3: Advanced Enhancements

7. **Add reward normalization** - More stable learning
8. **Implement adaptive clip_range** - Faster early convergence

---

## Proposed test43: PPO Hyperparameter Optimization

**Hypothesis**: Fixing PPO hyperparameters will improve sample efficiency and final performance.

**Configuration changes from test42**:
```yaml
# Critical fixes
gamma: 0.99                  # Was 0.90
batch_size: 128              # Was 256
entropy_coef: 0.1            # Was 0.05
entropy_coef_final: 0.01
entropy_decay_start: 500000

# High-impact improvements
vf_coef: 0.25                # Was 0.5
clip_range: 0.3              # Was 0.2

# Code changes needed:
# 1. Normalize advantages per rollout (not per minibatch)
# 2. Add KL monitoring and early stopping (target_kl=0.015)
# 3. Add reward normalization (RunningMeanStd)
```

**Expected outcomes**:
- Faster convergence (reach R=2.0 in <1M steps instead of 2M)
- Higher final AUROC (>0.65 vs current 0.62)
- More stable training (lower variance in episode rewards)

---

## References

- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Engstrom et al. (2020): "Implementation Matters in Deep RL"
- Huang et al. (2022): "The 37 Implementation Details of PPO"
