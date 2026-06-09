#!/usr/bin/env python3
"""
Deep dive analysis: Why is RL training so inefficient?
Analyze learning dynamics beyond just reward design.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

log_dir = Path("/share/liuyutian/tcrppo_v2/logs")

# Analyze trace53 (most mature)
log_file = log_dir / "trace53_terminal_trace29_reward_L2only_train.log"

print("="*80)
print("DEEP DIVE: Why is RL training inefficient?")
print("="*80)
print()

# Extract multiple metrics
data = defaultdict(list)

with open(log_file, 'r') as f:
    for line in f:
        # PPO Update metrics
        if 'PPO Update' in line:
            # Extract update number
            update_match = re.search(r'PPO Update (\d+)', line)
            if update_match:
                update_num = int(update_match.group(1))
                data['update_num'].append(update_num)

            # Extract losses and metrics
            for metric in ['policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction']:
                match = re.search(rf'{metric}=([+-]?\d+\.\d+)', line)
                if match:
                    data[metric].append(float(match.group(1)))

        # Episode metrics
        if 'Episode' in line and 'Step' in line:
            step_match = re.search(r'Step (\d+)', line)
            aff_match = re.search(r' A=([+-]?\d+\.\d+)', line)
            delta_match = re.search(r'DeltaA=([+-]?\d+\.\d+)', line)
            init_match = re.search(r'InitA=([+-]?\d+\.\d+)', line)

            if step_match and aff_match:
                data['episode_steps'].append(int(step_match.group(1)))
                data['episode_aff'].append(float(aff_match.group(1)))

                if delta_match:
                    data['episode_delta'].append(float(delta_match.group(1)))
                if init_match:
                    data['episode_init'].append(float(init_match.group(1)))

# Convert to numpy arrays
for key in data:
    data[key] = np.array(data[key])

print(f"Loaded {len(data['episode_aff'])} episodes, {len(data['update_num'])} PPO updates")
print()

# Analysis 1: Policy gradient strength
print("="*80)
print("ANALYSIS 1: Policy Gradient Strength")
print("="*80)

if len(data['approx_kl']) > 0:
    kl = data['approx_kl']
    print(f"KL divergence statistics:")
    print(f"  Mean: {np.mean(kl):.6f}")
    print(f"  Std:  {np.std(kl):.6f}")
    print(f"  Min:  {np.min(kl):.6f}")
    print(f"  Max:  {np.max(kl):.6f}")
    print()
    print(f"Healthy KL range: 0.01 - 0.05")
    print(f"Current KL: {np.mean(kl):.6f}")

    if np.mean(kl) < 0.01:
        print("⚠️  KL TOO LOW: Policy barely updating!")
        print("   → Policy gradient is too weak")
        print("   → Learning rate might be too low")
        print("   → Or advantage estimates are too noisy")
    elif np.mean(kl) > 0.05:
        print("⚠️  KL TOO HIGH: Policy updating too aggressively!")
        print("   → Might cause instability")
    else:
        print("✓ KL in healthy range")
print()

# Analysis 2: Value function learning
print("="*80)
print("ANALYSIS 2: Value Function Learning")
print("="*80)

if len(data['value_loss']) > 0:
    vf_loss = data['value_loss']

    # Check if VF loss is decreasing
    first_half = vf_loss[:len(vf_loss)//2]
    second_half = vf_loss[len(vf_loss)//2:]

    print(f"Value loss statistics:")
    print(f"  First half mean: {np.mean(first_half):.4f}")
    print(f"  Second half mean: {np.mean(second_half):.4f}")
    print(f"  Change: {np.mean(second_half) - np.mean(first_half):.4f}")
    print()

    if np.mean(second_half) >= np.mean(first_half):
        print("⚠️  VALUE FUNCTION NOT LEARNING!")
        print("   → VF loss not decreasing over time")
        print("   → Cannot provide accurate advantage estimates")
        print("   → This causes weak/wrong policy gradients")
    else:
        print("✓ Value function learning (loss decreasing)")
print()

# Analysis 3: Improvement ability
print("="*80)
print("ANALYSIS 3: Improvement Ability (DeltaA)")
print("="*80)

if len(data['episode_delta']) > 0:
    delta = data['episode_delta']

    print(f"DeltaA statistics:")
    print(f"  Mean: {np.mean(delta):.4f}")
    print(f"  Std:  {np.std(delta):.4f}")
    print(f"  Positive rate: {np.sum(delta > 0) / len(delta) * 100:.1f}%")
    print()

    # Check trend
    window = 1000
    if len(delta) > window * 2:
        early = delta[:window]
        late = delta[-window:]

        print(f"Trend analysis (first {window} vs last {window}):")
        print(f"  Early mean: {np.mean(early):.4f}")
        print(f"  Late mean:  {np.mean(late):.4f}")
        print(f"  Change: {np.mean(late) - np.mean(early):.4f}")
        print()

        if np.mean(late) < np.mean(early) - 0.5:
            print("⚠️  IMPROVEMENT ABILITY DEGRADING!")
            print("   → Policy is getting worse at improving TCRs")
            print("   → Possible catastrophic forgetting")
        elif abs(np.mean(late) - np.mean(early)) < 0.1:
            print("⚠️  NO IMPROVEMENT IN IMPROVEMENT ABILITY!")
            print("   → Policy not learning to improve better")
            print("   → Stuck in local optimum")
        else:
            print("✓ Improvement ability stable or improving")
print()

# Analysis 4: Exploration vs Exploitation
print("="*80)
print("ANALYSIS 4: Exploration vs Exploitation")
print("="*80)

if len(data['entropy']) > 0:
    entropy = data['entropy']

    print(f"Entropy statistics:")
    print(f"  Mean: {np.mean(entropy):.4f}")
    print(f"  Std:  {np.std(entropy):.4f}")
    print()

    # Check trend
    first_half = entropy[:len(entropy)//2]
    second_half = entropy[len(entropy)//2:]

    print(f"Entropy trend:")
    print(f"  First half: {np.mean(first_half):.4f}")
    print(f"  Second half: {np.mean(second_half):.4f}")
    print(f"  Change: {np.mean(second_half) - np.mean(first_half):.4f}")
    print()

    if np.mean(second_half) < np.mean(first_half) - 0.5:
        print("⚠️  ENTROPY DECREASING TOO FAST!")
        print("   → Policy converging prematurely")
        print("   → Might be stuck in local optimum")
        print("   → Need more exploration")
    elif np.mean(second_half) > np.mean(first_half):
        print("⚠️  ENTROPY INCREASING!")
        print("   → Policy becoming more random")
        print("   → Not converging properly")
    else:
        print("✓ Entropy decreasing gradually (healthy)")
print()

# Analysis 5: Plateau detection
print("="*80)
print("ANALYSIS 5: Plateau Detection")
print("="*80)

if len(data['episode_aff']) > 0:
    aff = data['episode_aff']

    # Compute moving average
    window = 500
    if len(aff) > window:
        ma = np.convolve(aff, np.ones(window)/window, mode='valid')

        # Find plateaus (where MA doesn't change much)
        ma_diff = np.abs(np.diff(ma))
        plateau_threshold = 0.01  # Less than 0.01 change per 500 episodes

        plateau_ratio = np.sum(ma_diff < plateau_threshold) / len(ma_diff)

        print(f"Plateau analysis (MA window={window}):")
        print(f"  Plateau ratio: {plateau_ratio * 100:.1f}%")
        print(f"  (Ratio of time where MA changes < {plateau_threshold})")
        print()

        if plateau_ratio > 0.5:
            print("⚠️  SPENDING >50% TIME IN PLATEAU!")
            print("   → Training is very inefficient")
            print("   → Policy not making progress")
            print()
            print("Possible causes:")
            print("  1. Weak policy gradient (low KL)")
            print("  2. Poor value function (inaccurate advantages)")
            print("  3. Stuck in local optimum (low entropy)")
            print("  4. Reward signal too sparse/noisy")
            print("  5. Architecture bottleneck (policy/value network too small)")
        else:
            print("✓ Reasonable progress (not stuck in plateau)")
print()

# Analysis 6: Sample efficiency
print("="*80)
print("ANALYSIS 6: Sample Efficiency")
print("="*80)

if len(data['episode_aff']) > 0 and len(data['episode_steps']) > 0:
    total_steps = data['episode_steps'][-1] - data['episode_steps'][0]
    total_episodes = len(data['episode_aff'])

    # Compute improvement per 10K steps
    aff = data['episode_aff']
    steps = data['episode_steps']

    # Bin by 10K steps
    step_bins = np.arange(steps[0], steps[-1], 10000)
    bin_means = []

    for i in range(len(step_bins) - 1):
        mask = (steps >= step_bins[i]) & (steps < step_bins[i+1])
        if np.sum(mask) > 0:
            bin_means.append(np.mean(aff[mask]))

    if len(bin_means) > 1:
        improvement_per_10k = (bin_means[-1] - bin_means[0]) / len(bin_means) * 10

        print(f"Sample efficiency:")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Total episodes: {total_episodes:,}")
        print(f"  Initial mean A: {bin_means[0]:.4f}")
        print(f"  Final mean A: {bin_means[-1]:.4f}")
        print(f"  Total improvement: {bin_means[-1] - bin_means[0]:.4f}")
        print(f"  Improvement per 10K steps: {improvement_per_10k:.4f}")
        print()

        if abs(improvement_per_10k) < 0.01:
            print("⚠️  VERY LOW SAMPLE EFFICIENCY!")
            print("   → Barely improving despite many samples")
            print("   → Need to fix learning dynamics")
        elif improvement_per_10k < 0:
            print("⚠️  NEGATIVE SAMPLE EFFICIENCY!")
            print("   → Getting worse over time")
            print("   → Catastrophic forgetting")
        else:
            print(f"✓ Positive sample efficiency: +{improvement_per_10k:.4f} per 10K steps")

print()
print("="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)
print()
print("Based on the analysis above, the main bottlenecks are likely:")
print()
print("1. **Weak policy gradient** (if KL < 0.01)")
print("   → Increase learning rate")
print("   → Increase entropy coefficient (more exploration)")
print()
print("2. **Poor value function** (if VF loss not decreasing)")
print("   → Increase vf_coef")
print("   → Use better value function architecture")
print("   → More training data (larger batch size)")
print()
print("3. **Architecture bottleneck** (if both above are OK)")
print("   → Increase hidden_dim (512 → 1024)")
print("   → Add more layers")
print("   → Better state representation")
print()
print("4. **Sparse reward signal** (if improvement ability low)")
print("   → Add dense rewards (per-step delta)")
print("   → Better reward shaping")
print()
print("5. **Local optimum** (if entropy too low)")
print("   → Increase entropy_coef")
print("   → Add exploration bonus")
print("   → Curriculum learning")
print()
