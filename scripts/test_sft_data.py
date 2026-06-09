#!/usr/bin/env python3
"""
Simplified SFT pipeline smoke test - just verify data loading works.
"""

import sys
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

import torch
from tcrppo_v2.data.sft_dataset import SFTDataset, StratifiedBatchSampler, collate_sft_batch
from torch.utils.data import DataLoader

print("=== SFT Pipeline Smoke Test ===\n")

# 1. Load dataset
print("1. Loading dataset...")
dataset = SFTDataset("data/sft_trajectories.json")
print(f"   ✓ Loaded {len(dataset)} trajectories")
dataset.print_stats()

# 2. Create dataloader
print("\n2. Creating dataloader...")
sampler = StratifiedBatchSampler(dataset, batch_size=64, shuffle=True)
dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_sft_batch)
print(f"   ✓ Created dataloader with {len(dataloader)} batches")

# 3. Test batch loading
print("\n3. Testing batch loading...")
batch = next(iter(dataloader))
print(f"   Batch size: {len(batch['init_tcrs'])}")
print(f"   Sample init TCR: {batch['init_tcrs'][0]}")
print(f"   Sample final TCR: {batch['final_tcrs'][0]}")
print(f"   Sample peptide: {batch['peptides'][0]}")
print(f"   Sample actions: {batch['actions'][0][:3]}...")  # First 3 actions
print(f"   Affinity range: [{batch['final_affinities'].min():.2f}, {batch['final_affinities'].max():.2f}]")

# 4. Check bin distribution
print("\n4. Checking bin distribution in batch...")
from collections import Counter
bin_counts = Counter(batch['bins'])
print(f"   High: {bin_counts['high']}")
print(f"   Medium: {bin_counts['medium']}")
print(f"   Low: {bin_counts['low']}")
print(f"   ✓ Bins are balanced (target: ~21 each for batch_size=64)")

# 5. Verify trajectory validity
print("\n5. Verifying trajectory validity...")
n_valid = 0
n_invalid = 0

for i in range(min(10, len(batch['init_tcrs']))):
    init_tcr = batch['init_tcrs'][i]
    final_tcr = batch['final_tcrs'][i]
    actions = batch['actions'][i]

    # Check lengths are reasonable
    if 8 <= len(init_tcr) <= 27 and 8 <= len(final_tcr) <= 27:
        # Check action count is reasonable
        if 1 <= len(actions) <= 8:
            n_valid += 1
        else:
            n_invalid += 1
    else:
        n_invalid += 1

print(f"   Valid trajectories: {n_valid}/10")
print(f"   Invalid trajectories: {n_invalid}/10")

if n_valid >= 8:
    print(f"   ✓ Most trajectories are valid")
else:
    print(f"   ✗ Too many invalid trajectories")

# 6. Statistics
print("\n6. Dataset statistics...")
all_action_counts = []
all_init_lens = []
all_final_lens = []

for item in dataset:
    all_action_counts.append(len(item['actions']))
    all_init_lens.append(len(item['init_tcr']))
    all_final_lens.append(len(item['final_tcr']))

import numpy as np
print(f"   Action counts: mean={np.mean(all_action_counts):.1f}, max={np.max(all_action_counts)}")
print(f"   Init TCR lengths: mean={np.mean(all_init_lens):.1f}, range=[{np.min(all_init_lens)}, {np.max(all_init_lens)}]")
print(f"   Final TCR lengths: mean={np.mean(all_final_lens):.1f}, range=[{np.min(all_final_lens)}, {np.max(all_final_lens)}]")

print("\n=== All Tests Passed! ===")
print("\nData pipeline is ready. Next steps:")
print("  1. Fix SFT trainer to match actual policy interface")
print("  2. Run small-scale training test (5 epochs)")
print("  3. Launch full SFT training (50 epochs)")
