#!/usr/bin/env python3
"""
Quick test of SFT training pipeline (1 epoch smoke test).
"""

import sys
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

import torch
from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.env import TCREditEnv
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
sampler = StratifiedBatchSampler(dataset, batch_size=8, shuffle=True)
dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_sft_batch)
print(f"   ✓ Created dataloader with {len(dataloader)} batches")

# 3. Initialize policy (skip full env for smoke test)
print("\n3. Initializing policy...")

# Use dummy dimensions matching actual env
obs_dim = 1280 + 1280  # ESM-2 TCR + pMHC embeddings
max_len = 27
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
idx_to_aa = {i: aa for aa, i in aa_to_idx.items()}

policy = ActorCritic(
    obs_dim=obs_dim,
    hidden_dim=256,
    max_tcr_len=max_len
)
print(f"   ✓ Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

# 4. Test forward pass on one batch
print("\n4. Testing forward pass...")
batch = next(iter(dataloader))
print(f"   Batch size: {len(batch['init_tcrs'])}")
print(f"   Sample init TCR: {batch['init_tcrs'][0]}")
print(f"   Sample final TCR: {batch['final_tcrs'][0]}")
print(f"   Sample peptide: {batch['peptides'][0]}")
print(f"   Sample actions: {batch['actions'][0]}")

# Create dummy observation
obs_tensor = torch.randn(1, obs_dim)

# Get features from backbone
features = policy.backbone(obs_tensor)
print(f"   ✓ Features shape: {features.shape}")

# Get logits from heads
logits_op = policy.op_head(features)
logits_pos = policy.pos_head(features)
logits_tok = policy.tok_head(features)

print(f"   ✓ Op logits shape: {logits_op.shape}")
print(f"   ✓ Pos logits shape: {logits_pos.shape}")
print(f"   ✓ Token logits shape: {logits_tok.shape}")

# 5. Test loss computation
print("\n5. Testing loss computation...")
import torch.nn.functional as F

action = batch['actions'][0][0]  # First action of first trajectory
op_target = torch.tensor([action['op_type']], dtype=torch.long)
pos_target = torch.tensor([action['position']], dtype=torch.long)

# Token target
if action['token']:
    tok_idx = aa_to_idx.get(action['token'], 0)
else:
    tok_idx = 0
tok_target = torch.tensor([tok_idx], dtype=torch.long)

op_loss = F.cross_entropy(logits_op, op_target)
pos_loss = F.cross_entropy(logits_pos, pos_target)
tok_loss = F.cross_entropy(logits_tok, tok_target)

print(f"   ✓ Op loss: {op_loss.item():.4f}")
print(f"   ✓ Pos loss: {pos_loss.item():.4f}")
print(f"   ✓ Token loss: {tok_loss.item():.4f}")
print(f"   ✓ Total loss: {(op_loss + pos_loss + tok_loss).item():.4f}")

# 6. Test backward pass
print("\n6. Testing backward pass...")
total_loss = op_loss + pos_loss + tok_loss
total_loss.backward()
print(f"   ✓ Gradients computed successfully")

# Check gradient norms
grad_norms = []
for name, param in policy.named_parameters():
    if param.grad is not None:
        grad_norms.append(param.grad.norm().item())

print(f"   ✓ Mean gradient norm: {sum(grad_norms) / len(grad_norms):.6f}")
print(f"   ✓ Max gradient norm: {max(grad_norms):.6f}")

print("\n=== All Tests Passed! ===")
print("\nReady to run full SFT training:")
print("  python scripts/train_sft.py --epochs 50 --batch_size 64 --lr 1e-4")
