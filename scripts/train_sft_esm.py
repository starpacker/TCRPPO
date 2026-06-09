#!/usr/bin/env python3
"""
SFT Trainer with real ESM-2 embeddings and high-quality data.

Key differences from train_sft.py:
1. Uses SFTEnvESM with real ESM-2 embeddings instead of dummy observations
2. Trains on high-quality trajectories (affinity >= -1.0)
3. More efficient batch processing with ESM caching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from typing import Dict, List

# Import project modules
import sys
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.sft_env_esm import SFTEnvESM
from tcrppo_v2.data.sft_dataset import SFTDataset, StratifiedBatchSampler, collate_sft_batch


class SFTTrainerESM:
    """Supervised fine-tuning trainer with real ESM-2 embeddings."""

    def __init__(
        self,
        policy: ActorCritic,
        env: SFTEnvESM,
        dataset: SFTDataset,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
        log_dir: str = 'output/sft_esm_logs'
    ):
        self.policy = policy.to(device)
        self.env = env
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

        # DataLoader with stratified sampling
        self.sampler = StratifiedBatchSampler(dataset, batch_size, shuffle=True)
        self.dataloader = DataLoader(
            dataset,
            batch_sampler=self.sampler,
            collate_fn=collate_sft_batch,
            num_workers=0
        )

        # Logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.policy.train()

        epoch_losses = {
            'total': [],
            'op_type': [],
            'position': [],
            'token': [],
            'repetition': []
        }

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            loss_dict = self.train_batch(batch)

            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'op': f"{loss_dict['op_type']:.4f}",
                'pos': f"{loss_dict['position']:.4f}",
                'tok': f"{loss_dict['token']:.4f}",
                'rep': f"{loss_dict.get('repetition', 0.0):.4f}"
            })

        # Average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        return avg_losses

    def train_batch(self, batch: Dict) -> Dict[str, float]:
        """Train on a single batch with teacher forcing."""
        init_tcrs = batch['init_tcrs']
        peptides = batch['peptides']
        action_sequences = batch['actions']  # List of action lists

        batch_size = len(init_tcrs)
        max_steps = max(len(actions) for actions in action_sequences)

        # Initialize losses
        total_loss = 0.0
        op_loss_sum = 0.0
        pos_loss_sum = 0.0
        tok_loss_sum = 0.0
        repetition_penalty_sum = 0.0
        n_steps = 0

        # Reset environments (get real ESM-2 observations)
        obs_list = []
        for tcr, peptide in zip(init_tcrs, peptides):
            obs = self.env.reset(init_tcr=tcr, peptide=peptide)
            obs_list.append(obs)

        # Track recent insertions for repetition penalty
        recent_insertions = [[] for _ in range(batch_size)]

        # Teacher forcing: step through trajectories
        for step_idx in range(max_steps):
            # Collect current observations
            obs_batch = torch.stack([torch.from_numpy(obs).float() for obs in obs_list]).to(self.device)

            # Collect targets for this step
            op_targets = []
            pos_targets = []
            tok_targets = []
            valid_mask = []

            for i in range(batch_size):
                if step_idx < len(action_sequences[i]):
                    action = action_sequences[i][step_idx]
                    op_targets.append(action['op_type'])
                    pos_targets.append(action['position'])

                    # Token target (convert AA to index)
                    if action['token']:
                        tok_idx = self.env.aa_to_idx.get(action['token'], 0)
                    else:
                        tok_idx = 0  # Dummy for DEL/STOP
                    tok_targets.append(tok_idx)

                    valid_mask.append(True)
                else:
                    # Padding
                    op_targets.append(0)
                    pos_targets.append(0)
                    tok_targets.append(0)
                    valid_mask.append(False)

            # Convert to tensors
            op_targets = torch.tensor(op_targets, dtype=torch.long, device=self.device)
            pos_targets = torch.tensor(pos_targets, dtype=torch.long, device=self.device)
            tok_targets = torch.tensor(tok_targets, dtype=torch.long, device=self.device)
            valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)

            # Compute losses (only for valid steps)
            if valid_mask.any():
                # Get features from backbone
                features = self.policy.backbone(obs_batch)

                # Head 1: op_type logits
                op_logits = self.policy.op_head(features)
                op_loss = F.cross_entropy(op_logits[valid_mask], op_targets[valid_mask])

                # Head 2: position logits (conditioned on ground-truth op)
                op_emb = self.policy.op_embed(op_targets)
                pos_input = torch.cat([features, op_emb], dim=-1)
                pos_logits = self.policy.pos_head(pos_input)
                pos_loss = F.cross_entropy(pos_logits[valid_mask], pos_targets[valid_mask])

                # Head 3: token logits (conditioned on ground-truth op and pos)
                pos_emb = self.policy.pos_embed(pos_targets)
                tok_input = torch.cat([features, op_emb, pos_emb], dim=-1)
                tok_logits = self.policy.token_head(tok_input)
                tok_loss = F.cross_entropy(tok_logits[valid_mask], tok_targets[valid_mask])

                # Repetition penalty: penalize consecutive insertions of same token
                repetition_penalty = 0.0
                for i in range(batch_size):
                    if valid_mask[i] and op_targets[i] == 1:  # INS operation
                        token = tok_targets[i].item()
                        if len(recent_insertions[i]) > 0 and recent_insertions[i][-1] == token:
                            # Penalize consecutive same insertions
                            repetition_penalty += 1.0
                        recent_insertions[i].append(token)
                        # Keep only last 3 insertions
                        if len(recent_insertions[i]) > 3:
                            recent_insertions[i].pop(0)

                repetition_penalty = torch.tensor(repetition_penalty, device=self.device)

                # Combined loss with repetition penalty
                step_loss = op_loss + pos_loss + tok_loss + 0.1 * repetition_penalty

                # Accumulate
                total_loss += step_loss
                op_loss_sum += op_loss.item()
                pos_loss_sum += pos_loss.item()
                tok_loss_sum += tok_loss.item()
                repetition_penalty_sum += repetition_penalty.item()
                n_steps += 1

            # Apply ground-truth actions to get next observations
            new_obs_list = []
            for i in range(batch_size):
                if step_idx < len(action_sequences[i]):
                    action = action_sequences[i][step_idx]
                    op = action['op_type']
                    pos = action['position']
                    tok = action['token'] or 'A'  # Dummy for DEL/STOP

                    # Apply action
                    obs, _, _, _ = self.env.step((op, pos, tok))
                    new_obs_list.append(obs)
                else:
                    # Keep old observation (padding)
                    new_obs_list.append(obs_list[i])

            obs_list = new_obs_list

        # Backward pass
        if n_steps > 0:
            avg_loss = total_loss / n_steps
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Log to tensorboard
            self.writer.add_scalar('train/loss_total', avg_loss.item(), self.global_step)
            self.writer.add_scalar('train/loss_op', op_loss_sum / n_steps, self.global_step)
            self.writer.add_scalar('train/loss_pos', pos_loss_sum / n_steps, self.global_step)
            self.writer.add_scalar('train/loss_tok', tok_loss_sum / n_steps, self.global_step)
            self.writer.add_scalar('train/repetition_penalty', repetition_penalty_sum / n_steps, self.global_step)
            self.global_step += 1

            return {
                'total': avg_loss.item(),
                'op_type': op_loss_sum / n_steps,
                'position': pos_loss_sum / n_steps,
                'token': tok_loss_sum / n_steps,
                'repetition': repetition_penalty_sum / n_steps,
            }
        else:
            return {'total': 0.0, 'op_type': 0.0, 'position': 0.0, 'token': 0.0, 'repetition': 0.0}

    def validate(self, n_samples: int = 100) -> Dict[str, float]:
        """Validate by generating TCRs and checking diversity."""
        self.policy.eval()

        peptides = ['GILGFVFTL', 'NLVPMVATV', 'GLCTLVAML']
        generated_tcrs = []

        with torch.no_grad():
            for peptide in peptides:
                for _ in range(n_samples // len(peptides)):
                    # Reset with random init
                    obs = self.env.reset(peptide=peptide)

                    done = False
                    for step in range(self.env.max_steps):
                        if done:
                            break

                        # Sample action
                        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                        op, pos, tok, _ = self.policy(obs_tensor, action_masks=None, actions=None)

                        op = op.item()
                        pos = pos.item()
                        tok_idx = tok.item()
                        token = self.env.idx_to_aa.get(tok_idx, 'A')

                        # Apply action
                        obs, _, done, info = self.env.step((op, pos, token))

                    generated_tcrs.append(self.env.current_tcr)

        # Compute diversity metrics
        unique_tcrs = set(generated_tcrs)
        diversity = len(unique_tcrs) / len(generated_tcrs)

        # Compute length statistics
        lengths = [len(tcr) for tcr in generated_tcrs]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        self.policy.train()

        return {
            'diversity': diversity,
            'mean_length': mean_length,
            'std_length': std_length,
            'mean_affinity': 0.0,  # Placeholder (would need tFold scoring)
            'std_affinity': 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description="Train SFT policy with real ESM-2 embeddings")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to high-quality SFT trajectories JSON')
    parser.add_argument('--esm_cache', type=str, default='data/esm2_embeddings.pt',
                        help='Path to precomputed ESM-2 embeddings')
    parser.add_argument('--output_dir', type=str, default='output/sft_esm_training',
                        help='Output directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='output/sft_esm_logs',
                        help='Tensorboard log directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (reduced from 64 due to ESM overhead)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for policy network')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Validate every N epochs')
    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data}")
    dataset = SFTDataset(args.data)
    print(f"  Loaded {len(dataset)} trajectories")
    print(f"  Affinity bins: {dataset.affinity_bins}")

    # Initialize environment with ESM-2
    print(f"Initializing SFT environment with ESM-2...")
    env = SFTEnvESM(
        esm_cache_path=args.esm_cache,
        max_steps=8,
        device=args.device
    )

    # Initialize policy
    print(f"Initializing policy (hidden_dim={args.hidden_dim})...")
    policy = ActorCritic(
        obs_dim=2560,  # ESM-2 650M: 1280 + 1280
        hidden_dim=args.hidden_dim,
        max_tcr_len=25
    )

    # Initialize trainer
    trainer = SFTTrainerESM(
        policy=policy,
        env=env,
        dataset=dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        log_dir=args.log_dir
    )

    print(f"\n{'='*80}")
    print(f"Starting SFT training with real ESM-2 embeddings")
    print(f"{'='*80}")
    print(f"Dataset: {args.data}")
    print(f"Trajectories: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")

    # Training loop
    best_diversity = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # Train
        losses = trainer.train_epoch(epoch)
        print(f"Train losses: total={losses['total']:.4f}, op={losses['op_type']:.4f}, "
              f"pos={losses['position']:.4f}, tok={losses['token']:.4f}")

        # Log to tensorboard
        trainer.writer.add_scalar('epoch/loss_total', losses['total'], epoch)

        # Validate
        if epoch % args.val_interval == 0:
            print("Validating...")
            val_metrics = trainer.validate(n_samples=100)
            print(f"Validation: diversity={val_metrics['diversity']:.4f}, "
                  f"mean_length={val_metrics['mean_length']:.1f} ± {val_metrics['std_length']:.1f}")

            trainer.writer.add_scalar('val/diversity', val_metrics['diversity'], epoch)
            trainer.writer.add_scalar('val/mean_length', val_metrics['mean_length'], epoch)

            # Save best model (by diversity)
            if val_metrics['diversity'] > best_diversity:
                best_diversity = val_metrics['diversity']
                checkpoint_path = output_dir / 'checkpoint_best.pt'
                torch.save({
                    'epoch': epoch,
                    'policy_state_dict': policy.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_diversity': best_diversity,
                    'config': vars(args)
                }, checkpoint_path)
                print(f"✓ Saved best checkpoint (diversity={best_diversity:.4f})")

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': vars(args)
            }, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {epoch}")

    # Final save
    final_path = output_dir / 'checkpoint_final.pt'
    torch.save({
        'epoch': args.epochs,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'config': vars(args)
    }, final_path)
    print(f"\n✓ Training complete! Final checkpoint saved to {final_path}")
    print(f"✓ Best validation diversity: {best_diversity:.4f}")


if __name__ == "__main__":
    main()
