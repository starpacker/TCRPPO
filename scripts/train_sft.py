#!/usr/bin/env python3
"""
SFT Trainer for TCR editing policy.

Trains the policy to imitate expert trajectories using teacher forcing.
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
from tcrppo_v2.sft_env import SFTEnv
from tcrppo_v2.data.sft_dataset import SFTDataset, StratifiedBatchSampler, collate_sft_batch


class SFTTrainer:
    """Supervised fine-tuning trainer with teacher forcing."""

    def __init__(
        self,
        policy: ActorCritic,
        env: SFTEnv,
        dataset: SFTDataset,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
        log_dir: str = 'output/sft_logs'
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
            'token': []
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
                'tok': f"{loss_dict['token']:.4f}"
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
        n_steps = 0

        # Reset environments
        obs_list = []
        for tcr, peptide in zip(init_tcrs, peptides):
            obs = self.env.reset(init_tcr=tcr, peptide=peptide)
            obs_list.append(obs)

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

                # Head 3: token logits (conditioned on ground-truth op + pos)
                pos_emb = self.policy.pos_embed(pos_targets)
                tok_input = torch.cat([features, op_emb, pos_emb], dim=-1)
                tok_logits = self.policy.token_head(tok_input)
                tok_loss = F.cross_entropy(tok_logits[valid_mask], tok_targets[valid_mask])

                step_loss = op_loss + pos_loss + tok_loss
                total_loss += step_loss

                op_loss_sum += op_loss.item()
                pos_loss_sum += pos_loss.item()
                tok_loss_sum += tok_loss.item()
                n_steps += 1

            # Apply actions to environments (teacher forcing)
            new_obs_list = []
            for i in range(batch_size):
                if step_idx < len(action_sequences[i]):
                    action = action_sequences[i][step_idx]
                    action_tuple = (action['op_type'], action['position'], action['token'])
                    obs, _, done, _ = self.env.step(action_tuple)
                    new_obs_list.append(obs)
                else:
                    new_obs_list.append(obs_list[i])  # Keep old obs

            obs_list = new_obs_list

        # Backward pass
        if n_steps > 0:
            avg_loss = total_loss / n_steps
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Log
            self.writer.add_scalar('train/loss_total', avg_loss.item(), self.global_step)
            self.writer.add_scalar('train/loss_op', op_loss_sum / n_steps, self.global_step)
            self.writer.add_scalar('train/loss_pos', pos_loss_sum / n_steps, self.global_step)
            self.writer.add_scalar('train/loss_tok', tok_loss_sum / n_steps, self.global_step)
            self.global_step += 1

            return {
                'total': avg_loss.item(),
                'op_type': op_loss_sum / n_steps,
                'position': pos_loss_sum / n_steps,
                'token': tok_loss_sum / n_steps
            }
        else:
            return {'total': 0.0, 'op_type': 0.0, 'position': 0.0, 'token': 0.0}

    def validate(self, n_samples: int = 100) -> Dict[str, float]:
        """
        Validate by generating TCRs and computing mean affinity.

        Returns:
            {'mean_affinity': float, 'std_affinity': float}
        """
        self.policy.eval()

        affinities = []
        peptides = ['GILGFVFTL', 'NLVPMVATV', 'GLCTLVAML']  # Sample peptides

        with torch.no_grad():
            for _ in range(n_samples):
                peptide = np.random.choice(peptides)
                obs = self.env.reset(peptide=peptide)

                done = False
                for _ in range(self.env.max_steps):
                    if done:
                        break

                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

                    # Sample action using policy's forward pass (no action masks for simplicity)
                    op, pos, tok, _ = self.policy(obs_tensor, action_masks=None, actions=None)

                    op = op.item()
                    pos = pos.item()
                    tok_idx = tok.item()
                    token = self.env.idx_to_aa.get(tok_idx, 'A')

                    obs, reward, done, info = self.env.step((op, pos, token))

                # Get final affinity
                affinity = info.get('affinity', -10.0)
                affinities.append(affinity)

        mean_aff = np.mean(affinities)
        std_aff = np.std(affinities)

        self.policy.train()
        return {'mean_affinity': mean_aff, 'std_affinity': std_aff}


def main():
    parser = argparse.ArgumentParser(description="Train SFT policy")
    parser.add_argument("--trajectories", type=str, default="data/sft_trajectories.json",
                        help="Path to trajectories JSON")
    parser.add_argument("--output_dir", type=str, default="output/sft_training",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Policy hidden dimension")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Validation interval (epochs)")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Checkpoint save interval (epochs)")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    print("Loading dataset...")
    dataset = SFTDataset(args.trajectories)
    dataset.print_stats()

    # Create environment and policy
    print("\nInitializing environment and policy...")
    env = SFTEnv(
        max_steps=8,
    )

    policy = ActorCritic(
        obs_dim=2560,  # ESM-2 650M: 1280 TCR + 1280 pMHC
        hidden_dim=args.hidden_dim,
        max_tcr_len=env.max_tcr_len
    )

    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create trainer
    trainer = SFTTrainer(
        policy=policy,
        env=env,
        dataset=dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        log_dir=str(output_dir / 'logs')
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_affinity = -float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # Train
        losses = trainer.train_epoch(epoch)
        print(f"Train loss: {losses['total']:.4f} (op={losses['op_type']:.4f}, pos={losses['position']:.4f}, tok={losses['token']:.4f})")

        # Validate
        if epoch % args.val_interval == 0:
            print("Validating...")
            val_metrics = trainer.validate(n_samples=100)
            print(f"Validation: mean_affinity={val_metrics['mean_affinity']:.4f} ± {val_metrics['std_affinity']:.4f}")

            trainer.writer.add_scalar('val/mean_affinity', val_metrics['mean_affinity'], epoch)

            # Save best model
            if val_metrics['mean_affinity'] > best_affinity:
                best_affinity = val_metrics['mean_affinity']
                checkpoint_path = output_dir / 'checkpoint_best.pt'
                torch.save({
                    'epoch': epoch,
                    'policy_state_dict': policy.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_affinity': best_affinity,
                    'config': vars(args)
                }, checkpoint_path)
                print(f"✓ Saved best checkpoint (affinity={best_affinity:.4f})")

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
    print(f"✓ Best validation affinity: {best_affinity:.4f}")


if __name__ == "__main__":
    main()
