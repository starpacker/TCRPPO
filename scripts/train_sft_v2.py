#!/usr/bin/env python3
"""
SFT v2 Trainer: SUB-only with 2-head loss (position + token).

Op type is always SUB, so we skip op_type prediction and only train
the position head and token head.
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
from typing import Dict

import sys
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.sft_env import SFTEnv
from tcrppo_v2.data.sft_dataset import SFTDataset, StratifiedBatchSampler, collate_sft_batch
from tcrppo_v2.utils.constants import AA_TO_IDX, OP_SUB


class SFTv2Trainer:
    """SUB-only SFT trainer with 2-head loss (position + token)."""

    def __init__(
        self,
        policy: ActorCritic,
        env: SFTEnv,
        dataset: SFTDataset,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
        log_dir: str = 'output/sft_v2_logs',
    ):
        self.policy = policy.to(device)
        self.env = env
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.sampler = StratifiedBatchSampler(dataset, batch_size, shuffle=True)
        self.dataloader = DataLoader(
            dataset,
            batch_sampler=self.sampler,
            collate_fn=collate_sft_batch,
            num_workers=0,
        )

        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.policy.train()

        epoch_losses = {'total': [], 'position': [], 'token': []}

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            loss_dict = self.train_batch(batch)
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'pos': f"{loss_dict['position']:.4f}",
                'tok': f"{loss_dict['token']:.4f}",
            })

        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        return avg_losses

    def train_batch(self, batch: Dict) -> Dict[str, float]:
        """Train on a single batch with teacher forcing. SUB-only."""
        init_tcrs = batch['init_tcrs']
        peptides = batch['peptides']
        action_sequences = batch['actions']

        batch_size = len(init_tcrs)
        max_steps = max(len(actions) for actions in action_sequences)

        total_loss = 0.0
        pos_loss_sum = 0.0
        tok_loss_sum = 0.0
        n_steps = 0

        # Reset environments
        obs_list = []
        for tcr, peptide in zip(init_tcrs, peptides):
            obs = self.env.reset(init_tcr=tcr, peptide=peptide)
            obs_list.append(obs)

        # Fixed op_type = SUB for all steps
        fixed_op = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # SUB=0

        for step_idx in range(max_steps):
            obs_batch = torch.stack(
                [torch.from_numpy(obs).float() for obs in obs_list]
            ).to(self.device)

            # Collect targets
            pos_targets = []
            tok_targets = []
            valid_mask = []

            for i in range(batch_size):
                if step_idx < len(action_sequences[i]):
                    action = action_sequences[i][step_idx]
                    pos_targets.append(action['position'])
                    tok_idx = AA_TO_IDX.get(action['token'], 0) if action['token'] else 0
                    tok_targets.append(tok_idx)
                    valid_mask.append(True)
                else:
                    pos_targets.append(0)
                    tok_targets.append(0)
                    valid_mask.append(False)

            pos_targets = torch.tensor(pos_targets, dtype=torch.long, device=self.device)
            tok_targets = torch.tensor(tok_targets, dtype=torch.long, device=self.device)
            valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)

            if valid_mask.any():
                features = self.policy.backbone(obs_batch)

                # Position head (conditioned on fixed SUB op)
                op_emb = self.policy.op_embed(fixed_op)
                pos_input = torch.cat([features, op_emb], dim=-1)
                pos_logits = self.policy.pos_head(pos_input)
                pos_loss = F.cross_entropy(pos_logits[valid_mask], pos_targets[valid_mask])

                # Token head (conditioned on fixed SUB op + ground-truth position)
                pos_emb = self.policy.pos_embed(pos_targets)
                tok_input = torch.cat([features, op_emb, pos_emb], dim=-1)
                tok_logits = self.policy.token_head(tok_input)
                tok_loss = F.cross_entropy(tok_logits[valid_mask], tok_targets[valid_mask])

                step_loss = pos_loss + tok_loss
                total_loss += step_loss

                pos_loss_sum += pos_loss.item()
                tok_loss_sum += tok_loss.item()
                n_steps += 1

            # Apply actions to environments (teacher forcing)
            new_obs_list = []
            for i in range(batch_size):
                if step_idx < len(action_sequences[i]):
                    action = action_sequences[i][step_idx]
                    action_tuple = (OP_SUB, action['position'], action['token'])
                    obs, _, done, _ = self.env.step(action_tuple)
                    new_obs_list.append(obs)
                else:
                    new_obs_list.append(obs_list[i])
            obs_list = new_obs_list

        # Backward pass
        if n_steps > 0:
            avg_loss = total_loss / n_steps
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.writer.add_scalar('train/loss_total', avg_loss.item(), self.global_step)
            self.writer.add_scalar('train/loss_pos', pos_loss_sum / n_steps, self.global_step)
            self.writer.add_scalar('train/loss_tok', tok_loss_sum / n_steps, self.global_step)
            self.global_step += 1

            return {
                'total': avg_loss.item(),
                'position': pos_loss_sum / n_steps,
                'token': tok_loss_sum / n_steps,
            }
        else:
            return {'total': 0.0, 'position': 0.0, 'token': 0.0}


def main():
    parser = argparse.ArgumentParser(description="Train SFT v2 (SUB-only)")
    parser.add_argument("--trajectories", type=str,
                        default="data/sft_v2_trajectories.json")
    parser.add_argument("--output_dir", type=str,
                        default="output/sft_v2_training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_interval", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    print("Loading dataset...")
    dataset = SFTDataset(args.trajectories)
    dataset.print_stats()

    # Create env and policy
    print("\nInitializing environment and policy...")
    env = SFTEnv(max_steps=8)
    policy = ActorCritic(
        obs_dim=2560,
        hidden_dim=args.hidden_dim,
        max_tcr_len=env.max_tcr_len,
    )
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create trainer
    trainer = SFTv2Trainer(
        policy=policy,
        env=env,
        dataset=dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        log_dir=str(output_dir / 'logs'),
    )

    # Training loop
    print(f"\nStarting SFT v2 training for {args.epochs} epochs...")
    print(f"Mode: SUB-only, 2-head loss (position + token)")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        losses = trainer.train_epoch(epoch)
        print(f"Loss: {losses['total']:.4f} (pos={losses['position']:.4f}, tok={losses['token']:.4f})")

        # Save best
        if losses['total'] < best_loss:
            best_loss = losses['total']
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_loss': best_loss,
                'config': vars(args),
            }, output_dir / 'checkpoint_best.pt')
            print(f"  New best loss: {best_loss:.4f}")

        # Periodic save
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': vars(args),
            }, output_dir / f'checkpoint_epoch{epoch}.pt')

    # Final save
    torch.save({
        'epoch': args.epochs,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'config': vars(args),
    }, output_dir / 'checkpoint_final.pt')

    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
