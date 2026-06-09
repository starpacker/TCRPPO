#!/usr/bin/env python3
"""
SFT v4 Trainer: Pure imitation learning (no target TCR in observation).

Key differences from v3:
1. No target_tcr in observation encoding
2. Relies on deterministic action order in data (left-to-right)
3. May need more epochs to converge
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
import math
from tqdm import tqdm
from typing import Dict, List, Optional

import sys
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.data.sft_dataset import SFTDataset, StratifiedBatchSampler, collate_sft_batch
from tcrppo_v2.utils.constants import (
    AMINO_ACIDS, AA_TO_IDX, NUM_AMINO_ACIDS,
    MAX_TCR_LEN, OP_SUB,
)


def encode_obs_v4(
    current_tcr: str,
    peptide: str,
    step_count: int,
    max_tcr_len: int = MAX_TCR_LEN,
    max_pep_len: int = 25,
    max_steps: int = 8,
    obs_dim: int = 2560,
) -> np.ndarray:
    """Encode observation WITHOUT target TCR (pure imitation learning).

    Layout:
      [0:500]      current TCR one-hot (25 * 20)
      [500:1000]   peptide one-hot (25 * 20)
      [1000]       tcr_len / max_tcr_len
      [1001]       step_count / max_steps
      [1002:2560]  zero padding (target slot removed)
    """
    obs = np.zeros(obs_dim, dtype=np.float32)

    # Current TCR one-hot
    for i, aa in enumerate(current_tcr[:max_tcr_len]):
        idx = AA_TO_IDX.get(aa, 0)
        obs[i * NUM_AMINO_ACIDS + idx] = 1.0

    # Peptide one-hot
    pep_offset = max_tcr_len * NUM_AMINO_ACIDS
    for i, aa in enumerate(peptide[:max_pep_len]):
        idx = AA_TO_IDX.get(aa, 0)
        obs[pep_offset + i * NUM_AMINO_ACIDS + idx] = 1.0

    # Scalar features
    scalar_offset = pep_offset + max_pep_len * NUM_AMINO_ACIDS
    obs[scalar_offset] = len(current_tcr) / max_tcr_len
    obs[scalar_offset + 1] = step_count / max_steps

    # No target TCR encoding!
    return obs


def apply_sub(tcr: str, position: int, token: str) -> str:
    """Apply SUB action to TCR string."""
    if 0 <= position < len(tcr):
        tcr_list = list(tcr)
        tcr_list[position] = token
        return ''.join(tcr_list)
    return tcr


class SFTv4Trainer:
    """Pure imitation learning trainer (no target conditioning)."""

    def __init__(
        self,
        policy: ActorCritic,
        dataset: SFTDataset,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
        log_dir: str = 'output/sft_v4_logs',
        max_steps: int = 8,
        warmup_steps: int = 100,
        total_steps: int = 10000,
    ):
        self.policy = policy.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.max_steps = max_steps

        self.optimizer = torch.optim.AdamW(
            policy.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        # Cosine LR schedule with warmup
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

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
        epoch_accs = {'position_acc': [], 'token_acc': []}

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            loss_dict = self.train_batch(batch)
            for key in epoch_losses:
                epoch_losses[key].append(loss_dict[key])
            for key in epoch_accs:
                if key in loss_dict:
                    epoch_accs[key].append(loss_dict[key])
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'pos': f"{loss_dict['position']:.4f}",
                'tok': f"{loss_dict['token']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        avg_accs = {key: np.mean(values) for key, values in epoch_accs.items() if values}
        avg_losses.update(avg_accs)
        return avg_losses

    def train_batch(self, batch: Dict) -> Dict[str, float]:
        """Train on a single batch with teacher forcing (no target TCR)."""
        init_tcrs = batch['init_tcrs']
        peptides = batch['peptides']
        action_sequences = batch['actions']

        batch_size = len(init_tcrs)
        max_steps = max(len(actions) for actions in action_sequences)

        total_loss = 0.0
        pos_loss_sum = 0.0
        tok_loss_sum = 0.0
        pos_correct = 0
        tok_correct = 0
        n_steps = 0
        n_valid = 0

        # Per-sample state tracking
        current_tcrs = list(init_tcrs)

        # Fixed op_type = SUB for all steps
        fixed_op = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for step_idx in range(max_steps):
            # Build observations WITHOUT target TCR
            obs_list = []
            for i in range(batch_size):
                obs = encode_obs_v4(
                    current_tcr=current_tcrs[i],
                    peptide=peptides[i],
                    step_count=step_idx,
                )
                obs_list.append(obs)

            obs_batch = torch.stack(
                [torch.from_numpy(obs).float() for obs in obs_list]
            ).to(self.device)

            # Collect targets and masks
            pos_targets = []
            tok_targets = []
            valid_mask = []
            tcr_lengths = []

            for i in range(batch_size):
                tcr_lengths.append(len(current_tcrs[i]))
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

            pos_targets_t = torch.tensor(pos_targets, dtype=torch.long, device=self.device)
            tok_targets_t = torch.tensor(tok_targets, dtype=torch.long, device=self.device)
            valid_mask_t = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)

            if valid_mask_t.any():
                features = self.policy.backbone(obs_batch)

                # Position head
                op_emb = self.policy.op_embed(fixed_op)
                pos_input = torch.cat([features, op_emb], dim=-1)
                pos_logits = self.policy.pos_head(pos_input)

                # Position masking
                pos_mask = torch.zeros(batch_size, MAX_TCR_LEN, dtype=torch.bool,
                                       device=self.device)
                for i in range(batch_size):
                    pos_mask[i, :tcr_lengths[i]] = True
                pos_logits = pos_logits.masked_fill(~pos_mask, float('-inf'))

                pos_loss = F.cross_entropy(
                    pos_logits[valid_mask_t], pos_targets_t[valid_mask_t]
                )

                # Token head
                pos_emb = self.policy.pos_embed(pos_targets_t)
                tok_input = torch.cat([features, op_emb, pos_emb], dim=-1)
                tok_logits = self.policy.token_head(tok_input)
                tok_loss = F.cross_entropy(
                    tok_logits[valid_mask_t], tok_targets_t[valid_mask_t]
                )

                step_loss = pos_loss + tok_loss
                total_loss += step_loss

                pos_loss_sum += pos_loss.item()
                tok_loss_sum += tok_loss.item()

                # Accuracy tracking
                with torch.no_grad():
                    pos_preds = pos_logits[valid_mask_t].argmax(dim=-1)
                    tok_preds = tok_logits[valid_mask_t].argmax(dim=-1)
                    pos_correct += (pos_preds == pos_targets_t[valid_mask_t]).sum().item()
                    tok_correct += (tok_preds == tok_targets_t[valid_mask_t]).sum().item()
                    n_valid += valid_mask_t.sum().item()

                n_steps += 1

            # Apply actions to per-sample state (teacher forcing)
            for i in range(batch_size):
                if step_idx < len(action_sequences[i]):
                    action = action_sequences[i][step_idx]
                    current_tcrs[i] = apply_sub(
                        current_tcrs[i], action['position'], action['token']
                    )

        # Backward pass
        if n_steps > 0:
            avg_loss = total_loss / n_steps
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            self.writer.add_scalar('train/loss_total', avg_loss.item(), self.global_step)
            self.writer.add_scalar('train/loss_pos', pos_loss_sum / n_steps, self.global_step)
            self.writer.add_scalar('train/loss_tok', tok_loss_sum / n_steps, self.global_step)
            self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
            if n_valid > 0:
                self.writer.add_scalar('train/pos_acc', pos_correct / n_valid, self.global_step)
                self.writer.add_scalar('train/tok_acc', tok_correct / n_valid, self.global_step)
            self.global_step += 1

            result = {
                'total': avg_loss.item(),
                'position': pos_loss_sum / n_steps,
                'token': tok_loss_sum / n_steps,
            }
            if n_valid > 0:
                result['position_acc'] = pos_correct / n_valid
                result['token_acc'] = tok_correct / n_valid
            return result
        else:
            return {'total': 0.0, 'position': 0.0, 'token': 0.0,
                    'position_acc': 0.0, 'token_acc': 0.0}


def main():
    parser = argparse.ArgumentParser(description="Train SFT v4 (pure imitation learning)")
    parser.add_argument("--trajectories", type=str,
                        default="data/sft_v4_trajectories.json")
    parser.add_argument("--output_dir", type=str,
                        default="output/sft_v4_training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="More epochs since no target TCR")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=float, default=2.0,
                        help="Longer warmup for harder task")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    print("Loading dataset...")
    dataset = SFTDataset(args.trajectories)
    dataset.print_stats()

    # Estimate total steps
    sampler = StratifiedBatchSampler(dataset, args.batch_size, shuffle=True)
    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    print(f"\nSteps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Create policy
    print("\nInitializing policy...")
    policy = ActorCritic(
        obs_dim=2560,
        hidden_dim=args.hidden_dim,
        max_tcr_len=MAX_TCR_LEN,
    )
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Resume if requested
    start_epoch = 1
    best_loss = float('inf')
    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        policy.load_state_dict(ckpt['policy_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"  Resumed from epoch {start_epoch - 1}, best_loss={best_loss:.4f}")

    # Create trainer
    trainer = SFTv4Trainer(
        policy=policy,
        dataset=dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        log_dir=str(output_dir / 'logs'),
        max_steps=8,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # Training loop
    print(f"\n{'='*60}")
    print(f"SFT v4 Training: Pure Imitation Learning")
    print(f"  No target TCR in observation")
    print(f"  Deterministic action order (left-to-right)")
    print(f"  LR: {args.lr} with cosine schedule + warmup")
    print(f"  Epochs: {start_epoch} to {args.epochs}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        losses = trainer.train_epoch(epoch)

        # Print results
        loss_str = f"Loss: {losses['total']:.4f} (pos={losses['position']:.4f}, tok={losses['token']:.4f})"
        acc_str = f"  Acc: pos={losses.get('position_acc', 0):.1%}, tok={losses.get('token_acc', 0):.1%}"
        print(f"{loss_str}{acc_str}")

        # Log to tensorboard
        trainer.writer.add_scalar('epoch/loss_total', losses['total'], epoch)
        trainer.writer.add_scalar('epoch/loss_pos', losses['position'], epoch)
        trainer.writer.add_scalar('epoch/loss_tok', losses['token'], epoch)
        if 'position_acc' in losses:
            trainer.writer.add_scalar('epoch/pos_acc', losses['position_acc'], epoch)
            trainer.writer.add_scalar('epoch/tok_acc', losses['token_acc'], epoch)

        # Save best
        if losses['total'] < best_loss:
            best_loss = losses['total']
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'best_loss': best_loss,
                'config': vars(args),
            }, output_dir / 'checkpoint_best.pt')
            print(f"  ** New best loss: {best_loss:.4f}")

        # Periodic save
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'best_loss': best_loss,
                'config': vars(args),
            }, output_dir / f'checkpoint_epoch{epoch}.pt')

    # Final save
    torch.save({
        'epoch': args.epochs,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'best_loss': best_loss,
        'config': vars(args),
    }, output_dir / 'checkpoint_final.pt')

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
