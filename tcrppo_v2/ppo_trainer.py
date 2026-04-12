"""Custom PPO trainer with autoregressive action space support.

Supports:
- VecEnv with n_envs parallel environments
- Autoregressive 3-head rollout collection
- GAE advantage estimation
- PPO clipped objective with entropy bonus
- Online SpecificityCallback
- Checkpointing at milestones
- TensorBoard logging
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.utils.constants import (
    ERGO_MODEL_DIR, ERGO_AE_FILE, NUM_OPS, NUM_AMINO_ACIDS,
    MAX_TCR_LEN, OP_SUB, OP_INS, OP_STOP, PROJECT_ROOT,
)
from tcrppo_v2.policy import ActorCritic


class RolloutBuffer:
    """Buffer for storing rollout data from VecEnv."""

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, device: str = "cpu"):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.device = device
        self.ptr = 0

        # Pre-allocate arrays
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.ops = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.positions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.tokens = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)

        # Action masks
        self.op_masks = np.zeros((n_steps, n_envs, NUM_OPS), dtype=bool)
        self.pos_masks = np.zeros((n_steps, n_envs, MAX_TCR_LEN), dtype=bool)

        # Computed after rollout
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        op: np.ndarray,
        pos: np.ndarray,
        tok: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        op_mask: np.ndarray,
        pos_mask: np.ndarray,
    ) -> None:
        """Add one step of data."""
        self.obs[self.ptr] = obs
        self.ops[self.ptr] = op
        self.positions[self.ptr] = pos
        self.tokens[self.ptr] = tok
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.op_masks[self.ptr] = op_mask
        self.pos_masks[self.ptr] = pos_mask
        self.ptr += 1

    def compute_gae(self, last_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and returns."""
        last_gae = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_value
            else:
                next_values = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int) -> List[dict]:
        """Flatten and yield minibatches."""
        total = self.n_steps * self.n_envs
        indices = np.arange(total)
        np.random.shuffle(indices)

        # Flatten
        flat_obs = self.obs.reshape(total, -1)
        flat_ops = self.ops.reshape(total)
        flat_pos = self.positions.reshape(total)
        flat_tok = self.tokens.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_op_masks = self.op_masks.reshape(total, NUM_OPS)
        flat_pos_masks = self.pos_masks.reshape(total, MAX_TCR_LEN)

        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": torch.FloatTensor(flat_obs[idx]).to(self.device),
                "ops": torch.LongTensor(flat_ops[idx]).to(self.device),
                "positions": torch.LongTensor(flat_pos[idx]).to(self.device),
                "tokens": torch.LongTensor(flat_tok[idx]).to(self.device),
                "old_log_probs": torch.FloatTensor(flat_log_probs[idx]).to(self.device),
                "advantages": torch.FloatTensor(flat_advantages[idx]).to(self.device),
                "returns": torch.FloatTensor(flat_returns[idx]).to(self.device),
                "op_masks": torch.BoolTensor(flat_op_masks[idx]).to(self.device),
                "pos_masks": torch.BoolTensor(flat_pos_masks[idx]).to(self.device),
            }

    def reset(self) -> None:
        """Reset buffer pointer."""
        self.ptr = 0


class PPOTrainer:
    """Custom PPO implementation for autoregressive TCR editing."""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cuda")
        self.total_timesteps = config.get("total_timesteps", 10_000_000)
        self.n_envs = config.get("n_envs", 20)
        self.n_steps = config.get("n_steps", 128)
        self.batch_size = config.get("batch_size", 256)
        self.n_epochs = config.get("n_epochs", 4)
        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.90)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_range = config.get("clip_range", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.05)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.seed = config.get("seed", 42)
        self.reward_mode = config.get("reward_mode", "v2_full")
        self.run_name = config.get("run_name", "v2_run")

        # Paths
        self.output_dir = config.get("output_dir", "output")
        self.results_dir = config.get("results_dir", "results")
        self.milestones = config.get("milestones", [500000, 1000000, 2000000, 5000000, 10000000])
        self.checkpoint_interval = config.get("checkpoint_interval", 100000)

        # Eval
        self.eval_interval = config.get("eval_interval", 100000)
        self.eval_n_tcrs = config.get("eval_n_tcrs", 5)
        self.eval_n_decoys = config.get("eval_n_decoys", 50)
        self.eval_abort_threshold = config.get("eval_abort_threshold", 0.40)
        self.eval_warmup = config.get("eval_warmup", 500000)

        # Set seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Will be initialized in setup()
        self.policy = None
        self.optimizer = None
        self.vec_env = None
        self.buffer = None
        self.logger = None

    def setup(self) -> None:
        """Initialize all components."""
        print(f"Setting up PPO trainer: {self.run_name}")
        print(f"  reward_mode={self.reward_mode}, n_envs={self.n_envs}")
        print(f"  weights: aff={self.config.get('w_affinity', 1.0)}, "
              f"decoy={self.config.get('w_decoy', 0.8)}, "
              f"nat={self.config.get('w_naturalness', 0.5)}, "
              f"div={self.config.get('w_diversity', 0.2)}")
        if self.config.get("min_steps", 0) > 0:
            print(f"  min_steps={self.config['min_steps']}, "
                  f"penalty={self.config.get('min_steps_penalty', 0.0)}")

        # Build scorers
        from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
        from tcrppo_v2.utils.esm_cache import ESMCache
        from tcrppo_v2.data.pmhc_loader import PMHCLoader, EVAL_TARGETS
        from tcrppo_v2.data.tcr_pool import TCRPool
        from tcrppo_v2.reward_manager import RewardManager
        from tcrppo_v2.env import VecTCREditEnv

        # pMHC loader — train mode loads all tc-hard targets (~163)
        pmhc_loader = PMHCLoader(mode="train")
        targets = pmhc_loader.get_target_list()

        # ERGO scorer
        model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
        affinity_scorer = AffinityERGOScorer(
            model_file=model_file,
            ae_file=ERGO_AE_FILE,
            device=self.device,
            mc_samples=self.config.get("affinity_mc_samples", 10),
        )
        print("  ERGO loaded")

        # ESM cache
        esm_cache = ESMCache(
            device=self.device,
            tcr_cache_size=self.config.get("esm_tcr_cache_size", 4096),
        )
        print(f"  ESM-2 loaded (dim={esm_cache.embed_dim})")

        print(f"  pMHC loader: {len(targets)} targets")

        # TCR pool (no L1 seeds — L1 curriculum is banned)
        tcr_pool = TCRPool(
            l1_seeds_dir=None,
            curriculum_schedule=self.config.get("curriculum_schedule"),
            seed=self.seed,
        )
        # Load L0 seeds from decoy D + tc-hard known binders
        decoy_lib_path = self.config.get("decoy_library_path", "/share/liuyutian/pMHC_decoy_library")
        tcr_pool.load_l0_from_decoy_d(decoy_lib_path, targets)
        # Also load tc-hard CDR3b binders as L0 seeds
        l0_tchard_dir = os.path.join(PROJECT_ROOT, "data", "l0_seeds_tchard")
        if os.path.isdir(l0_tchard_dir):
            tcr_pool.load_l0_from_dir(l0_tchard_dir)
        l0_targets = tcr_pool.get_l0_targets()
        print(f"  TCR pool: {tcr_pool.num_tcrdb_seqs} seqs, "
              f"L0 targets={len(l0_targets)}/{len(targets)}")

        # Decoy scorer (for reward, with LogSumExp penalty)
        decoy_scorer = None
        self.decoy_scorer = None
        if self.reward_mode in ("v2_full", "v2_decoy_only"):
            from tcrppo_v2.scorers.decoy import DecoyScorer
            decoy_scorer = DecoyScorer(
                decoy_library_path=decoy_lib_path,
                targets=targets,
                tier_weights=self.config.get("decoy_tier_weights"),
                K=self.config.get("decoy_K", 32),
                tau=self.config.get("decoy_tau", 10.0),
                affinity_scorer=affinity_scorer,
                rng=np.random.default_rng(self.seed),
            )
            # Start with only tier A unlocked
            decoy_scorer.set_unlocked_tiers(["A"])
            self.decoy_scorer = decoy_scorer
            print(f"  Decoy scorer loaded (K={decoy_scorer.K})")

        # Naturalness scorer
        naturalness_scorer = None
        if self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum"):
            from tcrppo_v2.scorers.naturalness import NaturalnessScorer
            stats_file = self.config.get("cdr3_ppl_stats", "data/cdr3_ppl_stats.json")
            naturalness_scorer = NaturalnessScorer(
                esm_model=esm_cache.model,
                esm_alphabet=esm_cache.alphabet,
                esm_batch_converter=esm_cache.batch_converter,
                device=self.device,
                stats_file=stats_file,
            )
            print("  Naturalness scorer loaded")

        # Diversity scorer
        diversity_scorer = None
        if self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum"):
            from tcrppo_v2.scorers.diversity import DiversityScorer
            diversity_scorer = DiversityScorer(
                buffer_size=self.config.get("diversity_buffer_size", 512),
                similarity_threshold=self.config.get("diversity_threshold", 0.85),
            )
            print("  Diversity scorer loaded")

        # Reward manager
        reward_manager = RewardManager(
            affinity_scorer=affinity_scorer,
            decoy_scorer=decoy_scorer,
            naturalness_scorer=naturalness_scorer,
            diversity_scorer=diversity_scorer,
            reward_mode=self.reward_mode,
            w_affinity=self.config.get("w_affinity", 1.0),
            w_decoy=self.config.get("w_decoy", 0.8),
            w_naturalness=self.config.get("w_naturalness", 0.5),
            w_diversity=self.config.get("w_diversity", 0.2),
            norm_window=self.config.get("norm_window", 10000),
            norm_warmup=self.config.get("norm_warmup", 1000),
        )

        # VecEnv
        self.vec_env = VecTCREditEnv(
            n_envs=self.n_envs,
            esm_cache=esm_cache,
            pmhc_loader=pmhc_loader,
            tcr_pool=tcr_pool,
            reward_manager=reward_manager,
            reward_mode=self.reward_mode,
            min_steps=self.config.get("min_steps", 0),
            min_steps_penalty=self.config.get("min_steps_penalty", 0.0),
        )
        print(f"  VecEnv: {self.n_envs} envs, obs_dim={self.vec_env.obs_dim}")

        # Policy
        self.policy = ActorCritic(
            obs_dim=self.vec_env.obs_dim,
            hidden_dim=self.config.get("hidden_dim", 512),
        ).to(self.device)
        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"  Policy: {n_params:,} parameters")

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

        # Buffer
        self.buffer = RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            obs_dim=self.vec_env.obs_dim,
            device=self.device,
        )

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(self.output_dir, self.run_name, "tb_logs")
            os.makedirs(log_dir, exist_ok=True)
            self.logger = SummaryWriter(log_dir)
        except ImportError:
            self.logger = None

        # Checkpoint dir
        self.ckpt_dir = os.path.join(self.output_dir, self.run_name, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(self) -> None:
        """Main training loop."""
        print(f"\nStarting training for {self.total_timesteps:,} timesteps...")
        self.setup()

        obs = self.vec_env.reset()
        global_step = 0
        n_updates = 0
        episode_rewards = []
        episode_lengths = []
        ep_reward_buf = np.zeros(self.n_envs)
        ep_length_buf = np.zeros(self.n_envs, dtype=int)
        next_milestone_idx = 0

        while global_step < self.total_timesteps:
            # Update decoy tier unlock schedule
            self._update_decoy_schedule(global_step)

            # Collect rollout
            self.buffer.reset()
            self.policy.eval()

            for step in range(self.n_steps):
                self.vec_env.set_global_step(global_step)

                # Get action masks
                masks = self.vec_env.get_action_masks()
                op_masks = np.stack([m["op_mask"] for m in masks])
                pos_masks = np.stack([m["pos_mask"] for m in masks])

                # Get actions from policy
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                mask_dict = {
                    "op_mask": torch.BoolTensor(op_masks).to(self.device),
                    "pos_mask": torch.BoolTensor(pos_masks).to(self.device),
                }

                with torch.no_grad():
                    ops, positions, tokens, values = self.policy(obs_tensor, mask_dict)

                ops_np = ops.cpu().numpy()
                pos_np = positions.cpu().numpy()
                tok_np = tokens.cpu().numpy()
                val_np = values.cpu().numpy()

                # Compute log-probs for the sampled actions
                with torch.no_grad():
                    log_probs, _, _, _ = self.policy(
                        obs_tensor, mask_dict,
                        actions=(ops, positions, tokens),
                    )
                    lp_np = log_probs.cpu().numpy()

                # Step environments
                actions = [(int(ops_np[i]), int(pos_np[i]), int(tok_np[i])) for i in range(self.n_envs)]
                next_obs, rewards, dones, infos = self.vec_env.step(actions)

                # Store in buffer
                self.buffer.add(obs, ops_np, pos_np, tok_np, lp_np,
                               rewards, dones, val_np, op_masks, pos_masks)

                # Track episode stats
                ep_reward_buf += rewards
                ep_length_buf += 1
                for i in range(self.n_envs):
                    if dones[i]:
                        episode_rewards.append(ep_reward_buf[i])
                        episode_lengths.append(ep_length_buf[i])
                        ep_reward_buf[i] = 0.0
                        ep_length_buf[i] = 0

                obs = next_obs
                global_step += self.n_envs

            # Compute GAE
            with torch.no_grad():
                last_values = self.policy.get_value(
                    torch.FloatTensor(obs).to(self.device)
                ).cpu().numpy()
            self.buffer.compute_gae(last_values, self.gamma, self.gae_lambda)

            # PPO update
            self.policy.train()
            total_pg_loss = 0
            total_vf_loss = 0
            total_entropy = 0
            n_batches = 0

            for epoch in range(self.n_epochs):
                for batch in self.buffer.get_batches(self.batch_size):
                    # Normalize advantages
                    adv = batch["advantages"]
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Evaluate current policy on stored data
                    log_probs, entropy, values, _ = self.policy(
                        batch["obs"],
                        action_masks={
                            "op_mask": batch["op_masks"],
                            "pos_mask": batch["pos_masks"],
                        },
                        actions=(batch["ops"], batch["positions"], batch["tokens"]),
                    )

                    # PPO clipped objective
                    ratio = torch.exp(log_probs - batch["old_log_probs"])
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    vf_loss = F.mse_loss(values, batch["returns"])

                    # Entropy bonus
                    entropy_loss = -entropy.mean()

                    # Total loss
                    loss = pg_loss + self.vf_coef * vf_loss + self.entropy_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    total_pg_loss += pg_loss.item()
                    total_vf_loss += vf_loss.item()
                    total_entropy += (-entropy_loss).item()
                    n_batches += 1

            n_updates += 1

            # Logging
            if n_batches > 0:
                avg_pg = total_pg_loss / n_batches
                avg_vf = total_vf_loss / n_batches
                avg_ent = total_entropy / n_batches

            if n_updates % 10 == 0 and episode_rewards:
                recent = episode_rewards[-100:]
                mean_r = np.mean(recent)
                mean_l = np.mean(episode_lengths[-100:])
                print(
                    f"Step {global_step:>10,} | "
                    f"Eps: {len(episode_rewards):>6} | "
                    f"R: {mean_r:>8.3f} | "
                    f"Len: {mean_l:>5.1f} | "
                    f"PG: {avg_pg:>8.4f} | "
                    f"VF: {avg_vf:>8.4f} | "
                    f"Ent: {avg_ent:>6.3f}"
                )

                if self.logger:
                    self.logger.add_scalar("train/mean_reward", mean_r, global_step)
                    self.logger.add_scalar("train/mean_episode_length", mean_l, global_step)
                    self.logger.add_scalar("train/pg_loss", avg_pg, global_step)
                    self.logger.add_scalar("train/vf_loss", avg_vf, global_step)
                    self.logger.add_scalar("train/entropy", avg_ent, global_step)

            # Checkpointing
            if next_milestone_idx < len(self.milestones) and global_step >= self.milestones[next_milestone_idx]:
                ms = self.milestones[next_milestone_idx]
                self.save_checkpoint(f"milestone_{ms}")
                print(f"  ** Milestone checkpoint saved: {ms:,} steps")
                next_milestone_idx += 1

            if global_step % self.checkpoint_interval < self.n_envs * self.n_steps:
                self.save_checkpoint("latest")

        # Final checkpoint
        self.save_checkpoint("final")
        print(f"\nTraining complete: {global_step:,} steps, {len(episode_rewards)} episodes")

        if self.logger:
            self.logger.close()

    def _update_decoy_schedule(self, global_step: int) -> None:
        """Update decoy tier unlock based on training progress."""
        if self.decoy_scorer is None:
            return
        if global_step < 2_000_000:
            tiers = ["A"]
        elif global_step < 5_000_000:
            tiers = ["A", "B"]
        elif global_step < 8_000_000:
            tiers = ["A", "B", "D"]
        else:
            tiers = ["A", "B", "D", "C"]
        self.decoy_scorer.set_unlocked_tiers(tiers)

    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.ckpt_dir, f"{name}.pt")
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="TCRPPO v2 Training")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--run_name", default=None, help="Run name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--reward_mode", default=None, help="Reward mode override")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total timesteps")
    parser.add_argument("--n_envs", type=int, default=None, help="Num envs")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--w_affinity", type=float, default=None, help="Affinity weight")
    parser.add_argument("--w_decoy", type=float, default=None, help="Decoy weight")
    parser.add_argument("--w_naturalness", type=float, default=None, help="Naturalness weight")
    parser.add_argument("--w_diversity", type=float, default=None, help="Diversity weight")
    parser.add_argument("--min_steps", type=int, default=None, help="Min steps before STOP")
    parser.add_argument("--min_steps_penalty", type=float, default=None, help="Penalty for early STOP")
    # Two-phase training support
    parser.add_argument("--resume_from", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--resume_change_reward_mode", default=None, help="Change reward mode on resume")
    parser.add_argument("--resume_reset_optimizer", action="store_true", help="Reset optimizer on resume")
    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.run_name:
        config["run_name"] = args.run_name
    if args.seed is not None:
        config["seed"] = args.seed
    if args.reward_mode:
        config["reward_mode"] = args.reward_mode
    if args.total_timesteps:
        config["total_timesteps"] = args.total_timesteps
    if args.n_envs:
        config["n_envs"] = args.n_envs
    config["device"] = args.device
    if args.w_affinity is not None:
        config["w_affinity"] = args.w_affinity
    if args.w_decoy is not None:
        config["w_decoy"] = args.w_decoy
    if args.w_naturalness is not None:
        config["w_naturalness"] = args.w_naturalness
    if args.w_diversity is not None:
        config["w_diversity"] = args.w_diversity
    if args.min_steps is not None:
        config["min_steps"] = args.min_steps
    if args.min_steps_penalty is not None:
        config["min_steps_penalty"] = args.min_steps_penalty

    config.setdefault("run_name", "v2_run")

    trainer = PPOTrainer(config)

    # Two-phase training: resume from checkpoint and optionally change reward mode
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

        if args.resume_change_reward_mode:
            print(f"Changing reward mode to: {args.resume_change_reward_mode}")
            trainer.reward_manager.reward_mode = args.resume_change_reward_mode
            config["reward_mode"] = args.resume_change_reward_mode

        if args.resume_reset_optimizer:
            print("Resetting optimizer")
            trainer.optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=config["learning_rate"])

    trainer.train()


if __name__ == "__main__":
    main()
