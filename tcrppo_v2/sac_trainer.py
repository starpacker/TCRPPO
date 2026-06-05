"""Off-policy SAC trainer for TCR sequence design.

This module ports the replay-buffer/actor-critic design loop from the
metasurface SAC code into the TCR editing stack.  It keeps the TCR-specific
parts from ``tcrppo_v2`` intact:

* observations are ESM-encoded TCR and pMHC embeddings;
* actions are masked autoregressive edits: op, position, amino-acid token;
* rewards are produced by ``RewardManager`` and can use tFold AMP.

The SAC implementation is the discrete/action-sampled variant: critics score a
sampled edit tuple, and actor/target updates use the sampled
``Q - alpha * log_pi`` objective.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.ppo_trainer import PPOTrainer, load_config
from tcrppo_v2.utils.constants import MAX_TCR_LEN, NUM_AMINO_ACIDS, NUM_OPS


class SACReplayBuffer:
    """Replay buffer for vectorized TCR edit transitions."""

    def __init__(self, capacity: int, obs_dim: int, device: str = "cpu"):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.ops = np.zeros(self.capacity, dtype=np.int64)
        self.positions = np.zeros(self.capacity, dtype=np.int64)
        self.tokens = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.op_masks = np.zeros((self.capacity, NUM_OPS), dtype=bool)
        self.pos_masks = np.zeros((self.capacity, MAX_TCR_LEN), dtype=bool)
        self.next_op_masks = np.zeros((self.capacity, NUM_OPS), dtype=bool)
        self.next_pos_masks = np.zeros((self.capacity, MAX_TCR_LEN), dtype=bool)

    def add_batch(
        self,
        obs: np.ndarray,
        actions: Tuple[np.ndarray, np.ndarray, np.ndarray],
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        op_masks: np.ndarray,
        pos_masks: np.ndarray,
        next_op_masks: np.ndarray,
        next_pos_masks: np.ndarray,
    ) -> None:
        """Add one vectorized environment step."""
        ops, positions, tokens = actions
        n = int(obs.shape[0])
        for i in range(n):
            idx = self.ptr
            self.obs[idx] = obs[i]
            self.next_obs[idx] = next_obs[i]
            self.ops[idx] = ops[i]
            self.positions[idx] = positions[i]
            self.tokens[idx] = tokens[i]
            self.rewards[idx] = rewards[i]
            self.dones[idx] = dones[i]
            self.op_masks[idx] = op_masks[i]
            self.pos_masks[idx] = pos_masks[i]
            self.next_op_masks[idx] = next_op_masks[i]
            self.next_pos_masks[idx] = next_pos_masks[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random minibatch."""
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=self.device),
            "ops": torch.as_tensor(self.ops[idx], dtype=torch.long, device=self.device),
            "positions": torch.as_tensor(self.positions[idx], dtype=torch.long, device=self.device),
            "tokens": torch.as_tensor(self.tokens[idx], dtype=torch.long, device=self.device),
            "rewards": torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=self.device),
            "dones": torch.as_tensor(self.dones[idx], dtype=torch.float32, device=self.device),
            "op_masks": torch.as_tensor(self.op_masks[idx], dtype=torch.bool, device=self.device),
            "pos_masks": torch.as_tensor(self.pos_masks[idx], dtype=torch.bool, device=self.device),
            "next_op_masks": torch.as_tensor(self.next_op_masks[idx], dtype=torch.bool, device=self.device),
            "next_pos_masks": torch.as_tensor(self.next_pos_masks[idx], dtype=torch.bool, device=self.device),
        }


class ActionQNetwork(nn.Module):
    """Q(s, op, position, token) for autoregressive edit actions."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 512,
        op_embed_dim: int = 16,
        pos_embed_dim: int = 32,
        token_embed_dim: int = 16,
        max_tcr_len: int = MAX_TCR_LEN,
    ):
        super().__init__()
        self.op_embed = nn.Embedding(NUM_OPS, op_embed_dim)
        self.pos_embed = nn.Embedding(max_tcr_len, pos_embed_dim)
        self.token_embed = nn.Embedding(NUM_AMINO_ACIDS, token_embed_dim)
        in_dim = obs_dim + op_embed_dim + pos_embed_dim + token_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)

    def forward(
        self,
        obs: torch.Tensor,
        ops: torch.Tensor,
        positions: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        pos = positions.clamp(min=0, max=self.pos_embed.num_embeddings - 1)
        tok = tokens.clamp(min=0, max=NUM_AMINO_ACIDS - 1)
        x = torch.cat(
            [obs, self.op_embed(ops), self.pos_embed(pos), self.token_embed(tok)],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


class TCRSACTrainer(PPOTrainer):
    """SAC trainer that reuses the existing tcrppo_v2 environment/scorers."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.replay_capacity = config.get("replay_buffer_capacity", 200_000)
        self.learning_starts = config.get("learning_starts", 2_000)
        self.gradient_steps = config.get("gradient_steps", 1)
        self.policy_delay = config.get("policy_delay", 1)
        self.tau = config.get("tau", 0.005)
        self.sac_batch_size = config.get("sac_batch_size", config.get("batch_size", 256))
        self.actor_lr = config.get("actor_learning_rate", config.get("learning_rate", 3e-4))
        self.critic_lr = config.get("critic_learning_rate", config.get("learning_rate", 3e-4))
        self.alpha_lr = config.get("alpha_learning_rate", 3e-4)
        self.init_temperature = config.get("init_temperature", 0.1)
        self.automatic_entropy_tuning = config.get("automatic_entropy_tuning", True)
        self.target_entropy = config.get("target_entropy", -4.0)
        self.random_steps = config.get("random_steps", self.learning_starts)

        self.q1: Optional[ActionQNetwork] = None
        self.q2: Optional[ActionQNetwork] = None
        self.q1_target: Optional[ActionQNetwork] = None
        self.q2_target: Optional[ActionQNetwork] = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.log_alpha = None
        self.alpha_optimizer = None
        self.replay_buffer: Optional[SACReplayBuffer] = None
        self._updates = 0
        self.best_affinity = -np.inf
        self.best_record: Dict[str, object] = {}
        self.best_records_by_peptide: Dict[str, Dict[str, object]] = {}
        self.best_live_update_interval = config.get("best_live_update_interval", 1)
        self.best_event_log = config.get("best_event_log", True)
        self._best_updates_since_flush = 0
        self._out_dir: Optional[str] = None

    @property
    def alpha(self) -> torch.Tensor:
        if self.log_alpha is None:
            return torch.tensor(self.init_temperature, device=self.device)
        return self.log_alpha.exp()

    def setup(self) -> None:
        """Build the shared TCR stack, then replace PPO optimizers with SAC."""
        print(f"Setting up TCR-SAC trainer: {self.run_name}")
        super().setup()

        obs_dim = self.vec_env.obs_dim
        hidden_dim = self.config.get("hidden_dim", 512)
        self.q1 = ActionQNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim).to(self.device)
        self.q2 = ActionQNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim).to(self.device)
        self.q1_target = ActionQNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim).to(self.device)
        self.q2_target = ActionQNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.critic_lr,
            eps=1e-5,
        )
        self.log_alpha = torch.tensor(
            np.log(self.init_temperature),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.replay_buffer = SACReplayBuffer(
            capacity=self.replay_capacity,
            obs_dim=obs_dim,
            device=self.device,
        )
        self._out_dir = os.path.join(self.output_dir, self.run_name)
        os.makedirs(self._out_dir, exist_ok=True)
        print(
            f"  SAC: replay={self.replay_capacity:,}, learning_starts={self.learning_starts:,}, "
            f"batch={self.sac_batch_size}, alpha={self.init_temperature}, "
            f"target_entropy={self.target_entropy}"
        )

    def _mask_dict(self, op_masks: np.ndarray, pos_masks: np.ndarray) -> Dict[str, torch.Tensor]:
        return {
            "op_mask": torch.as_tensor(op_masks, dtype=torch.bool, device=self.device),
            "pos_mask": torch.as_tensor(pos_masks, dtype=torch.bool, device=self.device),
        }

    def _sample_random_actions(
        self,
        op_masks: np.ndarray,
        pos_masks: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ops, positions, tokens = [], [], []
        for op_mask, pos_mask in zip(op_masks, pos_masks):
            valid_ops = np.flatnonzero(op_mask)
            valid_pos = np.flatnonzero(pos_mask)
            op = int(np.random.choice(valid_ops))
            pos = int(np.random.choice(valid_pos)) if len(valid_pos) else 0
            tok = int(np.random.randint(NUM_AMINO_ACIDS))
            ops.append(op)
            positions.append(pos)
            tokens.append(tok)
        return (
            np.asarray(ops, dtype=np.int64),
            np.asarray(positions, dtype=np.int64),
            np.asarray(tokens, dtype=np.int64),
        )

    def _sample_policy_actions(
        self,
        obs: np.ndarray,
        op_masks: np.ndarray,
        pos_masks: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        with torch.no_grad():
            ops, positions, tokens, _ = self.policy(
                obs_tensor,
                self._mask_dict(op_masks, pos_masks),
            )
        return (
            ops.cpu().numpy().astype(np.int64),
            positions.cpu().numpy().astype(np.int64),
            tokens.cpu().numpy().astype(np.int64),
        )

    def _sample_actions_and_logp(
        self,
        obs: torch.Tensor,
        op_masks: torch.Tensor,
        pos_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = {"op_mask": op_masks, "pos_mask": pos_masks}
        ops, positions, tokens, _ = self.policy(obs, masks)
        logp, _, _, _ = self.policy(obs, masks, actions=(ops, positions, tokens))
        return ops, positions, tokens, logp

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for src, dst in zip(source.parameters(), target.parameters()):
                dst.data.mul_(1.0 - tau).add_(src.data, alpha=tau)

    def _maybe_update_best_record(
        self,
        *,
        peptide: str,
        tcr: str,
        affinity: float,
        reward: float,
        global_step: int,
        episode_step: int,
        components: Dict[str, float],
    ) -> None:
        """Track global and per-target best TCR records."""
        record = {
            "step": int(global_step),
            "episode_step": int(episode_step),
            "tcr": tcr,
            "peptide": peptide,
            "affinity": float(affinity),
            "reward": float(reward),
            "components": components,
        }
        changed = False
        changed_fields = []
        if affinity > self.best_affinity:
            self.best_affinity = float(affinity)
            self.best_record = dict(record)
            changed = True
            changed_fields.append("global")

        current = self.best_records_by_peptide.get(peptide)
        if current is None or affinity > float(current.get("affinity", -np.inf)):
            self.best_records_by_peptide[peptide] = dict(record)
            changed = True
            changed_fields.append("peptide")

        if changed:
            self._best_updates_since_flush += 1
            if self.best_event_log:
                self._append_best_event(record, changed_fields)
            interval = max(1, int(self.best_live_update_interval))
            if self._best_updates_since_flush >= interval:
                self._flush_best_records()

    @staticmethod
    def _atomic_write_json(path: str, data) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)

    def _flush_best_records(self) -> None:
        """Persist live best records for monitoring a running job."""
        if not self._out_dir:
            return
        os.makedirs(self._out_dir, exist_ok=True)
        if self.best_record:
            self._atomic_write_json(
                os.path.join(self._out_dir, "best_tcr.json"),
                self.best_record,
            )
        if self.best_records_by_peptide:
            self._atomic_write_json(
                os.path.join(self._out_dir, "best_tcr_by_peptide.json"),
                self.best_records_by_peptide,
            )
        self._best_updates_since_flush = 0

    def _append_best_event(self, record: Dict[str, object], changed_fields: List[str]) -> None:
        """Append a JSONL event whenever a step-wise best record improves."""
        if not self._out_dir:
            return
        event = dict(record)
        event["changed"] = changed_fields
        path = os.path.join(self._out_dir, "best_tcr_events.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def _update_sac(self) -> Dict[str, float]:
        assert self.replay_buffer is not None
        batch = self.replay_buffer.sample(self.sac_batch_size)
        obs = torch.nan_to_num(batch["obs"], nan=0.0, posinf=1.0, neginf=-1.0)
        next_obs = torch.nan_to_num(batch["next_obs"], nan=0.0, posinf=1.0, neginf=-1.0)

        with torch.no_grad():
            next_ops, next_pos, next_tok, next_logp = self._sample_actions_and_logp(
                next_obs,
                batch["next_op_masks"],
                batch["next_pos_masks"],
            )
            next_q = torch.min(
                self.q1_target(next_obs, next_ops, next_pos, next_tok),
                self.q2_target(next_obs, next_ops, next_pos, next_tok),
            )
            target_q = batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * (
                next_q - self.alpha.detach() * next_logp
            )

        q1_pred = self.q1(obs, batch["ops"], batch["positions"], batch["tokens"])
        q2_pred = self.q2(obs, batch["ops"], batch["positions"], batch["tokens"])
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.max_grad_norm,
        )
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        alpha_loss_value = 0.0
        mean_logp = 0.0
        if self._updates % self.policy_delay == 0:
            ops, pos, tok, logp = self._sample_actions_and_logp(
                obs,
                batch["op_masks"],
                batch["pos_masks"],
            )
            q_pi = torch.min(self.q1(obs, ops, pos, tok), self.q2(obs, ops, pos, tok))
            actor_loss = (self.alpha.detach() * logp - q_pi).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha_loss_value = float(alpha_loss.item())

            actor_loss_value = float(actor_loss.item())
            mean_logp = float(logp.mean().item())

        self._soft_update(self.q1, self.q1_target, self.tau)
        self._soft_update(self.q2, self.q2_target, self.tau)
        self._updates += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.detach().item()),
            "logp": mean_logp,
            "q": float(q1_pred.mean().item()),
        }

    def train(self) -> None:
        """Main off-policy training loop."""
        print(f"\nStarting TCR-SAC training for {self.total_timesteps:,} env steps...")
        self.setup()
        self._save_experiment_json()

        resume_step = 0
        if getattr(self, "_resume_from", None):
            print(f"Resuming SAC from checkpoint: {self._resume_from}")
            resume_step = self.load_checkpoint(self._resume_from)
            print(f"  Resumed at step {resume_step:,}")

        obs = self.vec_env.reset()
        global_step = resume_step
        ep_reward_buf = np.zeros(self.n_envs, dtype=np.float32)
        ep_length_buf = np.zeros(self.n_envs, dtype=np.int64)
        episode_rewards: List[float] = []
        last_logged_episode = 0
        last_update_stats: Dict[str, float] = {}
        start_time = time.time()
        next_ckpt_step = None
        if self.checkpoint_interval and self.checkpoint_interval > 0:
            next_ckpt_step = (
                (global_step // self.checkpoint_interval) + 1
            ) * self.checkpoint_interval
        next_latest_step = None
        if self.latest_checkpoint_interval and self.latest_checkpoint_interval > 0:
            next_latest_step = (
                (global_step // self.latest_checkpoint_interval) + 1
            ) * self.latest_checkpoint_interval

        while global_step < self.total_timesteps:
            self._update_decoy_schedule(global_step)
            self._update_reward_schedule(global_step)
            self.vec_env.set_global_step(global_step)

            reset_indices, reset_obs = self.vec_env.reset_done()
            if reset_indices:
                for j, i in enumerate(reset_indices):
                    obs[i] = reset_obs[j]

            masks = self.vec_env.get_action_masks()
            op_masks = np.stack([m["op_mask"] for m in masks])
            pos_masks = np.stack([m["pos_mask"] for m in masks])

            if global_step < self.random_steps:
                ops_np, pos_np, tok_np = self._sample_random_actions(op_masks, pos_masks)
            else:
                ops_np, pos_np, tok_np = self._sample_policy_actions(obs, op_masks, pos_masks)

            was_done = np.array([env.done for env in self.vec_env.envs], dtype=bool)
            actions = [(int(ops_np[i]), int(pos_np[i]), int(tok_np[i])) for i in range(self.n_envs)]
            next_obs, rewards, dones, infos = self.vec_env.step(actions)
            effective_dones = dones | was_done

            next_masks = self.vec_env.get_action_masks()
            next_op_masks = np.stack([m["op_mask"] for m in next_masks])
            next_pos_masks = np.stack([m["pos_mask"] for m in next_masks])

            self.replay_buffer.add_batch(
                obs=obs,
                actions=(ops_np, pos_np, tok_np),
                rewards=rewards,
                next_obs=next_obs,
                dones=effective_dones.astype(np.float32),
                op_masks=op_masks,
                pos_masks=pos_masks,
                next_op_masks=next_op_masks,
                next_pos_masks=next_pos_masks,
            )

            for i in range(self.n_envs):
                components = infos[i].get("reward_components", {}) or {}
                if not was_done[i] and components and "affinity_raw" in components:
                    affinity = float(components.get("affinity_raw", 0.0))
                    tcr = infos[i].get("new_tcr", self.vec_env.envs[i].current_tcr)
                    peptide = self.vec_env.envs[i].peptide
                    self._maybe_update_best_record(
                        peptide=peptide,
                        tcr=tcr,
                        affinity=affinity,
                        reward=float(rewards[i]),
                        global_step=global_step,
                        episode_step=int(ep_length_buf[i] + 1),
                        components=components,
                    )
                if not was_done[i]:
                    ep_reward_buf[i] += rewards[i]
                    ep_length_buf[i] += 1
                if dones[i] and not was_done[i]:
                    components = infos[i].get("reward_components", {}) or {}
                    affinity = float(components.get("affinity_raw", 0.0))
                    tcr = infos[i].get("new_tcr", self.vec_env.envs[i].current_tcr)
                    peptide = self.vec_env.envs[i].peptide
                    if components and "affinity_raw" in components:
                        self._maybe_update_best_record(
                            peptide=peptide,
                            tcr=tcr,
                            affinity=affinity,
                            reward=float(ep_reward_buf[i]),
                            global_step=global_step,
                            episode_step=int(ep_length_buf[i]),
                            components=components,
                        )
                    episode_rewards.append(float(ep_reward_buf[i]))
                    print(
                        f"Episode {len(episode_rewards)} | Step {global_step} | "
                        f"R={ep_reward_buf[i]:.3f} | Len={ep_length_buf[i]} | "
                        f"A={affinity:.4f} | Peptide={peptide} | TCR={tcr}",
                        flush=True,
                    )
                    ep_reward_buf[i] = 0.0
                    ep_length_buf[i] = 0
                elif was_done[i]:
                    ep_reward_buf[i] = 0.0
                    ep_length_buf[i] = 0

            obs = next_obs
            global_step += self.n_envs

            if self.replay_buffer.size >= self.learning_starts:
                for _ in range(self.gradient_steps):
                    last_update_stats = self._update_sac()

            log_every = int(self.config.get("log_every_episodes", 20))
            if (
                episode_rewards
                and log_every > 0
                and len(episode_rewards) != last_logged_episode
                and len(episode_rewards) % log_every == 0
            ):
                last_logged_episode = len(episode_rewards)
                recent = episode_rewards[-100:]
                elapsed = time.time() - start_time
                stats = " ".join(f"{k}={v:.4f}" for k, v in last_update_stats.items())
                print(
                    f"[SAC] step={global_step:,} episodes={len(episode_rewards)} "
                    f"meanR100={np.mean(recent):.3f} replay={self.replay_buffer.size:,} "
                    f"{stats} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if next_latest_step is not None and global_step >= next_latest_step:
                self.save_checkpoint("latest", global_step)
                next_latest_step += self.latest_checkpoint_interval
            if next_ckpt_step is not None and global_step >= next_ckpt_step:
                self.save_checkpoint(f"checkpoint_{global_step}", global_step)
                next_ckpt_step += self.checkpoint_interval

        self.save_checkpoint("final", global_step)
        self._flush_best_records()
        if self.best_record:
            out_dir = os.path.join(self.output_dir, self.run_name)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "best_tcr.json"), "w") as f:
                json.dump(self.best_record, f, indent=2)
            with open(os.path.join(out_dir, "best_tcr_by_peptide.json"), "w") as f:
                json.dump(self.best_records_by_peptide, f, indent=2, sort_keys=True)
            print(f"Best TCR: {self.best_record}", flush=True)
            print(
                f"Per-peptide best records: {len(self.best_records_by_peptide)} targets",
                flush=True,
            )

    def save_checkpoint(self, name: str, global_step: int = 0) -> None:
        """Save SAC actor/critic checkpoint."""
        path = os.path.join(self.ckpt_dir, f"{name}.pt")
        tmp_path = f"{path}.tmp"
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "q1_target_state_dict": self.q1_target.state_dict(),
                "q2_target_state_dict": self.q2_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "config": self.config,
                "global_step": global_step,
                "best_record": self.best_record,
                "best_records_by_peptide": self.best_records_by_peptide,
            },
            tmp_path,
        )
        os.replace(tmp_path, path)

    def load_checkpoint(self, path: str) -> int:
        """Load a SAC checkpoint after setup()."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.q1.load_state_dict(ckpt["q1_state_dict"])
        self.q2.load_state_dict(ckpt["q2_state_dict"])
        self.q1_target.load_state_dict(ckpt["q1_target_state_dict"])
        self.q2_target.load_state_dict(ckpt["q2_target_state_dict"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])
        if "alpha_optimizer_state_dict" in ckpt:
            self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer_state_dict"])
        if "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.best_record = ckpt.get("best_record", {})
        self.best_records_by_peptide = ckpt.get("best_records_by_peptide", {})
        self.best_affinity = float(self.best_record.get("affinity", -np.inf))
        return int(ckpt.get("global_step", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="TCR-SAC Training")
    parser.add_argument("--config", default="configs/sac_tfold_amp.yaml", help="Config file")
    parser.add_argument("--run_name", default=None, help="Run name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total env steps")
    parser.add_argument("--n_envs", type=int, default=None, help="Number of vectorized envs")
    parser.add_argument("--device", default=None, help="Torch device")
    parser.add_argument("--affinity_scorer", default=None, help="Affinity scorer override")
    parser.add_argument("--encoder", default=None, choices=["esm2", "lightweight"], help="State encoder")
    parser.add_argument("--train_targets", default=None, help="Comma-separated peptides or target txt file")
    parser.add_argument("--resume_from", default=None, help="SAC checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.run_name:
        config["run_name"] = args.run_name
    if args.seed is not None:
        config["seed"] = args.seed
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        config["n_envs"] = args.n_envs
    if args.device is not None:
        config["device"] = args.device
    if args.affinity_scorer is not None:
        config["affinity_model"] = args.affinity_scorer
    if args.encoder is not None:
        config["encoder"] = args.encoder
    if args.train_targets is not None:
        config["train_targets"] = args.train_targets
    config.setdefault("run_name", "sac_tfold_amp")

    trainer = TCRSACTrainer(config)
    trainer._resume_from = args.resume_from
    try:
        trainer.train()
    except Exception as exc:
        print("\nFATAL ERROR during SAC training:")
        print(f"  {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
