"""TCR editing environment with indel actions and ESM-2 state encoding."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tcrppo_v2.utils.constants import (
    AMINO_ACIDS, NUM_AMINO_ACIDS, IDX_TO_AA,
    MIN_TCR_LEN, MAX_TCR_LEN, MAX_STEPS_PER_EPISODE,
    OP_SUB, OP_INS, OP_DEL, OP_STOP, NUM_OPS,
)
from tcrppo_v2.utils.encoding import is_valid_tcr


class TCREditEnv:
    """Single TCR editing environment.

    Observation: [TCR_emb | pMHC_emb | remaining_steps/max | cumulative_delta]
    Action: (op_type, position, token) — 3-head autoregressive
    """

    def __init__(
        self,
        esm_cache,
        pmhc_loader,
        tcr_pool,
        reward_manager,
        decoy_sampler=None,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        max_tcr_len: int = MAX_TCR_LEN,
        min_tcr_len: int = MIN_TCR_LEN,
        reward_mode: str = "v2_full",
        min_steps: int = 0,
        min_steps_penalty: float = 0.0,
        ban_stop: bool = False,
        terminal_reward_only: bool = False,
    ):
        """Initialize environment.

        Args:
            esm_cache: ESMCache instance for state encoding.
            pmhc_loader: PMHCLoader instance for target management.
            tcr_pool: TCRPool instance for curriculum sampling.
            reward_manager: RewardManager for reward computation.
            decoy_sampler: DecoySampler (optional, for decoy penalty).
            max_steps: Max steps per episode.
            max_tcr_len: Max TCR sequence length.
            min_tcr_len: Min TCR sequence length.
            reward_mode: Reward mode string.
            min_steps: Minimum steps before STOP is allowed.
            min_steps_penalty: Penalty for early STOP.
            ban_stop: If True, STOP action is never allowed.
            terminal_reward_only: If True, only compute reward at episode end.
        """
        self.esm_cache = esm_cache
        self.pmhc_loader = pmhc_loader
        self.tcr_pool = tcr_pool
        self.reward_manager = reward_manager
        self.decoy_sampler = decoy_sampler

        self.max_steps = max_steps
        self.max_tcr_len = max_tcr_len
        self.min_tcr_len = min_tcr_len
        self.reward_mode = reward_mode
        self.min_steps = min_steps
        self.min_steps_penalty = min_steps_penalty
        self.ban_stop = ban_stop
        self.terminal_reward_only = terminal_reward_only

        # State dimensions
        self.esm_dim = esm_cache.output_dim  # 1280
        # obs = [tcr_emb(1280) | pmhc_emb(1280) | remaining_steps(1) | cumulative_delta(1)]
        self.obs_dim = self.esm_dim * 2 + 2

        # Episode state
        self.tcr_seq: str = ""
        self.initial_tcr: str = ""
        self.peptide: str = ""
        self.target: str = ""
        self.pmhc_emb: Optional[torch.Tensor] = None
        self.step_count: int = 0
        self.cumulative_delta: float = 0.0
        self.initial_affinity: float = 0.0
        self.done: bool = True
        self.global_step: int = 0  # For curriculum/decoy schedule

    def set_global_step(self, step: int) -> None:
        """Update global training step for curriculum scheduling."""
        self.global_step = step

    def reset(
        self,
        peptide: Optional[str] = None,
        init_tcr: Optional[str] = None,
        target_weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Reset environment for a new episode.

        Args:
            peptide: Specific target peptide (random if None).
            init_tcr: Specific initial TCR (curriculum sampled if None).
            target_weights: Difficulty weights for target sampling.

        Returns:
            Observation array of shape [obs_dim].
        """
        # Select target
        if peptide is None:
            if target_weights:
                self.peptide = self.pmhc_loader.sample_target_weighted(target_weights)
            else:
                self.peptide = self.pmhc_loader.sample_target()
        else:
            self.peptide = peptide
        self.target = self.peptide

        # Encode pMHC (cached)
        pmhc_string = self.pmhc_loader.get_pmhc_string(self.peptide)
        self.pmhc_emb = self.esm_cache.encode_pmhc(pmhc_string)

        # Select initial TCR
        if init_tcr is not None:
            self.tcr_seq = init_tcr
        else:
            self.tcr_seq, _ = self.tcr_pool.sample_tcr(
                self.target, step=self.global_step, reward_mode=self.reward_mode
            )
        self.initial_tcr = self.tcr_seq

        # Compute initial affinity for delta reward
        if self.reward_manager.affinity_scorer is not None:
            if hasattr(self.reward_manager.affinity_scorer, 'score_batch_fast'):
                preds = self.reward_manager.affinity_scorer.score_batch_fast(
                    [self.tcr_seq], [self.peptide])
                self.initial_affinity = float(np.nan_to_num(preds[0], nan=0.0, posinf=0.0, neginf=0.0))
            else:
                aff, _ = self.reward_manager.affinity_scorer.score(self.tcr_seq, self.peptide)
                self.initial_affinity = float(np.nan_to_num(aff, nan=0.0, posinf=0.0, neginf=0.0))
        else:
            self.initial_affinity = 0.0

        self.step_count = 0
        self.cumulative_delta = 0.0
        self.done = False

        return self._get_obs()

    def step(self, action: Tuple[int, int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one editing action.

        Args:
            action: (op_type, position, token) tuple.

        Returns:
            (observation, reward, done, info) tuple.
        """
        assert not self.done, "Episode already done. Call reset()."

        op_type, position, token = action
        old_tcr = self.tcr_seq
        info = {"old_tcr": old_tcr, "op_type": op_type, "position": position, "token": token}

        # Apply action
        if op_type == OP_STOP:
            self.done = True
            info["action_name"] = "STOP"
        elif op_type == OP_SUB:
            self.tcr_seq = self._apply_sub(position, token)
            info["action_name"] = "SUB"
        elif op_type == OP_INS:
            self.tcr_seq = self._apply_ins(position, token)
            info["action_name"] = "INS"
        elif op_type == OP_DEL:
            self.tcr_seq = self._apply_del(position)
            info["action_name"] = "DEL"

        self.step_count += 1
        info["new_tcr"] = self.tcr_seq
        info["step"] = self.step_count

        # Check if max steps reached
        if self.step_count >= self.max_steps:
            self.done = True

        # Compute reward
        if self.terminal_reward_only:
            # Terminal-only: only compute reward at episode end
            if self.done:
                reward, components = self.reward_manager.compute_reward(
                    tcr=self.tcr_seq,
                    peptide=self.peptide,
                    initial_affinity=self.initial_affinity,
                    target=self.target,
                )
            else:
                reward = 0.0
                components = {}
        else:
            # Per-step: compute reward at every step
            reward, components = self.reward_manager.compute_reward(
                tcr=self.tcr_seq,
                peptide=self.peptide,
                initial_affinity=self.initial_affinity,
                target=self.target,
            )

        # Min-steps penalty: penalize STOP before min_steps
        if op_type == OP_STOP and self.min_steps > 0 and self.step_count < self.min_steps:
            reward += self.min_steps_penalty
            components["min_steps_penalty"] = self.min_steps_penalty

        self.cumulative_delta += reward
        info["reward_components"] = components

        obs = self._get_obs()
        return obs, reward, self.done, info

    def _apply_sub(self, position: int, token: int) -> str:
        """Substitute amino acid at position with token."""
        seq = list(self.tcr_seq)
        pos = min(position, len(seq) - 1)
        aa = IDX_TO_AA.get(token, AMINO_ACIDS[token % NUM_AMINO_ACIDS])
        seq[pos] = aa
        return "".join(seq)

    def _apply_ins(self, position: int, token: int) -> str:
        """Insert amino acid at position."""
        seq = list(self.tcr_seq)
        if len(seq) >= self.max_tcr_len:
            return self.tcr_seq  # Safety: shouldn't happen with masking
        pos = min(position, len(seq))
        aa = IDX_TO_AA.get(token, AMINO_ACIDS[token % NUM_AMINO_ACIDS])
        seq.insert(pos, aa)
        return "".join(seq)

    def _apply_del(self, position: int) -> str:
        """Delete amino acid at position."""
        seq = list(self.tcr_seq)
        if len(seq) <= self.min_tcr_len:
            return self.tcr_seq  # Safety: shouldn't happen with masking
        pos = min(position, len(seq) - 1)
        del seq[pos]
        return "".join(seq)

    def _reset_internal(self, peptide=None, init_tcr=None, target_weights=None) -> None:
        """Reset state without computing observation (for batched encoding).

        Note: initial_affinity is set to 0.0 here. For batched resets,
        VecTCREditEnv.reset() will batch-compute initial affinities after all
        envs have been reset.
        """
        if peptide is None:
            if target_weights:
                self.peptide = self.pmhc_loader.sample_target_weighted(target_weights)
            else:
                self.peptide = self.pmhc_loader.sample_target()
        else:
            self.peptide = peptide
        self.target = self.peptide

        pmhc_string = self.pmhc_loader.get_pmhc_string(self.peptide)
        self.pmhc_emb = self.esm_cache.encode_pmhc(pmhc_string)

        if init_tcr is not None:
            self.tcr_seq = init_tcr
        else:
            self.tcr_seq, _ = self.tcr_pool.sample_tcr(
                self.target, step=self.global_step, reward_mode=self.reward_mode
            )
        self.initial_tcr = self.tcr_seq

        # Initial affinity is set to 0.0 for now — VecEnv will batch-compute it
        self.initial_affinity = 0.0

        self.step_count = 0
        self.cumulative_delta = 0.0
        self.done = False

    def _step_action_only(self, action: Tuple[int, int, int]) -> Tuple[bool, dict]:
        """Apply action without computing reward or observation (for batched processing).

        Returns:
            (done, info) tuple. Reward must be computed externally via batch.
        """
        assert not self.done, "Episode already done. Call reset()."

        op_type, position, token = action
        old_tcr = self.tcr_seq
        info = {"old_tcr": old_tcr, "op_type": op_type, "position": position, "token": token}

        if op_type == OP_STOP:
            self.done = True
            info["action_name"] = "STOP"
        elif op_type == OP_SUB:
            self.tcr_seq = self._apply_sub(position, token)
            info["action_name"] = "SUB"
        elif op_type == OP_INS:
            self.tcr_seq = self._apply_ins(position, token)
            info["action_name"] = "INS"
        elif op_type == OP_DEL:
            self.tcr_seq = self._apply_del(position)
            info["action_name"] = "DEL"

        self.step_count += 1
        info["new_tcr"] = self.tcr_seq
        info["step"] = self.step_count

        if self.step_count >= self.max_steps:
            self.done = True

        return self.done, info

    def _apply_reward(self, reward: float, components: dict) -> None:
        """Apply externally-computed reward to env state."""
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        self.cumulative_delta += reward

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        # TCR embedding (re-computed each step)
        tcr_emb = self.esm_cache.encode_tcr(self.tcr_seq)

        # Concatenate: [tcr_emb | pmhc_emb | remaining_steps/max | cumulative_delta]
        remaining = (self.max_steps - self.step_count) / self.max_steps
        obs = torch.cat([
            tcr_emb,
            self.pmhc_emb,
            torch.tensor([remaining, self.cumulative_delta], device=tcr_emb.device),
        ])
        return obs.cpu().numpy()

    def get_action_mask(self) -> Dict[str, np.ndarray]:
        """Get action masks for current state.

        Returns:
            Dict with 'op_mask' (4,), 'pos_mask' (max_tcr_len,), both boolean (True=valid).
        """
        seq_len = len(self.tcr_seq)

        # Op mask
        op_mask = np.ones(NUM_OPS, dtype=bool)
        if seq_len >= self.max_tcr_len:
            op_mask[OP_INS] = False
        if seq_len <= self.min_tcr_len:
            op_mask[OP_DEL] = False
        if self.step_count == 0 or self.ban_stop:
            op_mask[OP_STOP] = False

        # Position mask: only valid positions
        pos_mask = np.zeros(self.max_tcr_len, dtype=bool)
        pos_mask[:seq_len] = True

        return {"op_mask": op_mask, "pos_mask": pos_mask}

    @property
    def current_tcr(self) -> str:
        """Current TCR sequence."""
        return self.tcr_seq


class VecTCREditEnv:
    """Vectorized wrapper for multiple TCREditEnv instances."""

    def __init__(
        self,
        n_envs: int,
        esm_cache,
        pmhc_loader,
        tcr_pool,
        reward_manager,
        decoy_sampler=None,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        reward_mode: str = "v2_full",
        min_steps: int = 0,
        min_steps_penalty: float = 0.0,
        ban_stop: bool = False,
        terminal_reward_only: bool = False,
    ):
        """Create n_envs parallel environments sharing scorers."""
        self.n_envs = n_envs
        self.envs = [
            TCREditEnv(
                esm_cache=esm_cache,
                pmhc_loader=pmhc_loader,
                tcr_pool=tcr_pool,
                reward_manager=reward_manager,
                decoy_sampler=decoy_sampler,
                max_steps=max_steps,
                reward_mode=reward_mode,
                min_steps=min_steps,
                min_steps_penalty=min_steps_penalty,
                ban_stop=ban_stop,
                terminal_reward_only=terminal_reward_only,
            )
            for _ in range(n_envs)
        ]
        self.obs_dim = self.envs[0].obs_dim

    def set_global_step(self, step: int) -> None:
        """Set global step for all environments."""
        for env in self.envs:
            env.set_global_step(step)

    def reset(self) -> np.ndarray:
        """Reset all environments with batched ESM encoding. Returns (n_envs, obs_dim)."""
        for env in self.envs:
            env._reset_internal()

        # Batch-compute initial affinities for all envs
        scorer = self.envs[0].reward_manager.affinity_scorer
        if scorer is not None:
            tcrs = [env.tcr_seq for env in self.envs]
            peps = [env.peptide for env in self.envs]
            if hasattr(scorer, 'score_batch_fast'):
                preds = scorer.score_batch_fast(tcrs, peps)
                for i, env in enumerate(self.envs):
                    env.initial_affinity = float(np.nan_to_num(preds[i], nan=0.0, posinf=0.0, neginf=0.0))
            else:
                for env in self.envs:
                    aff, _ = scorer.score(env.tcr_seq, env.peptide)
                    env.initial_affinity = float(np.nan_to_num(aff, nan=0.0, posinf=0.0, neginf=0.0))

        tcr_seqs = [env.tcr_seq for env in self.envs]
        esm_cache = self.envs[0].esm_cache
        tcr_embs = esm_cache.encode_tcr_batch(tcr_seqs)

        obs_list = []
        for i, env in enumerate(self.envs):
            remaining = (env.max_steps - env.step_count) / env.max_steps
            obs = torch.cat([
                tcr_embs[i],
                env.pmhc_emb,
                torch.tensor([remaining, env.cumulative_delta], device=tcr_embs[i].device),
            ]).cpu().numpy()
            obs_list.append(obs)

        return np.stack(obs_list)

    def reset_single(self, idx: int, **kwargs) -> np.ndarray:
        """Reset a single environment."""
        return self.envs[idx].reset(**kwargs)

    def step(self, actions: List[Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments with batched ESM encoding and batched reward.

        Args:
            actions: List of (op_type, position, token) per env.

        Returns:
            (obs, rewards, dones, infos) arrays.
        """
        assert len(actions) == self.n_envs

        rewards = np.zeros(self.n_envs)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = []
        stepped_indices = []  # indices that took a real step (need reward)
        reset_indices = []    # indices that auto-reset (reward=0)

        # Phase 1: Apply actions (no reward, no obs)
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if env.done:
                env._reset_internal()
                rewards[i] = 0.0
                dones[i] = False
                infos.append({"auto_reset": True})
                reset_indices.append(i)
            else:
                done, info = env._step_action_only(action)
                dones[i] = done
                infos.append(info)
                stepped_indices.append(i)

        # Phase 1.5: Batch-compute initial affinities for auto-reset envs
        if reset_indices:
            scorer = self.envs[0].reward_manager.affinity_scorer
            if scorer is not None:
                reset_tcrs = [self.envs[i].tcr_seq for i in reset_indices]
                reset_peps = [self.envs[i].peptide for i in reset_indices]
                if hasattr(scorer, 'score_batch_fast'):
                    preds = scorer.score_batch_fast(reset_tcrs, reset_peps)
                    for j, i in enumerate(reset_indices):
                        self.envs[i].initial_affinity = float(np.nan_to_num(preds[j], nan=0.0, posinf=0.0, neginf=0.0))
                else:
                    for i in reset_indices:
                        aff, _ = scorer.score(self.envs[i].tcr_seq, self.envs[i].peptide)
                        self.envs[i].initial_affinity = float(np.nan_to_num(aff, nan=0.0, posinf=0.0, neginf=0.0))

        # Phase 2: Batch reward computation for stepped envs. In terminal-only
        # mode, intermediate transitions must stay at reward=0.
        reward_indices = [
            i for i in stepped_indices
            if (not self.envs[i].terminal_reward_only) or dones[i]
        ]
        skipped_reward_indices = [i for i in stepped_indices if i not in reward_indices]
        for i in skipped_reward_indices:
            self.envs[i]._apply_reward(0.0, {})
            infos[i]["reward_components"] = {}

        if reward_indices:
            reward_manager = self.envs[0].reward_manager
            batch_tcrs = [self.envs[i].tcr_seq for i in reward_indices]
            batch_peps = [self.envs[i].peptide for i in reward_indices]
            batch_init_aff = [self.envs[i].initial_affinity for i in reward_indices]
            batch_targets = [self.envs[i].target for i in reward_indices]

            batch_rewards, batch_components = reward_manager.compute_reward_batch(
                batch_tcrs, batch_peps, batch_init_aff, batch_targets
            )

            for j, i in enumerate(reward_indices):
                r = batch_rewards[j]
                comp = batch_components[j]

                # Shaped reward: intermediate steps get delta signal, terminal gets full score
                if reward_manager.reward_mode == "v1_ergo_shaped":
                    aff_raw = comp.get("affinity_raw", 0.0)
                    init_aff = batch_init_aff[j]
                    if dones[i]:
                        # Terminal: full affinity score
                        r = aff_raw
                    else:
                        # Intermediate: small delta signal
                        r = 0.1 * (aff_raw - init_aff)

                rewards[i] = r
                self.envs[i]._apply_reward(r, comp)
                rewards[i] = float(np.nan_to_num(rewards[i], nan=0.0, posinf=0.0, neginf=0.0))
                infos[i]["reward_components"] = comp

        # Phase 3: Batch ESM encoding for all envs
        all_indices = reset_indices + stepped_indices
        all_indices.sort()
        tcr_seqs = [self.envs[i].tcr_seq for i in all_indices]
        esm_cache = self.envs[0].esm_cache
        tcr_embs = esm_cache.encode_tcr_batch(tcr_seqs)

        obs_list = []
        for j, i in enumerate(all_indices):
            env = self.envs[i]
            remaining = (env.max_steps - env.step_count) / env.max_steps
            obs = torch.cat([
                tcr_embs[j],
                env.pmhc_emb,
                torch.tensor([remaining, env.cumulative_delta], device=tcr_embs[j].device),
            ]).cpu().numpy()
            obs_list.append(obs)

        return np.stack(obs_list), rewards, dones, infos

    def get_action_masks(self) -> List[Dict[str, np.ndarray]]:
        """Get action masks for all environments."""
        return [env.get_action_mask() for env in self.envs]

    def get_current_tcrs(self) -> List[str]:
        """Get current TCR sequences for all environments."""
        return [env.current_tcr for env in self.envs]
