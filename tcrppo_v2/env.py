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
            aff, _ = self.reward_manager.affinity_scorer.score(self.tcr_seq, self.peptide)
            self.initial_affinity = aff
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
        reward, components = self.reward_manager.compute_reward(
            tcr=self.tcr_seq,
            peptide=self.peptide,
            initial_affinity=self.initial_affinity,
            target=self.target,
        )
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
        if self.step_count == 0:
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
            )
            for _ in range(n_envs)
        ]
        self.obs_dim = self.envs[0].obs_dim

    def set_global_step(self, step: int) -> None:
        """Set global step for all environments."""
        for env in self.envs:
            env.set_global_step(step)

    def reset(self) -> np.ndarray:
        """Reset all environments. Returns (n_envs, obs_dim)."""
        obs = np.stack([env.reset() for env in self.envs])
        return obs

    def reset_single(self, idx: int, **kwargs) -> np.ndarray:
        """Reset a single environment."""
        return self.envs[idx].reset(**kwargs)

    def step(self, actions: List[Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments.

        Args:
            actions: List of (op_type, position, token) per env.

        Returns:
            (obs, rewards, dones, infos) arrays.
        """
        assert len(actions) == self.n_envs

        obs_list = []
        rewards = np.zeros(self.n_envs)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if env.done:
                # Auto-reset done environments
                obs = env.reset()
                obs_list.append(obs)
                rewards[i] = 0.0
                dones[i] = False
                infos.append({"auto_reset": True})
            else:
                obs, reward, done, info = env.step(action)
                obs_list.append(obs)
                rewards[i] = reward
                dones[i] = done
                infos.append(info)

        return np.stack(obs_list), rewards, dones, infos

    def get_action_masks(self) -> List[Dict[str, np.ndarray]]:
        """Get action masks for all environments."""
        return [env.get_action_mask() for env in self.envs]

    def get_current_tcrs(self) -> List[str]:
        """Get current TCR sequences for all environments."""
        return [env.current_tcr for env in self.envs]
