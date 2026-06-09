"""TCR editing environment with indel actions and ESM-2 state encoding."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tcrppo_v2.utils.constants import (
    AMINO_ACIDS, NUM_AMINO_ACIDS, AA_TO_IDX, IDX_TO_AA,
    MIN_TCR_LEN, MAX_TCR_LEN, MAX_STEPS_PER_EPISODE,
    OP_SUB, OP_INS, OP_DEL, OP_STOP, NUM_OPS,
)
from tcrppo_v2.utils.encoding import is_valid_tcr
from tcrppo_v2.utils.biochem_features import compute_biochem_features


class TCREditEnv:
    """Single TCR editing environment.

    Observation: [TCR_emb | pMHC_emb], optionally plus 8D biochem features.
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
        sub_only: bool = False,
        terminal_reward_only: bool = False,
        use_biochem_features: bool = False,
        allow_stop_at_step0: bool = False,
        stop_at_step0_min_init_affinity: Optional[float] = None,
        pmhc_obs_transform: Optional[Dict[str, object]] = None,
        include_state_scalars: bool = False,
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
            sub_only: If True, only substitution edits are allowed.
            terminal_reward_only: If True, only compute reward at episode end.
            use_biochem_features: If True, append 8D biochemical features
                (charge and hydrophobicity statistics) to the observation.
            allow_stop_at_step0: If True, STOP is valid immediately after reset.
            stop_at_step0_min_init_affinity: Optional initial-affinity floor for
                immediate STOP. If set, low-init episodes must attempt editing.
            pmhc_obs_transform: Optional transform applied only to the pMHC
                observation embedding. Raw ESM cache/reward inputs stay unchanged.
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
        self.sub_only = sub_only
        self.terminal_reward_only = terminal_reward_only
        self.use_biochem_features = use_biochem_features
        # Accepted for forward-compat with newer trainers; currently a no-op
        # (obs is always [tcr_emb | pmhc_emb | optional biochem]).
        self.include_state_scalars = include_state_scalars
        self.allow_stop_at_step0 = bool(allow_stop_at_step0)
        self.stop_at_step0_min_init_affinity = (
            None
            if stop_at_step0_min_init_affinity is None
            else float(stop_at_step0_min_init_affinity)
        )
        self.pmhc_obs_transform = pmhc_obs_transform or {}

        # State dimensions
        self.esm_dim = esm_cache.output_dim  # 1280
        biochem_dim = 8 if self.use_biochem_features else 0
        # Legacy state scalars (remaining_steps_norm, cumulative_delta) — only
        # used to stay compatible with old checkpoints (trace61-era policies
        # were trained with these two extras). New experiments leave it off.
        state_scalar_dim = 2 if self.include_state_scalars else 0
        self.obs_dim = self.esm_dim * 2 + biochem_dim + state_scalar_dim

        # Episode state
        self.tcr_seq: str = ""
        self.initial_tcr: str = ""
        self.peptide: str = ""
        self.target: str = ""
        self.pmhc_emb: Optional[torch.Tensor] = None
        self.step_count: int = 0
        self.cumulative_delta: float = 0.0
        self.initial_affinity: float = 0.0
        self.initial_tcr_source: str = ""
        self.previous_affinity: float = 0.0
        self.done: bool = True
        self.global_step: int = 0  # For curriculum/decoy schedule

    def _reward_reference_affinity(self) -> float:
        """Return the affinity baseline for the next reward computation."""
        if getattr(self.reward_manager, "reward_mode", None) == "tfold_stepwise":
            return self.previous_affinity
        return self.initial_affinity

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
            self.initial_tcr_source = "provided"
        else:
            self.tcr_seq, self.initial_tcr_source = self.tcr_pool.sample_tcr(
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
        self.previous_affinity = self.initial_affinity

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
                    initial_affinity=self._reward_reference_affinity(),
                    initial_tcr=self.initial_tcr,
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
                initial_affinity=self._reward_reference_affinity(),
                initial_tcr=self.initial_tcr,
                target=self.target,
            )

        if components and "affinity_raw" in components:
            self.previous_affinity = float(components["affinity_raw"])

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
        if not seq or position <= 0:
            return self.tcr_seq
        pos = min(position, len(seq) - 1)
        aa = IDX_TO_AA.get(token, AMINO_ACIDS[token % NUM_AMINO_ACIDS])
        seq[pos] = aa
        return "".join(seq)

    def _apply_ins(self, position: int, token: int) -> str:
        """Insert amino acid at position."""
        seq = list(self.tcr_seq)
        if len(seq) >= self.max_tcr_len:
            return self.tcr_seq  # Safety: shouldn't happen with masking
        pos = min(max(position, 1), len(seq))
        aa = IDX_TO_AA.get(token, AMINO_ACIDS[token % NUM_AMINO_ACIDS])
        seq.insert(pos, aa)
        return "".join(seq)

    def _apply_del(self, position: int) -> str:
        """Delete amino acid at position."""
        seq = list(self.tcr_seq)
        if len(seq) <= self.min_tcr_len:
            return self.tcr_seq  # Safety: shouldn't happen with masking
        if position <= 0:
            return self.tcr_seq
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
        self.previous_affinity = 0.0

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
        obs = self._build_obs_from_tcr_emb(tcr_emb)
        return obs.cpu().numpy()

    def _build_obs_from_tcr_emb(self, tcr_emb: torch.Tensor) -> torch.Tensor:
        """Build observation from a precomputed TCR embedding."""
        parts = [tcr_emb, self._transform_pmhc_obs(self.pmhc_emb)]
        if self.use_biochem_features:
            biochem_feats = compute_biochem_features(self.tcr_seq)
            parts.append(torch.tensor(biochem_feats, device=tcr_emb.device, dtype=tcr_emb.dtype))
        if self.include_state_scalars:
            # Legacy 2-scalar tail: (remaining_steps_normalized, cumulative_delta).
            # Kept solely for backward compatibility with trace61-era checkpoints.
            remaining_norm = max(0.0, (self.max_steps - self.step_count) / max(1, self.max_steps))
            parts.append(torch.tensor(
                [remaining_norm, float(self.cumulative_delta)],
                device=tcr_emb.device, dtype=tcr_emb.dtype,
            ))
        return torch.cat(parts)

    def _transform_pmhc_obs(self, pmhc_emb: torch.Tensor) -> torch.Tensor:
        """Apply the configured observation-only pMHC transform."""
        if not self.pmhc_obs_transform:
            return pmhc_emb

        out = pmhc_emb
        center = self.pmhc_obs_transform.get("center")
        if center is not None:
            center = center.to(device=out.device, dtype=out.dtype)
            out = out - center

        if self.pmhc_obs_transform.get("layer_norm", False):
            out = F.layer_norm(out, out.shape)

        return out

    def get_action_mask(self) -> Dict[str, np.ndarray]:
        """Get action masks for current state.

        Returns:
            Dict with op/position/token masks, all boolean (True=valid).
        """
        seq_len = len(self.tcr_seq)

        # Op mask
        op_mask = np.ones(NUM_OPS, dtype=bool)
        if seq_len >= self.max_tcr_len:
            op_mask[OP_INS] = False
        if seq_len <= self.min_tcr_len:
            op_mask[OP_DEL] = False
        if self.sub_only:
            op_mask[OP_INS] = False
            op_mask[OP_DEL] = False
        block_step0_stop = self.step_count == 0 and not self.allow_stop_at_step0
        if (
            self.step_count == 0
            and self.stop_at_step0_min_init_affinity is not None
            and self.initial_affinity < self.stop_at_step0_min_init_affinity
        ):
            block_step0_stop = True
        if block_step0_stop or self.step_count < self.min_steps or self.ban_stop:
            op_mask[OP_STOP] = False

        # Position mask: only valid positions
        pos_mask = np.zeros(self.max_tcr_len, dtype=bool)
        # Protect the conserved leading Cys in CDR3-like sequences.
        if seq_len <= 1:
            pos_mask[:seq_len] = True
        else:
            pos_mask[1:seq_len] = True

        token_mask = np.ones((self.max_tcr_len, NUM_AMINO_ACIDS), dtype=bool)
        for pos, aa in enumerate(self.tcr_seq[:self.max_tcr_len]):
            aa_idx = AA_TO_IDX.get(aa)
            if aa_idx is not None:
                token_mask[pos, aa_idx] = False

        return {"op_mask": op_mask, "pos_mask": pos_mask, "token_mask": token_mask}

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
        max_tcr_len: int = MAX_TCR_LEN,
        min_tcr_len: int = MIN_TCR_LEN,
        reward_mode: str = "v2_full",
        min_steps: int = 0,
        min_steps_penalty: float = 0.0,
        ban_stop: bool = False,
        sub_only: bool = False,
        terminal_reward_only: bool = False,
        active_clipping: bool = False,
        use_biochem_features: bool = False,
        allow_stop_at_step0: bool = False,
        stop_at_step0_min_init_affinity: Optional[float] = None,
        pmhc_obs_transform: Optional[Dict[str, object]] = None,
        include_state_scalars: bool = False,
    ):
        """Create n_envs parallel environments sharing scorers.

        Note: ``include_state_scalars`` is accepted for forward compatibility
        with newer trainers but is currently a no-op (the per-env observation
        is always [tcr_emb | pmhc_emb | optional biochem]).
        """
        self.n_envs = n_envs
        self.active_clipping = active_clipping
        self.include_state_scalars = include_state_scalars
        self.envs = [
            TCREditEnv(
                esm_cache=esm_cache,
                pmhc_loader=pmhc_loader,
                tcr_pool=tcr_pool,
                reward_manager=reward_manager,
                decoy_sampler=decoy_sampler,
                max_steps=max_steps,
                max_tcr_len=max_tcr_len,
                min_tcr_len=min_tcr_len,
                reward_mode=reward_mode,
                min_steps=min_steps,
                min_steps_penalty=min_steps_penalty,
                ban_stop=ban_stop,
                sub_only=sub_only,
                terminal_reward_only=terminal_reward_only,
                use_biochem_features=use_biochem_features,
                allow_stop_at_step0=allow_stop_at_step0,
                stop_at_step0_min_init_affinity=stop_at_step0_min_init_affinity,
                pmhc_obs_transform=pmhc_obs_transform,
                include_state_scalars=include_state_scalars,
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
                    env.previous_affinity = env.initial_affinity
            else:
                for env in self.envs:
                    aff, _ = scorer.score(env.tcr_seq, env.peptide)
                    env.initial_affinity = float(np.nan_to_num(aff, nan=0.0, posinf=0.0, neginf=0.0))
                    env.previous_affinity = env.initial_affinity

        tcr_seqs = [env.tcr_seq for env in self.envs]
        esm_cache = self.envs[0].esm_cache
        tcr_embs = esm_cache.encode_tcr_batch(tcr_seqs)

        obs_list = []
        for i, env in enumerate(self.envs):
            obs_list.append(env._build_obs_from_tcr_emb(tcr_embs[i]).cpu().numpy())

        return np.stack(obs_list)

    def reset_done(self) -> Tuple[List[int], Optional[np.ndarray]]:
        """Reset environments that finished on the previous step.

        Returns reset indices plus fresh observations in the same order. This lets
        the trainer start the next sampled action from a real initial state instead
        of adding a reward=0 phantom transition from a terminal observation.
        """
        reset_indices = [i for i, env in enumerate(self.envs) if env.done]
        if not reset_indices:
            return [], None

        for i in reset_indices:
            self.envs[i]._reset_internal()

        # Batch-compute initial affinities for the reset envs.
        scorer = self.envs[0].reward_manager.affinity_scorer
        if scorer is not None:
            reset_tcrs = [self.envs[i].tcr_seq for i in reset_indices]
            reset_peps = [self.envs[i].peptide for i in reset_indices]
            if hasattr(scorer, 'score_batch_fast'):
                preds = scorer.score_batch_fast(reset_tcrs, reset_peps)
                for j, i in enumerate(reset_indices):
                    self.envs[i].initial_affinity = float(
                        np.nan_to_num(preds[j], nan=0.0, posinf=0.0, neginf=0.0)
                    )
                    self.envs[i].previous_affinity = self.envs[i].initial_affinity
            else:
                for i in reset_indices:
                    aff, _ = scorer.score(self.envs[i].tcr_seq, self.envs[i].peptide)
                    self.envs[i].initial_affinity = float(
                        np.nan_to_num(aff, nan=0.0, posinf=0.0, neginf=0.0)
                    )
                    self.envs[i].previous_affinity = self.envs[i].initial_affinity

        tcr_seqs = [self.envs[i].tcr_seq for i in reset_indices]
        esm_cache = self.envs[0].esm_cache
        tcr_embs = esm_cache.encode_tcr_batch(tcr_seqs)

        obs_list = []
        for j, i in enumerate(reset_indices):
            env = self.envs[i]
            obs_list.append(env._build_obs_from_tcr_emb(tcr_embs[j]).cpu().numpy())

        return reset_indices, np.stack(obs_list)

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
                        self.envs[i].previous_affinity = self.envs[i].initial_affinity
                else:
                    for i in reset_indices:
                        aff, _ = scorer.score(self.envs[i].tcr_seq, self.envs[i].peptide)
                        self.envs[i].initial_affinity = float(np.nan_to_num(aff, nan=0.0, posinf=0.0, neginf=0.0))
                        self.envs[i].previous_affinity = self.envs[i].initial_affinity

        # Phase 2: Batch reward computation for stepped envs. In terminal-only
        # mode, intermediate transitions must stay at reward=0.
        reward_indices = [
            i for i in stepped_indices
            if (
                ((not self.envs[i].terminal_reward_only) or dones[i])
                and not (self.active_clipping and self.envs[i].terminal_reward_only and dones[i])
            )
        ]
        skipped_reward_indices = [i for i in stepped_indices if i not in reward_indices]
        for i in skipped_reward_indices:
            self.envs[i]._apply_reward(0.0, {})
            infos[i]["reward_components"] = {}

        if reward_indices:
            reward_manager = self.envs[0].reward_manager
            batch_tcrs = [self.envs[i].tcr_seq for i in reward_indices]
            batch_peps = [self.envs[i].peptide for i in reward_indices]
            batch_initial_tcrs = [self.envs[i].initial_tcr for i in reward_indices]
            if reward_manager.reward_mode == "tfold_stepwise":
                batch_init_aff = [self.envs[i].previous_affinity for i in reward_indices]
            else:
                batch_init_aff = [self.envs[i].initial_affinity for i in reward_indices]
            batch_targets = [self.envs[i].target for i in reward_indices]

            batch_rewards, batch_components = reward_manager.compute_reward_batch(
                batch_tcrs, batch_peps, batch_init_aff, batch_initial_tcrs, batch_targets
            )

            for j, i in enumerate(reward_indices):
                r = batch_rewards[j]
                comp = batch_components[j]

                # Shaped reward: intermediate steps get delta signal, terminal gets full score
                if reward_manager.reward_mode == "v1_ergo_shaped":
                    aff_raw = comp.get("affinity_raw", 0.0)
                    init_aff = batch_init_aff[j]
                    aff_weight = reward_manager.weights.get("affinity", 1.0)
                    if dones[i]:
                        # Terminal: full affinity score
                        r = aff_weight * aff_raw
                    else:
                        # Intermediate: small delta signal
                        r = 0.1 * aff_weight * (aff_raw - init_aff)

                rewards[i] = r
                self.envs[i]._apply_reward(r, comp)
                if comp and "affinity_raw" in comp:
                    self.envs[i].previous_affinity = float(comp["affinity_raw"])
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
            obs_list.append(env._build_obs_from_tcr_emb(tcr_embs[j]).cpu().numpy())

        return np.stack(obs_list), rewards, dones, infos

    def get_action_masks(self) -> List[Dict[str, np.ndarray]]:
        """Get action masks for all environments."""
        return [env.get_action_mask() for env in self.envs]

    def get_current_tcrs(self) -> List[str]:
        """Get current TCR sequences for all environments."""
        return [env.current_tcr for env in self.envs]
