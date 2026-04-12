"""Reward manager: combine all scorers with running normalization."""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class RunningNormalizer:
    """Running mean/std normalizer with warmup period."""

    def __init__(self, window: int = 10000, warmup: int = 1000):
        self.buffer: deque = deque(maxlen=window)
        self.warmup = warmup

    def normalize(self, value: float) -> float:
        """Normalize value using running statistics."""
        self.buffer.append(value)
        if len(self.buffer) < self.warmup:
            return value
        mean = np.mean(self.buffer)
        std = np.std(self.buffer) + 1e-8
        return (value - mean) / std

    @property
    def is_warmed_up(self) -> bool:
        return len(self.buffer) >= self.warmup


class RewardManager:
    """Combine multiple scorer signals with z-score normalization.

    R_t = w1 * delta_affinity - w2 * R_decoy - w3 * R_naturalness - w4 * R_diversity
    """

    def __init__(
        self,
        affinity_scorer=None,
        decoy_scorer=None,
        naturalness_scorer=None,
        diversity_scorer=None,
        w_affinity: float = 1.0,
        w_decoy: float = 0.8,
        w_naturalness: float = 0.5,
        w_diversity: float = 0.2,
        norm_window: int = 10000,
        norm_warmup: int = 1000,
        use_delta_reward: bool = True,
        reward_mode: str = "v2_full",
        naturalness_eval_freq: int = 4,
        decoy_eval_freq: int = 2,
    ):
        self.affinity_scorer = affinity_scorer
        self.decoy_scorer = decoy_scorer
        self.naturalness_scorer = naturalness_scorer
        self.diversity_scorer = diversity_scorer

        self.weights = {
            "affinity": w_affinity,
            "decoy": w_decoy,
            "naturalness": w_naturalness,
            "diversity": w_diversity,
        }
        self.use_delta_reward = use_delta_reward
        self.reward_mode = reward_mode
        self.naturalness_eval_freq = naturalness_eval_freq
        self.decoy_eval_freq = decoy_eval_freq
        self._call_count = 0
        self._last_nat_score = 0.0
        self._last_decoy_score = 0.0

        # Per-component normalizers
        self.normalizers = {
            "affinity": RunningNormalizer(norm_window, norm_warmup),
            "decoy": RunningNormalizer(norm_window, norm_warmup),
            "naturalness": RunningNormalizer(norm_window, norm_warmup),
            "diversity": RunningNormalizer(norm_window, norm_warmup),
        }

    def compute_reward(
        self,
        tcr: str,
        peptide: str,
        initial_affinity: float = 0.0,
        target: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute combined reward for a TCR-peptide pair.

        Args:
            tcr: Current TCR CDR3beta sequence
            peptide: Target peptide
            initial_affinity: Affinity score at episode start (for delta reward)
            target: Target identifier for decoy sampling

        Returns:
            (total_reward, component_dict) where component_dict has raw scores.
        """
        target = target or peptide
        components = {}

        # Affinity
        if self.affinity_scorer is not None and self.reward_mode != "disabled":
            # Use fast scoring (no MC Dropout) for training speed
            if hasattr(self.affinity_scorer, 'score_batch_fast'):
                preds = self.affinity_scorer.score_batch_fast([tcr], [peptide])
                aff_score = preds[0]
                aff_conf = 1.0
            else:
                aff_score, aff_conf = self.affinity_scorer.score(tcr, peptide)
            if self.use_delta_reward:
                aff_delta = aff_score - initial_affinity
            else:
                aff_delta = aff_score
            components["affinity_raw"] = aff_score
            components["affinity_conf"] = aff_conf
            components["affinity_delta"] = aff_delta
        else:
            aff_delta = 0.0
            components["affinity_raw"] = 0.0

        # Decoy penalty (computed every N calls to reduce ERGO overhead)
        if (
            self.decoy_scorer is not None
            and self.reward_mode in ("v2_full", "v2_decoy_only")
        ):
            if self._call_count % self.decoy_eval_freq == 0:
                decoy_score, decoy_conf = self.decoy_scorer.score(tcr, peptide, target=target)
                self._last_decoy_score = decoy_score
            else:
                decoy_score = self._last_decoy_score
            components["decoy_raw"] = decoy_score
        else:
            decoy_score = 0.0
            components["decoy_raw"] = 0.0

        # Naturalness penalty (computed every N calls to save ESM forward passes)
        if (
            self.naturalness_scorer is not None
            and self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum")
        ):
            self._call_count += 1
            if self._call_count % self.naturalness_eval_freq == 0:
                nat_score, nat_conf = self.naturalness_scorer.score(tcr)
                self._last_nat_score = nat_score
            else:
                nat_score = self._last_nat_score
            components["naturalness_raw"] = nat_score
        else:
            nat_score = 0.0
            components["naturalness_raw"] = 0.0

        # Diversity penalty
        if (
            self.diversity_scorer is not None
            and self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum")
        ):
            div_score, div_conf = self.diversity_scorer.score(tcr)
            components["diversity_raw"] = div_score
        else:
            div_score = 0.0
            components["diversity_raw"] = 0.0

        # Normalize
        norm_aff = self.normalizers["affinity"].normalize(aff_delta)
        norm_decoy = self.normalizers["decoy"].normalize(decoy_score)
        norm_nat = self.normalizers["naturalness"].normalize(nat_score)
        norm_div = self.normalizers["diversity"].normalize(div_score)

        # v1_ergo_only: only affinity, terminal reward (not delta)
        if self.reward_mode == "v1_ergo_only":
            total = components.get("affinity_raw", 0.0)
        elif self.reward_mode == "v1_ergo_delta":
            total = aff_delta
        elif self.reward_mode == "v2_decoy_only":
            total = (
                self.weights["affinity"] * norm_aff
                - self.weights["decoy"] * norm_decoy
            )
        else:
            total = (
                self.weights["affinity"] * norm_aff
                - self.weights["decoy"] * norm_decoy
                - self.weights["naturalness"] * norm_nat
                - self.weights["diversity"] * norm_div
            )

        components["total"] = total
        components["norm_affinity"] = norm_aff
        components["norm_decoy"] = norm_decoy
        components["norm_naturalness"] = norm_nat
        components["norm_diversity"] = norm_div

        return total, components

    def compute_reward_batch(
        self,
        tcrs: List[str],
        peptides: List[str],
        initial_affinities: List[float],
        targets: Optional[List[str]] = None,
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """Compute rewards for a batch of TCR-peptide pairs.

        Batches ERGO inference across all envs for efficiency.
        """
        n = len(tcrs)
        targets = targets or peptides

        all_rewards = []
        all_components = []

        # Batch affinity scoring
        if self.affinity_scorer is not None and self.reward_mode != "disabled":
            if hasattr(self.affinity_scorer, 'score_batch_fast'):
                aff_scores = self.affinity_scorer.score_batch_fast(tcrs, peptides)
            else:
                aff_scores = []
                for tcr, pep in zip(tcrs, peptides):
                    s, _ = self.affinity_scorer.score(tcr, pep)
                    aff_scores.append(s)
        else:
            aff_scores = [0.0] * n

        # Process each sample (decoy/naturalness/diversity are per-sample)
        for i in range(n):
            components = {}
            aff_score = aff_scores[i]

            if self.use_delta_reward:
                aff_delta = aff_score - initial_affinities[i]
            else:
                aff_delta = aff_score
            components["affinity_raw"] = aff_score
            components["affinity_delta"] = aff_delta

            # Decoy
            if self.decoy_scorer is not None and self.reward_mode in ("v2_full", "v2_decoy_only"):
                if self._call_count % self.decoy_eval_freq == 0:
                    decoy_score, _ = self.decoy_scorer.score(tcrs[i], peptides[i], target=targets[i])
                    self._last_decoy_score = decoy_score
                else:
                    decoy_score = self._last_decoy_score
                components["decoy_raw"] = decoy_score
            else:
                decoy_score = 0.0
                components["decoy_raw"] = 0.0

            # Naturalness
            if (self.naturalness_scorer is not None
                    and self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum")):
                self._call_count += 1
                if self._call_count % self.naturalness_eval_freq == 0:
                    nat_score, _ = self.naturalness_scorer.score(tcrs[i])
                    self._last_nat_score = nat_score
                else:
                    nat_score = self._last_nat_score
                components["naturalness_raw"] = nat_score
            else:
                nat_score = 0.0
                components["naturalness_raw"] = 0.0

            # Diversity
            if (self.diversity_scorer is not None
                    and self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum")):
                div_score, _ = self.diversity_scorer.score(tcrs[i])
                components["diversity_raw"] = div_score
            else:
                div_score = 0.0
                components["diversity_raw"] = 0.0

            # Normalize
            norm_aff = self.normalizers["affinity"].normalize(aff_delta)
            norm_decoy = self.normalizers["decoy"].normalize(decoy_score)
            norm_nat = self.normalizers["naturalness"].normalize(nat_score)
            norm_div = self.normalizers["diversity"].normalize(div_score)

            if self.reward_mode == "v1_ergo_only":
                total = aff_score
            elif self.reward_mode == "v1_ergo_delta":
                total = aff_delta
            elif self.reward_mode == "v2_decoy_only":
                total = (
                    self.weights["affinity"] * norm_aff
                    - self.weights["decoy"] * norm_decoy
                )
            else:
                total = (
                    self.weights["affinity"] * norm_aff
                    - self.weights["decoy"] * norm_decoy
                    - self.weights["naturalness"] * norm_nat
                    - self.weights["diversity"] * norm_div
                )

            components["total"] = total
            all_rewards.append(total)
            all_components.append(components)

        return all_rewards, all_components
