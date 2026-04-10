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

        # Decoy penalty
        if (
            self.decoy_scorer is not None
            and self.reward_mode in ("v2_full",)
        ):
            decoy_score, decoy_conf = self.decoy_scorer.score(tcr, peptide, target=target)
            components["decoy_raw"] = decoy_score
            components["decoy_conf"] = decoy_conf
        else:
            decoy_score = 0.0
            components["decoy_raw"] = 0.0

        # Naturalness penalty
        if (
            self.naturalness_scorer is not None
            and self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum")
        ):
            nat_score, nat_conf = self.naturalness_scorer.score(tcr)
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
