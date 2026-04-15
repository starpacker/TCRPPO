"""Reward manager: combine all scorers into a single reward signal."""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class RewardManager:
    """Combine multiple scorer signals into reward.

    R_t = w1 * affinity - w2 * R_decoy - w3 * R_naturalness - w4 * R_diversity
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

    def compute_reward(
        self,
        tcr: str,
        peptide: str,
        initial_affinity: float = 0.0,
        target: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute combined reward for a TCR-peptide pair."""
        target = target or peptide
        components = {}

        # Affinity
        if self.affinity_scorer is not None and self.reward_mode != "disabled":
            if hasattr(self.affinity_scorer, 'score_batch_fast'):
                preds = self.affinity_scorer.score_batch_fast([tcr], [peptide])
                aff_score = preds[0]
            else:
                aff_score, _ = self.affinity_scorer.score(tcr, peptide)
            aff_delta = aff_score - initial_affinity if self.use_delta_reward else aff_score
            components["affinity_raw"] = aff_score
            components["affinity_delta"] = aff_delta
        else:
            aff_score = 0.0
            aff_delta = 0.0
            components["affinity_raw"] = 0.0

        # Decoy penalty — always computed (no frequency gating)
        if (self.decoy_scorer is not None
                and self.reward_mode in ("v2_full", "v2_decoy_only", "raw_decoy", "raw_multi_penalty", "threshold_penalty")):
            decoy_score, _ = self.decoy_scorer.score(tcr, peptide, target=target)
            components["decoy_raw"] = decoy_score
        else:
            decoy_score = 0.0
            components["decoy_raw"] = 0.0

        # Naturalness penalty — always computed (no frequency gating)
        if (self.naturalness_scorer is not None
                and self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty")):
            nat_score, _ = self.naturalness_scorer.score(tcr)
            components["naturalness_raw"] = nat_score
        else:
            nat_score = 0.0
            components["naturalness_raw"] = 0.0

        # Diversity penalty
        if (self.diversity_scorer is not None
                and self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty")):
            div_score, _ = self.diversity_scorer.score(tcr)
            components["diversity_raw"] = div_score
        else:
            div_score = 0.0
            components["diversity_raw"] = 0.0

        # Compute total reward — ALL modes use raw scores, NO z-norm
        if self.reward_mode == "v1_ergo_only":
            total = aff_score
        elif self.reward_mode == "v1_ergo_squared":
            total = aff_score ** 2
        elif self.reward_mode == "v1_ergo_delta":
            total = aff_delta
        elif self.reward_mode == "v1_ergo_stepwise":
            total = aff_score
        elif self.reward_mode in ("v2_decoy_only", "raw_decoy"):
            total = aff_score - self.weights["decoy"] * decoy_score
        elif self.reward_mode in ("raw_multi_penalty", "v2_full", "v2_no_decoy", "v2_no_curriculum"):
            total = (aff_score
                    - self.weights["decoy"] * decoy_score
                    - self.weights["naturalness"] * nat_score
                    - self.weights["diversity"] * div_score)
        elif self.reward_mode == "threshold_penalty":
            if aff_score < 0.5:
                total = aff_score
            else:
                total = (aff_score
                        - self.weights["decoy"] * decoy_score
                        - self.weights["naturalness"] * nat_score
                        - self.weights["diversity"] * div_score)
        else:
            total = aff_score  # Fallback: raw affinity

        components["total"] = total
        return total, components

    def compute_reward_batch(
        self,
        tcrs: List[str],
        peptides: List[str],
        initial_affinities: List[float],
        targets: Optional[List[str]] = None,
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """Compute rewards for a batch of TCR-peptide pairs."""
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

        # Process each sample — always compute all scorers (no frequency gating)
        for i in range(n):
            components = {}
            aff_score = aff_scores[i]
            aff_delta = aff_score - initial_affinities[i] if self.use_delta_reward else aff_score
            components["affinity_raw"] = aff_score
            components["affinity_delta"] = aff_delta

            # Decoy — always computed for every sample
            if (self.decoy_scorer is not None
                    and self.reward_mode in ("v2_full", "v2_decoy_only", "raw_decoy", "raw_multi_penalty", "threshold_penalty")):
                decoy_score, _ = self.decoy_scorer.score(tcrs[i], peptides[i], target=targets[i])
                components["decoy_raw"] = decoy_score
            else:
                decoy_score = 0.0
                components["decoy_raw"] = 0.0

            # Naturalness — always computed for every sample
            if (self.naturalness_scorer is not None
                    and self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty")):
                nat_score, _ = self.naturalness_scorer.score(tcrs[i])
                components["naturalness_raw"] = nat_score
            else:
                nat_score = 0.0
                components["naturalness_raw"] = 0.0

            # Diversity
            if (self.diversity_scorer is not None
                    and self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty")):
                div_score, _ = self.diversity_scorer.score(tcrs[i])
                components["diversity_raw"] = div_score
            else:
                div_score = 0.0
                components["diversity_raw"] = 0.0

            # Total reward — NO z-norm, all raw
            if self.reward_mode == "v1_ergo_only":
                total = aff_score
            elif self.reward_mode == "v1_ergo_squared":
                total = aff_score ** 2
            elif self.reward_mode == "v1_ergo_delta":
                total = aff_delta
            elif self.reward_mode == "v1_ergo_stepwise":
                total = aff_score
            elif self.reward_mode in ("v2_decoy_only", "raw_decoy"):
                total = aff_score - self.weights["decoy"] * decoy_score
            elif self.reward_mode in ("raw_multi_penalty", "v2_full", "v2_no_decoy", "v2_no_curriculum"):
                total = (aff_score
                        - self.weights["decoy"] * decoy_score
                        - self.weights["naturalness"] * nat_score
                        - self.weights["diversity"] * div_score)
            elif self.reward_mode == "threshold_penalty":
                if aff_score < 0.5:
                    total = aff_score
                else:
                    total = (aff_score
                            - self.weights["decoy"] * decoy_score
                            - self.weights["naturalness"] * nat_score
                            - self.weights["diversity"] * div_score)
            else:
                total = aff_score  # Fallback

            components["total"] = total
            all_rewards.append(total)
            all_components.append(components)

        return all_rewards, all_components
