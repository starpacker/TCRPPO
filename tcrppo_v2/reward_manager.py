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
        n_contrast_decoys: int = 4,
        convex_alpha: float = 3.0,
        contrastive_agg: str = "mean",
        ood_threshold: float = 0.15,
        ood_penalty_weight: float = 1.0,
        ood_penalty_mode: str = "soft",
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
        self.n_contrast_decoys = n_contrast_decoys
        self.convex_alpha = convex_alpha
        self.contrastive_agg = contrastive_agg  # "mean" or "max"
        self.ood_threshold = ood_threshold
        self.ood_penalty_weight = ood_penalty_weight
        self.ood_penalty_mode = ood_penalty_mode  # "soft" or "hard"
        # OOD stats tracking
        self._ood_triggered = 0
        self._ood_total = 0

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
            # For OOD penalty mode, use score_batch to get uncertainty
            if self.reward_mode == "v1_ergo_ood_penalty":
                if hasattr(self.affinity_scorer, 'score_batch'):
                    scores, confidences = self.affinity_scorer.score_batch([tcr], [peptide])
                    aff_score = scores[0]
                    confidence = confidences[0]
                    uncertainty = 1.0 - confidence
                    components["uncertainty"] = uncertainty

                    # Apply OOD penalty
                    self._ood_total += 1
                    if self.ood_penalty_mode == "soft":
                        # Soft penalty: only penalize the excess beyond threshold
                        if uncertainty > self.ood_threshold:
                            penalty = (uncertainty - self.ood_threshold) * self.ood_penalty_weight
                            aff_score = aff_score - penalty
                            components["ood_penalty"] = penalty
                            self._ood_triggered += 1
                        else:
                            components["ood_penalty"] = 0.0
                    else:  # hard
                        # Hard penalty: penalize full uncertainty
                        if uncertainty > self.ood_threshold:
                            penalty = uncertainty * self.ood_penalty_weight
                            aff_score = aff_score - penalty
                            components["ood_penalty"] = penalty
                            self._ood_triggered += 1
                        else:
                            components["ood_penalty"] = 0.0
                else:
                    # Fallback if scorer doesn't support score_batch
                    aff_score, _ = self.affinity_scorer.score(tcr, peptide)
                    components["uncertainty"] = 0.0
                    components["ood_penalty"] = 0.0
            elif hasattr(self.affinity_scorer, 'score_batch_fast'):
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
        elif self.reward_mode == "v1_ergo_ood_penalty":
            # OOD penalty already applied to aff_score above
            total = aff_score
        elif self.reward_mode == "v1_ergo_convex":
            # Convex reward: ERGO^alpha — amplifies gradient at high scores
            # alpha=3: 0.5→0.125, 0.7→0.343, 0.8→0.512, 0.9→0.729, 0.95→0.857
            total = aff_score ** self.convex_alpha
        elif self.reward_mode == "v1_ergo_squared":
            total = aff_score ** 2
        elif self.reward_mode == "v1_ergo_delta":
            total = aff_delta
        elif self.reward_mode == "v1_ergo_stepwise":
            total = aff_score
        elif self.reward_mode == "v1_ergo_shaped":
            # Shaped reward: intermediate steps get 0.1 * delta, terminal gets full score
            # Note: is_terminal flag must be passed via kwargs
            total = aff_score  # Default to full score (will be overridden in env for intermediate)
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
        elif self.reward_mode == "contrastive_ergo":
            # Contrastive: reward = ERGO(target) - agg(ERGO(decoys))
            # agg = "mean" (original) or "max" (worst-case specificity)
            if self.decoy_scorer is not None:
                decoy_peptides = self.decoy_scorer.sample_decoys(target, k=self.n_contrast_decoys)
                if not decoy_peptides:
                    total = aff_score  # No decoys available, fallback
                else:
                    # Score TCR against decoys using ERGO
                    if hasattr(self.affinity_scorer, 'score_batch_fast'):
                        decoy_scores = self.affinity_scorer.score_batch_fast(
                            [tcr] * len(decoy_peptides), decoy_peptides
                        )
                    else:
                        decoy_scores = [self.affinity_scorer.score(tcr, d)[0] for d in decoy_peptides]
                    if self.contrastive_agg == "max":
                        agg_decoy_score = float(np.max(decoy_scores))
                    else:
                        agg_decoy_score = float(np.mean(decoy_scores))
                    # Apply convex transformation if alpha != 1
                    if self.convex_alpha != 1.0 and self.convex_alpha > 0:
                        total = aff_score ** self.convex_alpha - agg_decoy_score ** self.convex_alpha
                    else:
                        total = aff_score - agg_decoy_score
                    components["decoy_mean"] = float(np.mean(decoy_scores))
                    components["decoy_max"] = float(np.max(decoy_scores))
                    components["contrast_margin"] = total
            else:
                total = aff_score  # Fallback if no decoy_scorer
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
            # For OOD penalty mode, use score_batch to get uncertainty
            if self.reward_mode == "v1_ergo_ood_penalty":
                if hasattr(self.affinity_scorer, 'score_batch'):
                    aff_scores, confidences = self.affinity_scorer.score_batch(tcrs, peptides)
                    uncertainties = [1.0 - c for c in confidences]
                else:
                    # Fallback
                    aff_scores = []
                    uncertainties = [0.0] * n
                    for tcr, pep in zip(tcrs, peptides):
                        s, _ = self.affinity_scorer.score(tcr, pep)
                        aff_scores.append(s)
            elif hasattr(self.affinity_scorer, 'score_batch_fast'):
                aff_scores = self.affinity_scorer.score_batch_fast(tcrs, peptides)
                uncertainties = [0.0] * n
            else:
                aff_scores = []
                uncertainties = [0.0] * n
                for tcr, pep in zip(tcrs, peptides):
                    s, _ = self.affinity_scorer.score(tcr, pep)
                    aff_scores.append(s)
        else:
            aff_scores = [0.0] * n
            uncertainties = [0.0] * n

        # Process each sample — always compute all scorers (no frequency gating)
        for i in range(n):
            components = {}
            aff_score = aff_scores[i]

            # Apply OOD penalty if in OOD mode
            if self.reward_mode == "v1_ergo_ood_penalty":
                uncertainty = uncertainties[i]
                components["uncertainty"] = uncertainty
                self._ood_total += 1

                if self.ood_penalty_mode == "soft":
                    if uncertainty > self.ood_threshold:
                        penalty = (uncertainty - self.ood_threshold) * self.ood_penalty_weight
                        aff_score = aff_score - penalty
                        components["ood_penalty"] = penalty
                        self._ood_triggered += 1
                    else:
                        components["ood_penalty"] = 0.0
                else:  # hard
                    if uncertainty > self.ood_threshold:
                        penalty = uncertainty * self.ood_penalty_weight
                        aff_score = aff_score - penalty
                        components["ood_penalty"] = penalty
                        self._ood_triggered += 1
                    else:
                        components["ood_penalty"] = 0.0

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
            elif self.reward_mode == "v1_ergo_ood_penalty":
                # OOD penalty already applied to aff_score above
                total = aff_score
            elif self.reward_mode == "v1_ergo_convex":
                total = aff_score ** self.convex_alpha
            elif self.reward_mode == "v1_ergo_squared":
                total = aff_score ** 2
            elif self.reward_mode == "v1_ergo_delta":
                total = aff_delta
            elif self.reward_mode == "v1_ergo_stepwise":
                total = aff_score
            elif self.reward_mode == "v1_ergo_shaped":
                total = aff_score  # env handles shaped vs terminal split
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
            elif self.reward_mode == "contrastive_ergo":
                if self.decoy_scorer is not None:
                    decoy_peptides = self.decoy_scorer.sample_decoys(targets[i], k=self.n_contrast_decoys)
                    if not decoy_peptides:
                        total = aff_score
                        components["decoy_mean"] = 0.0
                        components["decoy_max"] = 0.0
                        components["contrast_margin"] = aff_score
                    else:
                        if hasattr(self.affinity_scorer, 'score_batch_fast'):
                            decoy_scores = self.affinity_scorer.score_batch_fast(
                                [tcrs[i]] * len(decoy_peptides), decoy_peptides
                            )
                        else:
                            decoy_scores = [self.affinity_scorer.score(tcrs[i], d)[0] for d in decoy_peptides]
                        if self.contrastive_agg == "max":
                            agg_decoy_score = float(np.max(decoy_scores))
                        else:
                            agg_decoy_score = float(np.mean(decoy_scores))
                        total = aff_score - agg_decoy_score
                        components["decoy_mean"] = float(np.mean(decoy_scores))
                        components["decoy_max"] = float(np.max(decoy_scores))
                        components["contrast_margin"] = total
                else:
                    total = aff_score
            else:
                total = aff_score  # Fallback

            components["total"] = total
            all_rewards.append(total)
            all_components.append(components)

        return all_rewards, all_components

    def get_ood_stats(self) -> Dict[str, float]:
        """Get OOD penalty statistics."""
        if self._ood_total == 0:
            return {"ood_trigger_rate": 0.0, "ood_triggered": 0, "ood_total": 0}
        return {
            "ood_trigger_rate": self._ood_triggered / self._ood_total,
            "ood_triggered": self._ood_triggered,
            "ood_total": self._ood_total,
        }

    def reset_ood_stats(self):
        """Reset OOD statistics counters."""
        self._ood_triggered = 0
        self._ood_total = 0
