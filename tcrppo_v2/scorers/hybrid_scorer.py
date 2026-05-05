"""Hybrid scorer: Mix ERGO (fast) and tFold (accurate) with configurable ratio."""

import numpy as np
from typing import Tuple, List
from .base import BaseScorer


class HybridScorer(BaseScorer):
    """
    Hybrid scorer that mixes two scorers with a configurable ratio.

    Use case: 90% ERGO (fast, ~10ms) + 10% tFold (accurate, ~1s)
    This provides a balance between speed and accuracy.

    Args:
        primary_scorer: Fast scorer (e.g., ERGO)
        secondary_scorer: Accurate but slow scorer (e.g., tFold)
        secondary_ratio: Probability of using secondary scorer (default: 0.1)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        primary_scorer: BaseScorer,
        secondary_scorer: BaseScorer,
        secondary_ratio: float = 0.1,
        seed: int = 42
    ):
        self.primary = primary_scorer
        self.secondary = secondary_scorer
        self.secondary_ratio = secondary_ratio
        self.rng = np.random.RandomState(seed)

        # Statistics
        self.primary_calls = 0
        self.secondary_calls = 0
        self.total_calls = 0

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score a single TCR-peptide pair using hybrid strategy."""
        self.total_calls += 1

        # Randomly choose scorer based on ratio
        if self.rng.rand() < self.secondary_ratio:
            self.secondary_calls += 1
            score, conf = self.secondary.score(tcr, peptide, **kwargs)
            return score, conf
        else:
            self.primary_calls += 1
            score, conf = self.primary.score(tcr, peptide, **kwargs)
            return score, conf

    def score_batch(self, tcrs: List[str], peptides: List[str], **kwargs) -> Tuple[List[float], List[float]]:
        """Score a batch of TCR-peptide pairs using hybrid strategy."""
        n = len(tcrs)
        scores = []
        confidences = []

        # Randomly assign each sample to primary or secondary
        use_secondary = self.rng.rand(n) < self.secondary_ratio

        # Collect indices for each scorer
        primary_indices = [i for i in range(n) if not use_secondary[i]]
        secondary_indices = [i for i in range(n) if use_secondary[i]]

        # Score with primary scorer
        if primary_indices:
            primary_tcrs = [tcrs[i] for i in primary_indices]
            primary_peptides = [peptides[i] for i in primary_indices]
            primary_scores, primary_confs = self.primary.score_batch(
                primary_tcrs, primary_peptides, **kwargs
            )
            self.primary_calls += len(primary_indices)
        else:
            primary_scores, primary_confs = [], []

        # Score with secondary scorer
        if secondary_indices:
            secondary_tcrs = [tcrs[i] for i in secondary_indices]
            secondary_peptides = [peptides[i] for i in secondary_indices]
            secondary_scores, secondary_confs = self.secondary.score_batch(
                secondary_tcrs, secondary_peptides, **kwargs
            )
            self.secondary_calls += len(secondary_indices)
        else:
            secondary_scores, secondary_confs = [], []

        # Merge results in original order
        primary_iter = iter(zip(primary_scores, primary_confs))
        secondary_iter = iter(zip(secondary_scores, secondary_confs))

        for i in range(n):
            if use_secondary[i]:
                s, c = next(secondary_iter)
            else:
                s, c = next(primary_iter)
            scores.append(s)
            confidences.append(c)

        self.total_calls += n
        return scores, confidences

    def get_stats(self) -> dict:
        """Get statistics about scorer usage."""
        if self.total_calls == 0:
            return {
                'primary_calls': 0,
                'secondary_calls': 0,
                'total_calls': 0,
                'secondary_ratio_actual': 0.0,
                'secondary_ratio_target': self.secondary_ratio
            }

        return {
            'primary_calls': self.primary_calls,
            'secondary_calls': self.secondary_calls,
            'total_calls': self.total_calls,
            'secondary_ratio_actual': self.secondary_calls / self.total_calls,
            'secondary_ratio_target': self.secondary_ratio
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.primary_calls = 0
        self.secondary_calls = 0
        self.total_calls = 0
