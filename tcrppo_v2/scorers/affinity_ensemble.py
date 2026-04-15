"""Ensemble affinity scorer that averages multiple binding predictors.

Combines ERGO + NetTCR (or any BaseScorer-compatible scorers) to produce
a more robust training signal that's harder for the policy to exploit.
"""

from typing import Tuple, List

import numpy as np

from tcrppo_v2.scorers.base import BaseScorer


class EnsembleAffinityScorer(BaseScorer):
    """Weighted average of multiple affinity scorers."""

    def __init__(self, scorers: List[BaseScorer], weights: List[float] = None):
        if not scorers:
            raise ValueError("Need at least one scorer")
        self._scorers = scorers
        if weights is None:
            weights = [1.0 / len(scorers)] * len(scorers)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        self._weights = weights
        names = [type(s).__name__ for s in scorers]
        print(f"  Ensemble scorer: {names}, weights={[f'{w:.2f}' for w in self._weights]}")

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Weighted average score from all scorers."""
        total_score = 0.0
        total_conf = 0.0
        for scorer, w in zip(self._scorers, self._weights):
            s, c = scorer.score(tcr, peptide, **kwargs)
            total_score += w * s
            total_conf += w * c
        return total_score, total_conf

    def score_batch(self, tcrs: list, peptides: list, **kwargs) -> Tuple[list, list]:
        """Weighted average batch scores."""
        n = len(tcrs)
        total_scores = np.zeros(n, dtype=np.float64)
        total_confs = np.zeros(n, dtype=np.float64)
        for scorer, w in zip(self._scorers, self._weights):
            scores, confs = scorer.score_batch(tcrs, peptides, **kwargs)
            total_scores += w * np.array(scores, dtype=np.float64)
            total_confs += w * np.array(confs, dtype=np.float64)
        return total_scores.tolist(), total_confs.tolist()

    def score_batch_fast(self, tcrs: list, peptides: list) -> List[float]:
        """Weighted average fast batch scores for training."""
        n = len(tcrs)
        total_scores = np.zeros(n, dtype=np.float64)
        for scorer, w in zip(self._scorers, self._weights):
            scores = scorer.score_batch_fast(tcrs, peptides)
            total_scores += w * np.array(scores, dtype=np.float64)
        return total_scores.tolist()
