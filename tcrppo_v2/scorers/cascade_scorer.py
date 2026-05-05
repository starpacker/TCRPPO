"""Cascade scorer: ERGO pre-filter + tFold verify."""

from typing import Tuple, List
from .base import BaseScorer


class CascadeScorer(BaseScorer):
    """
    Cascade scorer that uses ERGO as fast pre-filter and tFold for verification.

    Strategy:
    - If ERGO score < threshold: return ERGO score (fast path)
    - If ERGO score >= threshold: call tFold and return weighted combination

    This is adaptive: fast early (most TCRs score low), accurate late (more high-scoring TCRs).

    Args:
        primary_scorer: Fast scorer (ERGO)
        secondary_scorer: Accurate scorer (tFold)
        threshold: ERGO score above which to call tFold (default: 0.3)
        tfold_weight: Weight for tFold in combination (default: 0.7)
        ergo_weight: Weight for ERGO in combination (default: 0.3)
    """

    def __init__(
        self,
        primary_scorer: BaseScorer,
        secondary_scorer: BaseScorer,
        threshold: float = 0.3,
        tfold_weight: float = 0.7,
        ergo_weight: float = 0.3,
    ):
        self.primary = primary_scorer
        self.secondary = secondary_scorer
        self.threshold = threshold
        self.tfold_weight = tfold_weight
        self.ergo_weight = ergo_weight

        # Statistics
        self.primary_only_calls = 0
        self.cascade_calls = 0
        self.total_calls = 0

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score with cascade strategy."""
        self.total_calls += 1

        # Stage 1: ERGO pre-filter
        ergo_score, ergo_conf = self.primary.score(tcr, peptide, **kwargs)

        # Stage 2: tFold verify if above threshold
        if ergo_score >= self.threshold:
            self.cascade_calls += 1
            tfold_score, tfold_conf = self.secondary.score(tcr, peptide, **kwargs)
            # Weighted combination
            final_score = self.tfold_weight * tfold_score + self.ergo_weight * ergo_score
            final_conf = self.tfold_weight * tfold_conf + self.ergo_weight * ergo_conf
            return final_score, final_conf
        else:
            self.primary_only_calls += 1
            return ergo_score, ergo_conf

    def score_batch(self, tcrs: List[str], peptides: List[str], **kwargs) -> Tuple[List[float], List[float]]:
        """Score batch with cascade strategy."""
        n = len(tcrs)
        scores = []
        confidences = []

        # Stage 1: Score all with ERGO
        ergo_scores, ergo_confs = self.primary.score_batch(tcrs, peptides, **kwargs)

        # Stage 2: Identify high-scoring samples for tFold
        above_threshold = [i for i in range(n) if ergo_scores[i] >= self.threshold]

        if above_threshold:
            # Score high-scoring samples with tFold
            tfold_tcrs = [tcrs[i] for i in above_threshold]
            tfold_peptides = [peptides[i] for i in above_threshold]
            tfold_scores, tfold_confs = self.secondary.score_batch(tfold_tcrs, tfold_peptides, **kwargs)
            tfold_dict = {i: (tfold_scores[j], tfold_confs[j]) for j, i in enumerate(above_threshold)}
        else:
            tfold_dict = {}

        # Combine results
        for i in range(n):
            if i in tfold_dict:
                tfold_score, tfold_conf = tfold_dict[i]
                final_score = self.tfold_weight * tfold_score + self.ergo_weight * ergo_scores[i]
                final_conf = self.tfold_weight * tfold_conf + self.ergo_weight * ergo_confs[i]
                scores.append(final_score)
                confidences.append(final_conf)
                self.cascade_calls += 1
            else:
                scores.append(ergo_scores[i])
                confidences.append(ergo_confs[i])
                self.primary_only_calls += 1

        self.total_calls += n
        return scores, confidences

    def get_stats(self) -> dict:
        """Get cascade statistics."""
        if self.total_calls == 0:
            return {
                'primary_only_calls': 0,
                'cascade_calls': 0,
                'total_calls': 0,
                'cascade_ratio': 0.0,
                'threshold': self.threshold
            }

        return {
            'primary_only_calls': self.primary_only_calls,
            'cascade_calls': self.cascade_calls,
            'total_calls': self.total_calls,
            'cascade_ratio': self.cascade_calls / self.total_calls,
            'threshold': self.threshold
        }

    def score_batch_fast(self, tcrs: List[str], peptides: List[str]) -> List[float]:
        """Fast batch scoring without confidence (for contrastive reward).

        Uses primary scorer's fast path for all samples, then selectively calls
        secondary scorer for above-threshold samples.
        """
        n = len(tcrs)

        # Stage 1: Fast scoring with primary (ERGO)
        if hasattr(self.primary, 'score_batch_fast'):
            ergo_scores = self.primary.score_batch_fast(tcrs, peptides)
        else:
            ergo_scores, _ = self.primary.score_batch(tcrs, peptides)

        # Stage 2: Identify high-scoring samples for secondary (tFold)
        above_threshold = [i for i in range(n) if ergo_scores[i] >= self.threshold]

        if above_threshold:
            # Score high-scoring samples with secondary
            tfold_tcrs = [tcrs[i] for i in above_threshold]
            tfold_peptides = [peptides[i] for i in above_threshold]

            if hasattr(self.secondary, 'score_batch_fast'):
                tfold_scores = self.secondary.score_batch_fast(tfold_tcrs, tfold_peptides)
            else:
                tfold_scores, _ = self.secondary.score_batch(tfold_tcrs, tfold_peptides)

            tfold_dict = {i: tfold_scores[j] for j, i in enumerate(above_threshold)}
        else:
            tfold_dict = {}

        # Combine results
        final_scores = []
        for i in range(n):
            if i in tfold_dict:
                final_score = self.tfold_weight * tfold_dict[i] + self.ergo_weight * ergo_scores[i]
                final_scores.append(final_score)
                self.cascade_calls += 1
            else:
                final_scores.append(ergo_scores[i])
                self.primary_only_calls += 1

        self.total_calls += n
        return final_scores

    def reset_stats(self):
        """Reset statistics."""
        self.primary_only_calls = 0
        self.cascade_calls = 0
        self.total_calls = 0
