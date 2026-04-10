"""Diversity penalty scorer based on recent-buffer similarity."""

from collections import deque
from typing import List, Tuple

from tcrppo_v2.scorers.base import BaseScorer
from tcrppo_v2.utils.encoding import levenshtein_similarity


class DiversityScorer(BaseScorer):
    """Penalize TCRs that are too similar to recently generated sequences."""

    def __init__(
        self,
        buffer_size: int = 512,
        similarity_threshold: float = 0.85,
    ):
        self.buffer: deque = deque(maxlen=buffer_size)
        self.threshold = similarity_threshold

    def _max_similarity(self, tcr: str) -> float:
        """Find max similarity to any sequence in the buffer."""
        if not self.buffer:
            return 0.0
        return max(levenshtein_similarity(tcr, other) for other in self.buffer)

    def add_to_buffer(self, tcr: str) -> None:
        """Add a generated TCR to the recent buffer."""
        self.buffer.append(tcr)

    def score(self, tcr: str, peptide: str = "", **kwargs) -> Tuple[float, float]:
        """Score a single TCR for diversity.

        Returns (penalty, confidence=1.0).
        Penalty is 0.0 if diverse enough, negative if too similar.
        """
        max_sim = self._max_similarity(tcr)
        if max_sim > self.threshold:
            penalty = -(max_sim - self.threshold)
        else:
            penalty = 0.0
        # Add to buffer after scoring
        self.add_to_buffer(tcr)
        return penalty, 1.0

    def score_batch(self, tcrs: list, peptides: list = None, **kwargs) -> Tuple[list, list]:
        """Score a batch of TCRs for diversity."""
        scores = []
        for tcr in tcrs:
            s, _ = self.score(tcr)
            scores.append(s)
        return scores, [1.0] * len(tcrs)

    def reset(self) -> None:
        """Clear the diversity buffer."""
        self.buffer.clear()
