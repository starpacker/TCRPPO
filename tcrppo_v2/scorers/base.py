"""Abstract base class for all reward scorers."""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseScorer(ABC):
    """All scorers return (score, confidence) where confidence in [0, 1]."""

    @abstractmethod
    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score a TCR-peptide pair.

        Returns:
            (score, confidence) where confidence in [0, 1].
        """
        pass

    @abstractmethod
    def score_batch(self, tcrs: list, peptides: list, **kwargs) -> Tuple[list, list]:
        """Score a batch of TCR-peptide pairs.

        Returns:
            (scores, confidences) - two lists of floats.
        """
        pass
