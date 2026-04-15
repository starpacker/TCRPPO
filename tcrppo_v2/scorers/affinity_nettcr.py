"""NetTCR-2.0 binding affinity scorer conforming to BaseScorer interface.

Wraps the existing NetTCR evaluation scorer for use as a training reward signal.
This breaks the ERGO train-eval coupling by using an independent binding predictor.
"""

import os
import sys
from typing import Tuple, List

import numpy as np

from tcrppo_v2.scorers.base import BaseScorer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AffinityNetTCRScorer(BaseScorer):
    """NetTCR-2.0 CNN binding predictor as training reward scorer."""

    def __init__(
        self,
        model_path: str = None,
        device: str = "cpu",
        batch_size: int = 256,
    ):
        # Suppress TF noise and force CPU to avoid GPU conflicts with PyTorch
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        # Hide GPUs from TF so it doesn't conflict with PyTorch
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass

        # Lazy import to avoid TF loading at module level
        from tcrppo_v2.evaluation.nettcr_scorer import NetTCRScorer

        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "data", "nettcr_model.weights.h5")

        self._scorer = NetTCRScorer(model_path=model_path, batch_size=batch_size)
        self._device = device  # NetTCR uses TF, not PyTorch — device is informational
        print(f"  NetTCR scorer loaded (weights: {model_path})")

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score a single TCR-peptide pair. Returns (score, confidence=1.0)."""
        s = self._scorer.score(tcr, peptide)
        return float(s), 1.0  # NetTCR has no built-in uncertainty estimation

    def score_batch(self, tcrs: list, peptides: list, **kwargs) -> Tuple[list, list]:
        """Score a batch of TCR-peptide pairs."""
        scores = self._scorer.score_batch(tcrs, peptides)
        confidences = [1.0] * len(scores)
        return scores.tolist() if isinstance(scores, np.ndarray) else list(scores), confidences

    def score_batch_fast(self, tcrs: list, peptides: list) -> List[float]:
        """Fast batch scoring for training (no uncertainty)."""
        scores = self._scorer.score_batch(tcrs, peptides)
        return scores.tolist() if isinstance(scores, np.ndarray) else list(scores)
