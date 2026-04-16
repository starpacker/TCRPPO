"""Fast binding scorer based on the sequence-only TCRBindingModel.

This scorer wraps the trained CDR3β×Peptide classifier (BiLSTM + CrossAttention)
for use as an RL training reward. It provides ERGO-compatible score_batch_fast()
interface returning binding probabilities in [0, 1].

Speed: ~0.5ms/sample on GPU (vs ERGO ~5ms, tFold ~8000ms)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch

from tcrppo_v2.scorers.base import BaseScorer
from tcrppo_v2.scorers.tcr_binding_model import (
    AA_VOCAB,
    build_model,
    encode_sequence,
)

logger = logging.getLogger(__name__)


class AffinityTCBindScorer(BaseScorer):
    """Fast sequence-only TCR-peptide binding scorer.

    Uses a BiLSTM + CrossAttention model trained on the tc-hard dataset
    (566K samples, 640 epitopes, hard negatives). Much faster than tFold
    while providing a diverse training signal from a different data source
    than ERGO.
    """

    def __init__(
        self,
        weights_path: str = "runs/binding_classifier_v1/best_model.pt",
        device: str = "cuda",
        max_cdr3_len: int = 30,
        max_pep_len: int = 25,
    ):
        """Initialize the scorer.

        Args:
            weights_path: Path to the trained model weights (.pt file).
            device: Device to run inference on.
            max_cdr3_len: Maximum CDR3β length for padding.
            max_pep_len: Maximum peptide length for padding.
        """
        self.device = device
        self.max_cdr3_len = max_cdr3_len
        self.max_pep_len = max_pep_len

        # Load model
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        model_config = checkpoint.get("model_config", {})

        self.model = build_model(model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        # Freeze
        for p in self.model.parameters():
            p.requires_grad = False

        val_auc = checkpoint.get("val_mean_epi_auc", "?")
        logger.info(
            f"AffinityTCBindScorer loaded: {self.model.n_params:,} params, "
            f"val_mean_epi_auc={val_auc}, device={device}"
        )

    def score(
        self,
        tcr: str,
        peptide: str,
        **kwargs,
    ) -> Tuple[float, float]:
        """Score a single TCR-peptide pair.

        Returns:
            (binding_prob, uncertainty) where uncertainty is 0.0
            (no MC dropout in this model).
        """
        scores = self.score_batch_fast([tcr], [peptide])
        return scores[0], 0.0

    def score_batch(
        self,
        tcrs: List[str],
        peptides: List[str],
        **kwargs,
    ) -> Tuple[List[float], List[float]]:
        """Score a batch with uncertainty estimates.

        Returns:
            (scores_list, uncertainties_list)
        """
        scores = self.score_batch_fast(tcrs, peptides)
        uncertainties = [0.0] * len(scores)
        return scores, uncertainties

    @torch.no_grad()
    def score_batch_fast(
        self,
        tcrs: List[str],
        peptides: List[str],
    ) -> List[float]:
        """Fast batch scoring. Returns binding probabilities in [0, 1].

        Args:
            tcrs: List of CDR3β sequences.
            peptides: List of peptide sequences (same length as tcrs).

        Returns:
            List of binding probabilities in [0, 1].
        """
        if not tcrs:
            return []

        # Encode sequences
        cdr3b_ids = torch.stack([
            encode_sequence(t, self.max_cdr3_len) for t in tcrs
        ]).to(self.device)
        pep_ids = torch.stack([
            encode_sequence(p, self.max_pep_len) for p in peptides
        ]).to(self.device)

        # Forward pass
        probs = self.model.predict_proba(cdr3b_ids, pep_ids)
        return probs.cpu().tolist()
