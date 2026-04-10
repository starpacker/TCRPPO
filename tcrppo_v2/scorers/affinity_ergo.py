"""ERGO binding affinity scorer with MC Dropout uncertainty."""

import sys
import os
from typing import Tuple, List
import copy

import numpy as np
import torch
import torch.nn as nn

from tcrppo_v2.scorers.base import BaseScorer
from tcrppo_v2.utils.constants import ERGO_DIR, ERGO_AE_FILE, ERGO_TCR_ATOX, ERGO_PEP_ATOX, ERGO_MAX_LEN

# Add ERGO to path for its internal imports
if ERGO_DIR not in sys.path:
    sys.path.insert(0, ERGO_DIR)

from ERGO_models import AutoencoderLSTMClassifier
import ae_utils as ae


class AffinityERGOScorer(BaseScorer):
    """ERGO AE-LSTM binding predictor with MC Dropout confidence."""

    def __init__(
        self,
        model_file: str,
        ae_file: str = ERGO_AE_FILE,
        device: str = "cuda",
        mc_samples: int = 10,
    ):
        self.device = device
        self.mc_samples = mc_samples
        self.model = self._load_model(model_file, ae_file, device)

    def _load_model(
        self, model_file: str, ae_file: str, device: str
    ) -> AutoencoderLSTMClassifier:
        model = AutoencoderLSTMClassifier(10, device, ERGO_MAX_LEN, 21, 100, 1, ae_file, False)
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    def _get_predictions(self, tcrs: List[str], peps: List[str]) -> List[float]:
        """Get ERGO binding predictions for a batch."""
        tcrs_copy = copy.deepcopy(tcrs)
        peps_copy = copy.deepcopy(peps)
        signs = [0] * len(tcrs_copy)
        batch_size = min(len(tcrs_copy), 4096) if len(tcrs_copy) > 0 else 1
        batches = ae.get_full_batches(
            tcrs_copy, peps_copy, signs, ERGO_TCR_ATOX, ERGO_PEP_ATOX, batch_size, ERGO_MAX_LEN
        )
        preds = ae.predict(self.model, batches, self.device)
        return preds[: len(tcrs)]

    def _build_gpu_batches(self, tcrs: List[str], peps: List[str]):
        """Build batches and push to GPU once for MC Dropout reuse."""
        tcrs_copy = copy.deepcopy(tcrs)
        peps_copy = copy.deepcopy(peps)
        signs = [0] * len(tcrs_copy)
        batch_size = min(len(tcrs_copy), 4096) if len(tcrs_copy) > 0 else 1
        batches = ae.get_full_batches(
            tcrs_copy, peps_copy, signs, ERGO_TCR_ATOX, ERGO_PEP_ATOX, batch_size, ERGO_MAX_LEN
        )
        gpu_batches = []
        for batch in batches:
            t, p, l, s = batch
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            if not isinstance(p, torch.Tensor):
                p = torch.tensor(p)
            if not isinstance(l, torch.Tensor):
                l = torch.tensor(l)
            gpu_batches.append((
                t.to(self.device, non_blocking=True),
                p.to(self.device, non_blocking=True),
                l.to(self.device, non_blocking=True),
                s,
            ))
        return gpu_batches

    def _predict_mc(self, gpu_batches, expected_n: int) -> List[float]:
        """Single MC Dropout forward pass without resetting eval mode."""
        all_probs = []
        for batch in gpu_batches:
            tcrs, padded_peps, pep_lens, _signs = batch
            with torch.no_grad():
                probs = self.model(tcrs, padded_peps, pep_lens)
            all_probs.append(probs.detach().squeeze(-1))
        preds = torch.cat(all_probs).cpu().numpy().tolist()
        return preds[:expected_n]

    def _enable_dropout(self) -> int:
        """Enable dropout layers for MC sampling."""
        n = 0
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
                n += 1
        return n

    def _disable_dropout(self):
        """Restore dropout layers to eval mode."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

    def mc_dropout_score(
        self, tcrs: List[str], peps: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run MC Dropout for uncertainty estimation.

        Returns (means, stds) arrays of shape (len(tcrs),).
        """
        n_drop = self._enable_dropout()
        if n_drop == 0:
            preds = np.array(self._get_predictions(tcrs, peps), dtype=np.float64)
            return preds, np.zeros_like(preds)

        expected_n = len(tcrs)
        gpu_batches = self._build_gpu_batches(tcrs, peps)
        samples = []
        try:
            for _ in range(self.mc_samples):
                preds = self._predict_mc(gpu_batches, expected_n)
                samples.append(np.array(preds, dtype=np.float64))
        finally:
            self._disable_dropout()

        stacked = np.stack(samples, axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score a single TCR-peptide pair with MC Dropout."""
        means, stds = self.mc_dropout_score([tcr], [peptide])
        confidence = 1.0 - float(stds[0])
        return float(means[0]), max(0.0, min(1.0, confidence))

    def score_batch(self, tcrs: list, peptides: list, **kwargs) -> Tuple[list, list]:
        """Score a batch of TCR-peptide pairs with MC Dropout."""
        means, stds = self.mc_dropout_score(tcrs, peptides)
        confidences = np.clip(1.0 - stds, 0.0, 1.0)
        return means.tolist(), confidences.tolist()

    def score_batch_fast(self, tcrs: list, peptides: list) -> List[float]:
        """Fast scoring without MC Dropout (single forward pass)."""
        return self._get_predictions(tcrs, peptides)
