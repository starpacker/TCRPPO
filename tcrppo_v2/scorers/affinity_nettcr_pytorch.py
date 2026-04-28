"""NetTCR-2.0 binding affinity scorer - PyTorch implementation.

Pure PyTorch reimplementation of NetTCR CNN architecture to avoid TF/PyTorch conflicts.
"""

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tcrppo_v2.scorers.base import BaseScorer
from tcrppo_v2.utils.constants import PROJECT_ROOT


# BLOSUM50 encoding (20 amino acids)
BLOSUM50_20AA = {
    'A': np.array((5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0), dtype=np.float32),
    'R': np.array((-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3), dtype=np.float32),
    'N': np.array((-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3), dtype=np.float32),
    'D': np.array((-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4), dtype=np.float32),
    'C': np.array((-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1), dtype=np.float32),
    'Q': np.array((-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3), dtype=np.float32),
    'E': np.array((-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3), dtype=np.float32),
    'G': np.array((0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4), dtype=np.float32),
    'H': np.array((-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4), dtype=np.float32),
    'I': np.array((-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4), dtype=np.float32),
    'L': np.array((-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1), dtype=np.float32),
    'K': np.array((-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3), dtype=np.float32),
    'M': np.array((-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1), dtype=np.float32),
    'F': np.array((-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1), dtype=np.float32),
    'P': np.array((-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3), dtype=np.float32),
    'S': np.array((1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2), dtype=np.float32),
    'T': np.array((0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0), dtype=np.float32),
    'W': np.array((-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3), dtype=np.float32),
    'Y': np.array((-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1), dtype=np.float32),
    'V': np.array((0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5), dtype=np.float32),
}

MAX_CDR3_LEN = 30
MAX_PEP_LEN = 12


def encode_sequences(sequences: List[str], max_len: int) -> torch.Tensor:
    """Encode amino acid sequences using BLOSUM50 with zero-padding."""
    n_seqs = len(sequences)
    n_features = 20
    encoded = np.zeros((n_seqs, max_len, n_features), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_len]):
            if aa in BLOSUM50_20AA:
                encoded[i, j] = BLOSUM50_20AA[aa]

    return torch.from_numpy(encoded)


class NetTCRModel(nn.Module):
    """NetTCR-2.0 one-chain (beta) CNN architecture in PyTorch."""

    def __init__(self):
        super().__init__()

        # Peptide tower - multi-kernel CNN
        self.pep_conv1 = nn.Conv1d(20, 16, kernel_size=1, padding='same')
        self.pep_conv3 = nn.Conv1d(20, 16, kernel_size=3, padding='same')
        self.pep_conv5 = nn.Conv1d(20, 16, kernel_size=5, padding='same')
        self.pep_conv7 = nn.Conv1d(20, 16, kernel_size=7, padding='same')
        self.pep_conv9 = nn.Conv1d(20, 16, kernel_size=9, padding='same')

        # CDR3 tower - multi-kernel CNN
        self.cdr_conv1 = nn.Conv1d(20, 16, kernel_size=1, padding='same')
        self.cdr_conv3 = nn.Conv1d(20, 16, kernel_size=3, padding='same')
        self.cdr_conv5 = nn.Conv1d(20, 16, kernel_size=5, padding='same')
        self.cdr_conv7 = nn.Conv1d(20, 16, kernel_size=7, padding='same')
        self.cdr_conv9 = nn.Conv1d(20, 16, kernel_size=9, padding='same')

        # Fully connected layers
        # 5 kernels * 16 filters * 2 towers = 160 features
        self.fc1 = nn.Linear(160, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, cdr_input, pep_input):
        """
        Args:
            cdr_input: (batch, max_cdr_len, 20)
            pep_input: (batch, max_pep_len, 20)
        Returns:
            predictions: (batch, 1)
        """
        # Conv1d expects (batch, channels, length)
        cdr = cdr_input.permute(0, 2, 1)  # (batch, 20, max_cdr_len)
        pep = pep_input.permute(0, 2, 1)  # (batch, 20, max_pep_len)

        # Peptide tower
        pep1 = torch.sigmoid(self.pep_conv1(pep))
        pep1 = F.adaptive_max_pool1d(pep1, 1).squeeze(-1)  # GlobalMaxPooling

        pep3 = torch.sigmoid(self.pep_conv3(pep))
        pep3 = F.adaptive_max_pool1d(pep3, 1).squeeze(-1)

        pep5 = torch.sigmoid(self.pep_conv5(pep))
        pep5 = F.adaptive_max_pool1d(pep5, 1).squeeze(-1)

        pep7 = torch.sigmoid(self.pep_conv7(pep))
        pep7 = F.adaptive_max_pool1d(pep7, 1).squeeze(-1)

        pep9 = torch.sigmoid(self.pep_conv9(pep))
        pep9 = F.adaptive_max_pool1d(pep9, 1).squeeze(-1)

        # CDR3 tower
        cdr1 = torch.sigmoid(self.cdr_conv1(cdr))
        cdr1 = F.adaptive_max_pool1d(cdr1, 1).squeeze(-1)

        cdr3 = torch.sigmoid(self.cdr_conv3(cdr))
        cdr3 = F.adaptive_max_pool1d(cdr3, 1).squeeze(-1)

        cdr5 = torch.sigmoid(self.cdr_conv5(cdr))
        cdr5 = F.adaptive_max_pool1d(cdr5, 1).squeeze(-1)

        cdr7 = torch.sigmoid(self.cdr_conv7(cdr))
        cdr7 = F.adaptive_max_pool1d(cdr7, 1).squeeze(-1)

        cdr9 = torch.sigmoid(self.cdr_conv9(cdr))
        cdr9 = F.adaptive_max_pool1d(cdr9, 1).squeeze(-1)

        # Concatenate all features
        combined = torch.cat([pep1, pep3, pep5, pep7, pep9,
                             cdr1, cdr3, cdr5, cdr7, cdr9], dim=1)

        # Fully connected layers
        x = torch.sigmoid(self.fc1(combined))
        x = torch.sigmoid(self.fc2(x))

        return x


class AffinityNetTCRPyTorchScorer(BaseScorer):
    """NetTCR-2.0 CNN binding predictor - pure PyTorch implementation."""

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        batch_size: int = 512,
    ):
        self.device = device
        self.batch_size = batch_size

        # Build model
        self.model = NetTCRModel().to(device)

        # Load weights if available
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "data", "nettcr_pytorch.pt")

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint)
            print(f"  NetTCR-PyTorch loaded (weights: {model_path})")
        else:
            print(f"  WARNING: NetTCR-PyTorch weights not found at {model_path}")
            print(f"  Model will use random initialization - need to convert TF weights!")

        self.model.eval()

    def _predict_batch(self, tcrs: List[str], peptides: List[str]) -> np.ndarray:
        """Run prediction on a batch of TCR-peptide pairs."""
        cdr_enc = encode_sequences(tcrs, MAX_CDR3_LEN).to(self.device)
        pep_enc = encode_sequences(peptides, MAX_PEP_LEN).to(self.device)

        with torch.no_grad():
            preds = self.model(cdr_enc, pep_enc)

        return preds.cpu().numpy().ravel()

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score a single TCR-peptide pair. Returns (score, confidence=1.0)."""
        scores = self._predict_batch([tcr], [peptide])
        return float(scores[0]), 1.0

    def score_batch(self, tcrs: list, peptides: list, **kwargs) -> Tuple[list, list]:
        """Score a batch of TCR-peptide pairs."""
        scores = self._predict_batch(tcrs, peptides)
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        confidences = [1.0] * len(scores)
        return list(scores), confidences

    def score_batch_fast(self, tcrs: list, peptides: list) -> List[float]:
        """Fast batch scoring for training (no uncertainty)."""
        scores = self._predict_batch(tcrs, peptides)
        if isinstance(scores, np.ndarray):
            return scores.tolist()
        return list(scores)
