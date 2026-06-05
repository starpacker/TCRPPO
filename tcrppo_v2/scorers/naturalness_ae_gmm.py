"""AE + GMM naturalness scorer ported from TCRPPO.

Uses a TCR-specific Autoencoder for reconstruction quality (edit-distance)
and a Gaussian Mixture Model on the latent space for distributional
likelihood.  Together they measure how "natural" a CDR3β sequence looks
relative to real TCR data – something that ESM-2 pseudo-perplexity misses
for short poly-C tails.

Standalone: no runtime dependency on the TCRPPO codebase; only needs the
three artefact files (blosum.txt, ae_model, gmm.pkl).
"""

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from tcrppo_v2.scorers.base import BaseScorer

# ── amino-acid constants ────────────────────────────────────────────────
AMINO_ACIDS = list("ARNDCEQGHILKMFPSTWYV")
_A2N = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
_N2A = {i + 1: aa for i, aa in enumerate(AMINO_ACIDS)}


# ── BLOSUM helpers ──────────────────────────────────────────────────────
def _load_blosum(path: str) -> np.ndarray:
    """Load a BLOSUM matrix from the TCRPPO-style TSV file.

    Returns shape (21, 20): 21 rows (0=PAD, 1-20=amino acids), 20 columns (20 AAs).
    Excludes the header line (i=0) and the last column (X).
    """
    blosum = np.zeros((21, 20), dtype=np.float32)
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i == 0 or i == 21:  # Skip header and non-existent row 21
                continue
            parts = line.strip().split("\t")
            blosum[i, :] = np.array(parts[1:-1], dtype=np.float32)  # Skip first (index) and last (X) columns
    return blosum


def _seq2num(sequences: List[str], max_len: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if max_len == 0:
        max_len = max(len(s) for s in sequences)
    arrays = np.zeros((len(sequences), max_len), dtype=np.float32)
    lengths = np.zeros(len(sequences), dtype=np.float32)
    for i, seq in enumerate(sequences):
        lengths[i] = len(seq)
        for j, aa in enumerate(seq):
            arrays[i, j] = _A2N.get(aa, 0)
    return arrays, lengths


def _num2seq(tensor: torch.Tensor) -> List[str]:
    seqs = []
    for i in range(tensor.shape[0]):
        seq = "".join(_N2A[idx.item()] for idx in tensor[i] if idx.item() != 0)
        seqs.append(seq)
    return seqs


def _blosum_encode(arrays: np.ndarray, blosum: np.ndarray) -> np.ndarray:
    return np.take(blosum, arrays.astype(int), axis=0)


# ── Autoencoder (architecture mirrors TCRPPO/code/reward/AE.py) ────────
class _AE(nn.Module):
    def __init__(
        self,
        embed_size: int = 20,
        hidden_size: int = 64,
        latent_size: int = 16,
        min_len: int = 8,
        max_len: int = 35,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.min_len = min_len
        self.max_len = max_len
        self.bidirectional = bidirectional
        self.embedding_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_size = len(AMINO_ACIDS) + 1      # 20 AAs + PAD/stop

        factor = 2 if bidirectional else 1

        self.encoder_rnn = nn.LSTM(
            embed_size, hidden_size, bidirectional=bidirectional, batch_first=True,
        )
        self.hidden2latent = nn.Linear(factor * hidden_size, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size)

        # Decoder LSTM gates (manual implementation from TCRPPO)
        self.decoder_i = nn.Sequential(nn.Linear(embed_size + hidden_size, hidden_size), nn.Sigmoid())
        self.decoder_o = nn.Sequential(nn.Linear(embed_size + hidden_size, hidden_size), nn.Sigmoid())
        self.decoder_f = nn.Sequential(nn.Linear(embed_size + hidden_size, hidden_size), nn.Sigmoid())
        self.decoder_g = nn.Sequential(nn.Linear(embed_size + hidden_size, hidden_size), nn.Tanh())

        self.hidden2out = nn.Sequential(
            nn.Linear(embed_size + hidden_size + latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_size),
        )

    # ---- encoder ----
    def encode(self, sequences: List[str], blosum: np.ndarray, device: torch.device) -> torch.Tensor:
        arrays, lengths = _seq2num(sequences)
        batch_size = arrays.shape[0]
        h0 = torch.zeros(2 if self.bidirectional else 1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros_like(h0)

        encoded = torch.tensor(_blosum_encode(arrays, blosum), dtype=torch.float32, device=device)
        packed = nn.utils.rnn.pack_padded_sequence(
            encoded, lengths, batch_first=True, enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder_rnn(packed, (h0, c0))

        if self.bidirectional:
            hidden = hidden.transpose(0, 1).flatten(start_dim=-2)
        else:
            hidden = hidden.squeeze(0)

        return self.hidden2latent(hidden)

    # ---- decoder (greedy generate) ----
    def generate(self, z: torch.Tensor, blosum: np.ndarray, device: torch.device) -> List[str]:
        hidden = self.latent2hidden(z)
        cell = torch.zeros_like(hidden)
        x_vecs = torch.zeros(z.shape[0], self.embedding_size, device=device)
        padded = torch.zeros(z.shape[0], self.max_len, dtype=torch.long, device=device)

        idxs = torch.arange(z.shape[0], device=device)
        cur_z, cur_h, cur_c, cur_x = z, hidden, cell, x_vecs

        for t in range(self.max_len):
            amino_hidden = torch.cat((cur_x, cur_h, cur_z), dim=-1)
            amino_scores = self.hidden2out(amino_hidden)

            if t <= self.min_len:
                amino_scores[:, 0] = -1000.0          # forbid early stop

            _, out = amino_scores.max(dim=1)

            nonstop = torch.arange(cur_h.shape[0], device=device)
            if t > self.min_len:
                nonstop = out.nonzero(as_tuple=False).squeeze(-1)
                out = out[nonstop]
                idxs = idxs[nonstop]

            if nonstop.shape[0] == 0:
                break

            padded[idxs, t] = out

            out_vec = np.take(blosum, out.cpu().numpy().astype(int), axis=0)
            out_vec = torch.tensor(out_vec, dtype=torch.float32, device=device)

            # Decoder LSTM step
            i_gate = self.decoder_i(torch.cat([cur_x[nonstop], cur_h[nonstop]], -1))
            o_gate = self.decoder_o(torch.cat([cur_x[nonstop], cur_h[nonstop]], -1))
            f_gate = self.decoder_f(torch.cat([cur_x[nonstop], cur_h[nonstop]], -1))
            g_gate = self.decoder_g(torch.cat([cur_x[nonstop], cur_h[nonstop]], -1))
            new_c = f_gate * cur_c[nonstop] + i_gate * g_gate
            new_h = o_gate * torch.tanh(new_c)

            cur_z = cur_z[nonstop]
            cur_h = new_h
            cur_c = new_c
            cur_x = out_vec

        seqs = _num2seq(padded)
        seqs = [s[::-1] for s in seqs]
        return seqs


# ── Public scorer ───────────────────────────────────────────────────────
class NaturalnessAEGMMScorer(BaseScorer):
    """AE reconstruction + GMM likelihood naturalness scorer.

    Parameters
    ----------
    ae_model_path : path to the saved AE state-dict.
    gmm_model_path : path to the pickled sklearn GMM.
    blosum_path : path to the BLOSUM TSV file.
    threshold : combined (edit_acc + gmm_like) below which a penalty fires.
    device : "cuda" or "cpu".
    """

    def __init__(
        self,
        ae_model_path: str,
        gmm_model_path: str,
        blosum_path: str,
        threshold: float = 0.8,
        device: str = "cuda",
    ):
        self.device = device
        self.threshold = threshold

        # BLOSUM matrix
        self.blosum = _load_blosum(blosum_path)

        # AE
        self.ae = _AE(embed_size=20, hidden_size=64, latent_size=16)
        state = torch.load(ae_model_path, map_location=device)
        self.ae.load_state_dict(state)
        self.ae.to(device)
        self.ae.eval()

        # GMM
        with open(gmm_model_path, "rb") as fh:
            self.gmm = pickle.load(fh)

    # ---- internals ----
    @torch.no_grad()
    def _encode(self, tcrs: List[str]) -> np.ndarray:
        z = self.ae.encode(tcrs, self.blosum, self.device)
        return z.cpu().numpy()

    @torch.no_grad()
    def _edit_accuracy(self, tcrs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (edit_accuracy_per_seq, latent_z)."""
        z_tensor = self.ae.encode(tcrs, self.blosum, self.device)
        recon = self.ae.generate(z_tensor, self.blosum, self.device)
        z_np = z_tensor.cpu().numpy()

        dists = np.zeros(len(tcrs))
        for i, (gen, orig) in enumerate(zip(recon, tcrs)):
            d = sum(1 for a, b in zip(gen, orig) if a != b)
            d += abs(len(gen) - len(orig))
            dists[i] = d / max(len(orig), 1)

        return 1.0 - dists, z_np

    def _gmm_likelihood(self, z: np.ndarray) -> np.ndarray:
        log_likes = self.gmm.score_samples(z)            # per-sample log-prob
        return np.exp((log_likes + 10.0) / 10.0)         # same transform as TCRPPO

    # ---- public API (BaseScorer) ----
    def score(self, tcr: str, peptide: str = "", **kwargs) -> Tuple[float, float]:
        """Return (penalty, confidence).

        penalty = 0.0  if the sequence is natural enough,
                  < 0   otherwise (magnitude = how unnatural).
        """
        edit_acc, z = self._edit_accuracy([tcr])
        gmm_like = self._gmm_likelihood(z)
        combined = float(edit_acc[0] + gmm_like[0])

        if combined >= self.threshold:
            return 0.0, 1.0
        return float(combined - self.threshold), 1.0     # negative penalty

    def score_batch(self, tcrs: list, peptides: list = None, **kwargs) -> Tuple[list, list]:
        """Score a batch of TCRs."""
        if len(tcrs) == 0:
            return [], []
        edit_acc, z = self._edit_accuracy(tcrs)
        gmm_like = self._gmm_likelihood(z)
        combined = edit_acc + gmm_like

        penalties = []
        for c in combined:
            if c >= self.threshold:
                penalties.append(0.0)
            else:
                penalties.append(float(c - self.threshold))
        return penalties, [1.0] * len(tcrs)

    # ---- diagnostics ----
    def diagnose(self, tcrs: List[str]) -> List[dict]:
        """Return detailed breakdown for debugging."""
        edit_acc, z = self._edit_accuracy(tcrs)
        gmm_like = self._gmm_likelihood(z)
        results = []
        for i, tcr in enumerate(tcrs):
            combined = edit_acc[i] + gmm_like[i]
            penalty = 0.0 if combined >= self.threshold else float(combined - self.threshold)
            results.append({
                "tcr": tcr,
                "edit_acc": float(edit_acc[i]),
                "gmm_like": float(gmm_like[i]),
                "combined": float(combined),
                "penalty": penalty,
            })
        return results
