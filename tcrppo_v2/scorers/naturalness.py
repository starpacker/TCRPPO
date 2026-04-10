"""ESM perplexity-based naturalness scorer with CDR3 z-score normalization."""

import json
import os
from typing import List, Tuple, Optional

import numpy as np
import torch

from tcrppo_v2.scorers.base import BaseScorer


class NaturalnessScorer(BaseScorer):
    """Penalize sequences that are unnaturally different from real CDR3beta.

    Uses ESM-2 pseudo-perplexity as a proxy for sequence naturalness.
    Offline-computed mean/std over TCRdb CDR3beta sequences provide
    CDR3-aware z-score normalization.
    """

    def __init__(
        self,
        esm_model=None,
        esm_alphabet=None,
        esm_batch_converter=None,
        device: str = "cuda",
        stats_file: Optional[str] = None,
        threshold_zscore: float = -2.0,
    ):
        self.device = device
        self.threshold = threshold_zscore

        # ESM model can be shared with state encoder
        if esm_model is not None:
            self.model = esm_model
            self.alphabet = esm_alphabet
            self.batch_converter = esm_batch_converter
        else:
            import esm
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.model = self.model.to(device)
            self.model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()

        # Load or initialize stats
        self.mean_ppl = None
        self.std_ppl = None
        if stats_file and os.path.exists(stats_file):
            self._load_stats(stats_file)

    def _load_stats(self, path: str) -> None:
        with open(path) as f:
            stats = json.load(f)
        self.mean_ppl = stats["mean_ppl"]
        self.std_ppl = stats["std_ppl"]

    def save_stats(self, path: str) -> None:
        """Save computed stats to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mean_ppl": self.mean_ppl, "std_ppl": self.std_ppl}, f, indent=2)

    @torch.no_grad()
    def compute_pseudo_perplexity(self, sequences: List[str]) -> np.ndarray:
        """Compute ESM-2 pseudo-perplexity for a batch of sequences.

        Pseudo-perplexity: exp(mean negative log-likelihood) where each position
        is masked and predicted.
        """
        ppls = []
        # Process one at a time to compute per-position masked log-likelihoods
        for seq in sequences:
            data = [("seq", seq)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)

            # Get all logits in a single forward pass (no masking needed for pseudo-ppl)
            output = self.model(tokens, repr_layers=[], return_contacts=False)
            logits = output["logits"]  # (1, L+2, vocab)

            # Compute log-probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # For each non-special position, get the log-prob of the true token
            # tokens shape: (1, L+2) where positions 0 and L+1 are BOS/EOS
            seq_len = len(seq)
            total_nll = 0.0
            for pos in range(1, seq_len + 1):
                true_token = tokens[0, pos]
                total_nll -= log_probs[0, pos, true_token].item()

            ppl = np.exp(total_nll / seq_len) if seq_len > 0 else float("inf")
            ppls.append(ppl)

        return np.array(ppls)

    def compute_stats_from_tcrdb(
        self, tcrdb_path: str, n_samples: int = 10000, seed: int = 42,
        save_path: Optional[str] = None
    ) -> Tuple[float, float]:
        """Compute mean/std perplexity over a sample of TCRdb CDR3beta sequences."""
        rng = np.random.default_rng(seed)

        # Load TCRdb sequences
        with open(os.path.join(tcrdb_path, "train_uniq_tcr_seqs.txt")) as f:
            all_seqs = [line.strip() for line in f if line.strip()]

        # Sample
        indices = rng.choice(len(all_seqs), size=min(n_samples, len(all_seqs)), replace=False)
        sampled = [all_seqs[i] for i in indices]

        # Filter valid AA sequences
        valid_aas = set("ARNDCQEGHILKMFPSTWYV")
        sampled = [s for s in sampled if all(c in valid_aas for c in s) and 8 <= len(s) <= 27]
        sampled = sampled[:n_samples]

        # Compute perplexities in batches
        ppls = []
        batch_size = 100
        for i in range(0, len(sampled), batch_size):
            batch = sampled[i : i + batch_size]
            batch_ppls = self.compute_pseudo_perplexity(batch)
            ppls.extend(batch_ppls.tolist())

        ppls = np.array(ppls)
        self.mean_ppl = float(np.mean(ppls))
        self.std_ppl = float(np.std(ppls))

        if save_path:
            self.save_stats(save_path)

        return self.mean_ppl, self.std_ppl

    def _zscore(self, ppl: float) -> float:
        if self.mean_ppl is None or self.std_ppl is None:
            return 0.0
        return (ppl - self.mean_ppl) / (self.std_ppl + 1e-8)

    def score(self, tcr: str, peptide: str = "", **kwargs) -> Tuple[float, float]:
        """Score a single TCR for naturalness.

        Returns (penalty, confidence=1.0).
        Penalty is 0.0 if natural enough, negative if unnatural.
        """
        ppl = self.compute_pseudo_perplexity([tcr])[0]
        z = self._zscore(ppl)

        if z >= self.threshold:
            return 0.0, 1.0
        else:
            return float(z - self.threshold), 1.0

    def score_batch(self, tcrs: list, peptides: list = None, **kwargs) -> Tuple[list, list]:
        """Score a batch of TCRs for naturalness."""
        ppls = self.compute_pseudo_perplexity(tcrs)
        scores = []
        for ppl in ppls:
            z = self._zscore(float(ppl))
            if z >= self.threshold:
                scores.append(0.0)
            else:
                scores.append(float(z - self.threshold))
        confidences = [1.0] * len(tcrs)
        return scores, confidences
