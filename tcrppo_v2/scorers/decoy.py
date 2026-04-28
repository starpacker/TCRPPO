"""Decoy contrastive penalty scorer using LogSumExp over sampled decoys."""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tcrppo_v2.scorers.base import BaseScorer


class DecoyScorer(BaseScorer):
    """LogSumExp contrastive penalty over tiered decoy library.

    R_decoy = (1/tau) * log(sum(exp(tau * affinity(TCR, p_neg))))
    """

    def __init__(
        self,
        decoy_library_path: str,
        targets: List[str],
        tier_weights: Dict[str, int] = None,
        K: int = 32,
        tau: float = 10.0,
        affinity_scorer=None,
        rng: np.random.Generator = None,
        decoy_a_min_count: int = 10,
        decoy_d_max_count: int = 50,
    ):
        self.decoy_library_path = decoy_library_path
        self.K = K
        self.tau = tau
        self.affinity_scorer = affinity_scorer
        self.tier_weights = tier_weights or {"A": 3, "B": 3, "D": 2, "C": 1}
        self.rng = rng or np.random.default_rng(42)
        self.decoy_a_min_count = decoy_a_min_count
        self.decoy_d_max_count = decoy_d_max_count

        # Load all decoy peptides per target per tier
        self.decoys: Dict[str, Dict[str, List[str]]] = {}
        self._load_decoys(targets)

        # Unlocked tiers (start with all for scoring; controlled externally)
        self.unlocked_tiers = ["A", "B", "D", "C"]

    def _load_decoys(self, targets: List[str]) -> None:
        """Load decoy peptide sequences for all targets and tiers."""
        data_dir = os.path.join(self.decoy_library_path, "data")

        # Load tier C (shared across all targets)
        tier_c_path = os.path.join(data_dir, "decoy_c", "decoy_library.json")
        tier_c_peptides = []
        if os.path.exists(tier_c_path):
            with open(tier_c_path) as f:
                tier_c_data = json.load(f)
            entries = tier_c_data.get("entries", tier_c_data)
            if isinstance(entries, list):
                for entry in entries:
                    pep_info = entry.get("peptide_info", {})
                    seq = pep_info.get("decoy_sequence", "")
                    if seq:
                        tier_c_peptides.append(seq)

        for target in targets:
            self.decoys[target] = {"A": [], "B": [], "C": tier_c_peptides, "D": []}

            # Tier A: graduated hamming distance (hd<=2 -> <=3 -> <=4)
            tier_a_path = os.path.join(data_dir, "decoy_a", target, "decoy_a_results.json")
            if os.path.exists(tier_a_path):
                with open(tier_a_path) as f:
                    entries = json.load(f)
                self.decoys[target]["A"] = self._filter_tier_a(entries)

            # Tier B
            tier_b_path = os.path.join(data_dir, "decoy_b", target, "decoy_b_results.json")
            if os.path.exists(tier_b_path):
                with open(tier_b_path) as f:
                    entries = json.load(f)
                self.decoys[target]["B"] = [e["sequence"] for e in entries if "sequence" in e]

            # Tier D: capped to decoy_d_max_count
            tier_d_path = os.path.join(data_dir, "decoy_d", target, "decoy_d_results.csv")
            if os.path.exists(tier_d_path):
                df = pd.read_csv(tier_d_path)
                if "sequence" in df.columns:
                    d_seqs = df["sequence"].dropna().tolist()
                    if len(d_seqs) > self.decoy_d_max_count:
                        indices = self.rng.choice(
                            len(d_seqs), size=self.decoy_d_max_count, replace=False
                        )
                        d_seqs = [d_seqs[i] for i in indices]
                    self.decoys[target]["D"] = d_seqs

    def _filter_tier_a(self, entries: list) -> List[str]:
        """Filter tier A entries using graduated hamming distance.

        Strategy: hd<=2 first; if < min_count, expand to hd<=3, then hd<=4.
        Falls back to all entries if hamming_distance field is missing.
        """
        has_hd = any("hamming_distance" in e for e in entries if isinstance(e, dict))
        if not has_hd:
            return [e["sequence"] for e in entries if isinstance(e, dict) and "sequence" in e]

        for max_hd in (2, 3, 4):
            seqs = [
                e["sequence"]
                for e in entries
                if isinstance(e, dict)
                and "sequence" in e
                and e.get("hamming_distance", 999) <= max_hd
            ]
            if len(seqs) >= self.decoy_a_min_count:
                return seqs
        # Return whatever we have at hd<=4 even if < min_count
        return seqs

    def set_unlocked_tiers(self, tiers: List[str]) -> None:
        """Update which tiers are unlocked (for curriculum)."""
        self.unlocked_tiers = tiers

    def sample_decoys(self, target: str, k: int = None) -> List[str]:
        """Sample K decoys using tier-weighted sampling."""
        k = k or self.K
        target_decoys = self.decoys.get(target, {})

        # Build weighted pool from unlocked tiers
        pool = []
        weights = []
        for tier in self.unlocked_tiers:
            tier_peps = target_decoys.get(tier, [])
            if tier_peps:
                w = self.tier_weights.get(tier, 1)
                pool.extend(tier_peps)
                weights.extend([w] * len(tier_peps))

        if not pool:
            return []

        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()

        k = min(k, len(pool))
        indices = self.rng.choice(len(pool), size=k, replace=False, p=weights)
        return [pool[i] for i in indices]

    def sample_decoys_by_tier(self, target: str, k_per_tier: int = 20) -> Dict[str, List[str]]:
        """Sample decoys grouped by tier for per-tier AUROC evaluation.

        Returns:
            Dict mapping tier name -> list of decoy peptide sequences.
        """
        target_decoys = self.decoys.get(target, {})
        result: Dict[str, List[str]] = {}
        for tier in self.unlocked_tiers:
            tier_peps = target_decoys.get(tier, [])
            if not tier_peps:
                continue
            k = min(k_per_tier, len(tier_peps))
            indices = self.rng.choice(len(tier_peps), size=k, replace=False)
            result[tier] = [tier_peps[i] for i in indices]
        return result

    def compute_logsumexp_penalty(
        self, tcr: str, decoy_peps: List[str]
    ) -> Tuple[float, float]:
        """Compute LogSumExp penalty for a TCR against sampled decoys."""
        if not decoy_peps or self.affinity_scorer is None:
            return 0.0, 0.0

        tcrs = [tcr] * len(decoy_peps)
        # Use fast scoring (no MC Dropout) for training speed
        if hasattr(self.affinity_scorer, 'score_batch_fast'):
            scores = self.affinity_scorer.score_batch_fast(tcrs, decoy_peps)
            scores = np.array(scores)
            mean_confidence = 1.0
        else:
            scores, confidences = self.affinity_scorer.score_batch(tcrs, decoy_peps)
            scores = np.array(scores)
            mean_confidence = float(np.mean(confidences))

        # LogSumExp: (1/tau) * log(sum(exp(tau * score)))
        scaled = self.tau * scores
        max_scaled = np.max(scaled)
        logsumexp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        penalty = logsumexp / self.tau

        return float(penalty), mean_confidence

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score a single TCR. Higher penalty = more cross-reactive (bad)."""
        target = kwargs.get("target", peptide)
        decoy_peps = self.sample_decoys(target)
        return self.compute_logsumexp_penalty(tcr, decoy_peps)

    def score_batch(
        self, tcrs: list, peptides: list, **kwargs
    ) -> Tuple[list, list]:
        """Score a batch of TCRs against their respective targets."""
        targets = kwargs.get("targets", peptides)
        scores = []
        confidences = []
        for tcr, target in zip(tcrs, targets):
            s, c = self.score(tcr, target, target=target)
            scores.append(s)
            confidences.append(c)
        return scores, confidences
