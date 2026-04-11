"""Tiered decoy sampling with unlock schedule."""

import csv
import json
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from tcrppo_v2.utils.constants import AMINO_ACIDS, DECOY_LIBRARY_PATH


class DecoySampler:
    """Phase-aware tiered decoy sampler.

    Samples decoy peptides from unlocked tiers based on training step.
    Tiers: A (point mutants) > B (2-3 AA mutants) > D (VDJdb/IEDB) > C (unrelated).
    """

    def __init__(
        self,
        decoy_library_path: str = DECOY_LIBRARY_PATH,
        targets: Optional[List[str]] = None,
        tier_weights: Optional[Dict[str, int]] = None,
        unlock_schedule: Optional[Dict[int, List[str]]] = None,
        seed: int = 42,
        decoy_a_min_count: int = 10,
        decoy_d_max_count: int = 50,
    ):
        """Initialize decoy sampler.

        Args:
            decoy_library_path: Root path to decoy library.
            targets: List of target peptides to load decoys for.
            tier_weights: Sampling weights per tier (default: A:3, B:3, D:2, C:1).
            unlock_schedule: Dict mapping step -> list of unlocked tiers.
            seed: Random seed.
            decoy_a_min_count: Minimum decoys before expanding hamming distance.
            decoy_d_max_count: Maximum tier D entries (random subsample if exceeded).
        """
        self.rng = np.random.default_rng(seed)
        self.decoy_library_path = decoy_library_path
        self.decoy_a_min_count = decoy_a_min_count
        self.decoy_d_max_count = decoy_d_max_count

        if tier_weights is None:
            tier_weights = {"A": 3, "B": 3, "D": 2, "C": 1}
        self.tier_weights = tier_weights

        if unlock_schedule is None:
            unlock_schedule = {
                0: ["A"],
                2_000_000: ["A", "B"],
                5_000_000: ["A", "B", "D"],
                8_000_000: ["A", "B", "D", "C"],
            }
        self.unlock_schedule = unlock_schedule

        # Current unlocked tiers
        self.unlocked_tiers: Set[str] = {"A"}

        # Load decoys per target per tier
        self.decoys: Dict[str, Dict[str, List[str]]] = {}
        self.tier_c_global: List[str] = []

        self._load_tier_c()
        if targets:
            for target in targets:
                self._load_target_decoys(target)

    def _load_tier_c(self) -> None:
        """Load tier C (1900 unrelated peptides, shared across all targets)."""
        c_path = os.path.join(
            self.decoy_library_path, "data", "decoy_c", "decoy_library.json"
        )
        if os.path.exists(c_path):
            with open(c_path) as f:
                data = json.load(f)
            entries = data.get("entries", [])
            for entry in entries:
                pep_info = entry.get("peptide_info", {})
                seq = pep_info.get("decoy_sequence", "")
                if seq and all(c in AMINO_ACIDS for c in seq):
                    self.tier_c_global.append(seq)

    def _load_target_decoys(self, target: str) -> None:
        """Load tier A, B, D decoys for a specific target."""
        self.decoys[target] = {"A": [], "B": [], "C": [], "D": []}

        # Tier A: graduated hamming distance (hd<=2 -> <=3 -> <=4)
        a_path = os.path.join(
            self.decoy_library_path, "data", "decoy_a", target, "decoy_a_results.json"
        )
        if os.path.exists(a_path):
            with open(a_path) as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else data.get("entries", data.get("results", []))
            self.decoys[target]["A"] = self._filter_tier_a(entries)

        # Tier B: 2-3 AA mutants
        b_path = os.path.join(
            self.decoy_library_path, "data", "decoy_b", target, "decoy_b_results.json"
        )
        if os.path.exists(b_path):
            with open(b_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    seq = entry.get("sequence", entry.get("decoy_sequence", ""))
                    if seq and all(c in AMINO_ACIDS for c in seq):
                        self.decoys[target]["B"].append(seq)
            elif isinstance(data, dict):
                for entry in data.get("entries", data.get("results", [])):
                    seq = entry.get("sequence", entry.get("decoy_sequence", ""))
                    if seq and all(c in AMINO_ACIDS for c in seq):
                        self.decoys[target]["B"].append(seq)

        # Tier C: use global pool
        self.decoys[target]["C"] = self.tier_c_global

        # Tier D: known binders (capped to decoy_d_max_count)
        d_path = os.path.join(
            self.decoy_library_path, "data", "decoy_d", target, "decoy_d_results.csv"
        )
        if os.path.exists(d_path):
            try:
                d_seqs = []
                with open(d_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        seq = row.get("sequence", "")
                        if seq and all(c in AMINO_ACIDS for c in seq):
                            d_seqs.append(seq)
                if len(d_seqs) > self.decoy_d_max_count:
                    indices = self.rng.choice(
                        len(d_seqs), size=self.decoy_d_max_count, replace=False
                    )
                    d_seqs = [d_seqs[i] for i in indices]
                self.decoys[target]["D"] = d_seqs
            except Exception:
                pass

    def _filter_tier_a(self, entries: list) -> List[str]:
        """Filter tier A entries using graduated hamming distance.

        Strategy: hd<=2 first; if < min_count, expand to hd<=3, then hd<=4.
        Falls back to all entries if hamming_distance field is missing.
        """
        has_hd = any("hamming_distance" in e for e in entries if isinstance(e, dict))
        if not has_hd:
            return [
                e.get("sequence", e.get("decoy_sequence", ""))
                for e in entries
                if isinstance(e, dict)
                and e.get("sequence", e.get("decoy_sequence", ""))
                and all(c in AMINO_ACIDS for c in e.get("sequence", e.get("decoy_sequence", "")))
            ]

        seqs = []
        for max_hd in (2, 3, 4):
            seqs = [
                e.get("sequence", e.get("decoy_sequence", ""))
                for e in entries
                if isinstance(e, dict)
                and e.get("sequence", e.get("decoy_sequence", ""))
                and all(c in AMINO_ACIDS for c in e.get("sequence", e.get("decoy_sequence", "")))
                and e.get("hamming_distance", 999) <= max_hd
            ]
            if len(seqs) >= self.decoy_a_min_count:
                return seqs
        return seqs

    def update_unlocked_tiers(self, step: int) -> None:
        """Update unlocked tiers based on training step."""
        current_tiers = {"A"}
        for threshold, tiers in sorted(self.unlock_schedule.items()):
            if step >= threshold:
                current_tiers = set(tiers)
        self.unlocked_tiers = current_tiers

    def sample_decoys(
        self,
        target: str,
        k: int = 32,
        step: Optional[int] = None,
    ) -> List[str]:
        """Sample K decoy peptides with tier weighting.

        Args:
            target: Target peptide.
            k: Number of decoys to sample.
            step: If provided, update unlocked tiers first.

        Returns:
            List of K decoy peptide sequences.
        """
        if step is not None:
            self.update_unlocked_tiers(step)

        if target not in self.decoys:
            self._load_target_decoys(target)

        target_decoys = self.decoys.get(target, {})

        # Build weighted pool from unlocked tiers
        pool: List[Tuple[str, str]] = []  # (sequence, tier)
        weights: List[float] = []

        for tier in self.unlocked_tiers:
            tier_seqs = target_decoys.get(tier, [])
            if not tier_seqs:
                continue
            w = self.tier_weights.get(tier, 1)
            for seq in tier_seqs:
                pool.append((seq, tier))
                weights.append(float(w))

        if not pool:
            # Fallback: sample from tier C global if nothing else available
            if self.tier_c_global:
                indices = self.rng.choice(len(self.tier_c_global), size=k, replace=True)
                return [self.tier_c_global[i] for i in indices]
            return []

        weights_arr = np.array(weights)
        weights_arr = weights_arr / weights_arr.sum()

        indices = self.rng.choice(len(pool), size=k, replace=True, p=weights_arr)
        return [pool[i][0] for i in indices]

    def get_available_tiers(self, target: str) -> Dict[str, int]:
        """Get count of available decoys per tier for a target."""
        if target not in self.decoys:
            return {}
        return {
            tier: len(seqs)
            for tier, seqs in self.decoys[target].items()
            if len(seqs) > 0
        }

    def get_tier_c_count(self) -> int:
        """Return number of tier C decoys."""
        return len(self.tier_c_global)
