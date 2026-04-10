"""TCR pool: TCRdb loading + curriculum sampler (L0/L1/L2)."""

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from tcrppo_v2.utils.constants import (
    AMINO_ACIDS, MIN_TCR_LEN, MAX_TCR_LEN, TCRDB_PATH,
)
from tcrppo_v2.utils.encoding import is_valid_tcr, mutate_sequence


class TCRPool:
    """TCRdb sequence pool with L0/L1/L2 curriculum sampling."""

    def __init__(
        self,
        tcrdb_path: str = TCRDB_PATH,
        l1_seeds_dir: Optional[str] = None,
        l0_mutation_range: Tuple[int, int] = (3, 5),
        l1_top_k: int = 500,
        curriculum_schedule: Optional[List[dict]] = None,
        seed: int = 42,
    ):
        """Initialize TCR pool.

        Args:
            tcrdb_path: Path to TCRdb data directory.
            l1_seeds_dir: Path to pre-computed L1 seed files.
            l0_mutation_range: (min, max) mutations for L0 seeding.
            l1_top_k: Number of top-K TCRs per target for L1.
            curriculum_schedule: List of {until, L0, L1, L2} dicts.
            seed: Random seed.
        """
        self.rng = np.random.default_rng(seed)
        self.l0_mutation_range = l0_mutation_range
        self.l1_top_k = l1_top_k

        # Default curriculum from design spec
        if curriculum_schedule is None:
            curriculum_schedule = [
                {"until": 1_000_000, "L0": 0.7, "L1": 0.2, "L2": 0.1},
                {"until": 3_000_000, "L0": 0.4, "L1": 0.4, "L2": 0.2},
                {"until": 6_000_000, "L0": 0.2, "L1": 0.4, "L2": 0.4},
                {"until": None, "L0": 0.1, "L1": 0.3, "L2": 0.6},
            ]
        self.curriculum_schedule = curriculum_schedule

        # Load TCRdb sequences (L2 pool)
        self.tcrdb_seqs = self._load_tcrdb(tcrdb_path)

        # L0: Known binders per target (from decoy_d VDJdb data or manual seeds)
        self.l0_seeds: Dict[str, List[str]] = {}

        # L1: Pre-computed top-K per target
        self.l1_seeds: Dict[str, List[str]] = {}
        if l1_seeds_dir and os.path.isdir(l1_seeds_dir):
            self._load_l1_seeds(l1_seeds_dir)

    def _load_tcrdb(self, tcrdb_path: str) -> List[str]:
        """Load CDR3beta sequences from TCRdb."""
        train_file = os.path.join(tcrdb_path, "train_uniq_tcr_seqs.txt")
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"TCRdb file not found: {train_file}")

        seqs = []
        with open(train_file) as f:
            for line in f:
                seq = line.strip()
                if seq and is_valid_tcr(seq) and MIN_TCR_LEN <= len(seq) <= MAX_TCR_LEN:
                    seqs.append(seq)
        return seqs

    def _load_l1_seeds(self, l1_seeds_dir: str) -> None:
        """Load pre-computed L1 seed files."""
        for fname in os.listdir(l1_seeds_dir):
            if fname.endswith(".txt"):
                target = fname.replace(".txt", "")
                filepath = os.path.join(l1_seeds_dir, fname)
                with open(filepath) as f:
                    seqs = [line.strip() for line in f if line.strip()]
                if seqs:
                    self.l1_seeds[target] = seqs

    def load_l0_from_decoy_d(self, decoy_library_path: str, targets: List[str]) -> None:
        """Load known binders from decoy tier D (VDJdb/IEDB) as L0 seeds.

        For targets without tier D data, L0 falls back to L1 or L2.
        """
        import pandas as pd

        for target in targets:
            csv_path = os.path.join(
                decoy_library_path, "data", "decoy_d", target, "decoy_d_results.csv"
            )
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if "sequence" in df.columns:
                        binders = df["sequence"].dropna().tolist()
                        # Filter to valid CDR3-like peptide lengths
                        binders = [
                            s for s in binders
                            if isinstance(s, str)
                            and all(c in AMINO_ACIDS for c in s)
                            and 5 <= len(s) <= 15
                        ]
                        if binders:
                            self.l0_seeds[target] = binders
                except Exception:
                    pass

    def get_curriculum_weights(self, step: int) -> Tuple[float, float, float]:
        """Get L0/L1/L2 sampling weights for current training step."""
        for entry in self.curriculum_schedule:
            until = entry.get("until")
            if until is None or step < until:
                return entry["L0"], entry["L1"], entry["L2"]
        # Fallback to last entry
        last = self.curriculum_schedule[-1]
        return last["L0"], last["L1"], last["L2"]

    def sample_tcr(
        self,
        target: str,
        step: int = 0,
        reward_mode: str = "v2_full",
    ) -> Tuple[str, str]:
        """Sample an initial TCR for an episode.

        Args:
            target: Target peptide sequence.
            step: Current training step (for curriculum).
            reward_mode: Reward mode (v2_no_curriculum uses only L2).

        Returns:
            (tcr_sequence, level) where level is "L0", "L1", or "L2".
        """
        if reward_mode == "v2_no_curriculum":
            return self._sample_l2(), "L2"

        w_l0, w_l1, w_l2 = self.get_curriculum_weights(step)

        # Check availability
        has_l0 = target in self.l0_seeds and len(self.l0_seeds[target]) > 0
        has_l1 = target in self.l1_seeds and len(self.l1_seeds[target]) > 0

        if not has_l0:
            w_l1 += w_l0
            w_l0 = 0.0
        if not has_l1:
            w_l2 += w_l1
            w_l1 = 0.0

        total = w_l0 + w_l1 + w_l2
        if total <= 0:
            return self._sample_l2(), "L2"

        roll = self.rng.random() * total
        if roll < w_l0:
            return self._sample_l0(target), "L0"
        elif roll < w_l0 + w_l1:
            return self._sample_l1(target), "L1"
        else:
            return self._sample_l2(), "L2"

    def _sample_l0(self, target: str) -> str:
        """Sample from L0: known binder with random mutations."""
        binder = self.rng.choice(self.l0_seeds[target])
        n_mutations = self.rng.integers(
            self.l0_mutation_range[0], self.l0_mutation_range[1] + 1
        )
        n_mutations = min(n_mutations, len(binder) - 1)
        return mutate_sequence(binder, n_mutations, self.rng)

    def _sample_l1(self, target: str) -> str:
        """Sample from L1: pre-computed ERGO top-K."""
        return self.rng.choice(self.l1_seeds[target])

    def _sample_l2(self) -> str:
        """Sample from L2: random TCRdb sequence."""
        return self.rng.choice(self.tcrdb_seqs)

    def get_random_tcr(self) -> str:
        """Get a random TCRdb sequence (for general use)."""
        return self.rng.choice(self.tcrdb_seqs)

    @property
    def num_tcrdb_seqs(self) -> int:
        """Number of loaded TCRdb sequences."""
        return len(self.tcrdb_seqs)

    def get_l0_targets(self) -> List[str]:
        """Return targets that have L0 seeds."""
        return [t for t in self.l0_seeds if len(self.l0_seeds[t]) > 0]

    def get_l1_targets(self) -> List[str]:
        """Return targets that have L1 seeds."""
        return [t for t in self.l1_seeds if len(self.l1_seeds[t]) > 0]
