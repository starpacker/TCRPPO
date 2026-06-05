"""TCR pool: TCRdb loading + curriculum sampler (L0/L1/L2)."""

import os
import json
from collections import deque
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
        online_pool_enabled: bool = False,
        online_pool_start_step: int = 0,
        online_pool_warmup_steps: int = 100_000,
        online_pool_max_ratio: float = 0.5,
        online_pool_max_per_target: int = 256,
        online_pool_mutate_prob: float = 0.0,
        online_pool_mutation_range: Tuple[int, int] = (1, 2),
        online_pool_min_hamming: int = 2,
        online_pool_sample_bands: Optional[List[dict]] = None,
        online_pool_snapshot_path: Optional[str] = None,
        online_pool_elite_ratio: float = 0.0,
        online_pool_elite_min_affinity: float = -0.5,
        online_pool_dynamic_mode: str = "band",
        online_pool_dynamic_below: float = 0.7,
        online_pool_dynamic_above: float = 0.0,
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
        self.online_pool_enabled = bool(online_pool_enabled)
        self.online_pool_start_step = max(0, int(online_pool_start_step))
        self.online_pool_warmup_steps = max(1, int(online_pool_warmup_steps))
        self.online_pool_max_ratio = min(1.0, max(0.0, float(online_pool_max_ratio)))
        self.online_pool_max_per_target = max(1, int(online_pool_max_per_target))
        self.online_pool_mutate_prob = min(1.0, max(0.0, float(online_pool_mutate_prob)))
        self.online_pool_mutation_range = online_pool_mutation_range
        self.online_pool_min_hamming = max(0, int(online_pool_min_hamming))
        self.online_pool_sample_bands = online_pool_sample_bands or []
        self.online_pool_snapshot_path = online_pool_snapshot_path
        self.online_pool_elite_ratio = min(1.0, max(0.0, float(online_pool_elite_ratio)))
        self.online_pool_elite_min_affinity = float(online_pool_elite_min_affinity)
        self.online_pool_dynamic_mode = str(online_pool_dynamic_mode or "band")
        self.online_pool_dynamic_below = max(0.0, float(online_pool_dynamic_below))
        self.online_pool_dynamic_above = max(0.0, float(online_pool_dynamic_above))

        # Ladder / adaptive band tracking (per-target recent final affinities)
        self.recent_affinities: Dict[str, deque] = {}
        self.recent_affinity_window = 50
        self.cached_bands: Dict[str, dict] = {}
        self.band_update_counters: Dict[str, int] = {}
        self.band_update_interval = 50

        # Enhanced curriculum: gradual L0→L1→L2 progression
        # 0-500K: pure L0 (learn what good TCRs look like)
        # 500K-1M: 70% L0 + 30% L1 (start exploring ERGO top-K)
        # 1M-2M: 50% L0 + 30% L1 + 20% L2 (add random exploration)
        if curriculum_schedule is None:
            curriculum_schedule = [
                {"until": 500_000, "L0": 1.0, "L1": 0.0, "L2": 0.0},
                {"until": 1_000_000, "L0": 0.7, "L1": 0.3, "L2": 0.0},
                {"until": 2_000_000, "L0": 0.5, "L1": 0.3, "L2": 0.2},
                {"until": None, "L0": 0.3, "L1": 0.3, "L2": 0.4},
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

        # Online pool: high-affinity final TCRs discovered during PPO,
        # stored separately per target peptide.
        self.online_seeds: Dict[str, List[dict]] = {}
        self._online_seen: Dict[str, set] = {}
        if self.online_pool_snapshot_path:
            self.load_online_pool_snapshot(self.online_pool_snapshot_path)
            if self.online_pool_dynamic_mode == "ladder":
                self._seed_ladder_from_pool()

    def _load_tcrdb(self, tcrdb_path: str) -> List[str]:
        """Load CDR3beta sequences from TCRdb."""
        train_file = os.path.join(tcrdb_path, "train_uniq_tcr_seqs.txt")
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"TCRdb file not found: {train_file}")

        seqs = []
        with open(train_file) as f:
            for line in f:
                seq = line.strip()
                if (
                    seq
                    and is_valid_tcr(seq)
                    and seq.startswith("C")
                    and MIN_TCR_LEN <= len(seq) <= MAX_TCR_LEN
                ):
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
        """Load CDR3beta binders from decoy tier D metadata as L0 seeds.

        The decoy-D library also contains generated *peptide* decoys in
        ``decoy_d_results.csv`` under a generic ``sequence`` column. Those are
        not TCRs and must not be used as CDR3 seeds.
        """
        import pandas as pd

        tcr_columns = ("cdr3b", "cdr3_beta", "cdr3.beta", "tcr", "tcr_sequence")
        for target in targets:
            csv_path = os.path.join(
                decoy_library_path, "data", "decoy_d", target, "decoy_d_results.csv"
            )
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    seq_col = next((col for col in tcr_columns if col in df.columns), None)
                    if seq_col is not None:
                        binders = df[seq_col].dropna().tolist()
                        binders = [
                            s for s in binders
                            if isinstance(s, str)
                            and all(c in AMINO_ACIDS for c in s)
                            and s.startswith("C")
                            and MIN_TCR_LEN <= len(s) <= MAX_TCR_LEN
                        ]
                        if binders:
                            self.l0_seeds[target] = binders
                except Exception:
                    pass

    def load_l0_from_dir(self, l0_dir: str) -> None:
        """Load L0 seeds from a directory of per-target text files.

        Each file is named <peptide>.txt with one CDR3b sequence per line.
        Merges with existing L0 seeds (deduplicates).
        """
        if not os.path.isdir(l0_dir):
            return
        for fname in os.listdir(l0_dir):
            if not fname.endswith(".txt"):
                continue
            target = fname[:-4]
            filepath = os.path.join(l0_dir, fname)
            with open(filepath) as f:
                seqs = [line.strip() for line in f if line.strip()]
            # Filter to valid TCR-like sequences
            valid = [
                s for s in seqs
                if is_valid_tcr(s)
                and s.startswith("C")
                and MIN_TCR_LEN <= len(s) <= MAX_TCR_LEN
            ]
            if not valid:
                continue
            if target in self.l0_seeds:
                # Merge and deduplicate
                existing = set(self.l0_seeds[target])
                for s in valid:
                    if s not in existing:
                        self.l0_seeds[target].append(s)
                        existing.add(s)
            else:
                self.l0_seeds[target] = valid

    def get_curriculum_weights(self, step: int) -> Tuple[float, float, float]:
        """Get L0/L1/L2 sampling weights for current training step."""
        for entry in self.curriculum_schedule:
            until = entry.get("until")
            if until is None or step < until:
                return entry["L0"], entry["L1"], entry["L2"]
        # Fallback to last entry
        last = self.curriculum_schedule[-1]
        return last["L0"], last["L1"], last["L2"]

    def get_online_pool_ratio(self, step: int) -> float:
        """Return current probability of sampling from the online target pool."""
        if not self.online_pool_enabled or step < self.online_pool_start_step:
            return 0.0
        progress = (step - self.online_pool_start_step) / self.online_pool_warmup_steps
        return self.online_pool_max_ratio * min(1.0, max(0.0, progress))

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
        online_ratio = self.get_online_pool_ratio(step)
        has_online = target in self.online_seeds and len(self._online_candidates(target, step)) > 0
        if has_online and self.rng.random() < online_ratio:
            return self._sample_online(target, step), "ONLINE"

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

    @staticmethod
    def _hamming_or_large(a: str, b: str) -> int:
        """Hamming distance for same-length TCRs; large distance otherwise."""
        if len(a) != len(b):
            return max(len(a), len(b))
        return sum(x != y for x, y in zip(a, b))

    def add_online_tcr(
        self,
        target: str,
        tcr: str,
        affinity: float,
        delta_affinity: float = 0.0,
        decoy_violation: float = 0.0,
    ) -> bool:
        """Add a discovered TCR to the per-target online pool."""
        if not self.online_pool_enabled:
            return False
        if not (
            target
            and tcr
            and is_valid_tcr(tcr)
            and tcr.startswith("C")
            and MIN_TCR_LEN <= len(tcr) <= MAX_TCR_LEN
        ):
            return False

        pool = self.online_seeds.setdefault(target, [])
        seen = self._online_seen.setdefault(target, set())
        if tcr in seen:
            return False

        incoming = {
            "tcr": tcr,
            "affinity": float(affinity),
            "delta_affinity": float(delta_affinity),
            "decoy_violation": float(decoy_violation),
        }
        incoming_score = float(incoming["affinity"])
        min_hamming = self.online_pool_min_hamming

        if min_hamming > 0:
            similar_indices = [
                i for i, item in enumerate(pool)
                if self._hamming_or_large(tcr, item["tcr"]) < min_hamming
            ]
            if similar_indices:
                worst_idx = min(
                    similar_indices,
                    key=lambda i: (
                        float(pool[i].get("affinity", -1e9)),
                        float(pool[i].get("delta_affinity", -1e9)),
                        -float(pool[i].get("decoy_violation", 1e9)),
                    ),
                )
                worst = pool[worst_idx]
                if incoming_score <= float(worst.get("affinity", -1e9)):
                    return False
                seen.discard(worst["tcr"])
                pool[worst_idx] = incoming
                seen.add(tcr)
                self._sort_online_pool(pool)
                return True

        seen.add(tcr)
        pool.append(incoming)
        self._sort_online_pool(pool)
        if len(pool) > self.online_pool_max_per_target:
            for removed in pool[self.online_pool_max_per_target:]:
                seen.discard(removed["tcr"])
            del pool[self.online_pool_max_per_target:]
        return True

    def _sort_online_pool(self, pool: List[dict]) -> None:
        """Sort online pool best first."""
        pool.sort(
            key=lambda item: (
                float(item.get("affinity", -1e9)),
                float(item.get("delta_affinity", -1e9)),
                -float(item.get("decoy_violation", 1e9)),
            ),
            reverse=True,
        )

    def load_online_pool_snapshot(self, path: str) -> int:
        """Load online seeds from a previous run snapshot."""
        if not path or not os.path.exists(path):
            return 0
        with open(path) as handle:
            snapshot = json.load(handle)
        targets = snapshot.get("targets", {})
        n_loaded = 0
        for target, payload in targets.items():
            entries = payload.get("entries", [])
            for entry in entries:
                tcr = entry.get("tcr")
                if not tcr:
                    continue
                if self.add_online_tcr(
                    target=target,
                    tcr=tcr,
                    affinity=float(entry.get("affinity", -1e9)),
                    delta_affinity=float(entry.get("delta_affinity", 0.0)),
                    decoy_violation=float(entry.get("decoy_violation", 0.0)),
                ):
                    n_loaded += 1
        return n_loaded

    def _online_elite_candidates(self, target: str) -> List[dict]:
        """Return high-affinity online seeds for preserve/polish replay."""
        pool = self.online_seeds.get(target, [])
        return [
            item for item in pool
            if float(item.get("affinity", -1e9)) >= self.online_pool_elite_min_affinity
        ]

    def _active_online_band(self, step: int) -> Optional[dict]:
        """Return the configured affinity band for online-pool sampling."""
        for band in self.online_pool_sample_bands:
            until = band.get("until")
            if until is None or step < int(until):
                return band
        if self.online_pool_sample_bands:
            return self.online_pool_sample_bands[-1]
        return None

    # ── Ladder-based adaptive band selection ──────────────────────────

    def record_episode_affinity(self, target: str, final_affinity: float) -> None:
        """Record a final affinity for adaptive / ladder band selection."""
        if target not in self.recent_affinities:
            self.recent_affinities[target] = deque(maxlen=self.recent_affinity_window)
            self.band_update_counters[target] = 0

        self.recent_affinities[target].append(float(final_affinity))
        self.band_update_counters[target] = self.band_update_counters.get(target, 0) + 1

        if self.band_update_counters[target] >= self.band_update_interval:
            self.cached_bands[target] = self._compute_ladder_band(target)
            self.band_update_counters[target] = 0

    def _compute_ladder_band(self, target: str) -> Optional[dict]:
        """Compute init TCR band one level below the model's current mean final affinity.

        Ladder rule (user-specified):
            mean final affinity < -2  → init from (<= -3)
            mean in [-2, -1)         → init from [-3, -2)
            mean in [-1,  0)         → init from [-2, -1)
            mean in [ 0,  1)         → init from [-1,  0)
            ... continuing upward in integer steps
        """
        recent = self.recent_affinities.get(target)
        if not recent or len(recent) == 0:
            return None

        mean_aff = float(np.mean(list(recent)))
        floor_level = int(np.floor(mean_aff))
        sample_max = float(floor_level)
        sample_min = sample_max - 1.0
        return {"min": sample_min, "max": sample_max, "mean_aff": mean_aff}

    def _seed_ladder_from_pool(self) -> None:
        """Seed recent_affinities from loaded pool so ladder bands are set from step 0.

        Uses the top half of each target's pool affinities as a proxy for
        the model's current performance on that target.
        """
        for target, pool in self.online_seeds.items():
            if not pool:
                continue
            affinities = sorted(
                [float(e.get("affinity", -1e9)) for e in pool], reverse=True
            )
            frontier = affinities[: max(1, len(affinities) // 2)]
            self.recent_affinities[target] = deque(
                frontier[: self.recent_affinity_window],
                maxlen=self.recent_affinity_window,
            )
            self.band_update_counters[target] = 0
            self.cached_bands[target] = self._compute_ladder_band(target)

    def _online_candidates(self, target: str, step: int) -> List[dict]:
        """Return online seeds that match the current sampling band.

        When dynamic_mode == "ladder", uses the per-target ladder band
        (one level below mean final affinity). Falls back to L2 (empty list)
        if no TCRs exist in the ladder band.
        """
        pool = self.online_seeds.get(target, [])
        if not pool:
            return []

        if self.online_pool_dynamic_mode == "ladder":
            band = self.cached_bands.get(target) or self._compute_ladder_band(target)
            if band is None:
                return pool

            min_aff = float(band["min"])
            max_aff = float(band["max"])
            candidates = [
                item for item in pool
                if min_aff <= float(item.get("affinity", -1e9)) < max_aff
            ]
            # Ladder: return empty if band has no candidates → caller falls back to L2
            return candidates

        # Default "band" mode: use step-based band config
        band = self._active_online_band(step)

        if band is None:
            default_min_affinity = -4.0
            return [
                item for item in pool
                if float(item.get("affinity", -1e9)) >= default_min_affinity
            ]

        min_aff = float(band.get("min", -1e9))
        max_aff = float(band.get("max", 1e9))
        candidates = [
            item for item in pool
            if min_aff <= float(item.get("affinity", -1e9)) < max_aff
        ]

        if candidates:
            return candidates

        below = [
            item for item in pool
            if float(item.get("affinity", -1e9)) < min_aff
        ]
        if below:
            return below
        return pool

    def _sample_online(self, target: str, step: int) -> str:
        """Sample from online per-target pool."""
        pool = self._online_candidates(target, step)
        if not pool:
            pool = self.online_seeds[target]
        elite = self._online_elite_candidates(target)
        if elite and self.rng.random() < self.online_pool_elite_ratio:
            pool = elite
        # Bias toward high-affinity entries within the active band.
        top_k = min(len(pool), max(1, int(np.sqrt(len(pool))) + 4))
        seq = pool[self.rng.integers(top_k)]["tcr"]
        return seq

    def get_online_pool_stats(self) -> Dict[str, int]:
        """Return online pool sizes by target."""
        return {target: len(pool) for target, pool in self.online_seeds.items()}

    def get_online_pool_snapshot(self, step: Optional[int] = None) -> dict:
        """Return a serializable snapshot of the online seed pool."""
        bands = getattr(self, "affinity_bands", None) or self.online_pool_sample_bands or []
        targets = {}
        total = 0
        for target, pool in sorted(self.online_seeds.items()):
            band_counts = {}
            for band in bands:
                name = band.get("name") or f"[{band.get('min')},{band.get('max')})"
                low = float(band.get("min", -1e9))
                high = float(band.get("max", 1e9))
                band_counts[name] = sum(
                    1 for item in pool
                    if low <= float(item.get("affinity", -1e9)) < high
                )
            below_bands = 0
            above_bands = 0
            if bands:
                min_band = min(float(b.get("min", -1e9)) for b in bands)
                max_band = max(float(b.get("max", 1e9)) for b in bands)
                below_bands = sum(1 for item in pool if float(item.get("affinity", -1e9)) < min_band)
                above_bands = sum(1 for item in pool if float(item.get("affinity", -1e9)) >= max_band)

            entries = []
            for rank, item in enumerate(pool, start=1):
                entries.append({
                    "rank": rank,
                    "tcr": item.get("tcr"),
                    "affinity": float(item.get("affinity", 0.0)),
                    "delta_affinity": float(item.get("delta_affinity", 0.0)),
                    "decoy_violation": float(item.get("decoy_violation", 0.0)),
                })

            recent_affinities = getattr(self, "recent_affinities", {}).get(target, [])
            cached_band = getattr(self, "cached_bands", {}).get(target)
            total += len(pool)
            targets[target] = {
                "size": len(pool),
                "best_affinity": float(pool[0].get("affinity", 0.0)) if pool else None,
                "worst_affinity": float(pool[-1].get("affinity", 0.0)) if pool else None,
                "band_counts": band_counts,
                "below_bands": below_bands,
                "above_bands": above_bands,
                "cached_dynamic_band": cached_band,
                "recent_affinities": [float(x) for x in recent_affinities],
                "entries": entries,
            }

        return {
            "step": int(step) if step is not None else None,
            "max_per_target": self.online_pool_max_per_target,
            "min_hamming": self.online_pool_min_hamming,
            "bands": bands,
            "total": total,
            "n_targets": len(targets),
            "targets": targets,
        }

    def write_online_pool_snapshot(self, path: str, step: Optional[int] = None) -> None:
        """Write the online seed pool snapshot as JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as handle:
            json.dump(self.get_online_pool_snapshot(step=step), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)

    def _sample_l0(self, target: str) -> str:
        """Sample from L0: known binder with random mutations."""
        seeds = self.l0_seeds[target]
        binder = seeds[self.rng.integers(len(seeds))]
        n_mutations = self.rng.integers(
            self.l0_mutation_range[0], self.l0_mutation_range[1] + 1
        )
        n_mutations = min(n_mutations, len(binder) - 1)
        # Preserve the conserved leading CDR3 cysteine; tFold scoring also
        # assumes CDR3-like input, so mutating it creates reward/state mismatch.
        return binder[0] + mutate_sequence(binder[1:], n_mutations, self.rng)

    def _sample_l1(self, target: str) -> str:
        """Sample from L1: pre-computed ERGO top-K."""
        seeds = self.l1_seeds[target]
        return seeds[self.rng.integers(len(seeds))]

    def _sample_l2(self) -> str:
        """Sample from L2: random TCRdb sequence."""
        return self.tcrdb_seqs[self.rng.integers(len(self.tcrdb_seqs))]

    def get_random_tcr(self) -> str:
        """Get a random TCRdb sequence (for general use)."""
        return self.tcrdb_seqs[self.rng.integers(len(self.tcrdb_seqs))]

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
