"""Tier 3 Sequence Analysis for generated TCRs.

Computes sequence-level metrics that don't require external ML models:
- Motif enrichment (CDR3 k-mer frequency vs TCRdb background)
- Distance to known binders (Levenshtein distance to VDJdb/tc-hard binders)
- Diversity metrics (unique sequences, mean pairwise distance)
- Length distribution comparison
- Amino acid composition comparison
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compute_kmer_frequencies(sequences: List[str], k: int = 3) -> Dict[str, float]:
    """Compute k-mer frequency distribution from sequences."""
    counts: Counter = Counter()
    total = 0
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            counts[kmer] += 1
            total += 1
    if total == 0:
        return {}
    return {kmer: count / total for kmer, count in counts.items()}


def compute_aa_composition(sequences: List[str]) -> Dict[str, float]:
    """Compute amino acid composition across sequences."""
    counts: Counter = Counter()
    total = 0
    for seq in sequences:
        for aa in seq:
            counts[aa] += 1
            total += 1
    if total == 0:
        return {}
    return {aa: count / total for aa, count in counts.items()}


def compute_length_distribution(sequences: List[str]) -> Dict[int, float]:
    """Compute length distribution of sequences."""
    if not sequences:
        return {}
    lengths = [len(s) for s in sequences]
    counts = Counter(lengths)
    total = len(sequences)
    return {length: count / total for length, count in sorted(counts.items())}


class SequenceAnalyzer:
    """Tier 3 sequence analysis for generated TCRs."""

    def __init__(
        self,
        tcrdb_path: Optional[str] = None,
        l0_seeds_dir: Optional[str] = None,
        background_sample_size: int = 10000,
        seed: int = 42,
    ):
        """Initialize with background TCRdb sequences and known binders.

        Args:
            tcrdb_path: Path to TCRdb data directory.
            l0_seeds_dir: Path to L0 seeds (known binders per target).
            background_sample_size: Number of TCRdb sequences for background stats.
            seed: Random seed.
        """
        self.rng = np.random.default_rng(seed)
        self.known_binders: Dict[str, List[str]] = {}

        # Load TCRdb background
        if tcrdb_path is None:
            tcrdb_path = "/share/liuyutian/TCRPPO/data/tcrdb"
        self.background_seqs = self._load_tcrdb_sample(tcrdb_path, background_sample_size)

        # Pre-compute background stats
        self.bg_kmer_freq = compute_kmer_frequencies(self.background_seqs, k=3)
        self.bg_aa_comp = compute_aa_composition(self.background_seqs)
        self.bg_length_dist = compute_length_distribution(self.background_seqs)

        # Load known binders
        if l0_seeds_dir is None:
            l0_seeds_dir = os.path.join(PROJECT_ROOT, "data", "l0_seeds_tchard")
        if os.path.isdir(l0_seeds_dir):
            self._load_known_binders(l0_seeds_dir)

    def _load_tcrdb_sample(self, tcrdb_path: str, n: int) -> List[str]:
        """Load random sample of TCRdb CDR3b sequences."""
        train_file = os.path.join(tcrdb_path, "train_uniq_tcr_seqs.txt")
        if not os.path.exists(train_file):
            return []
        seqs = []
        with open(train_file) as f:
            for line in f:
                seq = line.strip()
                if seq and 8 <= len(seq) <= 27:
                    seqs.append(seq)
        if len(seqs) > n:
            indices = self.rng.choice(len(seqs), size=n, replace=False)
            seqs = [seqs[i] for i in indices]
        return seqs

    def _load_known_binders(self, l0_dir: str) -> None:
        """Load known binders from L0 seed files."""
        for fname in os.listdir(l0_dir):
            if not fname.endswith(".txt"):
                continue
            target = fname[:-4]
            filepath = os.path.join(l0_dir, fname)
            with open(filepath) as f:
                seqs = [line.strip() for line in f if line.strip()]
            if seqs:
                self.known_binders[target] = seqs

    def analyze(
        self,
        generated_tcrs: List[str],
        target_peptide: str,
    ) -> Dict:
        """Run full Tier 3 analysis on generated TCRs for a target.

        Args:
            generated_tcrs: List of generated CDR3b sequences.
            target_peptide: Target peptide these TCRs were designed for.

        Returns:
            Dict with all analysis metrics.
        """
        results = {}

        # 1. Diversity metrics
        results["diversity"] = self._compute_diversity(generated_tcrs)

        # 2. Length distribution
        results["length_distribution"] = self._analyze_lengths(generated_tcrs)

        # 3. Amino acid composition
        results["aa_composition"] = self._analyze_aa_composition(generated_tcrs)

        # 4. K-mer enrichment
        results["kmer_enrichment"] = self._analyze_kmer_enrichment(generated_tcrs)

        # 5. Distance to known binders
        if target_peptide in self.known_binders:
            results["distance_to_binders"] = self._analyze_binder_distance(
                generated_tcrs, target_peptide
            )
        else:
            results["distance_to_binders"] = {"available": False}

        return results

    def _compute_diversity(self, tcrs: List[str]) -> Dict:
        """Compute diversity metrics."""
        n_total = len(tcrs)
        unique_tcrs = list(set(tcrs))
        n_unique = len(unique_tcrs)

        # Mean pairwise Levenshtein distance (sample if too many)
        if n_unique <= 100:
            sample = unique_tcrs
        else:
            idx = self.rng.choice(n_unique, size=100, replace=False)
            sample = [unique_tcrs[i] for i in idx]

        distances = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                distances.append(levenshtein_distance(sample[i], sample[j]))

        mean_dist = float(np.mean(distances)) if distances else 0.0

        return {
            "n_total": n_total,
            "n_unique": n_unique,
            "uniqueness_ratio": n_unique / max(n_total, 1),
            "mean_pairwise_levenshtein": mean_dist,
        }

    def _analyze_lengths(self, tcrs: List[str]) -> Dict:
        """Compare length distribution to TCRdb background."""
        gen_dist = compute_length_distribution(tcrs)
        gen_lengths = [len(s) for s in tcrs]

        return {
            "mean_length": float(np.mean(gen_lengths)) if gen_lengths else 0,
            "std_length": float(np.std(gen_lengths)) if gen_lengths else 0,
            "min_length": min(gen_lengths) if gen_lengths else 0,
            "max_length": max(gen_lengths) if gen_lengths else 0,
            "distribution": gen_dist,
            "bg_mean_length": float(np.mean([len(s) for s in self.background_seqs])) if self.background_seqs else 0,
        }

    def _analyze_aa_composition(self, tcrs: List[str]) -> Dict:
        """Compare amino acid composition to TCRdb background."""
        gen_comp = compute_aa_composition(tcrs)

        # KL divergence (gen || bg)
        aas = sorted(set(list(gen_comp.keys()) + list(self.bg_aa_comp.keys())))
        kl_div = 0.0
        for aa in aas:
            p = gen_comp.get(aa, 1e-10)
            q = self.bg_aa_comp.get(aa, 1e-10)
            if p > 0:
                kl_div += p * np.log(p / q)

        return {
            "generated": gen_comp,
            "background": self.bg_aa_comp,
            "kl_divergence": float(kl_div),
        }

    def _analyze_kmer_enrichment(self, tcrs: List[str], k: int = 3) -> Dict:
        """Find enriched k-mers compared to background."""
        gen_freq = compute_kmer_frequencies(tcrs, k=k)

        # Find top enriched k-mers (highest ratio of gen/bg)
        enrichments = {}
        for kmer, freq in gen_freq.items():
            bg_freq = self.bg_kmer_freq.get(kmer, 1e-6)
            enrichments[kmer] = freq / bg_freq

        # Sort by enrichment ratio
        top_enriched = sorted(enrichments.items(), key=lambda x: -x[1])[:20]
        top_depleted = sorted(enrichments.items(), key=lambda x: x[1])[:10]

        return {
            "top_enriched": [
                {"kmer": k, "enrichment": float(e), "gen_freq": float(gen_freq.get(k, 0)),
                 "bg_freq": float(self.bg_kmer_freq.get(k, 0))}
                for k, e in top_enriched
            ],
            "top_depleted": [
                {"kmer": k, "enrichment": float(e), "gen_freq": float(gen_freq.get(k, 0)),
                 "bg_freq": float(self.bg_kmer_freq.get(k, 0))}
                for k, e in top_depleted
            ],
        }

    def _analyze_binder_distance(
        self, tcrs: List[str], target: str
    ) -> Dict:
        """Compute distance between generated TCRs and known binders."""
        binders = self.known_binders[target]

        # For each generated TCR, find min distance to any known binder
        # Sample binders if too many
        if len(binders) > 200:
            idx = self.rng.choice(len(binders), size=200, replace=False)
            binder_sample = [binders[i] for i in idx]
        else:
            binder_sample = binders

        min_distances = []
        for tcr in tcrs:
            min_d = min(levenshtein_distance(tcr, b) for b in binder_sample)
            min_distances.append(min_d)

        # Also check for exact matches
        binder_set = set(binders)
        exact_matches = sum(1 for t in tcrs if t in binder_set)

        return {
            "available": True,
            "n_known_binders": len(binders),
            "mean_min_distance": float(np.mean(min_distances)),
            "median_min_distance": float(np.median(min_distances)),
            "min_distance": int(min(min_distances)) if min_distances else -1,
            "max_distance": int(max(min_distances)) if min_distances else -1,
            "exact_matches": exact_matches,
            "fraction_within_3_edits": float(
                sum(1 for d in min_distances if d <= 3) / len(min_distances)
            ) if min_distances else 0.0,
        }


def analyze_generated_tcrs(
    generated_tcrs: Dict[str, List[str]],
    tcrdb_path: Optional[str] = None,
    l0_seeds_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """Run Tier 3 analysis on generated TCRs for all targets.

    Args:
        generated_tcrs: Dict mapping target peptide -> list of generated CDR3b sequences.
        tcrdb_path: Path to TCRdb data.
        l0_seeds_dir: Path to L0 seed files.

    Returns:
        Dict mapping target -> analysis results.
    """
    analyzer = SequenceAnalyzer(tcrdb_path=tcrdb_path, l0_seeds_dir=l0_seeds_dir)

    results = {}
    for target, tcrs in generated_tcrs.items():
        results[target] = analyzer.analyze(tcrs, target)

    return results
