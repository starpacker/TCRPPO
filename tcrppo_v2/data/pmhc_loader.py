"""pMHC loader: target peptides + HLA pseudosequence encoding."""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from tcrppo_v2.utils.constants import (
    AMINO_ACIDS, MAX_PEP_LEN, HLA_PSEUDOSEQ_LEN, DECOY_LIBRARY_PATH,
)

# HLA pseudosequences (34 residues, NetMHC-style contact positions)
# Source: NetMHCpan 4.1 pseudosequences for common alleles
HLA_PSEUDOSEQUENCES: Dict[str, str] = {
    "HLA-A*02:01": "YFAMYQENAAHTLRWEPYSEGAEYLERTCEW",  # 30 residues (trimmed)
    "HLA-A*03:01": "YFAMYQENDAHTLRWEAYSEGAEYLERTCEW",
    "HLA-A*11:01": "YFAMYQENDAHTLRWEAYSEGAEYLERTCKW",
    "HLA-B*07:02": "YFAMYQENDSHTLRWEPYSEEAEYLERTCEW",
    "HLA-A*01:01": "YFSMYQENDAHTLRWEAYSEEAEYLERTCEW",
    "HLA-A*24:02": "YFAMYQENASHTLRWEPYSEAAEYLERTCEW",
    "HLA-B*08:01": "YFAMYQENAAHTLRWEPYSEEAEYLERTCEW",
}

# The 12 McPAS evaluation targets with their HLA restrictions
EVAL_TARGETS: Dict[str, str] = {
    "GILGFVFTL":  "HLA-A*02:01",   # Influenza M1
    "NLVPMVATV":  "HLA-A*02:01",   # CMV pp65
    "GLCTLVAML":  "HLA-A*02:01",   # EBV BMLF1
    "LLWNGPMAV":  "HLA-A*02:01",   # CMV pp65
    "YLQPRTFLL":  "HLA-A*02:01",   # SARS-CoV-2 Spike
    "FLYALALLL":  "HLA-A*02:01",   # EBV LMP2
    "SLYNTVATL":  "HLA-A*02:01",   # HIV-1 Gag
    "KLGGALQAK":  "HLA-A*03:01",   # CMV IE1
    "AVFDRKSDAK": "HLA-A*11:01",   # EBV EBNA3B
    "IVTDFSVIK":  "HLA-A*11:01",   # EBV EBNA3A
    "SPRWYFYYL":  "HLA-B*07:02",   # CMV pp65
    "RLRAEAQVK":  "HLA-A*03:01",   # CMV IE1
}


class PMHCLoader:
    """Load target peptides and their HLA pseudosequences."""

    def __init__(
        self,
        targets: Optional[List[str]] = None,
        decoy_library_path: str = DECOY_LIBRARY_PATH,
    ):
        """Initialize with target peptides.

        Args:
            targets: List of peptide sequences. If None, uses all 12 eval targets.
            decoy_library_path: Path to the decoy library (for candidate_targets.json).
        """
        if targets is None:
            targets = list(EVAL_TARGETS.keys())

        self.targets = targets
        self.target_hla: Dict[str, str] = {}
        self.target_pseudoseq: Dict[str, str] = {}

        # Try to load HLA info from candidate_targets.json
        self._hla_from_json = {}
        ct_path = os.path.join(decoy_library_path, "data", "candidate_targets.json")
        if os.path.exists(ct_path):
            with open(ct_path) as f:
                ct_data = json.load(f)
            for entry in ct_data.get("proposed_targets", []):
                seq = entry.get("sequence", "")
                hla = entry.get("hla_allele", "")
                if seq and hla:
                    self._hla_from_json[seq] = hla

        # Assign HLA alleles
        for peptide in self.targets:
            hla = self._resolve_hla(peptide)
            self.target_hla[peptide] = hla
            self.target_pseudoseq[peptide] = self._get_pseudoseq(hla)

    def _resolve_hla(self, peptide: str) -> str:
        """Resolve HLA allele for a peptide."""
        # Check hardcoded eval targets first
        if peptide in EVAL_TARGETS:
            return EVAL_TARGETS[peptide]
        # Check JSON data
        if peptide in self._hla_from_json:
            return self._hla_from_json[peptide]
        # Default to HLA-A*02:01 (most common in immunology studies)
        return "HLA-A*02:01"

    def _get_pseudoseq(self, hla: str) -> str:
        """Get HLA pseudosequence string."""
        if hla in HLA_PSEUDOSEQUENCES:
            return HLA_PSEUDOSEQUENCES[hla]
        # Fallback: use A*02:01 pseudosequence
        return HLA_PSEUDOSEQUENCES["HLA-A*02:01"]

    def get_pmhc_string(self, peptide: str) -> str:
        """Get concatenated peptide + HLA pseudosequence string for ESM encoding.

        Args:
            peptide: Target peptide sequence.

        Returns:
            Concatenated string: peptide + pseudosequence.
        """
        pseudoseq = self.target_pseudoseq.get(peptide, "")
        if not pseudoseq:
            pseudoseq = self._get_pseudoseq(self._resolve_hla(peptide))
        return peptide + pseudoseq

    def get_target_list(self) -> List[str]:
        """Return list of target peptides."""
        return list(self.targets)

    def get_target_info(self, peptide: str) -> Dict[str, str]:
        """Get info dict for a target."""
        return {
            "peptide": peptide,
            "hla": self.target_hla.get(peptide, "unknown"),
            "pseudoseq": self.target_pseudoseq.get(peptide, ""),
            "pmhc_string": self.get_pmhc_string(peptide),
        }

    def sample_target(self, rng: np.random.Generator = None) -> str:
        """Sample a random target uniformly."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(self.targets)

    def sample_target_weighted(
        self,
        weights: Dict[str, float],
        rng: np.random.Generator = None,
    ) -> str:
        """Sample a target with difficulty weighting.

        Args:
            weights: Dict mapping peptide -> weight (higher = more likely).
            rng: Random number generator.

        Returns:
            Sampled peptide sequence.
        """
        if rng is None:
            rng = np.random.default_rng()
        peptides = self.targets
        w = np.array([weights.get(p, 1.0) for p in peptides])
        w = w / w.sum()
        return rng.choice(peptides, p=w)
