"""pMHC loader: target peptides + HLA pseudosequence encoding.

Loads target peptides from tc-hard dataset (163 MHCI peptides with CDR3b data)
plus decoy library metadata. Supports eval-only (12 McPAS) and full training modes.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from tcrppo_v2.utils.constants import (
    AMINO_ACIDS, MAX_PEP_LEN, HLA_PSEUDOSEQ_LEN, DECOY_LIBRARY_PATH,
    PROJECT_ROOT,
)

# HLA pseudosequences (NetMHC-style contact positions)
# Source: NetMHCpan 4.1 pseudosequences for common alleles
HLA_PSEUDOSEQUENCES: Dict[str, str] = {
    "HLA-A*02:01": "YFAMYQENAAHTLRWEPYSEGAEYLERTCEW",
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

# Path to tc-hard extracted targets JSON
TC_HARD_TARGETS_PATH = os.path.join(PROJECT_ROOT, "data", "tc_hard_targets.json")


def load_tc_hard_targets(path: str = TC_HARD_TARGETS_PATH) -> Dict[str, str]:
    """Load peptide -> HLA mapping from tc-hard extracted data.

    Returns dict of {peptide: hla_allele} for all 163 MHCI peptides
    with >=10 unique CDR3b sequences in tc-hard.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {pep: info["hla"] for pep, info in data.items()}


# Legacy alias for backward compatibility with tests
TRAINING_TARGETS: Dict[str, str] = {
    pep: hla for pep, hla in load_tc_hard_targets().items()
    if pep not in EVAL_TARGETS
}


class PMHCLoader:
    """Load target peptides and their HLA pseudosequences."""

    def __init__(
        self,
        targets: Optional[List[str]] = None,
        decoy_library_path: str = DECOY_LIBRARY_PATH,
        mode: str = "eval",
        tc_hard_path: str = TC_HARD_TARGETS_PATH,
    ):
        """Initialize with target peptides.

        Args:
            targets: List of peptide sequences. If None, determined by mode.
            decoy_library_path: Path to the decoy library (for candidate_targets.json).
            mode: "eval" for 12 eval targets only,
                  "train" for all tc-hard targets (163 peptides).
            tc_hard_path: Path to tc-hard extracted targets JSON.
        """
        if targets is None:
            if mode == "train":
                tc_hard = load_tc_hard_targets(tc_hard_path)
                # All tc-hard targets (includes eval targets)
                all_targets = dict(EVAL_TARGETS)
                all_targets.update(tc_hard)
                targets = list(all_targets.keys())
            else:
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
        if peptide in EVAL_TARGETS:
            return EVAL_TARGETS[peptide]
        if peptide in TRAINING_TARGETS:
            return TRAINING_TARGETS[peptide]
        # Check tc-hard data loaded at init time
        tc_hard = load_tc_hard_targets()
        if peptide in tc_hard:
            return tc_hard[peptide]
        if peptide in self._hla_from_json:
            return self._hla_from_json[peptide]
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
        return self.targets[rng.integers(len(self.targets))]

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
