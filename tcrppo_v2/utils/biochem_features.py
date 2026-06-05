"""Biochemical features for amino acid sequences.

Provides charge and hydrophobicity statistics for TCR sequences.
"""

import numpy as np

# Charge properties: +1 for basic (K,R,H), -1 for acidic (D,E), 0 for others
AA_CHARGE = {
    'A': 0, 'R': +1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': +1, 'I': 0,
    'L': 0, 'K': +1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}

# Kyte-Doolittle hydrophobicity scale
AA_HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}


def compute_biochem_features(seq: str) -> np.ndarray:
    """Compute 8D biochemical feature vector for a protein sequence.
    
    Features (in order):
        1. Total charge (sum of all charges)
        2. Positive charge count (number of K,R,H)
        3. Negative charge count (number of D,E)
        4. Charge density (total_charge / length)
        5. Mean hydrophobicity
        6. Max hydrophobicity
        7. Min hydrophobicity
        8. Hydrophobicity std
    
    Args:
        seq: Amino acid sequence string.
    
    Returns:
        8D numpy array of features.
    """
    if not seq:
        return np.zeros(8, dtype=np.float32)
    
    charges = [AA_CHARGE.get(aa, 0) for aa in seq]
    hydros = [AA_HYDROPHOBICITY.get(aa, 0.0) for aa in seq]
    
    total_charge = sum(charges)
    pos_count = sum(1 for c in charges if c > 0)
    neg_count = sum(1 for c in charges if c < 0)
    charge_density = total_charge / len(seq)
    
    mean_hydro = np.mean(hydros)
    max_hydro = np.max(hydros)
    min_hydro = np.min(hydros)
    std_hydro = np.std(hydros) if len(hydros) > 1 else 0.0
    
    return np.array([
        total_charge,
        pos_count,
        neg_count,
        charge_density,
        mean_hydro,
        max_hydro,
        min_hydro,
        std_hydro,
    ], dtype=np.float32)


def compute_biochem_features_batch(seqs: list) -> np.ndarray:
    """Compute biochemical features for a batch of sequences.
    
    Args:
        seqs: List of amino acid sequence strings.
    
    Returns:
        [N, 8] numpy array of features.
    """
    return np.stack([compute_biochem_features(seq) for seq in seqs], axis=0)
