"""Sequence encoding utilities extracted from v1 data_utils.py."""

import numpy as np
from typing import List, Tuple

from tcrppo_v2.utils.constants import (
    AMINO_ACIDS, AA_TO_IDX, IDX_TO_AA,
    ERGO_AA_TO_IDX, ERGO_IDX_TO_AA,
)


def seq2num(sequences: List[str], max_len: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Convert AA sequences to 1-indexed integer arrays (0 = PAD)."""
    if max_len == 0:
        max_len = max(len(seq) for seq in sequences)
    arrays = np.zeros((len(sequences), max_len), dtype=np.int64)
    lengths = np.zeros(len(sequences), dtype=np.int64)
    for i, seq in enumerate(sequences):
        lengths[i] = len(seq)
        for j, aa in enumerate(seq):
            arrays[i, j] = ERGO_AA_TO_IDX[aa]
    return arrays, lengths


def num2seq(sequences: np.ndarray) -> List[str]:
    """Convert 1-indexed integer arrays back to AA strings."""
    result = []
    for i in range(sequences.shape[0]):
        seq = "".join(
            ERGO_IDX_TO_AA[idx.item()]
            for idx in sequences[i, :]
            if idx.item() != 0
        )
        result.append(seq)
    return result


def is_valid_tcr(seq: str) -> bool:
    """Check if a sequence contains only valid amino acids."""
    return all(aa in AA_TO_IDX for aa in seq)


def random_aa_sequence(length: int, rng: np.random.Generator = None) -> str:
    """Generate a random amino acid sequence of given length."""
    if rng is None:
        rng = np.random.default_rng()
    return "".join(rng.choice(AMINO_ACIDS, size=length))


def mutate_sequence(seq: str, n_mutations: int, rng: np.random.Generator = None) -> str:
    """Apply n random point mutations to a sequence."""
    if rng is None:
        rng = np.random.default_rng()
    seq_list = list(seq)
    positions = rng.choice(len(seq_list), size=min(n_mutations, len(seq_list)), replace=False)
    for pos in positions:
        original = seq_list[pos]
        candidates = [aa for aa in AMINO_ACIDS if aa != original]
        seq_list[pos] = rng.choice(candidates)
    return "".join(seq_list)


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (1.0 = identical)."""
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    # DP for edit distance
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return 1.0 - dp[n] / max_len
