"""Frozen ESM-2 inference with per-sequence caching."""

import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ESMCache:
    """Frozen ESM-2 encoder with LRU caching for TCR and pMHC embeddings.

    TCR embeddings are cached with LRU eviction (sequences change per step).
    pMHC embeddings are cached permanently (computed once per target).
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = "cuda",
        tcr_cache_size: int = 4096,
        frozen: bool = True,
    ):
        """Initialize ESM-2 model and caches.

        Args:
            model_name: ESM model identifier.
            device: Torch device.
            tcr_cache_size: Max entries in TCR embedding LRU cache.
            frozen: Whether to freeze ESM parameters.
        """
        import esm

        self.device = device
        self.tcr_cache_size = tcr_cache_size

        # Load ESM-2
        if model_name == "esm2_t33_650M_UR50D":
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        else:
            raise ValueError(f"Unsupported ESM model: {model_name}")

        self.model = self.model.to(device)
        self.model.eval()

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        self.batch_converter = self.alphabet.get_batch_converter()
        self.embed_dim = self.model.embed_dim  # 1280 for esm2_t33

        # Caches
        self._pmhc_cache: Dict[str, torch.Tensor] = {}
        self._tcr_cache: OrderedDict = OrderedDict()

    @property
    def output_dim(self) -> int:
        """Embedding dimension of ESM-2 outputs."""
        return self.embed_dim

    @torch.no_grad()
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode a single sequence, returning mean-pooled embedding.

        Args:
            sequence: Amino acid sequence string.

        Returns:
            Tensor of shape [embed_dim].
        """
        data = [("seq", sequence)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        results = self.model(tokens, repr_layers=[33], return_contacts=False)
        # Shape: [1, seq_len+2, embed_dim] (includes BOS/EOS tokens)
        token_repr = results["representations"][33]

        # Mean pool over non-special tokens (skip BOS at 0 and EOS at end)
        seq_len = len(sequence)
        embedding = token_repr[0, 1 : seq_len + 1, :].mean(dim=0)
        return embedding

    @torch.no_grad()
    def encode_sequences_batch(self, sequences: List[str]) -> torch.Tensor:
        """Encode multiple sequences, returning mean-pooled embeddings.

        Args:
            sequences: List of amino acid sequence strings.

        Returns:
            Tensor of shape [N, embed_dim].
        """
        if not sequences:
            return torch.zeros(0, self.embed_dim, device=self.device)

        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        results = self.model(tokens, repr_layers=[33], return_contacts=False)
        token_repr = results["representations"][33]

        embeddings = []
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            emb = token_repr[i, 1 : seq_len + 1, :].mean(dim=0)
            embeddings.append(emb)

        return torch.stack(embeddings)

    def encode_tcr(self, tcr_seq: str) -> torch.Tensor:
        """Encode a TCR with LRU caching.

        Args:
            tcr_seq: CDR3beta sequence.

        Returns:
            Tensor of shape [embed_dim].
        """
        if tcr_seq in self._tcr_cache:
            self._tcr_cache.move_to_end(tcr_seq)
            return self._tcr_cache[tcr_seq]

        embedding = self.encode_sequence(tcr_seq)

        # LRU eviction
        if len(self._tcr_cache) >= self.tcr_cache_size:
            self._tcr_cache.popitem(last=False)

        self._tcr_cache[tcr_seq] = embedding
        return embedding

    def encode_pmhc(self, pmhc_string: str) -> torch.Tensor:
        """Encode a pMHC string with permanent caching.

        Args:
            pmhc_string: Concatenated peptide + HLA pseudosequence.

        Returns:
            Tensor of shape [embed_dim].
        """
        if pmhc_string in self._pmhc_cache:
            return self._pmhc_cache[pmhc_string]

        embedding = self.encode_sequence(pmhc_string)
        self._pmhc_cache[pmhc_string] = embedding
        return embedding

    def encode_tcr_batch(self, tcr_seqs: List[str]) -> torch.Tensor:
        """Encode multiple TCRs, using cache where possible.

        Args:
            tcr_seqs: List of CDR3beta sequences.

        Returns:
            Tensor of shape [N, embed_dim].
        """
        results = []
        uncached_indices = []
        uncached_seqs = []

        for i, seq in enumerate(tcr_seqs):
            if seq in self._tcr_cache:
                self._tcr_cache.move_to_end(seq)
                results.append((i, self._tcr_cache[seq]))
            else:
                uncached_indices.append(i)
                uncached_seqs.append(seq)

        if uncached_seqs:
            new_embeddings = self.encode_sequences_batch(uncached_seqs)
            for j, idx in enumerate(uncached_indices):
                emb = new_embeddings[j]
                seq = uncached_seqs[j]
                # Cache
                if len(self._tcr_cache) >= self.tcr_cache_size:
                    self._tcr_cache.popitem(last=False)
                self._tcr_cache[seq] = emb
                results.append((idx, emb))

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return torch.stack([r[1] for r in results])

    def clear_tcr_cache(self) -> None:
        """Clear TCR embedding cache."""
        self._tcr_cache.clear()

    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self._tcr_cache.clear()
        self._pmhc_cache.clear()

    @property
    def tcr_cache_size_current(self) -> int:
        """Current number of cached TCR embeddings."""
        return len(self._tcr_cache)

    @property
    def pmhc_cache_size_current(self) -> int:
        """Current number of cached pMHC embeddings."""
        return len(self._pmhc_cache)
