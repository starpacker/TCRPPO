"""Lightweight TCR/pMHC encoder for GPU-free RL training.

Replaces ESM-2 with a trainable amino-acid embedding + BiLSTM encoder.
Same interface as ESMCache (encode_tcr, encode_tcr_batch, encode_pmhc, output_dim).

Key advantage: runs entirely on CPU, ~100x faster per sequence than ESM-2 on GPU.
Tradeoff: no pre-trained biochemical knowledge, but fine for RL exploration.

For pMHC encoding, uses pre-cached ESM-2 embeddings from disk (computed once
per target, ~163 targets total). Falls back to the same BiLSTM if no cache hit.
"""

import os
import string
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# Standard amino acid vocabulary (same as ESM)
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}  # 0 = padding
VOCAB_SIZE = len(AA_VOCAB) + 1  # +1 for padding token


def seq_to_indices(seq: str) -> List[int]:
    """Convert amino acid sequence to integer indices."""
    return [AA_TO_IDX.get(aa, 0) for aa in seq]


class BiLSTMEncoder(nn.Module):
    """Small BiLSTM sequence encoder."""

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        self.output_dim = output_dim

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Encode sequences to fixed-size vectors.

        Args:
            indices: LongTensor of shape [batch, seq_len].

        Returns:
            Tensor of shape [batch, output_dim].
        """
        emb = self.embedding(indices)  # [B, L, embed_dim]
        output, _ = self.lstm(emb)     # [B, L, hidden*2]
        # Mean pool over non-padding positions
        mask = (indices != 0).unsqueeze(-1).float()  # [B, L, 1]
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, hidden*2]
        return self.proj(pooled)  # [B, output_dim]


class LightweightEncoder:
    """Drop-in replacement for ESMCache using a small BiLSTM.

    Same interface: encode_tcr(), encode_tcr_batch(), encode_pmhc(), output_dim.
    Runs on CPU by default. No ESM-2 model loading required.
    """

    def __init__(
        self,
        device: str = "cpu",
        embed_dim: int = 64,
        hidden_dim: int = 128,
        encoder_output_dim: int = 256,
        tcr_cache_size: int = 4096,
        disk_cache_path: Optional[str] = None,
        **kwargs,  # Accept and ignore ESMCache-specific args
    ):
        self.device = device
        self.tcr_cache_size = tcr_cache_size

        # Initialize the BiLSTM encoder
        self.encoder = BiLSTMEncoder(embed_dim, hidden_dim, encoder_output_dim).to(device)
        self.encoder.eval()  # Start in eval mode (will be set to train during RL)

        self.embed_dim = encoder_output_dim  # Match ESMCache interface

        # In-memory caches (same as ESMCache)
        self._pmhc_cache: Dict[str, torch.Tensor] = {}
        self._tcr_cache: OrderedDict = OrderedDict()

        # Disk cache for pre-computed ESM-2 pMHC embeddings
        self._disk_cache = None
        if disk_cache_path is None:
            from tcrppo_v2.utils.constants import PROJECT_ROOT
            disk_cache_path = os.path.join(PROJECT_ROOT, "data", "esm_cache.db")

        if os.path.exists(disk_cache_path):
            try:
                from tcrppo_v2.utils.esm_cache import DiskEmbeddingCache
                self._disk_cache = DiskEmbeddingCache(disk_cache_path, 1280)
                # Note: disk cache has ESM-2 dim (1280), we'll project if needed
                self._esm_proj = None
                if encoder_output_dim != 1280:
                    self._esm_proj = nn.Linear(1280, encoder_output_dim).to(device)
            except Exception:
                self._disk_cache = None

        print(f"  Lightweight encoder: BiLSTM({embed_dim}→{hidden_dim}→{encoder_output_dim})")
        params = sum(p.numel() for p in self.encoder.parameters())
        print(f"  Parameters: {params:,} (vs ESM-2: 650M)")

    @property
    def output_dim(self) -> int:
        """Embedding dimension (matches ESMCache interface)."""
        return self.embed_dim

    def _encode_seqs(self, sequences: List[str]) -> torch.Tensor:
        """Encode sequences using the BiLSTM."""
        if not sequences:
            return torch.zeros(0, self.embed_dim, device=self.device)

        # Convert to padded tensor
        max_len = max(len(s) for s in sequences)
        indices = torch.zeros(len(sequences), max_len, dtype=torch.long, device=self.device)
        for i, seq in enumerate(sequences):
            idx = seq_to_indices(seq)
            indices[i, :len(idx)] = torch.tensor(idx, dtype=torch.long)

        with torch.no_grad():
            return self.encoder(indices)

    def encode_tcr(self, tcr_seq: str) -> torch.Tensor:
        """Encode a single TCR with caching."""
        if tcr_seq in self._tcr_cache:
            self._tcr_cache.move_to_end(tcr_seq)
            return self._tcr_cache[tcr_seq]

        embedding = self._encode_seqs([tcr_seq])[0]

        if len(self._tcr_cache) >= self.tcr_cache_size:
            self._tcr_cache.popitem(last=False)
        self._tcr_cache[tcr_seq] = embedding
        return embedding

    def encode_tcr_batch(self, tcr_seqs: List[str]) -> torch.Tensor:
        """Encode multiple TCRs with caching."""
        results = {}
        need_compute = []

        for i, seq in enumerate(tcr_seqs):
            if seq in self._tcr_cache:
                self._tcr_cache.move_to_end(seq)
                results[i] = self._tcr_cache[seq]
            else:
                need_compute.append((i, seq))

        if need_compute:
            seqs = [s for _, s in need_compute]
            new_embs = self._encode_seqs(seqs)
            for j, (orig_i, seq) in enumerate(need_compute):
                emb = new_embs[j]
                if len(self._tcr_cache) >= self.tcr_cache_size:
                    self._tcr_cache.popitem(last=False)
                self._tcr_cache[seq] = emb
                results[orig_i] = emb

        ordered = [results[i] for i in range(len(tcr_seqs))]
        return torch.stack(ordered)

    def encode_pmhc(self, pmhc_string: str) -> torch.Tensor:
        """Encode a pMHC string. Uses cached ESM-2 embeddings if available."""
        if pmhc_string in self._pmhc_cache:
            return self._pmhc_cache[pmhc_string]

        # Try disk cache (pre-computed ESM-2 embeddings)
        if self._disk_cache is not None:
            cached_np = self._disk_cache.get(pmhc_string)
            if cached_np is not None:
                emb = torch.from_numpy(cached_np).to(self.device)
                if self._esm_proj is not None:
                    with torch.no_grad():
                        emb = self._esm_proj(emb)
                self._pmhc_cache[pmhc_string] = emb
                return emb

        # Fallback: encode with BiLSTM
        embedding = self._encode_seqs([pmhc_string])[0]
        self._pmhc_cache[pmhc_string] = embedding
        return embedding
