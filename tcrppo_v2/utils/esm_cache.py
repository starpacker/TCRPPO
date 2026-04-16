"""Frozen ESM-2 inference with per-sequence caching.

Two-tier caching strategy:
  1. In-memory LRU cache (fast, limited size)
  2. Persistent disk cache via sqlite3 (slower, unlimited, survives restarts)

On cache miss: check disk → compute with ESM-2 → store in both caches.
"""

import os
import sqlite3
import struct
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class DiskEmbeddingCache:
    """SQLite-backed persistent cache for ESM-2 embeddings.

    Stores embeddings as raw bytes (float32) keyed by sequence string.
    Thread-safe via WAL mode. Supports batch lookups.
    """

    def __init__(self, db_path: str, embed_dim: int = 1280):
        self.db_path = db_path
        self.embed_dim = embed_dim
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(seq TEXT PRIMARY KEY, emb BLOB)"
        )
        self._conn.commit()
        self._n_bytes = embed_dim * 4  # float32

    def get(self, seq: str) -> Optional[np.ndarray]:
        """Lookup a single embedding. Returns float32 ndarray or None."""
        row = self._conn.execute(
            "SELECT emb FROM embeddings WHERE seq=?", (seq,)
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def get_batch(self, seqs: List[str]) -> Dict[str, np.ndarray]:
        """Batch lookup. Returns dict of seq→ndarray for found entries."""
        if not seqs:
            return {}
        # SQLite IN clause with parameterized placeholders
        results = {}
        # Process in chunks of 500 to avoid SQLite variable limit
        for i in range(0, len(seqs), 500):
            chunk = seqs[i : i + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT seq, emb FROM embeddings WHERE seq IN ({placeholders})",
                chunk,
            ).fetchall()
            for seq, blob in rows:
                results[seq] = np.frombuffer(blob, dtype=np.float32).copy()
        return results

    def put(self, seq: str, emb: np.ndarray) -> None:
        """Store a single embedding."""
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (seq, emb) VALUES (?, ?)",
            (seq, emb.astype(np.float32).tobytes()),
        )
        self._conn.commit()

    def put_batch(self, items: List[Tuple[str, np.ndarray]]) -> None:
        """Store multiple embeddings in one transaction."""
        if not items:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (seq, emb) VALUES (?, ?)",
            [(seq, emb.astype(np.float32).tobytes()) for seq, emb in items],
        )
        self._conn.commit()

    def size(self) -> int:
        """Number of cached embeddings."""
        row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return row[0]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


class ESMCache:
    """Frozen ESM-2 encoder with two-tier caching.

    Tier 1: In-memory LRU cache (fast, up to tcr_cache_size entries)
    Tier 2: Persistent SQLite disk cache (survives restarts, unlimited)

    Lookup order: memory → disk → ESM-2 compute.
    pMHC embeddings are cached permanently in memory (small number of targets).
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = "cuda",
        tcr_cache_size: int = 4096,
        frozen: bool = True,
        disk_cache_path: Optional[str] = None,
    ):
        """Initialize ESM-2 model and caches.

        Args:
            model_name: ESM model identifier.
            device: Torch device.
            tcr_cache_size: Max entries in TCR embedding LRU cache.
            frozen: Whether to freeze ESM parameters.
            disk_cache_path: Path to SQLite cache file. If None, uses
                data/esm_cache.db relative to project root.
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

        # In-memory caches
        self._pmhc_cache: Dict[str, torch.Tensor] = {}
        self._tcr_cache: OrderedDict = OrderedDict()

        # Persistent disk cache
        if disk_cache_path is None:
            from tcrppo_v2.utils.constants import PROJECT_ROOT
            disk_cache_path = os.path.join(PROJECT_ROOT, "data", "esm_cache.db")
        self._disk_cache = DiskEmbeddingCache(disk_cache_path, self.embed_dim)
        self._disk_hits = 0
        self._disk_misses = 0

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
        """Encode a TCR with two-tier caching (memory → disk → compute).

        Args:
            tcr_seq: CDR3beta sequence.

        Returns:
            Tensor of shape [embed_dim].
        """
        # Tier 1: memory cache
        if tcr_seq in self._tcr_cache:
            self._tcr_cache.move_to_end(tcr_seq)
            return self._tcr_cache[tcr_seq]

        # Tier 2: disk cache
        cached_np = self._disk_cache.get(tcr_seq)
        if cached_np is not None:
            self._disk_hits += 1
            embedding = torch.from_numpy(cached_np).to(self.device)
            if len(self._tcr_cache) >= self.tcr_cache_size:
                self._tcr_cache.popitem(last=False)
            self._tcr_cache[tcr_seq] = embedding
            return embedding

        # Tier 3: compute
        self._disk_misses += 1
        embedding = self.encode_sequence(tcr_seq)

        # Store in both caches
        if len(self._tcr_cache) >= self.tcr_cache_size:
            self._tcr_cache.popitem(last=False)
        self._tcr_cache[tcr_seq] = embedding
        self._disk_cache.put(tcr_seq, embedding.cpu().numpy())
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
        """Encode multiple TCRs, using two-tier cache where possible.

        Lookup order: memory → disk → ESM-2 batch compute.

        Args:
            tcr_seqs: List of CDR3beta sequences.

        Returns:
            Tensor of shape [N, embed_dim].
        """
        results = []
        need_disk_check = []  # (original_index, seq)
        need_compute = []     # (original_index, seq)

        # Phase 1: check memory cache
        for i, seq in enumerate(tcr_seqs):
            if seq in self._tcr_cache:
                self._tcr_cache.move_to_end(seq)
                results.append((i, self._tcr_cache[seq]))
            else:
                need_disk_check.append((i, seq))

        # Phase 2: batch disk lookup for memory misses
        if need_disk_check:
            disk_seqs = [seq for _, seq in need_disk_check]
            disk_results = self._disk_cache.get_batch(disk_seqs)
            for idx, seq in need_disk_check:
                if seq in disk_results:
                    self._disk_hits += 1
                    emb = torch.from_numpy(disk_results[seq]).to(self.device)
                    if len(self._tcr_cache) >= self.tcr_cache_size:
                        self._tcr_cache.popitem(last=False)
                    self._tcr_cache[seq] = emb
                    results.append((idx, emb))
                else:
                    need_compute.append((idx, seq))

        # Phase 3: batch ESM-2 compute for disk misses
        if need_compute:
            self._disk_misses += len(need_compute)
            compute_seqs = [seq for _, seq in need_compute]
            new_embeddings = self.encode_sequences_batch(compute_seqs)
            disk_items = []
            for j, (idx, seq) in enumerate(need_compute):
                emb = new_embeddings[j]
                if len(self._tcr_cache) >= self.tcr_cache_size:
                    self._tcr_cache.popitem(last=False)
                self._tcr_cache[seq] = emb
                disk_items.append((seq, emb.cpu().numpy()))
                results.append((idx, emb))
            # Batch write to disk
            self._disk_cache.put_batch(disk_items)

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return torch.stack([r[1] for r in results])

    def clear_tcr_cache(self) -> None:
        """Clear in-memory TCR embedding cache (disk cache persists)."""
        self._tcr_cache.clear()

    def clear_all_caches(self) -> None:
        """Clear all in-memory caches (disk cache persists)."""
        self._tcr_cache.clear()
        self._pmhc_cache.clear()

    @property
    def tcr_cache_size_current(self) -> int:
        """Current number of in-memory cached TCR embeddings."""
        return len(self._tcr_cache)

    @property
    def pmhc_cache_size_current(self) -> int:
        """Current number of cached pMHC embeddings."""
        return len(self._pmhc_cache)

    @property
    def disk_cache_size(self) -> int:
        """Number of embeddings in persistent disk cache."""
        return self._disk_cache.size()

    @property
    def disk_cache_stats(self) -> Dict[str, int]:
        """Disk cache hit/miss statistics."""
        return {
            "hits": self._disk_hits,
            "misses": self._disk_misses,
            "total": self._disk_cache.size(),
        }
