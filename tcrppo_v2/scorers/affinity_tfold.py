"""tFold V3.4 structure-aware TCR-pMHC binding scorer.

Two-tier scoring approach:
  1. Fast path: V3.4 classifier on cached tFold features (< 1ms per sample)
  2. Slow path: Full tFold pipeline via subprocess for cache misses (~1s per sample)

The V3.4 classifier (1.57M params) runs directly in the tcrppo_v2 process.
Feature extraction (735M tFold model) runs in a subprocess using the `tfold`
conda env to avoid dependency conflicts.

Feature cache: SQLite database keyed by (cdr3b, peptide, hla) tuple.
"""

import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from tcrppo_v2.scorers.base import BaseScorer

logger = logging.getLogger(__name__)

# Paths
TFOLD_ROOT = "/share/liuyutian/tfold"
TFOLD_PYTHON = "/home/liuyutian/server/miniconda3/envs/tfold/bin/python"
WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
    "tfold_feature_worker.py",
)
V34_WEIGHTS = os.path.join(
    TFOLD_ROOT, "TCR_PMHC_pred", "4_16", "weights", "best_v34.pth"
)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


# ============================================================================
# V3.4 Model — local copy of architecture (no tFold dependency)
# ============================================================================

# We inline the model architecture to avoid importing from the tfold env.
# This is identical to TCR_PMHC_pred/4_16/model.py but self-contained.

class RBFDistanceEncoding(nn.Module):
    """Encode pairwise Cα distances into RBF features."""

    def __init__(self, n_rbf: int = 16, cutoff: float = 50.0):
        super().__init__()
        self.n_rbf = n_rbf
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.width = (cutoff / (n_rbf - 1)) * 0.5

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.unsqueeze(-1)
        return torch.exp(-((d - self.centers) ** 2) / (2 * self.width**2))


class StructureCrossAttention(nn.Module):
    """Cross-attention between CDR3 residues and peptide residues."""

    def __init__(self, d_model=192, n_heads=4, n_rbf=16, pfea_dim=128, dropout=0.1):
        super().__init__()
        import math
        self._math = math
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.rbf = RBFDistanceEncoding(n_rbf=n_rbf, cutoff=50.0)
        self.dist_bias = nn.Linear(n_rbf, n_heads, bias=False)
        self.pfea_bias = nn.Sequential(
            nn.Linear(pfea_dim, pfea_dim // 2),
            nn.ReLU(),
            nn.Linear(pfea_dim // 2, n_heads, bias=False),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q_feats, kv_feats, ca_q, ca_kv, pfea, mask_q, mask_kv):
        import math
        B, Lq, D = q_feats.shape
        Lk = kv_feats.shape[1]
        H = self.n_heads
        dh = self.d_head

        q_feats = self.norm_q(q_feats)
        kv_feats = self.norm_kv(kv_feats)

        Q = self.W_q(q_feats).reshape(B, Lq, H, dh).permute(0, 2, 1, 3)
        K = self.W_k(kv_feats).reshape(B, Lk, H, dh).permute(0, 2, 1, 3)
        V = self.W_v(kv_feats).reshape(B, Lk, H, dh).permute(0, 2, 1, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dh)

        distances = torch.cdist(ca_q, ca_kv)
        rbf_feats = self.rbf(distances)
        dist_b = self.dist_bias(rbf_feats).permute(0, 3, 1, 2)
        attn = attn + dist_b

        pfea_b = self.pfea_bias(pfea).permute(0, 3, 1, 2)
        attn = attn + pfea_b

        kv_mask = mask_kv.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(~kv_mask, float("-inf"))
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, Lq, D)
        out = self.W_o(out)
        out = self.dropout(out)

        attended = q_feats + out
        attended = attended + self.ffn(attended)
        q_mask = mask_q.unsqueeze(-1)
        attended = attended * q_mask.float()
        return attended


class ClassifierV34Local(nn.Module):
    """V3.4 classifier — local copy without tFold dependencies."""

    def __init__(self, d_sfea=192, n_heads=4, n_rbf=16, pfea_dim=128,
                 n_attn_layers=2, mlp_hidden=256, dropout=0.1):
        super().__init__()
        self.cross_attn_beta = nn.ModuleList([
            StructureCrossAttention(d_sfea, n_heads, n_rbf, pfea_dim, dropout)
            for _ in range(n_attn_layers)
        ])
        self.cross_attn_alpha = nn.ModuleList([
            StructureCrossAttention(d_sfea, n_heads, n_rbf, pfea_dim, dropout)
            for _ in range(n_attn_layers)
        ])

        pool_dim = d_sfea * 2
        v33_dim = 448
        agg_dim = pool_dim * 2 + v33_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(agg_dim),
            nn.Linear(agg_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )
        self._count_params()

    def _count_params(self):
        self._n_params = sum(p.numel() for p in self.parameters())

    def _masked_pool(self, x, mask):
        mask_f = mask.unsqueeze(-1).float()
        x_masked = x * mask_f
        lengths = mask_f.sum(dim=1).clamp(min=1)
        mean_pool = x_masked.sum(dim=1) / lengths
        x_for_max = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_pool = x_for_max.max(dim=1)[0]
        max_pool = max_pool.masked_fill(max_pool == float("-inf"), 0.0)
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        sfea_cdr3b = batch["sfea_cdr3b"]
        sfea_cdr3a = batch["sfea_cdr3a"]
        sfea_pep = batch["sfea_pep"]
        ca_cdr3b = batch["ca_cdr3b"]
        ca_cdr3a = batch["ca_cdr3a"]
        ca_pep = batch["ca_pep"]
        pfea_bp = batch["pfea_cdr3b_pep"]
        pfea_ap = batch["pfea_cdr3a_pep"]
        v33_feat = batch["v33_feat"]
        len_cdr3b = batch["len_cdr3b"]
        len_cdr3a = batch["len_cdr3a"]
        len_pep = batch["len_pep"]

        device = sfea_cdr3b.device
        B = sfea_cdr3b.shape[0]
        Lb = sfea_cdr3b.shape[1]
        La = sfea_cdr3a.shape[1]
        Lp = sfea_pep.shape[1]

        mask_cdr3b = torch.arange(Lb, device=device).unsqueeze(0) < len_cdr3b.unsqueeze(1)
        mask_cdr3a = torch.arange(La, device=device).unsqueeze(0) < len_cdr3a.unsqueeze(1)
        mask_pep = torch.arange(Lp, device=device).unsqueeze(0) < len_pep.unsqueeze(1)

        h_beta = sfea_cdr3b
        for layer in self.cross_attn_beta:
            h_beta = layer(h_beta, sfea_pep, ca_cdr3b, ca_pep, pfea_bp, mask_cdr3b, mask_pep)

        h_alpha = sfea_cdr3a
        for layer in self.cross_attn_alpha:
            h_alpha = layer(h_alpha, sfea_pep, ca_cdr3a, ca_pep, pfea_ap, mask_cdr3a, mask_pep)

        pool_beta = self._masked_pool(h_beta, mask_cdr3b)
        pool_alpha = self._masked_pool(h_alpha, mask_cdr3a)

        aggregated = torch.cat([pool_beta, pool_alpha, v33_feat], dim=-1)
        gate_logits = self.classifier(aggregated).squeeze(-1)
        return gate_logits


# ============================================================================
# Feature Cache (SQLite-backed)
# ============================================================================

def _make_cache_key(cdr3b: str, peptide: str, hla: str) -> str:
    """Create a deterministic cache key from input triplet."""
    return f"{cdr3b}|{peptide}|{hla}"


class TFoldFeatureCache:
    """SQLite cache for tFold V3.4 extracted features.

    Key: (cdr3b, peptide, hla) → stored as "cdr3b|peptide|hla"
    Value: Padded feature dict serialized as bytes via torch.save.
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS features "
            "(cache_key TEXT PRIMARY KEY, data BLOB)"
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT data FROM features WHERE cache_key=?", (key,)
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        import io
        return torch.load(io.BytesIO(row[0]), map_location="cpu", weights_only=False)

    def get_batch(self, keys: List[str]) -> Dict[str, Dict]:
        import io
        results = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT cache_key, data FROM features WHERE cache_key IN ({placeholders})",
                chunk,
            ).fetchall()
            for k, blob in rows:
                results[k] = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        self.hits += len(results)
        self.misses += len(keys) - len(results)
        return results

    def put(self, key: str, features: Dict) -> None:
        import io
        buf = io.BytesIO()
        torch.save(features, buf)
        self._conn.execute(
            "INSERT OR REPLACE INTO features (cache_key, data) VALUES (?, ?)",
            (key, buf.getvalue()),
        )
        self._conn.commit()

    def put_batch(self, items: List[Tuple[str, Dict]]) -> None:
        import io
        rows = []
        for key, feats in items:
            buf = io.BytesIO()
            torch.save(feats, buf)
            rows.append((key, buf.getvalue()))
        self._conn.executemany(
            "INSERT OR REPLACE INTO features (cache_key, data) VALUES (?, ?)",
            rows,
        )
        self._conn.commit()

    def size(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]

    def close(self):
        self._conn.close()


# ============================================================================
# Main Scorer
# ============================================================================


class AffinityTFoldScorer(BaseScorer):
    """tFold V3.4 structure-aware TCR-pMHC binding scorer.

    Scores CDR3β-peptide binding using the pre-trained V3.4 classifier,
    which uses structure features from the tFold pipeline.

    Two-tier architecture:
      1. Feature cache (SQLite): Stores tFold-extracted features keyed by
         (cdr3b, peptide, hla). Cache hits avoid the expensive tFold pipeline.
      2. Feature extraction (subprocess): On cache miss, calls the tfold env
         subprocess to run the full 735M-param tFold pipeline.
      3. V3.4 classifier: Runs in-process on cached features. 1.57M params,
         < 1ms per batch of 256 samples.

    For RL training, most scoring calls will be cache hits after the initial
    exploration phase, since the CDR3 edit space is bounded.
    """

    def __init__(
        self,
        device: str = "cuda",
        cache_path: Optional[str] = None,
        default_hla: str = "HLA-A*02:01",
        gpu_id: int = 0,
        max_subprocess_batch: int = 32,
        server_socket_path: str = "/tmp/tfold_server.sock",
        cache_only: bool = False,
        cache_miss_score: float = 0.5,
    ):
        """Initialize tFold V3.4 scorer.

        Args:
            device: Torch device for the V3.4 classifier.
            cache_path: Path to SQLite feature cache. Default: data/tfold_feature_cache.db
            default_hla: Default HLA allele for scoring (most McPAS targets use A*02:01).
            gpu_id: GPU ID for the tFold feature extraction subprocess.
            max_subprocess_batch: Max samples per server request.
            server_socket_path: Path to tFold feature server Unix socket.
            cache_only: If True, never call tFold server; cache misses return None
                       (scored as cache_miss_score). Useful for fast training.
            cache_miss_score: Score to return for cache misses in cache_only mode.
                            Default 0.5 (neutral). Set to 0.0 for pessimistic.
        """
        self.device = device
        self.default_hla = default_hla
        self.gpu_id = gpu_id
        self.max_subprocess_batch = max_subprocess_batch
        self.server_socket_path = server_socket_path
        self.cache_only = cache_only
        self.cache_miss_score = cache_miss_score

        # Load V3.4 classifier
        self.model = self._load_classifier(device)

        # Feature cache
        if cache_path is None:
            cache_path = os.path.join(PROJECT_ROOT, "data", "tfold_feature_cache.db")
        self._cache = TFoldFeatureCache(cache_path)

        # Stats
        self._n_subprocess_calls = 0
        self._n_scored = 0

        logger.info(
            f"tFold V3.4 scorer initialized: {self.model._n_params:,} params, "
            f"cache={self._cache.size()} entries, device={device}"
        )

    def _load_classifier(self, device: str) -> ClassifierV34Local:
        """Load the pre-trained V3.4 classifier."""
        ckpt = torch.load(V34_WEIGHTS, map_location="cpu", weights_only=False)
        config = ckpt.get("model_config", {})

        model = ClassifierV34Local(
            d_sfea=config.get("d_sfea", 192),
            n_heads=config.get("n_heads", 4),
            n_rbf=config.get("n_rbf", 16),
            pfea_dim=config.get("pfea_dim", 128),
            n_attn_layers=config.get("n_attn_layers", 2),
            mlp_hidden=config.get("mlp_hidden", 256),
            dropout=config.get("dropout", 0.1),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        for p in model.parameters():
            p.requires_grad = False

        return model

    def _extract_features_server(
        self, requests: List[Dict[str, str]]
    ) -> List[Optional[Dict]]:
        """Request feature extraction from the persistent tFold server.

        The server must be running (scripts/tfold_feature_server.py).
        Falls back to returning None for all if the server is unavailable.

        Args:
            requests: List of {"cdr3b", "peptide", "hla"} dicts.

        Returns:
            List of feature dicts (or None for failed extractions).
        """
        import base64
        import io
        import socket
        import struct

        if not requests:
            return []

        self._n_subprocess_calls += 1
        sock_path = self.server_socket_path

        if not os.path.exists(sock_path):
            logger.warning(
                f"tFold server socket not found at {sock_path}. "
                f"Start the server with: {TFOLD_PYTHON} scripts/tfold_feature_server.py --socket {sock_path}"
            )
            return [None] * len(requests)

        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(600)  # 10 min timeout for large batches
            sock.connect(sock_path)

            # Send extract request
            request_json = json.dumps({
                "cmd": "extract",
                "samples": requests,
            }).encode("utf-8")
            header = struct.pack(">I", len(request_json))
            sock.sendall(header + request_json)

            # Receive response
            resp_header = b""
            while len(resp_header) < 4:
                chunk = sock.recv(4 - len(resp_header))
                if not chunk:
                    raise ConnectionError("Server closed connection")
                resp_header += chunk
            resp_len = struct.unpack(">I", resp_header)[0]

            resp_data = b""
            while len(resp_data) < resp_len:
                chunk = sock.recv(min(resp_len - len(resp_data), 65536))
                if not chunk:
                    raise ConnectionError("Server closed connection during response")
                resp_data += chunk

            response = json.loads(resp_data.decode("utf-8"))
            sock.close()

            if response.get("status") != "ok":
                logger.error(f"tFold server error: {response.get('error', 'unknown')}")
                return [None] * len(requests)

            # Decode features from base64
            features_b64 = response.get("features", [])
            errors = response.get("errors", [None] * len(features_b64))
            features = []

            for i, (fb64, err) in enumerate(zip(features_b64, errors)):
                if fb64 is None or err is not None:
                    features.append(None)
                    if err:
                        logger.warning(f"tFold extraction error for {requests[i]['cdr3b']}: {err}")
                else:
                    feat_bytes = base64.b64decode(fb64)
                    feat = torch.load(io.BytesIO(feat_bytes), map_location="cpu", weights_only=False)
                    features.append(feat)

            return features

        except (ConnectionRefusedError, FileNotFoundError):
            logger.warning(f"tFold server not running at {sock_path}")
            return [None] * len(requests)
        except socket.timeout:
            logger.error("tFold server request timed out (10 min)")
            return [None] * len(requests)
        except Exception as e:
            logger.error(f"tFold server communication error: {e}")
            return [None] * len(requests)

    def _get_features_batch(
        self,
        cdr3bs: List[str],
        peptides: List[str],
        hlas: Optional[List[str]] = None,
    ) -> List[Optional[Dict]]:
        """Get features for a batch, using cache where possible.

        Returns:
            List of feature dicts (padded, ready for classifier).
        """
        if hlas is None:
            hlas = [self.default_hla] * len(cdr3bs)

        # Build cache keys
        keys = [_make_cache_key(c, p, h) for c, p, h in zip(cdr3bs, peptides, hlas)]

        # Batch cache lookup
        cached = self._cache.get_batch(keys)

        # Find misses
        results = [None] * len(keys)
        need_extract = []  # (original_index, request_dict)

        for i, key in enumerate(keys):
            if key in cached:
                results[i] = cached[key]
            else:
                need_extract.append((i, {
                    "cdr3b": cdr3bs[i],
                    "peptide": peptides[i],
                    "hla": hlas[i],
                }))

        # Extract missing features via server (or skip in cache_only mode)
        if need_extract:
            if self.cache_only:
                logger.debug(
                    f"tFold cache miss (cache_only): {len(need_extract)}/{len(keys)} skipped"
                )
                # Leave results[i] = None for misses; handled by score_batch_fast()
                all_extracted = []
            else:
                logger.info(
                    f"tFold cache miss: {len(need_extract)}/{len(keys)} need extraction"
                )
                # Process in batches
                all_extracted = []
                for batch_start in range(0, len(need_extract), self.max_subprocess_batch):
                    batch = need_extract[batch_start : batch_start + self.max_subprocess_batch]
                    requests = [item[1] for item in batch]
                    extracted = self._extract_features_server(requests)
                    all_extracted.extend(zip(batch, extracted))

            # Store extracted features
            to_cache = []
            for (orig_idx, req), feats in all_extracted:
                if feats is not None:
                    results[orig_idx] = feats
                    key = keys[orig_idx]
                    to_cache.append((key, feats))

            if to_cache:
                self._cache.put_batch(to_cache)
                logger.info(f"Cached {len(to_cache)} new tFold features")

        return results

    @torch.no_grad()
    def _classify_batch(self, features_list: List[Dict]) -> torch.Tensor:
        """Run V3.4 classifier on a batch of pre-extracted features.

        Args:
            features_list: List of padded feature dicts.

        Returns:
            Tensor of binding scores (higher = more likely binding).
        """
        if not features_list:
            return torch.tensor([], device=self.device)

        # Stack into batch tensors
        batch = {}
        tensor_keys = [
            "sfea_cdr3b", "sfea_cdr3a", "sfea_pep",
            "ca_cdr3b", "ca_cdr3a", "ca_pep",
            "pfea_cdr3b_pep", "pfea_cdr3a_pep", "v33_feat",
        ]
        int_keys = ["len_cdr3b", "len_cdr3a", "len_pep"]

        for key in tensor_keys:
            batch[key] = torch.stack([f[key] for f in features_list]).to(self.device)
        for key in int_keys:
            batch[key] = torch.tensor(
                [f[key] for f in features_list], dtype=torch.long, device=self.device
            )

        gate_logits = self.model(batch)  # [B] — higher = more non-binding
        # Convert to binding score: higher = more binding
        binding_scores = -torch.sigmoid(gate_logits)

        return binding_scores

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score a single TCR-peptide pair.

        Args:
            tcr: CDR3beta sequence.
            peptide: Peptide sequence.

        Returns:
            (binding_score, confidence) where binding_score is in [-1, 0]
            (higher = more binding) and confidence is 1.0 (no uncertainty est).
        """
        hla = kwargs.get("hla", self.default_hla)
        features = self._get_features_batch([tcr], [peptide], [hla])

        if features[0] is None:
            # Feature extraction failed — return neutral score
            return 0.0, 0.0

        score = self._classify_batch([features[0]])
        self._n_scored += 1
        return float(score[0].item()), 1.0

    def score_batch(self, tcrs: list, peptides: list, **kwargs) -> Tuple[list, list]:
        """Score a batch of TCR-peptide pairs with confidence.

        Returns:
            (scores, confidences) — two lists of floats.
        """
        hlas = kwargs.get("hlas", [self.default_hla] * len(tcrs))
        features = self._get_features_batch(tcrs, peptides, hlas)

        # Separate successful and failed
        valid_features = []
        valid_indices = []
        for i, f in enumerate(features):
            if f is not None:
                valid_features.append(f)
                valid_indices.append(i)

        # Failed extractions get a low score (mapped 0.1) to discourage unscoreable sequences.
        # In cache_only mode, cache misses get cache_miss_score (default 0.5).
        default_raw = self.cache_miss_score - 1.0 if self.cache_only else -0.9
        scores = [default_raw] * len(tcrs)
        confidences = [0.0] * len(tcrs)

        if valid_features:
            binding_scores = self._classify_batch(valid_features)
            for j, orig_idx in enumerate(valid_indices):
                scores[orig_idx] = float(binding_scores[j].item())
                confidences[orig_idx] = 1.0

        self._n_scored += len(tcrs)
        return scores, confidences

    def score_batch_fast(self, tcrs: list, peptides: list) -> List[float]:
        """Fast batch scoring (no confidence estimation).

        For RL training reward, we map the binding score from [-1, 0] to [0, 1]
        to be consistent with ERGO's output range.

        Cache misses in cache_only mode return self.cache_miss_score (default 0.5).
        """
        scores, _ = self.score_batch(tcrs, peptides)
        # Map from [-1, 0] (raw) to [0, 1] (ERGO-compatible)
        # raw = -sigmoid(gate_logit), so raw is in [-1, 0]
        # mapped = raw + 1 → [0, 1], where 1 = strong binding
        # For cache misses (cache_only mode): raw = cache_miss_score - 1.0,
        # so mapped = raw + 1.0 = cache_miss_score. Correct by construction.
        return [s + 1.0 for s in scores]

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Return cache performance statistics."""
        return {
            "cache_size": self._cache.size(),
            "cache_hits": self._cache.hits,
            "cache_misses": self._cache.misses,
            "subprocess_calls": self._n_subprocess_calls,
            "total_scored": self._n_scored,
        }
