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
INVALID_FEATURE_SENTINEL = {"_invalid": True}

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

    def delete_batch(self, keys: List[str]) -> None:
        if not keys:
            return
        self._conn.executemany(
            "DELETE FROM features WHERE cache_key=?",
            [(key,) for key in keys],
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
    ):
        """Initialize tFold V3.4 scorer.

        Args:
            device: Torch device for the V3.4 classifier.
            cache_path: Path to SQLite feature cache. Default: data/tfold_feature_cache.db
            default_hla: Default HLA allele for scoring (most McPAS targets use A*02:01).
            gpu_id: GPU ID for the tFold feature extraction subprocess.
            max_subprocess_batch: Max samples per server request.
            server_socket_path: Path to tFold feature server Unix socket.
        """
        self.device = device
        self.default_hla = default_hla
        self.gpu_id = gpu_id
        self.max_subprocess_batch = max_subprocess_batch
        self.server_socket_path = server_socket_path

        # Load V3.4 classifier
        self.model = self._load_classifier(device)

        # Feature cache
        if cache_path is None:
            cache_path = os.path.join(PROJECT_ROOT, "data", "tfold_feature_cache.db")
        self._cache = TFoldFeatureCache(cache_path)

        # Stats
        self._n_subprocess_calls = 0
        self._n_scored = 0
        # tFold classifier predicts a gate logit where larger means more non-binding.
        # We expose the pre-sigmoid binding logit (-gate_logit) as the affinity reward.
        self._default_raw_score = -20.0
        self._last_batch_trace: List[Dict[str, object]] = []

        logger.info(
            f"tFold V3.4 scorer initialized: {self.model._n_params:,} params, "
            f"cache={self._cache.size()} entries, device={device}"
        )

    def _log_prediction_trace(
        self,
        trace_rows: List[Dict[str, object]],
        scores: List[float],
        confidences: List[float],
        total_elapsed_s: float,
    ) -> None:
        if not trace_rows:
            return

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        end_to_end_ms = (total_elapsed_s * 1000.0) / max(len(trace_rows), 1)
        for idx, trace in enumerate(trace_rows):
            score = scores[idx] if idx < len(scores) else self._default_raw_score
            confidence = confidences[idx] if idx < len(confidences) else 0.0
            print(
                f"[tFoldScore] ts={ts} source={trace.get('source', 'unknown')} "
                f"path_ms={float(trace.get('path_ms', 0.0)):.2f} "
                f"classify_ms={float(trace.get('classify_ms', 0.0)):.2f} "
                f"end_to_end_ms={end_to_end_ms:.2f} affinity_logit={score:.4f} "
                f"conf={confidence:.2f} cdr3b={trace.get('cdr3b', '')[:18]} "
                f"peptide={trace.get('peptide', '')} hla={trace.get('hla', self.default_hla)}",
                flush=True,
            )

    @staticmethod
    def _feature_is_valid(features: Optional[Dict]) -> bool:
        if features is None:
            return False
        if features.get("_invalid") is True:
            return False

        tensor_keys = [
            "sfea_cdr3b", "sfea_cdr3a", "sfea_pep",
            "ca_cdr3b", "ca_cdr3a", "ca_pep",
            "pfea_cdr3b_pep", "pfea_cdr3a_pep", "v33_feat",
        ]
        int_keys = ["len_cdr3b", "len_cdr3a", "len_pep"]

        try:
            for key in tensor_keys:
                value = features.get(key)
                if value is None or not torch.isfinite(value).all():
                    return False
            for key in int_keys:
                value = int(features.get(key, 0))
                if value <= 0:
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def _feature_is_invalid_sentinel(features: Optional[Dict]) -> bool:
        return isinstance(features, dict) and features.get("_invalid") is True

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
            print(f"[tFold] Connecting to server at {sock_path}...", flush=True)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1800)  # 30 min timeout for cold-start structure predictions
            sock.connect(sock_path)
            print(f"[tFold] Connected. Sending {len(requests)} extraction requests...", flush=True)

            # Send extract request
            request_json = json.dumps({
                "cmd": "extract",
                "samples": requests,
            }).encode("utf-8")
            header = struct.pack(">I", len(request_json))
            sock.sendall(header + request_json)
            print(f"[tFold] Request sent. Waiting for response (timeout=30min)...", flush=True)

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
            print(f"[tFold] Response received. Status: {response.get('status')}", flush=True)

            if response.get("status") != "ok":
                logger.error(f"tFold server error: {response.get('error', 'unknown')}")
                print(f"[tFold] ERROR: {response.get('error', 'unknown')}", flush=True)
                return [None] * len(requests)

            # Decode features from base64
            features_b64 = response.get("features", [])
            errors = response.get("errors", [None] * len(features_b64))
            features = []

            success_count = 0
            for i, (fb64, err) in enumerate(zip(features_b64, errors)):
                if fb64 is None or err is not None:
                    features.append(None)
                    if err:
                        logger.warning(f"tFold extraction error for {requests[i]['cdr3b']}: {err}")
                else:
                    feat_bytes = base64.b64decode(fb64)
                    feat = torch.load(io.BytesIO(feat_bytes), map_location="cpu", weights_only=False)
                    features.append(feat)
                    success_count += 1

            print(f"[tFold] Extraction complete: {success_count}/{len(requests)} successful", flush=True)
            return features

        except (ConnectionRefusedError, FileNotFoundError):
            logger.warning(f"tFold server not running at {sock_path}")
            return [None] * len(requests)
        except socket.timeout:
            logger.error("tFold server request timed out (30 min)")
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

        # Prepend 'C' to CDR3β sequences if missing (tFold expects conserved Cys)
        cdr3bs_normalized = []
        for cdr3b in cdr3bs:
            if not cdr3b.startswith('C'):
                cdr3bs_normalized.append('C' + cdr3b)
            else:
                cdr3bs_normalized.append(cdr3b)

        # Build cache keys
        keys = [_make_cache_key(c, p, h) for c, p, h in zip(cdr3bs_normalized, peptides, hlas)]

        lookup_t0 = time.perf_counter()
        # Batch cache lookup
        cached = self._cache.get_batch(keys)
        cache_lookup_ms = ((time.perf_counter() - lookup_t0) * 1000.0) / max(len(keys), 1)

        # Find misses
        results = [None] * len(keys)
        need_extract = []  # (original_index, request_dict)
        trace_rows = [
            {
                "cdr3b": cdr3bs_normalized[i],
                "peptide": peptides[i],
                "hla": hlas[i],
                "source": "cache_hit_pending",
            }
            for i in range(len(keys))
        ]

        invalid_cached_keys = []
        for i, key in enumerate(keys):
            cached_features = cached.get(key)
            if self._feature_is_invalid_sentinel(cached_features):
                invalid_cached_keys.append(key)
                logger.info("Retrying negative-cached tFold features for %s", key)
                trace_rows[i]["source"] = "negative_cache_reextract"
                need_extract.append((i, {
                    "cdr3b": cdr3bs_normalized[i],
                    "peptide": peptides[i],
                    "hla": hlas[i],
                }))
                continue
            if cached_features is not None and self._feature_is_valid(cached_features):
                results[i] = cached_features
                trace_rows[i]["source"] = "cache_hit"
                trace_rows[i]["path_ms"] = cache_lookup_ms
            else:
                if cached_features is not None:
                    invalid_cached_keys.append(key)
                    logger.warning("Invalid cached tFold features for %s; re-extracting", key)
                    trace_rows[i]["source"] = "cache_invalid_reextract"
                else:
                    trace_rows[i]["source"] = "cache_miss"
                need_extract.append((i, {
                    "cdr3b": cdr3bs_normalized[i],
                    "peptide": peptides[i],
                    "hla": hlas[i],
                }))

        if invalid_cached_keys:
            self._cache.delete_batch(invalid_cached_keys)

        # Extract missing features via server
        if need_extract:
            print(f"[tFold] Cache miss: {len(need_extract)}/{len(keys)} need extraction", flush=True)
            logger.info(
                f"tFold cache miss: {len(need_extract)}/{len(keys)} need extraction"
            )
            # Process in batches
            all_extracted = []
            extract_t0 = time.perf_counter()
            for batch_start in range(0, len(need_extract), self.max_subprocess_batch):
                batch = need_extract[batch_start : batch_start + self.max_subprocess_batch]
                requests = [item[1] for item in batch]
                print(f"[tFold] Extracting batch {batch_start//self.max_subprocess_batch + 1} ({len(batch)} samples)...", flush=True)
                extracted = self._extract_features_server(requests)
                all_extracted.extend(zip(batch, extracted))
                print(f"[tFold] Batch {batch_start//self.max_subprocess_batch + 1} done", flush=True)
            extract_path_ms = ((time.perf_counter() - extract_t0) * 1000.0) / max(len(need_extract), 1)

            # Store extracted features
            to_cache = []
            invalid_to_cache = []
            for (orig_idx, req), feats in all_extracted:
                if self._feature_is_valid(feats):
                    results[orig_idx] = feats
                    key = keys[orig_idx]
                    to_cache.append((key, feats))
                    trace_rows[orig_idx]["source"] = "extract_ok"
                    trace_rows[orig_idx]["path_ms"] = extract_path_ms
                elif feats is not None:
                    key = keys[orig_idx]
                    invalid_to_cache.append((key, INVALID_FEATURE_SENTINEL))
                    trace_rows[orig_idx]["source"] = "extract_invalid"
                    trace_rows[orig_idx]["path_ms"] = extract_path_ms
                    logger.warning(
                        "Discarding non-finite extracted tFold features for %s|%s|%s",
                        req["cdr3b"], req["peptide"], req["hla"],
                    )
                else:
                    trace_rows[orig_idx]["source"] = "extract_failed"
                    trace_rows[orig_idx]["path_ms"] = extract_path_ms

            if to_cache:
                self._cache.put_batch(to_cache)
                print(f"[tFold] Cached {len(to_cache)} new features (total cache size: {self._cache.size()})", flush=True)
                logger.info(f"Cached {len(to_cache)} new tFold features")
            if invalid_to_cache:
                self._cache.put_batch(invalid_to_cache)
                logger.info("Negative-cached %d invalid tFold feature entries", len(invalid_to_cache))

        self._last_batch_trace = trace_rows
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

        for key, value in batch.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                logger.warning("Non-finite tensor detected in tFold classifier input: %s", key)
                batch[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        gate_logits = self.model(batch)  # [B] — higher = more non-binding
        if not torch.isfinite(gate_logits).all():
            logger.warning("Non-finite logits from tFold classifier; replacing with zeros")
            gate_logits = torch.nan_to_num(gate_logits, nan=0.0, posinf=0.0, neginf=0.0)
        # Convert to a pre-sigmoid binding logit: higher = more binding.
        binding_scores = -gate_logits

        return binding_scores

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score a single TCR-peptide pair.

        Args:
            tcr: CDR3beta sequence.
            peptide: Peptide sequence.

        Returns:
            (binding_logit, confidence) where binding_logit is -gate_logit
            (higher = more binding) and confidence is 1.0 (no uncertainty est).
        """
        hla = kwargs.get("hla", self.default_hla)
        features = self._get_features_batch([tcr], [peptide], [hla])

        if features[0] is None:
            # Feature extraction failed — treat as non-binding, not neutral.
            return self._default_raw_score, 0.0

        score = self._classify_batch([features[0]])
        value = float(score[0].item())
        if not np.isfinite(value):
            logger.warning(
                "Non-finite single tFold score for tcr=%s peptide=%s; using fallback %.3f",
                tcr, peptide, self._default_raw_score,
            )
            value = self._default_raw_score
            confidence = 0.0
        else:
            confidence = 1.0
        self._n_scored += 1
        return value, confidence

    def score_batch(self, tcrs: list, peptides: list, **kwargs) -> Tuple[list, list]:
        """Score a batch of TCR-peptide pairs with confidence.

        Returns:
            (scores, confidences) — two lists of floats.
        """
        t0 = time.perf_counter()
        hlas = kwargs.get("hlas", [self.default_hla] * len(tcrs))
        features = self._get_features_batch(tcrs, peptides, hlas)
        trace_rows = list(self._last_batch_trace)

        # Separate successful and failed
        valid_features = []
        valid_indices = []
        for i, f in enumerate(features):
            if f is not None:
                valid_features.append(f)
                valid_indices.append(i)

        # Failed extractions get a low score to discourage unscoreable sequences.
        default_raw = self._default_raw_score
        scores = [default_raw] * len(tcrs)
        confidences = [0.0] * len(tcrs)

        if valid_features:
            classify_t0 = time.perf_counter()
            binding_scores = self._classify_batch(valid_features)
            classify_ms = ((time.perf_counter() - classify_t0) * 1000.0) / max(len(valid_features), 1)
            for j, orig_idx in enumerate(valid_indices):
                score = float(binding_scores[j].item())
                trace_rows[orig_idx]["classify_ms"] = classify_ms
                if np.isfinite(score):
                    scores[orig_idx] = score
                    confidences[orig_idx] = 1.0
                else:
                    logger.warning(
                        "Non-finite batch tFold score for tcr=%s peptide=%s hla=%s; using fallback %.3f",
                        tcrs[orig_idx], peptides[orig_idx], hlas[orig_idx], default_raw,
                    )

        self._n_scored += len(tcrs)
        self._log_prediction_trace(
            trace_rows=trace_rows,
            scores=scores,
            confidences=confidences,
            total_elapsed_s=time.perf_counter() - t0,
        )
        return scores, confidences

    def score_batch_fast(self, tcrs: list, peptides: list) -> List[float]:
        """Fast batch scoring (no confidence estimation).

        For RL training reward, return the pre-sigmoid binding logit directly.
        This keeps the high-sensitivity gate-logit signal instead of compressing
        it through sigmoid.
        """
        scores, _ = self.score_batch(tcrs, peptides)
        return [float(s) for s in scores]

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


# ============================================================================
# Cascade Scorer: ERGO (fast) + tFold (oracle for uncertain cases)
# ============================================================================


class TFoldCascadeScorer(BaseScorer):
    """ERGO + tFold cascade scorer with uncertainty-gated arbitration.

    Architecture:
      1. ERGO MC Dropout: fast scoring (0.05s) + uncertainty estimation
      2. If ERGO std > threshold → invoke tFold for arbitration
      3. tFold: high-quality binding prediction (PerEpiAUC=0.82)

    This provides strong reward signal while maintaining reasonable training speed.
    Expected tFold invocation rate: ~10-20% of scoring calls.
    """

    def __init__(
        self,
        ergo_model_file: str,
        tfold_scorer: AffinityTFoldScorer,
        uncertainty_threshold: float = 0.15,
        mc_samples: int = 10,
        ergo_device: str = "cuda",
    ):
        """Initialize cascade scorer.

        Args:
            ergo_model_file: Path to ERGO model checkpoint.
            tfold_scorer: Pre-initialized AffinityTFoldScorer instance.
            uncertainty_threshold: ERGO std above which tFold is invoked.
            mc_samples: Number of MC Dropout samples for ERGO.
            ergo_device: Device for ERGO model.
        """
        from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer

        self.uncertainty_threshold = uncertainty_threshold
        self.tfold_scorer = tfold_scorer

        # Initialize ERGO scorer
        self.ergo_scorer = AffinityERGOScorer(
            model_file=ergo_model_file,
            device=ergo_device,
            mc_samples=mc_samples,
        )

        # Telemetry
        self._n_ergo_calls = 0
        self._n_tfold_calls = 0

        logger.info(
            f"TFold cascade scorer initialized: threshold={uncertainty_threshold}, "
            f"mc_samples={mc_samples}"
        )

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score with cascade logic.

        Returns (score, confidence). ERGO scores remain probability-like;
        tFold fallback scores are pre-sigmoid binding logits.
        """
        # Step 1: ERGO MC Dropout
        means, stds = self.ergo_scorer.mc_dropout_score([tcr], [peptide])
        ergo_score = float(means[0])
        ergo_std = float(stds[0])
        self._n_ergo_calls += 1

        # Step 2: Check uncertainty
        if ergo_std < self.uncertainty_threshold:
            # Low uncertainty → trust ERGO
            confidence = 1.0 - ergo_std
            return ergo_score, max(0.0, min(1.0, confidence))

        # Step 3: High uncertainty → invoke tFold
        tfold_scores = self.tfold_scorer.score_batch_fast([tcr], [peptide])
        self._n_tfold_calls += 1

        # tFold score is a pre-sigmoid binding logit from score_batch_fast.
        tfold_score = tfold_scores[0]

        # Return tFold score with high confidence
        return tfold_score, 0.95

    def score_batch(self, tcrs: list, peptides: list, **kwargs) -> Tuple[list, list]:
        """Score batch with cascade logic.

        First scores all with ERGO MC Dropout, then invokes tFold only for
        uncertain cases.
        """
        n = len(tcrs)
        if n == 0:
            return [], []

        # Step 1: ERGO MC Dropout for all
        means, stds = self.ergo_scorer.mc_dropout_score(tcrs, peptides)
        self._n_ergo_calls += n

        scores = means.tolist()
        confidences = (1.0 - stds).tolist()

        # Step 2: Identify uncertain cases
        uncertain_indices = [
            i for i in range(n) if stds[i] >= self.uncertainty_threshold
        ]

        if not uncertain_indices:
            return scores, [max(0.0, min(1.0, c)) for c in confidences]

        # Step 3: Score uncertain cases with tFold (batched)
        uncertain_tcrs = [tcrs[i] for i in uncertain_indices]
        uncertain_peps = [peptides[i] for i in uncertain_indices]
        tfold_scores = self.tfold_scorer.score_batch_fast(uncertain_tcrs, uncertain_peps)
        self._n_tfold_calls += len(uncertain_indices)

        # Update scores and confidences
        for j, i in enumerate(uncertain_indices):
            scores[i] = tfold_scores[j]
            confidences[i] = 0.95

        return scores, [max(0.0, min(1.0, c)) for c in confidences]

    def score_batch_fast(self, tcrs: list, peptides: list) -> List[float]:
        """Fast scoring with cascade logic (no confidence estimation).

        Use for training reward computation.
        """
        scores, _ = self.score_batch(tcrs, peptides)
        return scores

    def get_telemetry(self) -> dict:
        """Get cascade telemetry statistics."""
        total = self._n_ergo_calls
        tfold_pct = (self._n_tfold_calls / total * 100) if total > 0 else 0.0
        tfold_stats = self.tfold_scorer.cache_stats

        return {
            "total_calls": total,
            "tfold_calls": self._n_tfold_calls,
            "tfold_pct": tfold_pct,
            "tfold_cache_hits": tfold_stats["cache_hits"],
            "tfold_cache_misses": tfold_stats["cache_misses"],
            "tfold_cache_size": tfold_stats["cache_size"],
        }

    def reset_telemetry(self) -> None:
        """Reset telemetry counters."""
        self._n_ergo_calls = 0
        self._n_tfold_calls = 0
