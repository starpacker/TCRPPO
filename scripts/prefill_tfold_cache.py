#!/usr/bin/env python
"""Pre-populate tFold feature cache by extracting features for all L0 seed TCRs.

This script reads L0 seed CDR3beta sequences from data/l0_seeds_tchard/ and
sends them to the running tFold feature server for extraction. Results are
stored in the SQLite feature cache used by AffinityTFoldScorer.

Usage:
    python scripts/prefill_tfold_cache.py \
        --socket /tmp/tfold_server.sock \
        --batch_size 16 \
        --targets GILGFVFTL NLVPMVATV  # optional, default: all eval targets

Prerequisites:
    The tFold feature server must be running:
        bash scripts/start_tfold_server.sh 3
"""

import argparse
import base64
import io
import json
import os
import socket
import struct
import sys
import time
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcrppo_v2.scorers.affinity_tfold import TFoldFeatureCache, _make_cache_key

# 12 McPAS eval targets
EVAL_TARGETS = [
    "GILGFVFTL", "NLVPMVATV", "GLCTLVAML", "LLWNGPMAV",
    "YLQPRTFLL", "FLYALALLL", "SLYNTVATL", "KLGGALQAK",
    "AVFDRKSDAK", "IVTDFSVIK", "SPRWYFYYL", "RLRAEAQVK",
]

DEFAULT_HLA = "HLA-A*02:01"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def send_request(sock_path: str, request: dict, timeout: int = 600) -> dict:
    """Send a request to the tFold feature server and return the response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(sock_path)

    data = json.dumps(request).encode("utf-8")
    header = struct.pack(">I", len(data))
    sock.sendall(header + data)

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

    sock.close()
    return json.loads(resp_data.decode("utf-8"))


def load_l0_seeds(targets: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Load L0 seed CDR3beta sequences for each target peptide."""
    l0_dir = os.path.join(PROJECT_ROOT, "data", "l0_seeds_tchard")
    if not os.path.isdir(l0_dir):
        print(f"L0 seeds directory not found: {l0_dir}")
        return {}

    seeds = {}
    for fname in os.listdir(l0_dir):
        if not fname.endswith(".txt"):
            continue
        peptide = fname.replace(".txt", "")
        if targets and peptide not in targets:
            continue
        fpath = os.path.join(l0_dir, fname)
        with open(fpath) as f:
            seqs = [line.strip() for line in f if line.strip()]
        if seqs:
            seeds[peptide] = seqs

    return seeds


def main():
    parser = argparse.ArgumentParser(description="Pre-populate tFold feature cache")
    parser.add_argument("--socket", default="/tmp/tfold_server.sock",
                       help="tFold server socket path")
    parser.add_argument("--cache_path", default=None,
                       help="Feature cache DB path (default: data/tfold_feature_cache.db)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Samples per server request")
    parser.add_argument("--targets", nargs="*", default=None,
                       help="Target peptides to process (default: 12 eval targets)")
    parser.add_argument("--max_per_target", type=int, default=None,
                       help="Max L0 seeds per target (for testing)")
    args = parser.parse_args()

    targets = args.targets or EVAL_TARGETS

    # Check server
    print(f"Checking tFold server at {args.socket}...")
    try:
        resp = send_request(args.socket, {"cmd": "ping"}, timeout=10)
        print(f"  Server: {resp}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to tFold server: {e}")
        print(f"  Start it with: bash scripts/start_tfold_server.sh")
        return

    # Open cache
    cache_path = args.cache_path or os.path.join(PROJECT_ROOT, "data", "tfold_feature_cache.db")
    cache = TFoldFeatureCache(cache_path)
    print(f"Feature cache at {cache_path} ({cache.size()} existing entries)")

    # Load seeds
    seeds = load_l0_seeds(targets)
    total_seeds = sum(len(v) for v in seeds.values())
    print(f"Loaded {total_seeds} L0 seeds for {len(seeds)} targets")

    # Process each target
    n_extracted = 0
    n_cached = 0
    n_failed = 0
    t_start = time.time()

    for target_idx, (peptide, cdr3bs) in enumerate(seeds.items()):
        if args.max_per_target:
            cdr3bs = cdr3bs[:args.max_per_target]

        # Check which are already cached
        uncached = []
        for cdr3b in cdr3bs:
            key = _make_cache_key(cdr3b, peptide, DEFAULT_HLA)
            existing = cache.get(key)
            if existing is not None:
                n_cached += 1
            else:
                uncached.append(cdr3b)

        if not uncached:
            print(f"  [{target_idx+1}/{len(seeds)}] {peptide}: all {len(cdr3bs)} already cached")
            continue

        print(f"  [{target_idx+1}/{len(seeds)}] {peptide}: {len(uncached)}/{len(cdr3bs)} need extraction")

        # Batch extract
        for batch_start in range(0, len(uncached), args.batch_size):
            batch = uncached[batch_start : batch_start + args.batch_size]
            samples = [
                {"cdr3b": cdr3b, "peptide": peptide, "hla": DEFAULT_HLA}
                for cdr3b in batch
            ]

            try:
                t0 = time.time()
                resp = send_request(args.socket, {"cmd": "extract", "samples": samples})
                elapsed = time.time() - t0

                if resp.get("status") != "ok":
                    print(f"    Server error: {resp.get('error', 'unknown')}")
                    n_failed += len(batch)
                    continue

                features_b64 = resp.get("features", [])
                errors = resp.get("errors", [None] * len(features_b64))

                batch_ok = 0
                to_cache = []
                for i, (fb64, err) in enumerate(zip(features_b64, errors)):
                    if fb64 is not None and err is None:
                        feat_bytes = base64.b64decode(fb64)
                        feat = torch.load(io.BytesIO(feat_bytes), map_location="cpu", weights_only=False)
                        key = _make_cache_key(batch[i], peptide, DEFAULT_HLA)
                        to_cache.append((key, feat))
                        batch_ok += 1
                    else:
                        n_failed += 1

                if to_cache:
                    cache.put_batch(to_cache)
                    n_extracted += len(to_cache)

                print(f"    batch {batch_start//args.batch_size + 1}: "
                      f"{batch_ok}/{len(batch)} ok ({elapsed:.1f}s, "
                      f"{elapsed/len(batch):.1f}s/sample)")

            except Exception as e:
                print(f"    Batch error: {e}")
                n_failed += len(batch)

    elapsed_total = time.time() - t_start
    print(f"\nDone in {elapsed_total:.0f}s:")
    print(f"  Extracted: {n_extracted}")
    print(f"  Already cached: {n_cached}")
    print(f"  Failed: {n_failed}")
    print(f"  Cache total: {cache.size()}")
    cache.close()


if __name__ == "__main__":
    main()
