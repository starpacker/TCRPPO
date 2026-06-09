#!/usr/bin/env python3
"""
Controlled experiment: tFold scorer drift diagnosis.

Tests whether the running trace93 tFold server (GPU 3, under training load)
produces different scores than a fresh tFold server (GPU 2, idle).

Protocol:
  1. Generate 50 fixed TCR-peptide pairs from trace93 L0 seeds.
  2. Score all 50 through the RUNNING trace93 server (GPU 3).
  3. Score all 50 through a FRESH control server (GPU 2).
  4. Compare scores.

If scores differ systematically, GPU contention / server runtime state
is confirmed as the drift cause.
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

import numpy as np
import torch

# Add project root
sys.path.insert(0, "/share/liuyutian/tcrppo_v2")

from tcrppo_v2.utils.encoding import mutate_sequence

# ── V3.4 classifier (runs locally, deterministic) ──────────────────────

V34_WEIGHTS = "/share/liuyutian/tfold/TCR_PMHC_pred/4_16/weights/best_v34.pth"


def load_classifier(device="cpu"):
    """Load the V3.4 classifier for local scoring."""
    from tcrppo_v2.scorers.affinity_tfold import ClassifierV34Local
    ckpt = torch.load(V34_WEIGHTS, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = ClassifierV34Local(
        d_sfea=cfg.get("d_sfea", 192),
        n_heads=cfg.get("n_heads", 4),
        n_rbf=cfg.get("n_rbf", 16),
        pfea_dim=cfg.get("pfea_dim", 128),
        n_attn_layers=cfg.get("n_attn_layers", 2),
        mlp_hidden=cfg.get("mlp_hidden", 256),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ── Socket communication ───────────────────────────────────────────────

def extract_features_via_socket(socket_path, samples, timeout=600):
    """Send samples to tFold feature server, return raw feature dicts."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(socket_path)

    request_json = json.dumps({
        "cmd": "extract",
        "samples": samples,
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
            raise ConnectionError("Server closed during response")
        resp_data += chunk

    response = json.loads(resp_data.decode("utf-8"))
    sock.close()

    if response.get("status") != "ok":
        raise RuntimeError(f"Server error: {response.get('error')}")

    features_b64 = response.get("features", [])
    errors = response.get("errors", [None] * len(features_b64))
    features = []
    for fb64, err in zip(features_b64, errors):
        if fb64 is None or err is not None:
            features.append(None)
        else:
            feat_bytes = base64.b64decode(fb64)
            feat = torch.load(io.BytesIO(feat_bytes), map_location="cpu", weights_only=False)
            features.append(feat)
    return features


def classify_features(model, features, device="cpu"):
    """Run V3.4 classifier on feature dicts → binding scores."""
    # Filter out None features
    valid_features = [f for f in features if f is not None]
    if not valid_features:
        return [-20.0] * len(features)

    # Build batch
    batch = {}
    tensor_keys = [
        "sfea_cdr3b", "sfea_cdr3a", "sfea_pep",
        "ca_cdr3b", "ca_cdr3a", "ca_pep",
        "pfea_cdr3b_pep", "pfea_cdr3a_pep", "v33_feat",
    ]
    int_keys = ["len_cdr3b", "len_cdr3a", "len_pep"]

    for key in tensor_keys:
        batch[key] = torch.stack([f[key] for f in valid_features]).to(device)
    for key in int_keys:
        batch[key] = torch.tensor(
            [f[key] for f in valid_features], dtype=torch.long, device=device
        )

    with torch.inference_mode():
        gate_logits = model(batch)  # [B]
        binding_scores = -gate_logits  # higher = more binding

    # Map back to original order
    scores = []
    valid_idx = 0
    for feat in features:
        if feat is None:
            scores.append(-20.0)
        else:
            scores.append(binding_scores[valid_idx].item())
            valid_idx += 1

    return scores


# ── Generate fixed test pairs ──────────────────────────────────────────

def generate_test_pairs(n=50, seed=999):
    """Generate n fixed TCR-peptide pairs from trace93 L0 seeds."""
    rng = np.random.default_rng(seed)
    seeds_dir = "/share/liuyutian/tcrppo_v2/data/l0_seeds_trace93"

    # Load all peptide seed files
    peptide_seeds = {}
    for fname in sorted(os.listdir(seeds_dir)):
        if fname.endswith(".txt"):
            peptide = fname.replace(".txt", "")
            with open(os.path.join(seeds_dir, fname)) as f:
                seeds = [line.strip() for line in f if line.strip()]
            if seeds:
                peptide_seeds[peptide] = seeds

    peptides = sorted(peptide_seeds.keys())
    pairs = []
    mut_rng = np.random.default_rng(seed + 1)

    for i in range(n):
        pep = peptides[rng.integers(len(peptides))]
        seeds = peptide_seeds[pep]
        seed_tcr = seeds[rng.integers(len(seeds))]
        # Apply 3-5 random mutations (same as trace93 L0 pipeline)
        n_mut = rng.integers(3, 6)  # [3, 5] inclusive
        # Remove leading C if present (mutate_sequence expects CDR3 without C)
        tcr_body = seed_tcr[1:] if seed_tcr.startswith("C") else seed_tcr
        mutated = mutate_sequence(tcr_body, n_mut, mut_rng)
        # Add C back
        cdr3b = "C" + mutated
        pairs.append({"cdr3b": cdr3b, "peptide": pep, "hla": "HLA-A*02:01"})

    return pairs


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="tFold scorer drift experiment")
    parser.add_argument("--trace93-socket", default="/tmp/tfold_server_trace93.sock",
                        help="Running trace93 tFold server socket")
    parser.add_argument("--control-socket", default="/tmp/tfold_server_control.sock",
                        help="Fresh control tFold server socket")
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--output", default="logs/tfold_drift_experiment.json")
    parser.add_argument("--only-trace93", action="store_true",
                        help="Only score through trace93 server (skip control)")
    parser.add_argument("--only-control", action="store_true",
                        help="Only score through control server")
    parser.add_argument("--repeat-trace93", type=int, default=1,
                        help="Repeat scoring through trace93 server N times")
    args = parser.parse_args()

    print("=" * 60)
    print("tFold Scorer Drift Experiment")
    print("=" * 60)

    # Step 1: Generate fixed test pairs
    print(f"\n[1] Generating {args.n_pairs} fixed test pairs...")
    pairs = generate_test_pairs(n=args.n_pairs)
    samples = [{"cdr3b": p["cdr3b"], "peptide": p["peptide"], "hla": p["hla"]} for p in pairs]
    print(f"    Generated {len(pairs)} pairs across {len(set(p['peptide'] for p in pairs))} peptides")
    for i in range(min(5, len(pairs))):
        print(f"    Sample {i}: {pairs[i]['cdr3b'][:20]}... + {pairs[i]['peptide']}")

    # Load classifier once
    print("\n[2] Loading V3.4 classifier...")
    classifier = load_classifier("cpu")
    print("    Classifier loaded")

    results = {"pairs": [{"cdr3b": p["cdr3b"], "peptide": p["peptide"]} for p in pairs]}

    # Step 2: Score through trace93 server
    if not args.only_control:
        for rep in range(args.repeat_trace93):
            label = f"trace93" + (f"_rep{rep+1}" if args.repeat_trace93 > 1 else "")
            print(f"\n[3.{rep+1}] Scoring through trace93 server ({args.trace93_socket})...")
            t0 = time.time()
            feats_trace93 = extract_features_via_socket(args.trace93_socket, samples)
            extract_time = time.time() - t0
            scores_trace93 = classify_features(classifier, feats_trace93)
            n_ok = sum(1 for f in feats_trace93 if f is not None)
            print(f"    Extracted {n_ok}/{len(samples)} features in {extract_time:.1f}s")
            print(f"    Scores: mean={np.mean(scores_trace93):.4f}, "
                  f"std={np.std(scores_trace93):.4f}, "
                  f"min={np.min(scores_trace93):.4f}, max={np.max(scores_trace93):.4f}")
            results[label] = {
                "scores": scores_trace93,
                "n_ok": n_ok,
                "extract_time_s": extract_time,
                "mean": float(np.mean(scores_trace93)),
                "std": float(np.std(scores_trace93)),
            }

    # Step 3: Score through control server
    if not args.only_trace93:
        if os.path.exists(args.control_socket):
            print(f"\n[4] Scoring through control server ({args.control_socket})...")
            t0 = time.time()
            feats_control = extract_features_via_socket(args.control_socket, samples)
            extract_time = time.time() - t0
            scores_control = classify_features(classifier, feats_control)
            n_ok = sum(1 for f in feats_control if f is not None)
            print(f"    Extracted {n_ok}/{len(samples)} features in {extract_time:.1f}s")
            print(f"    Scores: mean={np.mean(scores_control):.4f}, "
                  f"std={np.std(scores_control):.4f}, "
                  f"min={np.min(scores_control):.4f}, max={np.max(scores_control):.4f}")
            results["control"] = {
                "scores": scores_control,
                "n_ok": n_ok,
                "extract_time_s": extract_time,
                "mean": float(np.mean(scores_control)),
                "std": float(np.std(scores_control)),
            }

            # Compare
            if "trace93" in results:
                s_t = np.array(results["trace93"]["scores"])
                s_c = np.array(scores_control)
                diff = s_t - s_c
                print(f"\n[5] COMPARISON (trace93 - control):")
                print(f"    Mean diff:   {np.mean(diff):.4f}")
                print(f"    Std diff:    {np.std(diff):.4f}")
                print(f"    Max |diff|:  {np.max(np.abs(diff)):.4f}")
                print(f"    Pairs with |diff| > 0.1: {np.sum(np.abs(diff) > 0.1)}/{len(diff)}")
                print(f"    Pairs with |diff| > 0.5: {np.sum(np.abs(diff) > 0.5)}/{len(diff)}")
                print(f"    Correlation: {np.corrcoef(s_t, s_c)[0,1]:.6f}")
                results["comparison"] = {
                    "mean_diff": float(np.mean(diff)),
                    "std_diff": float(np.std(diff)),
                    "max_abs_diff": float(np.max(np.abs(diff))),
                    "correlation": float(np.corrcoef(s_t, s_c)[0, 1]),
                    "n_diff_gt_0.1": int(np.sum(np.abs(diff) > 0.1)),
                    "n_diff_gt_0.5": int(np.sum(np.abs(diff) > 0.5)),
                }
        else:
            print(f"\n[4] SKIP: Control socket not found at {args.control_socket}")
            print(f"    Start control server first:")
            print(f"    python scripts/tfold_feature_server.py --socket {args.control_socket} --gpu 2 --use-amp-wrapper --chunk-size 64")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
