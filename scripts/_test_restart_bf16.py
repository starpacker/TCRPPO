#!/usr/bin/env python3
"""Test: Does restarting BF16 AMP server between rounds eliminate drift?

Protocol:
  For each round:
    1. Start a fresh BF16 AMP server
    2. Score 9 fixed pairs
    3. Kill the server
  Compare scores across rounds.
"""

import sys, time, json, os, signal, subprocess
import numpy as np
sys.path.insert(0, "/share/liuyutian/tcrppo_v2")

from scripts.test_tfold_scorer_drift import (
    generate_test_pairs, extract_features_via_socket, load_classifier, classify_features
)

SOCKET = "/tmp/tfold_server_restart_test.sock"
GPU = 3
N_REPS = 5
N_PAIRS = 9

pairs = generate_test_pairs(n=N_PAIRS, seed=999)
samples = [{"cdr3b": p["cdr3b"], "peptide": p["peptide"], "hla": p["hla"]} for p in pairs]

print("Loading classifier...", flush=True)
classifier = load_classifier("cpu")

print(f"\nTest: Restart BF16 server between each scoring round", flush=True)
print(f"  {N_PAIRS} pairs x {N_REPS} rounds (fresh server each round)", flush=True)
print(flush=True)

all_scores = []

for rep in range(N_REPS):
    # Clean up socket
    if os.path.exists(SOCKET):
        os.remove(SOCKET)

    # Start fresh server
    print(f"  [Round {rep+1}] Starting fresh BF16 AMP server on GPU {GPU}...", flush=True)
    proc = subprocess.Popen(
        ["/home/liuyutian/server/miniconda3/envs/tfold/bin/python",
         "scripts/tfold_feature_server.py",
         "--socket", SOCKET,
         "--gpu", str(GPU),
         "--use-amp-wrapper",
         "--chunk-size", "64"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    # Wait for socket
    for i in range(60):
        if os.path.exists(SOCKET):
            break
        time.sleep(1)
    else:
        print(f"    ERROR: Server failed to start", flush=True)
        continue

    print(f"    Server ready (PID={proc.pid})", flush=True)

    # Score
    t0 = time.time()
    feats = extract_features_via_socket(SOCKET, samples, timeout=600)
    scores = classify_features(classifier, feats)
    elapsed = time.time() - t0
    all_scores.append(scores)
    m = np.mean(scores)
    print(f"    Scores: mean={m:.4f}  std={np.std(scores):.4f}  ({elapsed:.0f}s)", flush=True)

    # Kill server
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait()
    if os.path.exists(SOCKET):
        os.remove(SOCKET)
    print(f"    Server stopped", flush=True)
    print(flush=True)

# Analysis
means = [np.mean(s) for s in all_scores]
arr = np.array(all_scores)

print("=" * 60, flush=True)
print("BF16 WITH RESTART: VARIANCE ANALYSIS", flush=True)
print("=" * 60, flush=True)
print(f"  Means across rounds:  {[f'{m:.4f}' for m in means]}", flush=True)
print(f"  Mean of means:        {np.mean(means):.4f}", flush=True)
print(f"  Std of means:         {np.std(means):.4f}", flush=True)
print(f"  Max spread (max-min): {max(means)-min(means):.4f}", flush=True)
print(flush=True)

per_pair_std = arr.std(axis=0)
print(f"  Per-pair std:", flush=True)
print(f"    Mean:  {per_pair_std.mean():.6f}", flush=True)
print(f"    Max:   {per_pair_std.max():.6f}", flush=True)
print(flush=True)

# Compare with no-restart BF16
print("COMPARISON:", flush=True)
try:
    with open("logs/tfold_rapid_fire_variance.json") as f:
        bf16_data = json.load(f)
    bf16_means_9 = [np.mean(bf16_data["all_scores"][i][:9]) for i in range(5)]
    print(f"  BF16 NO restart:   std={np.std(bf16_means_9):.4f}  spread={max(bf16_means_9)-min(bf16_means_9):.4f}", flush=True)
    print(f"  BF16 WITH restart: std={np.std(means):.4f}  spread={max(means)-min(means):.4f}", flush=True)
    print(f"  FP32 (reference):  std=0.0000  spread=0.0000", flush=True)
except Exception as e:
    print(f"  (Could not load comparison data: {e})", flush=True)

print(flush=True)
if np.std(means) < 0.05:
    print(f"VERDICT: RESTART ELIMINATES DRIFT (std={np.std(means):.4f})", flush=True)
elif np.std(means) < 0.2:
    print(f"VERDICT: RESTART REDUCES DRIFT (std={np.std(means):.4f})", flush=True)
else:
    print(f"VERDICT: RESTART DOES NOT HELP (std={np.std(means):.4f})", flush=True)

with open("logs/tfold_restart_test.json", "w") as f:
    json.dump({"means": means, "all_scores": all_scores}, f, indent=2)
print(f"\nSaved to logs/tfold_restart_test.json", flush=True)
