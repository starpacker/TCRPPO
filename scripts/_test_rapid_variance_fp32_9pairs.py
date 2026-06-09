#!/usr/bin/env python3
"""Quick test: score 9 pairs through FP32 tFold server N times rapidly."""

import sys, time, json
import numpy as np
sys.path.insert(0, "/share/liuyutian/tcrppo_v2")

from scripts.test_tfold_scorer_drift import (
    generate_test_pairs, extract_features_via_socket, load_classifier, classify_features
)

SOCKET = "/tmp/tfold_server_fp32_test.sock"
N_REPS = 5
N_PAIRS = 9  # Reduced from 50

pairs = generate_test_pairs(n=N_PAIRS, seed=999)
samples = [{"cdr3b": p["cdr3b"], "peptide": p["peptide"], "hla": p["hla"]} for p in pairs]

print("Loading classifier...", flush=True)
classifier = load_classifier("cpu")

print(f"\nScoring {N_PAIRS} pairs x {N_REPS} repetitions through FP32 server", flush=True)
print(f"Socket: {SOCKET}", flush=True)
print(flush=True)

all_scores = []
for rep in range(N_REPS):
    t0 = time.time()
    feats = extract_features_via_socket(SOCKET, samples, timeout=1800)
    scores = classify_features(classifier, feats)
    elapsed = time.time() - t0
    all_scores.append(scores)
    m = np.mean(scores)
    print(f"  Rep {rep+1}/{N_REPS}: mean={m:.4f}  std={np.std(scores):.4f}  ({elapsed:.0f}s)", flush=True)

# Analysis
means = [np.mean(s) for s in all_scores]
arr = np.array(all_scores)

print(flush=True)
print("=" * 60, flush=True)
print("FP32 RAPID-FIRE VARIANCE (9 pairs)", flush=True)
print("=" * 60, flush=True)
print(f"  Means across reps:    {[f'{m:.4f}' for m in means]}", flush=True)
print(f"  Mean of means:        {np.mean(means):.4f}", flush=True)
print(f"  Std of means:         {np.std(means):.4f}", flush=True)
print(f"  Max spread (max-min): {max(means)-min(means):.4f}", flush=True)
print(flush=True)

per_pair_std = arr.std(axis=0)
print(f"  Per-pair std (across reps):", flush=True)
print(f"    Mean:  {per_pair_std.mean():.6f}", flush=True)
print(f"    Max:   {per_pair_std.max():.6f}", flush=True)
print(flush=True)

for i in range(1, N_REPS):
    diff = np.array(all_scores[0]) - np.array(all_scores[i])
    print(f"  Rep1 vs Rep{i+1}: mean_diff={np.mean(diff):+.4f}  max_abs={np.max(np.abs(diff)):.4f}", flush=True)

if np.std(means) < 0.05:
    print(f"\nVERDICT: DETERMINISTIC (std={np.std(means):.4f} < 0.05)", flush=True)
elif np.std(means) < 0.2:
    print(f"\nVERDICT: MINOR VARIANCE (std={np.std(means):.4f})", flush=True)
else:
    print(f"\nVERDICT: NON-DETERMINISTIC (std={np.std(means):.4f})", flush=True)

# Compare with BF16
print(flush=True)
print("=" * 60, flush=True)
print("COMPARISON: FP32 vs BF16 AMP (first 9 pairs)", flush=True)
print("=" * 60, flush=True)
try:
    with open("logs/tfold_rapid_fire_variance.json") as f:
        bf16_data = json.load(f)
    # Extract first 9 pairs from each BF16 rep
    bf16_means_9 = [np.mean(bf16_data["all_scores"][i][:9]) for i in range(5)]
    print(f"  BF16 means (9 pairs):  {[f'{m:.4f}' for m in bf16_means_9]}", flush=True)
    print(f"  BF16 std:              {np.std(bf16_means_9):.4f}", flush=True)
    print(f"  BF16 spread:           {max(bf16_means_9)-min(bf16_means_9):.4f}", flush=True)
    print(flush=True)
    print(f"  FP32 means (9 pairs):  {[f'{m:.4f}' for m in means]}", flush=True)
    print(f"  FP32 std:              {np.std(means):.4f}", flush=True)
    print(f"  FP32 spread:           {max(means)-min(means):.4f}", flush=True)
    print(flush=True)
    if np.std(means) < np.std(bf16_means_9) / 2:
        print(f"  → FP32 is MUCH MORE STABLE than BF16 AMP", flush=True)
    elif np.std(means) < np.std(bf16_means_9):
        print(f"  → FP32 is MORE STABLE than BF16 AMP", flush=True)
    else:
        print(f"  → FP32 drift is SIMILAR to BF16 (not an AMP-only issue)", flush=True)
except Exception as e:
    print(f"  (Could not load BF16 data: {e})", flush=True)

with open("logs/tfold_rapid_fire_variance_fp32_9pairs.json", "w") as f:
    json.dump({"means": means, "all_scores": all_scores, "per_pair_std": per_pair_std.tolist()}, f, indent=2)
print(f"\nSaved to logs/tfold_rapid_fire_variance_fp32_9pairs.json", flush=True)
