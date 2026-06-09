#!/usr/bin/env python3
"""Test whether BF16 bias is systematic across different samples.

Protocol:
  1. Generate N diverse samples (different CDR3/peptide/HLA combinations)
  2. Score each sample with both FP32 and BF16 (fresh restart for BF16)
  3. Calculate per-sample bias: (BF16_score - FP32_score)
  4. Analyze bias distribution:
     - If std(bias) is small → systematic bias, can calibrate
     - If std(bias) is large → non-systematic, must use FP32
"""

import sys, time, json, os, signal, subprocess
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "/share/liuyutian/tcrppo_v2")

from scripts.test_tfold_scorer_drift import (
    generate_test_pairs, extract_features_via_socket, load_classifier, classify_features
)

SOCKET_FP32 = "/tmp/tfold_server_bias_test_fp32.sock"
SOCKET_BF16 = "/tmp/tfold_server_bias_test_bf16.sock"
GPU = 3
N_SAMPLES = 30  # Test 30 diverse samples

# Generate diverse samples with different seeds
print("Generating 30 diverse test samples...", flush=True)
all_pairs = []
for seed in range(30):
    pair = generate_test_pairs(n=1, seed=seed)[0]
    all_pairs.append(pair)

samples = [{"cdr3b": p["cdr3b"], "peptide": p["peptide"], "hla": p["hla"]} for p in all_pairs]

print("Loading classifier...", flush=True)
classifier = load_classifier("cpu")

def start_server(socket_path, use_amp=True):
    if os.path.exists(socket_path):
        os.remove(socket_path)
    args = ["/home/liuyutian/server/miniconda3/envs/tfold/bin/python",
            "scripts/tfold_feature_server.py",
            "--socket", socket_path,
            "--gpu", str(GPU),
            "--chunk-size", "64"]
    if use_amp:
        args.append("--use-amp-wrapper")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    for i in range(90):
        if os.path.exists(socket_path):
            return proc
        time.sleep(1)
    print(f"    ERROR: Server failed to start at {socket_path}", flush=True)
    return None

def kill_server(proc, socket_path):
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait()
    if os.path.exists(socket_path):
        os.remove(socket_path)

# ============================================================
# TEST 1: FP32 scores (ground truth)
# ============================================================
print(f"\n[TEST 1] Scoring {N_SAMPLES} samples with FP32...", flush=True)
proc_fp32 = start_server(SOCKET_FP32, use_amp=False)
if not proc_fp32:
    print("FAILED to start FP32 server", flush=True)
    sys.exit(1)

print(f"  FP32 server ready (PID={proc_fp32.pid})", flush=True)
t0 = time.time()
feats_fp32 = extract_features_via_socket(SOCKET_FP32, samples, timeout=3600)
scores_fp32 = classify_features(classifier, feats_fp32)
elapsed_fp32 = time.time() - t0
print(f"  FP32 done: mean={np.mean(scores_fp32):.4f} ({elapsed_fp32:.0f}s total, {elapsed_fp32/N_SAMPLES:.1f}s/sample)", flush=True)
kill_server(proc_fp32, SOCKET_FP32)

# ============================================================
# TEST 2: BF16 scores (fresh restart)
# ============================================================
print(f"\n[TEST 2] Scoring {N_SAMPLES} samples with BF16 (fresh server)...", flush=True)
proc_bf16 = start_server(SOCKET_BF16, use_amp=True)
if not proc_bf16:
    print("FAILED to start BF16 server", flush=True)
    sys.exit(1)

print(f"  BF16 server ready (PID={proc_bf16.pid})", flush=True)
t0 = time.time()
feats_bf16 = extract_features_via_socket(SOCKET_BF16, samples, timeout=3600)
scores_bf16 = classify_features(classifier, feats_bf16)
elapsed_bf16 = time.time() - t0
print(f"  BF16 done: mean={np.mean(scores_bf16):.4f} ({elapsed_bf16:.0f}s total, {elapsed_bf16/N_SAMPLES:.1f}s/sample)", flush=True)
kill_server(proc_bf16, SOCKET_BF16)

# ============================================================
# ANALYSIS: Bias distribution
# ============================================================
print(f"\n{'='*60}", flush=True)
print("BIAS ANALYSIS", flush=True)
print(f"{'='*60}", flush=True)

bias = np.array(scores_bf16) - np.array(scores_fp32)
ratio = np.array(scores_bf16) / np.array(scores_fp32)

print(f"Per-sample bias (BF16 - FP32):", flush=True)
print(f"  Mean bias:      {np.mean(bias):.4f}", flush=True)
print(f"  Std of bias:    {np.std(bias):.4f}", flush=True)
print(f"  Min bias:       {np.min(bias):.4f}", flush=True)
print(f"  Max bias:       {np.max(bias):.4f}", flush=True)
print(f"  Bias range:     {np.max(bias)-np.min(bias):.4f}", flush=True)
print(flush=True)

print(f"Per-sample ratio (BF16 / FP32):", flush=True)
print(f"  Mean ratio:     {np.mean(ratio):.4f}", flush=True)
print(f"  Std of ratio:   {np.std(ratio):.4f}", flush=True)
print(f"  Min ratio:      {np.min(ratio):.4f}", flush=True)
print(f"  Max ratio:      {np.max(ratio):.4f}", flush=True)
print(flush=True)

# Correlation between FP32 and BF16
corr = np.corrcoef(scores_fp32, scores_bf16)[0, 1]
print(f"Correlation (FP32 vs BF16):  {corr:.6f}", flush=True)
print(flush=True)

# Ranking agreement (Spearman correlation)
from scipy.stats import spearmanr
spearman_corr, spearman_p = spearmanr(scores_fp32, scores_bf16)
print(f"Spearman rank correlation:   {spearman_corr:.6f} (p={spearman_p:.2e})", flush=True)
print(flush=True)

# ============================================================
# VERDICT
# ============================================================
print(f"{'='*60}", flush=True)
print("VERDICT", flush=True)
print(f"{'='*60}", flush=True)

bias_cv = np.std(bias) / abs(np.mean(bias)) if np.mean(bias) != 0 else float('inf')
print(f"Coefficient of variation (CV) of bias: {bias_cv:.4f}", flush=True)
print(flush=True)

if bias_cv < 0.15:
    print("✓ SYSTEMATIC BIAS (CV < 0.15)", flush=True)
    print("  → Bias is consistent across samples", flush=True)
    print("  → Can use BF16 + linear calibration:", flush=True)
    print(f"      scores_calibrated = scores_bf16 - {np.mean(bias):.4f}", flush=True)
    print(f"  → Or use ratio calibration:", flush=True)
    print(f"      scores_calibrated = scores_bf16 / {np.mean(ratio):.4f}", flush=True)
elif bias_cv < 0.30:
    print("⚠ MODERATE BIAS VARIABILITY (0.15 ≤ CV < 0.30)", flush=True)
    print("  → Bias varies moderately across samples", flush=True)
    print("  → Linear calibration may help, but test carefully", flush=True)
    print(f"  → Ranking correlation is still strong: {spearman_corr:.4f}", flush=True)
else:
    print("✗ NON-SYSTEMATIC BIAS (CV ≥ 0.30)", flush=True)
    print("  → Bias is highly variable across samples", flush=True)
    print("  → BF16 cannot be reliably calibrated", flush=True)
    print("  → Recommendation: Use FP32 for accuracy-critical applications", flush=True)

print(flush=True)
print(f"Speed comparison:", flush=True)
print(f"  FP32: {elapsed_fp32/N_SAMPLES:.1f}s/sample", flush=True)
print(f"  BF16: {elapsed_bf16/N_SAMPLES:.1f}s/sample", flush=True)
print(f"  Speedup: {elapsed_fp32/elapsed_bf16:.2f}x", flush=True)

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    "n_samples": N_SAMPLES,
    "scores_fp32": scores_fp32 if isinstance(scores_fp32, list) else scores_fp32.tolist(),
    "scores_bf16": scores_bf16 if isinstance(scores_bf16, list) else scores_bf16.tolist(),
    "bias": bias.tolist(),
    "ratio": ratio.tolist(),
    "statistics": {
        "mean_bias": float(np.mean(bias)),
        "std_bias": float(np.std(bias)),
        "cv_bias": float(bias_cv),
        "mean_ratio": float(np.mean(ratio)),
        "std_ratio": float(np.std(ratio)),
        "pearson_corr": float(corr),
        "spearman_corr": float(spearman_corr),
        "spearman_p": float(spearman_p),
    },
    "timing": {
        "fp32_total": elapsed_fp32,
        "bf16_total": elapsed_bf16,
        "fp32_per_sample": elapsed_fp32 / N_SAMPLES,
        "bf16_per_sample": elapsed_bf16 / N_SAMPLES,
        "speedup": elapsed_fp32 / elapsed_bf16,
    }
}

with open("logs/tfold_bf16_bias_systematic.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to logs/tfold_bf16_bias_systematic.json", flush=True)

# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Scatter: FP32 vs BF16
ax = axes[0, 0]
ax.scatter(scores_fp32, scores_bf16, alpha=0.6, s=50)
min_val = min(scores_fp32.min(), scores_bf16.min())
max_val = max(scores_fp32.max(), scores_bf16.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
ax.set_xlabel('FP32 Score')
ax.set_ylabel('BF16 Score')
ax.set_title(f'FP32 vs BF16 (r={corr:.4f}, ρ={spearman_corr:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Histogram: Bias distribution
ax = axes[0, 1]
ax.hist(bias, bins=20, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(bias), color='r', linestyle='--', linewidth=2, label=f'Mean={np.mean(bias):.4f}')
ax.set_xlabel('Bias (BF16 - FP32)')
ax.set_ylabel('Frequency')
ax.set_title(f'Bias Distribution (std={np.std(bias):.4f}, CV={bias_cv:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Sample index vs Bias
ax = axes[1, 0]
ax.plot(range(N_SAMPLES), bias, 'o-', alpha=0.6)
ax.axhline(np.mean(bias), color='r', linestyle='--', linewidth=2, label=f'Mean={np.mean(bias):.4f}')
ax.fill_between(range(N_SAMPLES),
                 np.mean(bias) - np.std(bias),
                 np.mean(bias) + np.std(bias),
                 alpha=0.2, color='red', label=f'±1 std')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Bias (BF16 - FP32)')
ax.set_title('Per-Sample Bias')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Histogram: Ratio distribution
ax = axes[1, 1]
ax.hist(ratio, bins=20, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(ratio), color='r', linestyle='--', linewidth=2, label=f'Mean={np.mean(ratio):.4f}')
ax.set_xlabel('Ratio (BF16 / FP32)')
ax.set_ylabel('Frequency')
ax.set_title(f'Ratio Distribution (std={np.std(ratio):.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/tfold_bf16_bias_systematic.png', dpi=150, bbox_inches='tight')
print(f"Saved plot to figures/tfold_bf16_bias_systematic.png", flush=True)
