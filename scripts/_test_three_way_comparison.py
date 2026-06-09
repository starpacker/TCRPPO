#!/usr/bin/env python3
"""Three-way comparison: FP32 vs BF16+Restart vs BF16 No-Restart
Uses 5 NEW samples (seed=42) scored 5 times each."""

import sys, time, json, os, signal, subprocess
import numpy as np
sys.path.insert(0, "/share/liuyutian/tcrppo_v2")

from scripts.test_tfold_scorer_drift import (
    generate_test_pairs, extract_features_via_socket, load_classifier, classify_features
)

SOCKET_FP32 = "/tmp/tfold_server_3way_fp32.sock"
SOCKET_BF16 = "/tmp/tfold_server_3way_bf16.sock"
GPU = 3
N_REPS = 5
N_PAIRS = 5

# New samples with different seed
pairs = generate_test_pairs(n=N_PAIRS, seed=42)
samples = [{"cdr3b": p["cdr3b"], "peptide": p["peptide"], "hla": p["hla"]} for p in pairs]

print("Loading classifier...", flush=True)
classifier = load_classifier("cpu")

print(f"\n{'='*60}", flush=True)
print(f"THREE-WAY COMPARISON: 5 new pairs (seed=42) x 5 reps", flush=True)
print(f"{'='*60}\n", flush=True)

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
# TEST 1: FP32 (single server, score 5 times)
# ============================================================
print("[TEST 1] FP32 - single server, 5 reps", flush=True)
proc = start_server(SOCKET_FP32, use_amp=False)
if proc:
    print(f"  Server ready (PID={proc.pid})", flush=True)
    fp32_scores = []
    for rep in range(N_REPS):
        t0 = time.time()
        feats = extract_features_via_socket(SOCKET_FP32, samples, timeout=600)
        scores = classify_features(classifier, feats)
        elapsed = time.time() - t0
        fp32_scores.append(scores)
        print(f"  Rep {rep+1}: mean={np.mean(scores):.4f} ({elapsed:.0f}s)", flush=True)
    kill_server(proc, SOCKET_FP32)
    print(f"  Server stopped\n", flush=True)
else:
    fp32_scores = []
    print(f"  FAILED\n", flush=True)

# ============================================================
# TEST 2: BF16 WITH RESTART (fresh server each rep)
# ============================================================
print("[TEST 2] BF16 + Restart - fresh server each rep", flush=True)
bf16_restart_scores = []
for rep in range(N_REPS):
    proc = start_server(SOCKET_BF16, use_amp=True)
    if proc:
        print(f"  Rep {rep+1}: server PID={proc.pid}", flush=True, end="")
        t0 = time.time()
        feats = extract_features_via_socket(SOCKET_BF16, samples, timeout=600)
        scores = classify_features(classifier, feats)
        elapsed = time.time() - t0
        bf16_restart_scores.append(scores)
        print(f"  mean={np.mean(scores):.4f} ({elapsed:.0f}s)", flush=True)
        kill_server(proc, SOCKET_BF16)
    else:
        print(f"  Rep {rep+1}: FAILED", flush=True)
print(flush=True)

# ============================================================
# TEST 3: BF16 NO RESTART (single server, score 5 times)
# ============================================================
print("[TEST 3] BF16 No Restart - single server, 5 reps", flush=True)
proc = start_server(SOCKET_BF16, use_amp=True)
if proc:
    print(f"  Server ready (PID={proc.pid})", flush=True)
    bf16_norestart_scores = []
    for rep in range(N_REPS):
        t0 = time.time()
        feats = extract_features_via_socket(SOCKET_BF16, samples, timeout=600)
        scores = classify_features(classifier, feats)
        elapsed = time.time() - t0
        bf16_norestart_scores.append(scores)
        print(f"  Rep {rep+1}: mean={np.mean(scores):.4f} ({elapsed:.0f}s)", flush=True)
    kill_server(proc, SOCKET_BF16)
    print(f"  Server stopped\n", flush=True)
else:
    bf16_norestart_scores = []
    print(f"  FAILED\n", flush=True)

# ============================================================
# SUMMARY
# ============================================================
print("=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)

def summarize(name, all_scores):
    if not all_scores:
        print(f"  {name}: NO DATA", flush=True)
        return
    means = [np.mean(s) for s in all_scores]
    print(f"  {name}:", flush=True)
    print(f"    Means:  {[f'{m:.4f}' for m in means]}", flush=True)
    print(f"    Std:    {np.std(means):.6f}", flush=True)
    print(f"    Spread: {max(means)-min(means):.6f}", flush=True)

summarize("FP32", fp32_scores)
summarize("BF16+Restart", bf16_restart_scores)
summarize("BF16 NoRestart", bf16_norestart_scores)

print(flush=True)
print("VERDICT:", flush=True)
fp32_std = np.std([np.mean(s) for s in fp32_scores]) if fp32_scores else 0
restart_std = np.std([np.mean(s) for s in bf16_restart_scores]) if bf16_restart_scores else 0
norestart_std = np.std([np.mean(s) for s in bf16_norestart_scores]) if bf16_norestart_scores else 0
print(f"  FP32 std:           {fp32_std:.6f}", flush=True)
print(f"  BF16+Restart std:   {restart_std:.6f}", flush=True)
print(f"  BF16 NoRestart std: {norestart_std:.6f}", flush=True)

results = {
    "fp32_scores": fp32_scores if fp32_scores else [],
    "bf16_restart_scores": bf16_restart_scores if bf16_restart_scores else [],
    "bf16_norestart_scores": bf16_norestart_scores if bf16_norestart_scores else [],
}
with open("logs/tfold_three_way_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to logs/tfold_three_way_comparison.json", flush=True)
