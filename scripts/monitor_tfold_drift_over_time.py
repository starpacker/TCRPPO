#!/usr/bin/env python3
"""
Long-running experiment: Monitor tFold scorer drift over time.

Setup:
  1. Start a fresh tFold server on an isolated GPU (no training load).
  2. Score a fixed test set of 50 TCR-peptide pairs every N minutes.
  3. Log scores to file and plot drift over time.

Hypothesis:
  If scores remain stable over hours, the drift is caused by GPU contention
  from concurrent training. If scores still degrade, the issue is intrinsic
  to long-running tFold servers (e.g., numerical precision drift, cache
  pollution, etc.).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, "/share/liuyutian/tcrppo_v2")

from scripts.test_tfold_scorer_drift import (
    generate_test_pairs,
    extract_features_via_socket,
    load_classifier,
    classify_features,
)


def main():
    parser = argparse.ArgumentParser(description="Monitor tFold scorer drift over time")
    parser.add_argument("--socket", required=True, help="tFold server socket path")
    parser.add_argument("--n-pairs", type=int, default=50, help="Number of test pairs")
    parser.add_argument("--interval-minutes", type=int, default=30, help="Scoring interval in minutes")
    parser.add_argument("--duration-hours", type=int, default=12, help="Total monitoring duration in hours")
    parser.add_argument("--output-dir", default="logs/tfold_drift_monitoring", help="Output directory")
    parser.add_argument("--seed", type=int, default=999, help="Random seed for test pair generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate fixed test pairs once
    print("=" * 70)
    print("tFold Scorer Drift Monitoring (Long-Running)")
    print("=" * 70)
    print(f"\nSocket:       {args.socket}")
    print(f"Test pairs:   {args.n_pairs}")
    print(f"Interval:     {args.interval_minutes} min")
    print(f"Duration:     {args.duration_hours} hours")
    print(f"Output dir:   {args.output_dir}")
    print()

    print(f"[Setup] Generating {args.n_pairs} fixed test pairs (seed={args.seed})...")
    pairs = generate_test_pairs(n=args.n_pairs, seed=args.seed)
    samples = [{"cdr3b": p["cdr3b"], "peptide": p["peptide"], "hla": p["hla"]} for p in pairs]

    # Save test pairs
    pairs_file = os.path.join(args.output_dir, "test_pairs.json")
    with open(pairs_file, "w") as f:
        json.dump([{"cdr3b": p["cdr3b"], "peptide": p["peptide"]} for p in pairs], f, indent=2)
    print(f"    Saved {len(pairs)} pairs to {pairs_file}")

    # Load classifier once
    print("[Setup] Loading V3.4 classifier...")
    classifier = load_classifier("cpu")
    print("    Classifier loaded")

    # Check socket exists
    if not os.path.exists(args.socket):
        print(f"\nERROR: Socket not found at {args.socket}")
        print("Start the tFold server first:")
        print(f"  python scripts/tfold_feature_server.py --socket {args.socket} --gpu <GPU_ID> --use-amp-wrapper --chunk-size 64")
        sys.exit(1)

    print(f"\n[Ready] Starting monitoring loop...")
    print(f"    Will score every {args.interval_minutes} minutes for {args.duration_hours} hours")
    print()

    # Monitoring loop
    results = []
    start_time = time.time()
    end_time = start_time + (args.duration_hours * 3600)
    round_num = 0

    while time.time() < end_time:
        round_num += 1
        elapsed_hours = (time.time() - start_time) / 3600

        print(f"[Round {round_num}] Elapsed: {elapsed_hours:.2f} hours | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Score through server
            t0 = time.time()
            feats = extract_features_via_socket(args.socket, samples, timeout=600)
            extract_time = time.time() - t0
            scores = classify_features(classifier, feats)
            n_ok = sum(1 for f in feats if f is not None)

            # Compute stats
            scores_arr = np.array(scores)
            result = {
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                "elapsed_hours": elapsed_hours,
                "scores": scores,
                "n_ok": n_ok,
                "extract_time_s": extract_time,
                "mean": float(scores_arr.mean()),
                "std": float(scores_arr.std()),
                "min": float(scores_arr.min()),
                "max": float(scores_arr.max()),
            }
            results.append(result)

            print(f"    Extracted: {n_ok}/{len(samples)} in {extract_time:.1f}s")
            print(f"    Scores: mean={result['mean']:.4f}, std={result['std']:.4f}, range=[{result['min']:.4f}, {result['max']:.4f}]")

            # Save results after each round
            results_file = os.path.join(args.output_dir, "monitoring_results.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            # Drift analysis (if we have baseline)
            if len(results) > 1:
                baseline_mean = results[0]["mean"]
                current_mean = result["mean"]
                drift = current_mean - baseline_mean
                print(f"    Drift from baseline: {drift:+.4f} (baseline={baseline_mean:.4f})")

            print()

        except Exception as e:
            print(f"    ERROR: {e}")
            print()

        # Wait for next interval
        if time.time() < end_time:
            sleep_seconds = args.interval_minutes * 60
            next_time = time.time() + sleep_seconds
            print(f"[Sleep] Waiting {args.interval_minutes} min until next round...")
            print(f"        Next round at: {datetime.fromtimestamp(next_time).strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            time.sleep(sleep_seconds)

    print("=" * 70)
    print("Monitoring Complete")
    print("=" * 70)
    print(f"\nTotal rounds: {len(results)}")
    print(f"Output: {os.path.join(args.output_dir, 'monitoring_results.json')}")

    if len(results) >= 2:
        baseline_mean = results[0]["mean"]
        final_mean = results[-1]["mean"]
        total_drift = final_mean - baseline_mean
        print(f"\nBaseline mean (round 1): {baseline_mean:.4f}")
        print(f"Final mean (round {len(results)}):    {final_mean:.4f}")
        print(f"Total drift:             {total_drift:+.4f}")

        # Per-round drift
        drifts = [r["mean"] - baseline_mean for r in results]
        print(f"\nDrift over time:")
        for i, (r, d) in enumerate(zip(results, drifts), 1):
            print(f"  Round {i:2d} ({r['elapsed_hours']:5.2f}h): mean={r['mean']:.4f}, drift={d:+.4f}")


if __name__ == "__main__":
    main()
