#!/usr/bin/env python3
"""Quick check of tFold drift monitoring progress."""

import json
import sys

results_file = "logs/tfold_drift_isolated_gpu2/monitoring_results.json"

try:
    with open(results_file) as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Results file not found: {results_file}")
    sys.exit(1)

if not data:
    print("No rounds completed yet")
    sys.exit(0)

baseline = data[0]["mean"]
print("=" * 70)
print("tFold Drift Monitoring Progress")
print("=" * 70)
print(f"Total rounds: {len(data)}")
print(f"Baseline (round 1): {baseline:.4f}")
print()

print(f"{'Round':<6} {'Elapsed':>8} {'Mean':>8} {'Drift':>8} {'StdDev':>8} {'Range':>20}")
print("-" * 70)

for r in data:
    drift = r["mean"] - baseline
    range_str = f"[{r['min']:.2f}, {r['max']:.2f}]"
    print(f"{r['round']:<6} {r['elapsed_hours']:>8.2f}h {r['mean']:>8.4f} {drift:>+8.4f} {r['std']:>8.4f} {range_str:>20}")

if len(data) > 1:
    print()
    print("Drift Analysis:")
    final_mean = data[-1]["mean"]
    total_drift = final_mean - baseline
    max_drift = max(abs(r["mean"] - baseline) for r in data)
    print(f"  Total drift (baseline → latest): {total_drift:+.4f}")
    print(f"  Max absolute drift:              {max_drift:.4f}")

    # Check stability (within ±0.2 is considered stable)
    if max_drift < 0.2:
        print(f"  Status: ✓ STABLE (drift < 0.2)")
    elif max_drift < 0.5:
        print(f"  Status: ⚠ MINOR DRIFT (0.2 < drift < 0.5)")
    else:
        print(f"  Status: ✗ SIGNIFICANT DRIFT (drift > 0.5)")

print()
print(f"Last update: {data[-1]['timestamp']}")
print(f"Next round in: ~30 minutes")
