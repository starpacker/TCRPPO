#!/usr/bin/env python3
"""Compare trace91 vs trace92 configurations side-by-side."""

import yaml

configs = {
    "trace91_ultimate_fresh_start": "configs/trace91_ultimate_fresh_start.yaml",
    "trace92_ultimate_sft_rl": "configs/trace92_ultimate_sft_rl.yaml",
}

print("=" * 120)
print(f"{'Parameter':<40} {'trace91 (Fresh Start)':<40} {'trace92 (SFT+RL)':<40}")
print("=" * 120)

# Load configs
data = {}
for name, path in configs.items():
    with open(path) as f:
        data[name] = yaml.safe_load(f)

# Key parameters to compare
params = [
    ("total_timesteps", "Total Timesteps"),
    ("learning_rate", "Learning Rate"),
    ("entropy_coef", "Entropy Coefficient"),
    ("clip_range", "Clip Range"),
    ("w_naturalness", "Naturalness Weight"),
    ("naturalness_threshold_zscore", "Naturalness Threshold"),
    ("target_affinity_gate", "Starting Gate"),
    ("online_tcr_pool_start_step", "Pool Start Step"),
    ("online_tcr_pool_max_ratio", "Pool Max Ratio"),
    ("online_tcr_pool_min_affinity", "Pool Min Affinity"),
    ("resume_from", "Resume From"),
]

for key, label in params:
    v1 = data["trace91_ultimate_fresh_start"].get(key, "N/A")
    v2 = data["trace92_ultimate_sft_rl"].get(key, "N/A")

    # Format values
    v1_str = str(v1) if v1 != "N/A" else "N/A"
    v2_str = str(v2) if v2 != "N/A" else "N/A"

    # Highlight differences
    diff = " **" if v1 != v2 else ""

    print(f"{label:<40} {v1_str:<40} {v2_str:<40}{diff}")

print("=" * 120)
print("\n** = Different between configs")

# Gate schedules
print("\n" + "=" * 120)
print("Gate Schedules")
print("=" * 120)

print("\ntrace91 (Fresh Start):")
for item in data["trace91_ultimate_fresh_start"].get("gate_schedule", []):
    print(f"  Step {item['step']:>7,}: gate={item['gate']:>5.1f}")

print("\ntrace92 (SFT+RL):")
for item in data["trace92_ultimate_sft_rl"].get("gate_schedule", []):
    print(f"  Step {item['step']:>7,}: gate={item['gate']:>5.1f}")

print("\n" + "=" * 120)
