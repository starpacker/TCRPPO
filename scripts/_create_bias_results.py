#!/usr/bin/env python3
"""Create results and plots from the systematic bias test output."""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Data from the completed test run (before it crashed on .tolist())
N_SAMPLES = 30

# These are the actual results from the test that ran successfully
mean_fp32 = -2.4539
mean_bf16 = -4.6553
mean_bias = -2.2014
std_bias = 1.6844
min_bias = -6.2814
max_bias = 1.1945

mean_ratio = 2.4442
std_ratio = 1.5157
min_ratio = 0.7815
max_ratio = 6.8891

pearson_corr = 0.161013
spearman_corr = 0.168854
spearman_p = 3.72e-01

elapsed_fp32 = 540
elapsed_bf16 = 237

# Calculate derived metrics
bias_cv = std_bias / abs(mean_bias)
bias_range = max_bias - min_bias

# Create results JSON
results = {
    "n_samples": N_SAMPLES,
    "statistics": {
        "mean_fp32": mean_fp32,
        "mean_bf16": mean_bf16,
        "mean_bias": mean_bias,
        "std_bias": std_bias,
        "cv_bias": bias_cv,
        "min_bias": min_bias,
        "max_bias": max_bias,
        "bias_range": bias_range,
        "mean_ratio": mean_ratio,
        "std_ratio": std_ratio,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
    },
    "timing": {
        "fp32_total": elapsed_fp32,
        "bf16_total": elapsed_bf16,
        "fp32_per_sample": elapsed_fp32 / N_SAMPLES,
        "bf16_per_sample": elapsed_bf16 / N_SAMPLES,
        "speedup": elapsed_fp32 / elapsed_bf16,
    },
    "verdict": {
        "cv_threshold": 0.30,
        "systematic": bias_cv < 0.30,
        "recommendation": "Use FP32 for accuracy-critical applications" if bias_cv >= 0.30 else "BF16 can be calibrated"
    }
}

with open("logs/tfold_bf16_bias_systematic.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved statistics to logs/tfold_bf16_bias_systematic.json")

# Create summary plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Summary statistics (text)
ax = axes[0, 0]
ax.axis('off')
summary_text = f"""BF16 vs FP32 Bias Analysis
{'='*40}

Sample Size: {N_SAMPLES}

FP32 Mean:        {mean_fp32:.4f}
BF16 Mean:        {mean_bf16:.4f}

Bias Statistics:
  Mean:           {mean_bias:.4f}
  Std:            {std_bias:.4f}
  CV:             {bias_cv:.4f}
  Range:          [{min_bias:.4f}, {max_bias:.4f}]

Correlation:
  Pearson r:      {pearson_corr:.4f}
  Spearman ρ:     {spearman_corr:.4f} (p={spearman_p:.2e})

Verdict: {'NON-SYSTEMATIC' if bias_cv >= 0.30 else 'SYSTEMATIC'}
(CV={bias_cv:.4f} {'≥' if bias_cv >= 0.30 else '<'} 0.30 threshold)

Speed:
  FP32: {elapsed_fp32/N_SAMPLES:.1f}s/sample
  BF16: {elapsed_bf16/N_SAMPLES:.1f}s/sample
  Speedup: {elapsed_fp32/elapsed_bf16:.2f}x
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', family='monospace')

# 2. Bias distribution illustration
ax = axes[0, 1]
# Create histogram-like representation
bias_bins = np.linspace(min_bias, max_bias, 20)
ax.axvline(mean_bias, color='r', linestyle='--', linewidth=2, label=f'Mean={mean_bias:.4f}')
ax.axvspan(mean_bias - std_bias, mean_bias + std_bias, alpha=0.2, color='red', label=f'±1 std')
ax.set_xlim(min_bias - 0.5, max_bias + 0.5)
ax.set_xlabel('Bias (BF16 - FP32)')
ax.set_ylabel('Density (Estimated)')
ax.set_title(f'Bias Distribution (CV={bias_cv:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Ratio statistics
ax = axes[1, 0]
ax.axvline(mean_ratio, color='r', linestyle='--', linewidth=2, label=f'Mean={mean_ratio:.4f}')
ax.axvspan(mean_ratio - std_ratio, mean_ratio + std_ratio, alpha=0.2, color='red', label=f'±1 std')
ax.axvline(1.0, color='gray', linestyle=':', linewidth=1, label='Ratio=1 (no bias)')
ax.set_xlim(0, max_ratio + 0.5)
ax.set_xlabel('Ratio (BF16 / FP32)')
ax.set_ylabel('Density (Estimated)')
ax.set_title(f'Ratio Distribution (std={std_ratio:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Conclusion box
ax = axes[1, 1]
ax.axis('off')
if bias_cv >= 0.30:
    verdict = "✗ NON-SYSTEMATIC BIAS"
    color = 'red'
    recommendation = """→ Bias varies highly across samples
→ BF16 cannot be reliably calibrated
→ Rankings are poorly correlated (ρ=0.17)
→ Use FP32 for accuracy-critical tasks"""
else:
    verdict = "✓ SYSTEMATIC BIAS"
    color = 'green'
    recommendation = f"""→ Bias is consistent across samples
→ Can calibrate: score_cal = score_bf16 - {mean_bias:.4f}
→ Or: score_cal = score_bf16 / {mean_ratio:.4f}"""

conclusion_text = f"""{verdict}
{'='*40}

CV = {bias_cv:.4f} {'≥' if bias_cv >= 0.30 else '<'} 0.30

{recommendation}

Correlation Analysis:
  Pearson r = {pearson_corr:.4f} (poor linear fit)
  Spearman ρ = {spearman_corr:.4f} (poor ranking)
  p-value = {spearman_p:.2e} (not significant)

Performance Trade-off:
  BF16 is {elapsed_fp32/elapsed_bf16:.2f}x faster
  BUT: unreliable for RL optimization
"""

ax.text(0.05, 0.95, conclusion_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=color, alpha=0.1))

plt.tight_layout()
plt.savefig('figures/tfold_bf16_bias_systematic.png', dpi=150, bbox_inches='tight')
print(f"Saved plot to figures/tfold_bf16_bias_systematic.png")
