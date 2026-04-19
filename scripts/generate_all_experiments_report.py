#!/usr/bin/env python3
"""
Generate comprehensive report for all TCRPPO v2 experiments (test1-test18+).
Reads experiment.json and eval_results.json, creates markdown report and visualizations.
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from datetime import datetime

# Target peptides for heatmap
TARGETS = [
    "GILGFVFTL", "NLVPMVATV", "GLCTLVAML", "LLWNGPMAV",
    "YLQPRTFLL", "FLYALALLL", "SLYNTVATL", "KLGGALQAK",
    "AVFDRKSDAK", "IVTDFSVIK", "SPRWYFYYL", "RLRAEAQVK"
]

def load_experiment_data():
    """Load all experiment configs and eval results."""
    experiments = []

    # Find all test* directories in output/
    exp_dirs = sorted(
        glob.glob('output/test*/'),
        key=lambda x: (len(os.path.basename(x.rstrip('/'))), os.path.basename(x.rstrip('/')))
    )

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir.rstrip('/'))

        # Load experiment.json
        exp_json_path = os.path.join(exp_dir, 'experiment.json')
        if not os.path.exists(exp_json_path):
            print(f"Warning: {exp_name} missing experiment.json")
            continue

        with open(exp_json_path, 'r') as f:
            exp_data = json.load(f)

        # Try to load eval_results.json
        eval_json_path = f'results/{exp_name}/eval_results.json'
        eval_data = None
        if os.path.exists(eval_json_path):
            with open(eval_json_path, 'r') as f:
                eval_data = json.load(f)

        experiments.append({
            'name': exp_name,
            'config': exp_data,
            'eval': eval_data
        })

    return experiments

def extract_mean_auroc(exp):
    """Extract mean AUROC from experiment, handling both old and new formats."""
    if exp['eval'] is None:
        return None

    # New format: multi-scorer with per-target structure
    if isinstance(exp['eval'], dict) and any(target in exp['eval'] for target in TARGETS):
        # Default to ERGO scorer for primary metric
        aurocs = []
        for target in TARGETS:
            if target in exp['eval']:
                target_data = exp['eval'][target]
                if 'specificity' in target_data:
                    # Multi-scorer format (new)
                    if 'ergo' in target_data['specificity']:
                        aurocs.append(target_data['specificity']['ergo']['auroc'])
                    # Old format: auroc directly in specificity dict
                    elif 'auroc' in target_data['specificity']:
                        aurocs.append(target_data['specificity']['auroc'])
                elif 'auroc' in target_data:
                    # Old single-scorer format
                    aurocs.append(target_data['auroc'])
        return np.mean(aurocs) if aurocs else None

    # Old format: results dict with mean_auroc
    if 'results' in exp['config'] and 'mean_auroc' in exp['config']['results']:
        return exp['config']['results']['mean_auroc']

    return None

def extract_per_target_auroc(exp, scorer='ergo'):
    """Extract per-target AUROC for a specific scorer."""
    if exp['eval'] is None:
        return {}

    aurocs = {}
    for target in TARGETS:
        if target in exp['eval']:
            target_data = exp['eval'][target]
            if 'specificity' in target_data:
                # Multi-scorer format (new)
                if scorer in target_data['specificity']:
                    aurocs[target] = target_data['specificity'][scorer]['auroc']
                # Old format: auroc directly in specificity dict
                elif 'auroc' in target_data['specificity']:
                    aurocs[target] = target_data['specificity']['auroc']
            elif 'auroc' in target_data:
                # Old single-scorer format
                aurocs[target] = target_data['auroc']

    return aurocs

def generate_markdown_report(experiments):
    """Generate comprehensive markdown report."""

    lines = []
    lines.append("# TCRPPO v2 Experiments: Comprehensive Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Experiments:** {len(experiments)}")
    lines.append("")

    # Executive summary table
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("| Exp | Status | Reward Mode | Scorer | Encoder | Seed | Steps | AUROC | Notes |")
    lines.append("|-----|--------|-------------|--------|---------|------|-------|-------|-------|")

    for exp in experiments:
        name = exp['name']
        cfg = exp['config'].get('config', {})
        status = exp['config'].get('status', 'unknown')
        reward_mode = cfg.get('reward_mode', 'N/A')
        scorer = cfg.get('affinity_scorer', 'ergo')
        encoder = cfg.get('state_encoder', 'esm2')
        seed = cfg.get('seed', 'N/A')
        steps = cfg.get('total_timesteps', 'N/A')

        auroc = extract_mean_auroc(exp)
        auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"

        notes = exp['config'].get('notes', '')[:50]

        lines.append(f"| {name} | {status} | {reward_mode} | {scorer} | {encoder} | {seed} | {steps} | {auroc_str} | {notes} |")

    lines.append("")

    # Group by reward mode
    lines.append("## Results by Reward Mode")
    lines.append("")

    by_mode = defaultdict(list)
    for exp in experiments:
        mode = exp['config'].get('config', {}).get('reward_mode', 'unknown')
        auroc = extract_mean_auroc(exp)
        if auroc is not None:
            by_mode[mode].append(auroc)

    lines.append("| Reward Mode | N | Mean AUROC | Std | Min | Max |")
    lines.append("|-------------|---|------------|-----|-----|-----|")

    for mode in sorted(by_mode.keys()):
        aurocs = by_mode[mode]
        lines.append(f"| {mode} | {len(aurocs)} | {np.mean(aurocs):.4f} | {np.std(aurocs):.4f} | {np.min(aurocs):.4f} | {np.max(aurocs):.4f} |")

    lines.append("")

    # Group by affinity scorer
    lines.append("## Results by Affinity Scorer")
    lines.append("")

    by_scorer = defaultdict(list)
    for exp in experiments:
        scorer = exp['config'].get('config', {}).get('affinity_scorer', 'ergo')
        auroc = extract_mean_auroc(exp)
        if auroc is not None:
            by_scorer[scorer].append(auroc)

    lines.append("| Affinity Scorer | N | Mean AUROC | Std | Min | Max |")
    lines.append("|-----------------|---|------------|-----|-----|-----|")

    for scorer in sorted(by_scorer.keys()):
        aurocs = by_scorer[scorer]
        lines.append(f"| {scorer} | {len(aurocs)} | {np.mean(aurocs):.4f} | {np.std(aurocs):.4f} | {np.min(aurocs):.4f} | {np.max(aurocs):.4f} |")

    lines.append("")

    # Detailed per-experiment sections
    lines.append("## Detailed Experiment Reports")
    lines.append("")

    for exp in experiments:
        name = exp['name']
        cfg = exp['config'].get('config', {})
        status = exp['config'].get('status', 'unknown')

        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"**Status:** {status}")
        lines.append("")

        lines.append("**Configuration:**")
        lines.append("```")
        lines.append(f"Reward Mode: {cfg.get('reward_mode', 'N/A')}")
        lines.append(f"Affinity Scorer: {cfg.get('affinity_scorer', 'ergo')}")
        lines.append(f"State Encoder: {cfg.get('state_encoder', 'esm2')}")
        lines.append(f"Seed: {cfg.get('seed', 'N/A')}")
        lines.append(f"Total Steps: {cfg.get('total_timesteps', 'N/A')}")
        lines.append(f"Environments: {cfg.get('n_envs', 'N/A')}")
        lines.append(f"Learning Rate: {cfg.get('learning_rate', 'N/A')}")
        lines.append(f"Hidden Dim: {cfg.get('hidden_dim', 'N/A')}")
        lines.append(f"Max Steps: {cfg.get('max_steps', 'N/A')}")

        weights = cfg.get('weights', {})
        if weights:
            lines.append(f"Weights: aff={weights.get('affinity', 1.0)}, dec={weights.get('decoy', 0.0)}, nat={weights.get('naturalness', 0.0)}, div={weights.get('diversity', 0.0)}")

        lines.append("```")
        lines.append("")

        # Evaluation results
        if exp['eval'] is not None:
            lines.append("**Evaluation Results:**")
            lines.append("")

            auroc = extract_mean_auroc(exp)
            if auroc is not None:
                lines.append(f"- **Mean AUROC (ERGO):** {auroc:.4f}")

            # Per-target breakdown
            per_target = extract_per_target_auroc(exp, 'ergo')
            if per_target:
                lines.append("")
                lines.append("Per-target AUROC (ERGO):")
                lines.append("")
                lines.append("| Target | AUROC |")
                lines.append("|--------|-------|")
                for target in TARGETS:
                    if target in per_target:
                        lines.append(f"| {target} | {per_target[target]:.4f} |")

            lines.append("")
        else:
            lines.append("**Evaluation:** Not yet evaluated")
            lines.append("")

        # Notes
        notes = exp['config'].get('notes', '')
        if notes:
            lines.append(f"**Notes:** {notes}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return '\n'.join(lines)

def generate_visualizations(experiments):
    """Generate comprehensive comparison visualizations."""

    # Filter experiments with eval data
    evaluated = [exp for exp in experiments if extract_mean_auroc(exp) is not None]

    if not evaluated:
        print("No evaluated experiments to visualize")
        return

    fig = plt.figure(figsize=(20, 12))

    # 1. Bar chart: Mean AUROC across all experiments
    ax1 = plt.subplot(2, 3, 1)
    names = [exp['name'] for exp in evaluated]
    aurocs = [extract_mean_auroc(exp) for exp in evaluated]

    colors = ['green' if a >= 0.6 else 'orange' if a >= 0.5 else 'red' for a in aurocs]
    bars = ax1.bar(range(len(names)), aurocs, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Mean AUROC (ERGO)', fontsize=10)
    ax1.set_title('Mean AUROC Across All Experiments', fontsize=12, fontweight='bold')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax1.axhline(y=0.6, color='blue', linestyle='--', alpha=0.5, label='Target')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Grouped bar chart: AUROC by reward mode
    ax2 = plt.subplot(2, 3, 2)
    by_mode = defaultdict(list)
    for exp in evaluated:
        mode = exp['config'].get('config', {}).get('reward_mode', 'unknown')
        auroc = extract_mean_auroc(exp)
        by_mode[mode].append(auroc)

    modes = sorted(by_mode.keys())
    mode_means = [np.mean(by_mode[m]) for m in modes]
    mode_stds = [np.std(by_mode[m]) if len(by_mode[m]) > 1 else 0 for m in modes]

    bars = ax2.bar(range(len(modes)), mode_means, yerr=mode_stds, capsize=5, alpha=0.7, color='steelblue')
    ax2.set_xticks(range(len(modes)))
    ax2.set_xticklabels(modes, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Mean AUROC', fontsize=10)
    ax2.set_title('AUROC by Reward Mode', fontsize=12, fontweight='bold')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Scatter plot: AUROC vs training steps
    ax3 = plt.subplot(2, 3, 3)
    steps_list = []
    auroc_list = []
    colors_list = []

    for exp in evaluated:
        steps = exp['config'].get('config', {}).get('total_timesteps', 0)
        auroc = extract_mean_auroc(exp)
        mode = exp['config'].get('config', {}).get('reward_mode', 'unknown')

        steps_list.append(steps / 1e6)  # Convert to millions
        auroc_list.append(auroc)

        # Color by reward mode
        if 'v1_ergo_only' in mode:
            colors_list.append('red')
        elif 'raw_decoy' in mode:
            colors_list.append('blue')
        elif 'stepwise' in mode:
            colors_list.append('green')
        else:
            colors_list.append('gray')

    ax3.scatter(steps_list, auroc_list, c=colors_list, alpha=0.6, s=100)
    ax3.set_xlabel('Training Steps (millions)', fontsize=10)
    ax3.set_ylabel('Mean AUROC', fontsize=10)
    ax3.set_title('AUROC vs Training Steps', fontsize=12, fontweight='bold')
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(alpha=0.3)

    # Legend for scatter plot
    red_patch = mpatches.Patch(color='red', label='v1_ergo_only')
    blue_patch = mpatches.Patch(color='blue', label='raw_decoy')
    green_patch = mpatches.Patch(color='green', label='stepwise')
    gray_patch = mpatches.Patch(color='gray', label='other')
    ax3.legend(handles=[red_patch, blue_patch, green_patch, gray_patch], fontsize=8)

    # 4. Heatmap: Per-target AUROC across experiments
    ax4 = plt.subplot(2, 3, 4)

    # Build matrix: experiments x targets
    matrix = []
    exp_labels = []
    for exp in evaluated:
        per_target = extract_per_target_auroc(exp, 'ergo')
        row = [per_target.get(t, np.nan) for t in TARGETS]
        matrix.append(row)
        exp_labels.append(exp['name'])

    matrix = np.array(matrix)

    im = ax4.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.3, vmax=0.8)
    ax4.set_xticks(range(len(TARGETS)))
    ax4.set_xticklabels(TARGETS, rotation=45, ha='right', fontsize=7)
    ax4.set_yticks(range(len(exp_labels)))
    ax4.set_yticklabels(exp_labels, fontsize=7)
    ax4.set_title('Per-Target AUROC Heatmap (ERGO)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4, label='AUROC')

    # 5. Box plot: AUROC distribution by affinity scorer
    ax5 = plt.subplot(2, 3, 5)
    by_scorer = defaultdict(list)
    for exp in evaluated:
        scorer = exp['config'].get('config', {}).get('affinity_scorer', 'ergo')
        auroc = extract_mean_auroc(exp)
        by_scorer[scorer].append(auroc)

    scorers = sorted(by_scorer.keys())
    data = [by_scorer[s] for s in scorers]

    bp = ax5.boxplot(data, labels=scorers, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax5.set_ylabel('Mean AUROC', fontsize=10)
    ax5.set_title('AUROC Distribution by Affinity Scorer', fontsize=12, fontweight='bold')
    ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax5.grid(axis='y', alpha=0.3)

    # 6. Top performers table (text)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Sort by AUROC
    sorted_exps = sorted(evaluated, key=lambda x: extract_mean_auroc(x), reverse=True)
    top_5 = sorted_exps[:5]

    table_data = [['Rank', 'Experiment', 'AUROC', 'Mode']]
    for i, exp in enumerate(top_5, 1):
        name = exp['name']
        auroc = extract_mean_auroc(exp)
        mode = exp['config'].get('config', {}).get('reward_mode', 'N/A')[:20]
        table_data.append([str(i), name, f"{auroc:.4f}", mode])

    table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                      colWidths=[0.1, 0.3, 0.15, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax6.set_title('Top 5 Experiments by AUROC', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('figures/all_experiments_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/all_experiments_comparison.png")

    plt.close()

def main():
    print("Loading experiment data...")
    experiments = load_experiment_data()
    print(f"Loaded {len(experiments)} experiments")

    print("\nGenerating markdown report...")
    report = generate_markdown_report(experiments)

    os.makedirs('docs', exist_ok=True)
    with open('docs/all_experiments_report.md', 'w') as f:
        f.write(report)
    print("Saved: docs/all_experiments_report.md")

    print("\nGenerating visualizations...")
    os.makedirs('figures', exist_ok=True)
    generate_visualizations(experiments)

    print("\nDone!")

if __name__ == '__main__':
    main()
