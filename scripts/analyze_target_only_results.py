#!/usr/bin/env python
"""分析multi-peptide评估结果，只关注目标peptides的亲和力。"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_log_file(log_path: str, target_peptides_file: str = "data/tfold_excellent_peptides.txt"):
    """解析日志文件，只提取目标peptides的结果。"""

    # 加载目标peptide列表
    target_peptides = set()
    if Path(target_peptides_file).exists():
        with open(target_peptides_file) as f:
            target_peptides = set(line.strip() for line in f if line.strip())

    # 从日志中提取实际使用的目标peptides
    with open(log_path, 'r') as f:
        for line in f:
            if line.startswith("Targets:"):
                targets_str = line.split("Targets:")[1].strip()
                actual_targets = eval(targets_str)  # ['GILGFVFTL', ...]
                break

    # 解析评估结果
    results = defaultdict(lambda: defaultdict(list))
    pattern = r'\[tFoldScore\].*affinity_logit=([-\d.]+).*cdr3b=(\S+) peptide=(\S+)'

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                affinity_logit = float(match.group(1))
                cdr3b = match.group(2)
                peptide = match.group(3)

                # 只保留目标peptides的结果
                if peptide in actual_targets:
                    results[peptide][cdr3b].append(affinity_logit)

    return results, actual_targets


def analyze_trace(trace_name: str, log_path: str):
    """分析单个trace的结果。"""

    if not Path(log_path).exists():
        return None

    results, target_peptides = parse_log_file(log_path)

    if not results:
        return None

    peptide_stats = {}

    for peptide in target_peptides:
        if peptide not in results:
            continue

        tcrs = results[peptide]

        # 对每个TCR取最大affinity（如果有多次评估）
        max_affinities = [max(scores) for scores in tcrs.values()]

        if max_affinities:
            peptide_stats[peptide] = {
                'n_tcrs': len(tcrs),
                'mean_affinity': np.mean(max_affinities),
                'max_affinity': np.max(max_affinities),
                'std_affinity': np.std(max_affinities),
            }

    if not peptide_stats:
        return None

    return {
        'trace_name': trace_name,
        'peptide_stats': peptide_stats,
        'overall_mean': np.mean([s['mean_affinity'] for s in peptide_stats.values()]),
        'overall_max': np.max([s['max_affinity'] for s in peptide_stats.values()]),
        'n_peptides_evaluated': len(peptide_stats),
    }


def main():
    output_dir = "results/fp32_eval_parallel_simple_20260603_114658"

    traces = ['trace61', 'trace72', 'trace73']
    all_results = {}

    print("="*80)
    print("FP32评估结果分析（仅目标peptides）")
    print("="*80)

    for trace_name in traces:
        log_path = f"{output_dir}/{trace_name}.log"

        print(f"\n{trace_name}:")
        print("-"*80)

        result = analyze_trace(trace_name, log_path)

        if result:
            all_results[trace_name] = result

            print(f"已评估peptides: {result['n_peptides_evaluated']}/20")
            print(f"整体平均亲和力: {result['overall_mean']:.4f}")
            print(f"整体最大亲和力: {result['overall_max']:.4f}")

            print(f"\nPer-peptide结果:")
            print(f"{'Peptide':<15} {'N TCRs':<10} {'Mean':<12} {'Max':<12}")
            print("-"*55)

            for peptide in sorted(result['peptide_stats'].keys()):
                stats = result['peptide_stats'][peptide]
                print(f"{peptide:<15} {stats['n_tcrs']:<10} {stats['mean_affinity']:<12.4f} {stats['max_affinity']:<12.4f}")
        else:
            print("无结果数据")

    if len(all_results) > 1:
        print(f"\n\n{'='*80}")
        print("对比")
        print(f"{'='*80}")
        print(f"\n{'Trace':<15} {'Peptides':<12} {'Mean':<15} {'Max':<15}")
        print("-"*60)

        for trace_name in sorted(all_results.keys()):
            result = all_results[trace_name]
            print(f"{trace_name:<15} {result['n_peptides_evaluated']:<12} {result['overall_mean']:<15.4f} {result['overall_max']:<15.4f}")

        # 找到最佳trace
        if all_results:
            best_trace = max(all_results.items(), key=lambda x: x[1]['overall_mean'])
            print("-"*60)
            print(f"\n最佳trace（按平均亲和力）: {best_trace[0]} ({best_trace[1]['overall_mean']:.4f})")


if __name__ == "__main__":
    main()
