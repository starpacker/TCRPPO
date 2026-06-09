#!/usr/bin/env python3
"""
Score a TCR against target and all decoy peptides to identify which decoys have high affinity.
"""

import json
import sys
import os

sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.scorers.affinity_tfold import TFoldAffinityScorer

def load_decoys(target_peptide, decoy_lib_path='/share/liuyutian/pMHC_decoy_library'):
    """Load all decoy peptides for a target."""
    decoys = []

    # Tier A
    tier_a_dir = f'{decoy_lib_path}/data/decoy_a/{target_peptide}'
    if os.path.exists(tier_a_dir):
        for fname in os.listdir(tier_a_dir):
            if fname.endswith('.json'):
                with open(f'{tier_a_dir}/{fname}') as f:
                    data = json.load(f)
                    for item in data:
                        decoys.append({
                            'peptide': item['sequence'],
                            'tier': 'A',
                            'similarity': item.get('similarity_score', 0),
                            'hamming': item.get('hamming_distance', 0)
                        })

    # Tier B
    tier_b_dir = f'{decoy_lib_path}/data/decoy_b/{target_peptide}'
    if os.path.exists(tier_b_dir):
        for fname in os.listdir(tier_b_dir):
            if fname.endswith('.json'):
                with open(f'{tier_b_dir}/{fname}') as f:
                    data = json.load(f)
                    for item in data:
                        decoys.append({
                            'peptide': item['sequence'],
                            'tier': 'B',
                            'similarity': item.get('similarity_score', 0),
                            'hamming': item.get('hamming_distance', 0)
                        })

    # Tier C (sample from large library)
    tier_c_file = f'{decoy_lib_path}/data/decoy_c/decoy_library.json'
    if os.path.exists(tier_c_file):
        with open(tier_c_file) as f:
            tier_c_data = json.load(f)
            # Sample first 20 from tier C
            for item in tier_c_data[:20]:
                decoys.append({
                    'peptide': item['sequence'],
                    'tier': 'C',
                    'similarity': 0.0,
                    'hamming': 999
                })

    # Tier D (known binders)
    tier_d_dir = f'{decoy_lib_path}/data/decoy_d/{target_peptide}'
    if os.path.exists(tier_d_dir):
        for fname in os.listdir(tier_d_dir):
            if fname.endswith('.json'):
                with open(f'{tier_d_dir}/{fname}') as f:
                    data = json.load(f)
                    for item in data:
                        decoys.append({
                            'peptide': item['sequence'],
                            'tier': 'D',
                            'similarity': item.get('similarity_score', 0),
                            'hamming': item.get('hamming_distance', 0)
                        })

    return decoys


def main():
    # Best TCR from trace61
    tcr = "CALTGWTYNEQAFYYCCCCF"
    target_peptide = "LLLDRLNQL"
    hla = "HLA-A*02:01"

    print(f"Analyzing TCR: {tcr}")
    print(f"Target peptide: {target_peptide}")
    print(f"HLA: {hla}")
    print()

    # Initialize tFold scorer
    print("Initializing tFold scorer...")
    scorer = TFoldAffinityScorer(
        server_socket="/tmp/tfold_server_trace90_single_peptide.sock",
        use_cache=True,
        cache_path="data/tfold_feature_cache_trace90_single_peptide.db"
    )

    # Score target
    print("Scoring target peptide...")
    target_score = scorer.score_batch_fast([tcr], [target_peptide])[0]
    print(f"Target affinity: {target_score:.4f}")
    print()

    # Load decoys
    print("Loading decoy peptides...")
    decoys = load_decoys(target_peptide)
    print(f"Found {len(decoys)} decoy peptides")
    print()

    # Score all decoys
    print("Scoring decoys...")
    decoy_results = []
    for i, decoy in enumerate(decoys):
        score = scorer.score_batch_fast([tcr], [decoy['peptide']])[0]
        decoy_results.append({
            **decoy,
            'affinity': score
        })
        if (i + 1) % 10 == 0:
            print(f"  Scored {i+1}/{len(decoys)} decoys...")

    print()

    # Sort by affinity (highest first)
    decoy_results.sort(key=lambda x: x['affinity'], reverse=True)

    # Show top decoys with high affinity
    print("=" * 80)
    print("TOP DECOYS WITH HIGH AFFINITY (explaining the high decoy_affinity)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Tier':<6} {'Affinity':<10} {'vs Target':<10} {'Peptide':<15} {'Similarity':<12}")
    print("-" * 80)

    for i, result in enumerate(decoy_results[:20], 1):
        delta = result['affinity'] - target_score
        print(f"{i:<6} {result['tier']:<6} {result['affinity']:<10.4f} {delta:+10.4f} {result['peptide']:<15} {result['similarity']:<12.3f}")

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    high_affinity_decoys = [r for r in decoy_results if r['affinity'] > 0.6]
    print(f"Decoys with affinity > 0.6: {len(high_affinity_decoys)}")

    if high_affinity_decoys:
        print("\nThese decoys explain the high decoy_affinity score:")
        for r in high_affinity_decoys:
            print(f"  - {r['peptide']} (tier {r['tier']}): {r['affinity']:.4f}")

    print()
    print("Conclusion:")
    if len(high_affinity_decoys) > 0:
        print(f"  This TCR binds to {len(high_affinity_decoys)} decoys with affinity > 0.6")
        print(f"  It is a PROMISCUOUS BINDER (lacks specificity)")
    else:
        print(f"  This TCR has good specificity (no decoys with affinity > 0.6)")


if __name__ == "__main__":
    main()
