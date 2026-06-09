#!/usr/bin/env python3
"""
Build SFT v2 dataset: SUB-only trajectories with reverse augmentation.

Strategy:
1. Collect unique final TCRs from raw pairs with their best affinity/peptide/hla
2. Filter out finals with CCCC+ (4+ consecutive C)
3. For each unique final TCR, generate N random init TCRs via random substitution
4. Each trajectory = list of SUB actions to go from init -> final
5. Shuffle action orderings for path diversity
"""

import json
import random
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")


def collect_unique_finals(pairs: List[Dict]) -> Dict[str, Dict]:
    """Collect unique final TCRs, keeping best affinity and all associated peptides."""
    finals = {}
    for p in pairs:
        ft = p['final_tcr']
        a = p['final_affinity']
        if ft not in finals or a > finals[ft]['affinity']:
            finals[ft] = {
                'tcr': ft,
                'affinity': a,
                'peptide': p['peptide'],
                'hla': p['hla'],
            }
    return finals


def filter_cccc(finals: Dict[str, Dict]) -> Dict[str, Dict]:
    """Remove final TCRs with 4+ consecutive identical characters."""
    return {k: v for k, v in finals.items() if not re.search(r'(.)\1{3,}', k)}


def generate_init_tcr(final_tcr: str, n_subs: int) -> Tuple[str, List[Dict]]:
    """Generate a random init TCR by substituting n_subs positions in final_tcr.

    Returns:
        (init_tcr, actions) where actions are SUB ops to go from init -> final.
    """
    L = len(final_tcr)
    if n_subs > L:
        n_subs = L

    # Pick random positions to mutate
    positions = random.sample(range(L), n_subs)

    # Build init TCR by replacing those positions with random different AAs
    init_list = list(final_tcr)
    mutations = {}  # pos -> (init_aa, final_aa)
    for pos in positions:
        original_aa = final_tcr[pos]
        # Pick a different AA
        candidates = [aa for aa in AMINO_ACIDS if aa != original_aa]
        new_aa = random.choice(candidates)
        init_list[pos] = new_aa
        mutations[pos] = (new_aa, original_aa)

    init_tcr = ''.join(init_list)

    # Generate actions in random order (path diversity)
    shuffled_positions = list(mutations.keys())
    random.shuffle(shuffled_positions)

    actions = []
    for pos in shuffled_positions:
        _, final_aa = mutations[pos]
        actions.append({
            'op_type': 0,  # SUB
            'position': pos,
            'token': final_aa,
        })

    return init_tcr, actions


def verify_trajectory(init_tcr: str, final_tcr: str, actions: List[Dict]) -> bool:
    """Verify that applying SUB actions to init produces final."""
    current = list(init_tcr)
    for a in actions:
        pos = a['position']
        if pos >= len(current):
            return False
        current[pos] = a['token']
    return ''.join(current) == final_tcr


def build_dataset(
    pairs: List[Dict],
    n_inits_per_final: int = 20,
    max_steps: int = 8,
    seed: int = 42,
) -> List[Dict]:
    """Build SUB-only SFT v2 dataset with reverse augmentation."""
    random.seed(seed)

    # Step 1: Collect unique finals
    finals = collect_unique_finals(pairs)
    print(f"Unique final TCRs: {len(finals)}")

    # Step 2: Filter CCCC+
    finals = filter_cccc(finals)
    print(f"After CCCC+ filter: {len(finals)}")

    # Collect all peptide/hla associations per final TCR (for diversity)
    final_contexts = defaultdict(list)
    for p in pairs:
        ft = p['final_tcr']
        if ft in finals:
            final_contexts[ft].append({
                'peptide': p['peptide'],
                'hla': p['hla'],
            })

    # Step 3: Generate trajectories
    trajectories = []
    n_verified = 0
    n_failed = 0

    for tcr, info in finals.items():
        contexts = final_contexts[tcr]

        for i in range(n_inits_per_final):
            # Random number of substitutions (1 to max_steps)
            n_subs = random.randint(1, min(max_steps, len(tcr)))

            init_tcr, actions = generate_init_tcr(tcr, n_subs)

            # Verify
            if not verify_trajectory(init_tcr, tcr, actions):
                n_failed += 1
                continue
            n_verified += 1

            # Pick a context (peptide/hla) - rotate through available ones
            ctx = contexts[i % len(contexts)]

            trajectories.append({
                'init_tcr': init_tcr,
                'final_tcr': tcr,
                'peptide': ctx['peptide'],
                'hla': ctx['hla'],
                'final_affinity': info['affinity'],
                'init_affinity': -10.0,  # unknown, placeholder
                'delta_affinity': info['affinity'] + 10.0,
                'actions': actions,
                'reconstruction_method': 'reverse_augment',
                'n_subs': n_subs,
            })

    print(f"Generated {len(trajectories)} trajectories (verified: {n_verified}, failed: {n_failed})")

    # Step 4: Also include original same-length pairs reconstructed as SUB-only
    n_original = 0
    for p in pairs:
        if len(p['init_tcr']) != len(p['final_tcr']):
            continue
        if re.search(r'(.)\1{3,}', p['final_tcr']):
            continue

        # Reconstruct as SUB-only
        init_tcr = p['init_tcr']
        final_tcr = p['final_tcr']
        actions = []
        positions = []
        for pos in range(len(init_tcr)):
            if init_tcr[pos] != final_tcr[pos]:
                positions.append(pos)

        if len(positions) == 0 or len(positions) > max_steps:
            continue

        # Random order for path diversity
        random.shuffle(positions)
        for pos in positions:
            actions.append({
                'op_type': 0,
                'position': pos,
                'token': final_tcr[pos],
            })

        if verify_trajectory(init_tcr, final_tcr, actions):
            trajectories.append({
                'init_tcr': init_tcr,
                'final_tcr': final_tcr,
                'peptide': p['peptide'],
                'hla': p['hla'],
                'final_affinity': p['final_affinity'],
                'init_affinity': p['init_affinity'],
                'delta_affinity': p['delta_affinity'],
                'actions': actions,
                'reconstruction_method': 'original_sub_only',
                'n_subs': len(positions),
            })
            n_original += 1

    print(f"Added {n_original} original same-length pairs")
    print(f"Total trajectories: {len(trajectories)}")

    # Shuffle
    random.shuffle(trajectories)
    return trajectories


def compute_stats(trajectories: List[Dict]) -> Dict:
    """Compute dataset statistics."""
    by_method = defaultdict(int)
    by_bin = {'high': 0, 'medium': 0, 'low': 0}
    n_subs_dist = defaultdict(int)
    lengths = []

    for t in trajectories:
        by_method[t['reconstruction_method']] += 1

        a = t['final_affinity']
        if a >= 0:
            by_bin['high'] += 1
        elif a >= -2:
            by_bin['medium'] += 1
        else:
            by_bin['low'] += 1

        n_subs_dist[t['n_subs']] += 1
        lengths.append(len(t['final_tcr']))

    return {
        'total': len(trajectories),
        'by_method': dict(by_method),
        'by_bin': by_bin,
        'n_subs_distribution': dict(sorted(n_subs_dist.items())),
        'mean_final_len': sum(lengths) / len(lengths) if lengths else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Build SFT v2 dataset")
    parser.add_argument("--input", type=str,
                        default="/share/liuyutian/tcrppo_v2/data/sft_raw_pairs.json")
    parser.add_argument("--output", type=str,
                        default="/share/liuyutian/tcrppo_v2/data/sft_v2_trajectories.json")
    parser.add_argument("--n_inits_per_final", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load raw pairs
    print(f"Loading pairs from {args.input}")
    with open(args.input) as f:
        data = json.load(f)
    pairs = data['pairs']
    print(f"Loaded {len(pairs)} pairs")

    # Build dataset
    trajectories = build_dataset(
        pairs,
        n_inits_per_final=args.n_inits_per_final,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # Statistics
    stats = compute_stats(trajectories)
    print(f"\n=== Dataset Statistics ===")
    print(f"Total trajectories: {stats['total']}")
    print(f"By method: {stats['by_method']}")
    print(f"By bin: {stats['by_bin']}")
    print(f"N_subs distribution: {stats['n_subs_distribution']}")
    print(f"Mean final TCR length: {stats['mean_final_len']:.1f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'n_inits_per_final': args.n_inits_per_final,
            'max_steps': args.max_steps,
            'seed': args.seed,
            'stats': stats,
        },
        'trajectories': trajectories,
    }

    print(f"\nSaving to {args.output}")
    with open(output_path, 'w') as f:
        json.dump(output_data, f)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved {len(trajectories)} trajectories ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
