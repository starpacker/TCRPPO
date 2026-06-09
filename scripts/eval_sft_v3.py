#!/usr/bin/env python3
"""
Quick evaluation of SFT v3 model: can it design better TCRs?

Tests:
1. Oracle mode: given target TCR, can the model reach it?
2. Design mode: starting from random, does the model improve affinity?
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.utils.constants import (
    AMINO_ACIDS, AA_TO_IDX, IDX_TO_AA, NUM_AMINO_ACIDS,
    MAX_TCR_LEN, OP_SUB,
)

# Reuse the encode_obs from train_sft_v3
from scripts.train_sft_v3 import encode_obs, apply_sub


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained policy from checkpoint."""
    policy = ActorCritic(obs_dim=2560, hidden_dim=512, max_tcr_len=MAX_TCR_LEN)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy = policy.to(device)
    policy.eval()
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('best_loss', '?'):.4f}")
    return policy


@torch.no_grad()
def generate_tcr(
    policy, init_tcr: str, peptide: str, target_tcr: str = None,
    n_steps: int = 8, device: str = 'cuda', greedy: bool = True,
):
    """Generate TCR by running policy for n_steps.

    Args:
        target_tcr: If provided, included in observation (oracle mode).
                    If None, target is zeroed out (design mode).
    """
    current_tcr = init_tcr
    actions_taken = []

    fixed_op = torch.zeros(1, dtype=torch.long, device=device)

    for step in range(n_steps):
        obs = encode_obs(
            current_tcr=current_tcr,
            peptide=peptide,
            step_count=step,
            target_tcr=target_tcr,
            target_dropout=0.0 if target_tcr else 1.0,  # 0 dropout = always show target
        )
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        features = policy.backbone(obs_t)

        # Position prediction
        op_emb = policy.op_embed(fixed_op)
        pos_input = torch.cat([features, op_emb], dim=-1)
        pos_logits = policy.pos_head(pos_input)

        # Mask invalid positions
        tcr_len = len(current_tcr)
        pos_mask = torch.zeros(1, MAX_TCR_LEN, dtype=torch.bool, device=device)
        pos_mask[0, :tcr_len] = True
        pos_logits = pos_logits.masked_fill(~pos_mask, float('-inf'))

        if greedy:
            pos = pos_logits.argmax(dim=-1)
        else:
            pos = torch.distributions.Categorical(logits=pos_logits).sample()

        # Token prediction
        pos_emb = policy.pos_embed(pos)
        tok_input = torch.cat([features, op_emb, pos_emb], dim=-1)
        tok_logits = policy.token_head(tok_input)

        if greedy:
            tok = tok_logits.argmax(dim=-1)
        else:
            tok = torch.distributions.Categorical(logits=tok_logits).sample()

        pos_idx = pos.item()
        tok_idx = tok.item()
        token = IDX_TO_AA[tok_idx]

        # Apply substitution
        current_tcr = apply_sub(current_tcr, pos_idx, token)
        actions_taken.append({'position': pos_idx, 'token': token})

    return current_tcr, actions_taken


def edit_distance(s1: str, s2: str) -> int:
    """Simple substitution distance (same length assumed)."""
    if len(s1) != len(s2):
        return max(len(s1), len(s2))
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def test_oracle_mode(policy, device='cuda'):
    """Test: given target TCR, can the model reach it?"""
    print("\n" + "="*60)
    print("TEST 1: Oracle Mode (target visible)")
    print("="*60)

    # Load some test trajectories
    data_path = '/share/liuyutian/tcrppo_v2/data/sft_v2_trajectories.json'
    with open(data_path) as f:
        data = json.load(f)

    # Sample 20 trajectories with varying n_subs
    trajs = data['trajectories']
    np.random.seed(42)
    indices = np.random.choice(len(trajs), size=100, replace=False)

    results_by_nsubs = {}
    total_correct = 0
    total_tested = 0

    for idx in indices:
        t = trajs[idx]
        init_tcr = t['init_tcr']
        final_tcr = t['final_tcr']
        peptide = t['peptide']
        n_subs = t['n_subs']

        # Run model with target visible
        result_tcr, actions = generate_tcr(
            policy, init_tcr, peptide,
            target_tcr=final_tcr,
            n_steps=n_subs,  # Give it exactly n_subs steps
            device=device,
            greedy=True,
        )

        dist = edit_distance(result_tcr, final_tcr)
        exact_match = (result_tcr == final_tcr)

        if n_subs not in results_by_nsubs:
            results_by_nsubs[n_subs] = {'exact': 0, 'total': 0, 'dists': []}
        results_by_nsubs[n_subs]['total'] += 1
        results_by_nsubs[n_subs]['dists'].append(dist)
        if exact_match:
            results_by_nsubs[n_subs]['exact'] += 1
            total_correct += 1
        total_tested += 1

    print(f"\nOverall exact match: {total_correct}/{total_tested} = {total_correct/total_tested:.1%}")
    print(f"\nBy n_subs:")
    for n in sorted(results_by_nsubs.keys()):
        r = results_by_nsubs[n]
        mean_dist = np.mean(r['dists'])
        print(f"  n_subs={n}: exact={r['exact']}/{r['total']} ({r['exact']/r['total']:.0%}), "
              f"mean_remaining_dist={mean_dist:.1f}")


def test_design_mode(policy, device='cuda'):
    """Test: without target, does the model improve TCRs?"""
    print("\n" + "="*60)
    print("TEST 2: Design Mode (no target, free generation)")
    print("="*60)

    peptides = ['GILGFVFTL', 'NLVPMVATV', 'GLCTLVAML', 'KLWASPLHV', 'FPRPWLHGL']

    print("\nGenerating TCRs from random starts (8 steps, greedy)...")
    all_results = []

    for peptide in peptides:
        print(f"\n  Peptide: {peptide}")
        for i in range(10):
            # Random init TCR
            length = np.random.randint(12, 20)
            init_tcr = ''.join(np.random.choice(AMINO_ACIDS, size=length))

            result_tcr, actions = generate_tcr(
                policy, init_tcr, peptide,
                target_tcr=None,  # No target = design mode
                n_steps=8,
                device=device,
                greedy=True,
            )

            n_changes = edit_distance(init_tcr, result_tcr)
            all_results.append({
                'peptide': peptide,
                'init_tcr': init_tcr,
                'final_tcr': result_tcr,
                'n_changes': n_changes,
            })

            if i < 3:  # Print first 3
                print(f"    {init_tcr} -> {result_tcr} ({n_changes} changes)")

    # Check diversity
    print(f"\n  Total generated: {len(all_results)}")
    unique_finals = set(r['final_tcr'] for r in all_results)
    print(f"  Unique final TCRs: {len(unique_finals)}/{len(all_results)}")

    # Check if it's just making random changes or has preferences
    print("\n  Amino acid frequency in generated positions (last step):")
    from collections import Counter
    aa_counts = Counter()
    for r in all_results:
        for aa in r['final_tcr']:
            aa_counts[aa] += 1
    total_aa = sum(aa_counts.values())
    top_5 = aa_counts.most_common(5)
    bottom_5 = aa_counts.most_common()[-5:]
    print(f"    Most common:  {', '.join(f'{aa}={c/total_aa:.1%}' for aa, c in top_5)}")
    print(f"    Least common: {', '.join(f'{aa}={c/total_aa:.1%}' for aa, c in bottom_5)}")

    return all_results


def test_affinity_scoring(results, device='cuda'):
    """Score generated TCRs with tFold (if available)."""
    print("\n" + "="*60)
    print("TEST 3: Affinity Scoring (tFold)")
    print("="*60)

    try:
        from tcrppo_v2.scorers.affinity_tfold import TFoldScorer
        scorer = TFoldScorer(device=device)
        print("  tFold scorer loaded successfully")
    except Exception as e:
        print(f"  Could not load tFold scorer: {e}")
        print("  Skipping affinity evaluation")
        return

    # Score a subset
    peptide_results = {}
    for peptide in set(r['peptide'] for r in results):
        pep_items = [r for r in results if r['peptide'] == peptide]
        init_tcrs = [r['init_tcr'] for r in pep_items]
        final_tcrs = [r['final_tcr'] for r in pep_items]
        peptides_list = [peptide] * len(pep_items)

        try:
            init_scores = scorer.score_batch_fast(init_tcrs, peptides_list)
            final_scores = scorer.score_batch_fast(final_tcrs, peptides_list)

            mean_init = np.mean(init_scores)
            mean_final = np.mean(final_scores)
            improved = sum(1 for i, f in zip(init_scores, final_scores) if f > i)

            peptide_results[peptide] = {
                'mean_init': mean_init,
                'mean_final': mean_final,
                'delta': mean_final - mean_init,
                'improved_pct': improved / len(pep_items),
            }
            print(f"\n  {peptide}:")
            print(f"    Init score:  {mean_init:.4f}")
            print(f"    Final score: {mean_final:.4f}")
            print(f"    Delta:       {mean_final - mean_init:+.4f}")
            print(f"    Improved:    {improved}/{len(pep_items)} ({improved/len(pep_items):.0%})")
        except Exception as e:
            print(f"\n  {peptide}: scoring failed - {e}")

    if peptide_results:
        mean_delta = np.mean([v['delta'] for v in peptide_results.values()])
        mean_improved = np.mean([v['improved_pct'] for v in peptide_results.values()])
        print(f"\n  Overall: mean_delta={mean_delta:+.4f}, mean_improved={mean_improved:.0%}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = '/share/liuyutian/tcrppo_v2/output/sft_v3_training/checkpoint_best.pt'

    print("SFT v3 Evaluation")
    print("="*60)
    policy = load_model(checkpoint, device=device)

    # Test 1: Oracle mode
    test_oracle_mode(policy, device)

    # Test 2: Design mode
    results = test_design_mode(policy, device)

    # Test 3: Affinity scoring
    test_affinity_scoring(results, device)

    print("\n" + "="*60)
    print("Done!")


if __name__ == "__main__":
    main()
