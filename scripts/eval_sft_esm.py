#!/usr/bin/env python3
"""Evaluate SFT-ESM checkpoint by generating TCRs and scoring with tFold."""

import sys
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

import argparse
import json
import numpy as np
import torch
import time
from pathlib import Path
from typing import List, Dict

from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.sft_env_esm import SFTEnvESM
from tcrppo_v2.utils.constants import MAX_TCR_LEN, OP_STOP, IDX_TO_AA


def load_policy(checkpoint_path: str, device: str = 'cuda'):
    """Load policy from SFT checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    hidden_dim = ckpt['config'].get('hidden_dim', 512)
    # SFT training used max_tcr_len=25 (not the default 20)
    max_tcr_len = 25
    policy = ActorCritic(
        obs_dim=2560,
        hidden_dim=hidden_dim,
        max_tcr_len=max_tcr_len
    ).to(device)

    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()

    print(f"✓ Loaded checkpoint: epoch={ckpt.get('epoch', '?')}, hidden_dim={hidden_dim}")
    return policy


def generate_tcrs(
    policy,
    env: SFTEnvESM,
    peptide: str,
    n_tcrs: int = 50,
    max_steps: int = 8,
    device: str = 'cuda',
    deterministic: bool = False,
) -> List[Dict]:
    """Generate TCRs for a peptide using the SFT policy."""
    results = []

    with torch.no_grad():
        for i in range(n_tcrs):
            obs = env.reset(peptide=peptide)
            init_tcr = env.current_tcr
            actions_taken = []

            for step in range(max_steps):
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)

                # Get action from policy
                op_logits = policy.op_head(policy.backbone(obs_tensor))

                # Full forward pass
                features = policy.backbone(obs_tensor)
                op_logits_raw = policy.op_head(features)

                if deterministic:
                    op = op_logits_raw.argmax(dim=-1).item()
                else:
                    op_probs = torch.softmax(op_logits_raw, dim=-1)
                    op = torch.multinomial(op_probs, 1).item()

                if op == OP_STOP and step > 0:
                    break
                elif op == OP_STOP and step == 0:
                    op = 0  # Force SUB if STOP at step 0

                # Position
                op_emb = policy.op_embed(torch.tensor([op], device=device))
                pos_input = torch.cat([features, op_emb], dim=-1)
                pos_logits = policy.pos_head(pos_input)

                # Mask invalid positions
                tcr_len = len(env.current_tcr)
                pos_mask = torch.zeros(pos_logits.shape[-1], device=device)
                pos_mask[tcr_len:] = -1e9
                pos_logits = pos_logits + pos_mask.unsqueeze(0)

                if deterministic:
                    pos = pos_logits.argmax(dim=-1).item()
                else:
                    pos_probs = torch.softmax(pos_logits, dim=-1)
                    pos = torch.multinomial(pos_probs, 1).item()

                pos = min(pos, tcr_len - 1)

                # Token
                pos_emb = policy.pos_embed(torch.tensor([pos], device=device))
                tok_input = torch.cat([features, op_emb, pos_emb], dim=-1)
                tok_logits = policy.token_head(tok_input)

                if deterministic:
                    tok_idx = tok_logits.argmax(dim=-1).item()
                else:
                    tok_probs = torch.softmax(tok_logits, dim=-1)
                    tok_idx = torch.multinomial(tok_probs, 1).item()

                token = IDX_TO_AA.get(tok_idx, 'A')
                actions_taken.append((op, pos, token))

                obs, _, done, info = env.step((op, pos, token))
                if done:
                    break

            results.append({
                'tcr': env.current_tcr,
                'init_tcr': init_tcr,
                'peptide': peptide,
                'n_steps': len(actions_taken),
                'actions': [(a[0], a[1], a[2]) for a in actions_taken],
            })

    return results


def score_with_tfold(
    tcrs: List[str],
    peptides: List[str],
    socket_path: str = '/tmp/tfold_server_trace73_curriculum_exploration.sock',
) -> List[float]:
    """Score TCR-peptide pairs using tFold server."""
    import socket
    import struct

    scores = []
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)

    for tcr, pep in zip(tcrs, peptides):
        request = json.dumps({'tcr': tcr, 'peptide': pep, 'hla': 'HLA-A*02:01'})
        request_bytes = request.encode('utf-8')
        sock.sendall(struct.pack('!I', len(request_bytes)))
        sock.sendall(request_bytes)

        # Read response
        length_bytes = sock.recv(4)
        if len(length_bytes) < 4:
            scores.append(float('nan'))
            continue
        length = struct.unpack('!I', length_bytes)[0]
        response_bytes = b''
        while len(response_bytes) < length:
            chunk = sock.recv(length - len(response_bytes))
            if not chunk:
                break
            response_bytes += chunk

        response = json.loads(response_bytes.decode('utf-8'))
        scores.append(response.get('score', float('nan')))

    sock.close()
    return scores


def score_with_tfold_batch(
    tcrs: List[str],
    peptide: str,
    socket_path: str,
) -> List[float]:
    """Score a batch of TCRs against one peptide using tFold AffinityScorer."""
    from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer

    scorer = AffinityTFoldScorer(server_socket_path=socket_path)
    scores = []
    for tcr in tcrs:
        try:
            value, confidence = scorer.score(tcr, peptide, hla='HLA-A*02:01')
            scores.append(value)
        except Exception as e:
            print(f"  Warning: scoring failed for {tcr[:10]}...: {e}")
            scores.append(float('nan'))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='output/sft_esm_training/checkpoint_best.pt')
    parser.add_argument('--esm_cache', type=str,
                        default='data/esm2_embeddings_sft.pt')
    parser.add_argument('--n_tcrs', type=int, default=20,
                        help='Number of TCRs to generate per peptide')
    parser.add_argument('--tfold_socket', type=str,
                        default='/tmp/tfold_server_trace73_curriculum_exploration.sock')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--output', type=str, default='results/sft_esm_eval')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Evaluation targets
    in_domain_peptides = [
        'GILGFVFTL', 'NLVPMVATV', 'GLCTLVAML', 'YLQPRTFLL', 'LLLDRLNQL',
    ]
    ood_peptides = [
        'FLYALALLL', 'SLYNTVATL', 'KLGGALQAK', 'IVTDFSVIK', 'SPRWYFYYL',
    ]

    print("=" * 60)
    print("SFT-ESM Model Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"N TCRs per peptide: {args.n_tcrs}")
    print(f"tFold socket: {args.tfold_socket}")
    print(f"Deterministic: {args.deterministic}")
    print()

    # Load model
    policy = load_policy(args.checkpoint, args.device)

    # Load environment (max_tcr_len=25 to match training)
    env = SFTEnvESM(
        esm_cache_path=args.esm_cache,
        max_steps=8,
        max_tcr_len=25,
        device=args.device,
    )

    # Generate and evaluate
    all_results = {}

    print("\n--- In-Domain Peptides ---")
    for pep in in_domain_peptides:
        print(f"\n[{pep}] Generating {args.n_tcrs} TCRs...")
        t0 = time.time()
        gen_results = generate_tcrs(
            policy, env, pep,
            n_tcrs=args.n_tcrs, device=args.device,
            deterministic=args.deterministic,
        )
        gen_time = time.time() - t0

        tcrs = [r['tcr'] for r in gen_results]
        unique_tcrs = set(tcrs)

        print(f"  Generated {len(tcrs)} TCRs ({len(unique_tcrs)} unique) in {gen_time:.1f}s")
        print(f"  Lengths: {np.mean([len(t) for t in tcrs]):.1f} ± {np.std([len(t) for t in tcrs]):.1f}")
        print(f"  Steps: {np.mean([r['n_steps'] for r in gen_results]):.1f}")

        # Score with tFold
        print(f"  Scoring with tFold...")
        t0 = time.time()
        try:
            scores = score_with_tfold_batch(tcrs, pep, args.tfold_socket)
            score_time = time.time() - t0

            valid_scores = [s for s in scores if np.isfinite(s)]
            if valid_scores:
                print(f"  tFold Affinity: mean={np.mean(valid_scores):.4f}, "
                      f"max={np.max(valid_scores):.4f}, "
                      f"min={np.min(valid_scores):.4f}")
                print(f"  >0: {sum(1 for s in valid_scores if s > 0)}/{len(valid_scores)} "
                      f"({sum(1 for s in valid_scores if s > 0)/len(valid_scores)*100:.1f}%)")
                print(f"  >-0.5: {sum(1 for s in valid_scores if s > -0.5)}/{len(valid_scores)} "
                      f"({sum(1 for s in valid_scores if s > -0.5)/len(valid_scores)*100:.1f}%)")
                print(f"  Scoring time: {score_time:.1f}s ({score_time/len(tcrs):.2f}s/TCR)")
            else:
                print(f"  No valid scores returned!")
        except Exception as e:
            print(f"  tFold scoring failed: {e}")
            scores = [float('nan')] * len(tcrs)

        all_results[pep] = {
            'type': 'in_domain',
            'generated': gen_results,
            'scores': scores,
            'stats': {
                'n_tcrs': len(tcrs),
                'n_unique': len(unique_tcrs),
                'mean_length': float(np.mean([len(t) for t in tcrs])),
                'mean_steps': float(np.mean([r['n_steps'] for r in gen_results])),
                'mean_affinity': float(np.nanmean(scores)),
                'max_affinity': float(np.nanmax(scores)) if scores else float('nan'),
                'pct_above_0': float(sum(1 for s in scores if s > 0) / len(scores)) if scores else 0,
                'pct_above_neg05': float(sum(1 for s in scores if s > -0.5) / len(scores)) if scores else 0,
            }
        }

    print("\n\n--- OOD Peptides ---")
    for pep in ood_peptides:
        print(f"\n[{pep}] Generating {args.n_tcrs} TCRs...")
        t0 = time.time()
        gen_results = generate_tcrs(
            policy, env, pep,
            n_tcrs=args.n_tcrs, device=args.device,
            deterministic=args.deterministic,
        )
        gen_time = time.time() - t0

        tcrs = [r['tcr'] for r in gen_results]
        unique_tcrs = set(tcrs)

        print(f"  Generated {len(tcrs)} TCRs ({len(unique_tcrs)} unique) in {gen_time:.1f}s")
        print(f"  Lengths: {np.mean([len(t) for t in tcrs]):.1f} ± {np.std([len(t) for t in tcrs]):.1f}")
        print(f"  Steps: {np.mean([r['n_steps'] for r in gen_results]):.1f}")

        # Score with tFold
        print(f"  Scoring with tFold...")
        t0 = time.time()
        try:
            scores = score_with_tfold_batch(tcrs, pep, args.tfold_socket)
            score_time = time.time() - t0

            valid_scores = [s for s in scores if np.isfinite(s)]
            if valid_scores:
                print(f"  tFold Affinity: mean={np.mean(valid_scores):.4f}, "
                      f"max={np.max(valid_scores):.4f}, "
                      f"min={np.min(valid_scores):.4f}")
                print(f"  >0: {sum(1 for s in valid_scores if s > 0)}/{len(valid_scores)} "
                      f"({sum(1 for s in valid_scores if s > 0)/len(valid_scores)*100:.1f}%)")
                print(f"  >-0.5: {sum(1 for s in valid_scores if s > -0.5)}/{len(valid_scores)} "
                      f"({sum(1 for s in valid_scores if s > -0.5)/len(valid_scores)*100:.1f}%)")
                print(f"  Scoring time: {score_time:.1f}s ({score_time/len(tcrs):.2f}s/TCR)")
            else:
                print(f"  No valid scores returned!")
        except Exception as e:
            print(f"  tFold scoring failed: {e}")
            scores = [float('nan')] * len(tcrs)

        all_results[pep] = {
            'type': 'ood',
            'generated': gen_results,
            'scores': scores,
            'stats': {
                'n_tcrs': len(tcrs),
                'n_unique': len(unique_tcrs),
                'mean_length': float(np.mean([len(t) for t in tcrs])),
                'mean_steps': float(np.mean([r['n_steps'] for r in gen_results])),
                'mean_affinity': float(np.nanmean(scores)),
                'max_affinity': float(np.nanmax(scores)) if scores else float('nan'),
                'pct_above_0': float(sum(1 for s in scores if s > 0) / len(scores)) if scores else 0,
                'pct_above_neg05': float(sum(1 for s in scores if s > -0.5) / len(scores)) if scores else 0,
            }
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Peptide':<15}{'Type':<10}{'Mean A':<10}{'Max A':<10}{'%>0':<8}{'%>-0.5':<8}{'Unique':<8}")
    print("-" * 69)

    in_domain_means = []
    ood_means = []

    for pep, data in all_results.items():
        stats = data['stats']
        print(f"{pep:<15}{data['type']:<10}"
              f"{stats['mean_affinity']:<10.4f}"
              f"{stats['max_affinity']:<10.4f}"
              f"{stats['pct_above_0']*100:<8.1f}"
              f"{stats['pct_above_neg05']*100:<8.1f}"
              f"{stats['n_unique']:<8}")

        if data['type'] == 'in_domain':
            in_domain_means.append(stats['mean_affinity'])
        else:
            ood_means.append(stats['mean_affinity'])

    print("-" * 69)
    if in_domain_means:
        print(f"{'IN-DOMAIN AVG':<25}{np.nanmean(in_domain_means):<10.4f}")
    if ood_means:
        print(f"{'OOD AVG':<25}{np.nanmean(ood_means):<10.4f}")
    if in_domain_means or ood_means:
        all_means = in_domain_means + ood_means
        print(f"{'OVERALL AVG':<25}{np.nanmean(all_means):<10.4f}")

    # Comparison with baselines
    print("\n--- Comparison with Baselines ---")
    print(f"  Previous SFT (dummy obs):  mean affinity = -7.10")
    print(f"  trace73 RL (mean):         mean affinity = -1.172")
    print(f"  trace73 RL (best single):  max affinity  = 1.093")
    print(f"  SFT training data:         mean affinity = -0.22")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    with open(output_dir / 'eval_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✓ Results saved to {output_dir}/eval_results.json")


if __name__ == '__main__':
    main()
