#!/usr/bin/env python
"""Parallel FP32 evaluation of trace checkpoints.

Evaluates three traces on 20 target peptides using FP32 tFold scorer.
Generates 20 TCRs per peptide and computes max/mean affinity scores.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer


def load_checkpoint(ckpt_path: str, device: str) -> Tuple[ActorCritic, dict]:
    """Load policy from checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Extract config
    config = ckpt.get("config", {})

    # Infer obs_dim from checkpoint weights
    obs_dim = ckpt["policy_state_dict"]["backbone.0.weight"].shape[1]
    hidden_dim = ckpt["policy_state_dict"]["backbone.0.weight"].shape[0]

    print(f"Detected obs_dim={obs_dim}, hidden_dim={hidden_dim} from checkpoint")

    # Create policy
    policy = ActorCritic(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        max_tcr_len=config.get("max_tcr_len", 27),
    ).to(device)

    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    return policy, config


def generate_tcr(
    policy: ActorCritic,
    initial_seq: str,
    max_steps: int = 8,
    device: str = "cuda",
) -> str:
    """Generate a single TCR using the policy."""
    from tcrppo_v2.env import TCRDesignEnv
    from tcrppo_v2.utils.esm_cache import ESMCache

    # Create minimal env for sequence generation
    esm_cache = ESMCache(device=device)
    env = TCRDesignEnv(
        target_peptides=["GILGFVFTL"],  # Dummy peptide
        reward_manager=None,  # Not needed for generation
        esm_cache=esm_cache,
        max_steps=max_steps,
        device=device,
    )

    env.current_tcr = initial_seq
    env.steps_taken = 0

    for _ in range(max_steps):
        state = env._get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action_mask = env._get_action_mask()
            action = policy.select_action(state_tensor, action_mask, deterministic=True)

        _, _, done, _ = env.step(action)

        if done:
            break

    return env.current_tcr


def evaluate_trace(
    trace_name: str,
    ckpt_path: str,
    target_peptides: List[str],
    n_tcrs_per_peptide: int,
    gpu_id: int,
    output_dir: str,
) -> Dict:
    """Evaluate a single trace checkpoint."""
    device = f"cuda:{gpu_id}"
    print(f"[{trace_name}] Starting evaluation on GPU {gpu_id}")
    print(f"[{trace_name}] Checkpoint: {ckpt_path}")

    # Load policy
    policy, config = load_checkpoint(ckpt_path, device)

    # Initialize FP32 tFold scorer
    scorer = AffinityTFoldScorer(
        device=device,
        gpu_id=gpu_id,
        use_cache=True,
    )

    results = {
        "trace_name": trace_name,
        "checkpoint": ckpt_path,
        "config": config,
        "per_peptide": {},
        "summary": {},
    }

    all_max_affinities = []
    all_mean_affinities = []

    for peptide in target_peptides:
        print(f"[{trace_name}] Evaluating peptide: {peptide}")

        # Generate TCRs
        tcrs = []
        for i in range(n_tcrs_per_peptide):
            # Start from random TCRdb sequence
            initial_seq = "CASSLGQAYEQYF"  # Default seed
            tcr = generate_tcr(policy, initial_seq, max_steps=config.get("max_steps", 8), device=device)
            tcrs.append(tcr)

        # Score TCRs with FP32 tFold
        print(f"[{trace_name}] Scoring {len(tcrs)} TCRs for {peptide}...")
        scores = []
        for tcr in tcrs:
            score, _ = scorer.score(tcr, peptide)
            scores.append(score)

        scores = np.array(scores)
        max_aff = float(np.max(scores))
        mean_aff = float(np.mean(scores))

        results["per_peptide"][peptide] = {
            "max_affinity": max_aff,
            "mean_affinity": mean_aff,
            "std_affinity": float(np.std(scores)),
            "tcrs": tcrs,
            "scores": scores.tolist(),
        }

        all_max_affinities.append(max_aff)
        all_mean_affinities.append(mean_aff)

        print(f"[{trace_name}] {peptide}: max={max_aff:.4f}, mean={mean_aff:.4f}")

    # Summary statistics
    results["summary"] = {
        "overall_max_affinity": float(np.max(all_max_affinities)),
        "overall_mean_affinity": float(np.mean(all_mean_affinities)),
        "mean_of_max_affinities": float(np.mean(all_max_affinities)),
        "mean_of_mean_affinities": float(np.mean(all_mean_affinities)),
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{trace_name}_fp32_eval.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[{trace_name}] Results saved to: {result_path}")
    print(f"[{trace_name}] Summary: mean_of_mean_affinities={results['summary']['mean_of_mean_affinities']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Parallel FP32 evaluation of traces")
    parser.add_argument("--trace-name", required=True, help="Trace name")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--peptides", required=True, help="Path to peptide list file")
    parser.add_argument("--n-tcrs", type=int, default=20, help="TCRs per peptide")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    # Load peptides
    with open(args.peptides, "r") as f:
        peptides = [line.strip() for line in f if line.strip()]

    print(f"Evaluating {args.trace_name} with {len(peptides)} peptides")

    results = evaluate_trace(
        trace_name=args.trace_name,
        ckpt_path=args.checkpoint,
        target_peptides=peptides,
        n_tcrs_per_peptide=args.n_tcrs,
        gpu_id=args.gpu,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*80}")
    print(f"Evaluation complete: {args.trace_name}")
    print(f"Mean of mean affinities: {results['summary']['mean_of_mean_affinities']:.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
