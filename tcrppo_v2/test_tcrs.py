"""Generate TCRs using trained policy and evaluate specificity.

Usage:
    python tcrppo_v2/test_tcrs.py \
        --checkpoint output/v2_full_run1/checkpoints/final.pt \
        --config configs/default.yaml \
        --n_tcrs 50 \
        --output_dir results/v2_full_run1
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.utils.esm_cache import ESMCache
from tcrppo_v2.data.pmhc_loader import PMHCLoader, EVAL_TARGETS
from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
from tcrppo_v2.scorers.decoy import DecoyScorer
from tcrppo_v2.utils.constants import (
    ERGO_MODEL_DIR, ERGO_AE_FILE, MAX_TCR_LEN,
    OP_SUB, OP_INS, OP_DEL, OP_STOP, NUM_OPS, NUM_AMINO_ACIDS,
    MAX_STEPS_PER_EPISODE, MIN_TCR_LEN,
)
from tcrppo_v2.utils.encoding import is_valid_tcr
from tcrppo_v2.env import TCREditEnv
from tcrppo_v2.data.tcr_pool import TCRPool
from tcrppo_v2.reward_manager import RewardManager


def load_policy(checkpoint_path: str, obs_dim: int, hidden_dim: int, device: str) -> ActorCritic:
    """Load trained policy from checkpoint."""
    policy = ActorCritic(obs_dim=obs_dim, hidden_dim=hidden_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


def generate_tcrs(
    policy: ActorCritic,
    env: TCREditEnv,
    peptide: str,
    n_tcrs: int,
    device: str,
) -> List[Dict]:
    """Generate TCRs for a target peptide using the trained policy.

    Returns list of dicts with tcr, initial_tcr, n_steps, reward_components.
    """
    results = []
    for i in range(n_tcrs):
        obs = env.reset(peptide=peptide)
        trajectory = {"initial_tcr": env.initial_tcr, "peptide": peptide, "steps": []}

        while not env.done:
            mask = env.get_action_mask()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mask_dict = {
                "op_mask": torch.BoolTensor(mask["op_mask"]).unsqueeze(0).to(device),
                "pos_mask": torch.BoolTensor(mask["pos_mask"]).unsqueeze(0).to(device),
            }

            with torch.no_grad():
                op, pos, tok, value = policy(obs_tensor, mask_dict)

            action = (int(op[0]), int(pos[0]), int(tok[0]))
            obs, reward, done, info = env.step(action)
            trajectory["steps"].append({
                "action": info.get("action_name", ""),
                "tcr": info.get("new_tcr", ""),
                "reward": float(reward),
            })

        trajectory["final_tcr"] = env.current_tcr
        trajectory["n_steps"] = env.step_count
        trajectory["cumulative_reward"] = env.cumulative_delta
        results.append(trajectory)

    return results


def evaluate_specificity(
    tcrs: List[str],
    target_peptide: str,
    affinity_scorer: AffinityERGOScorer,
    decoy_scorer: DecoyScorer,
    n_decoys: int = 50,
) -> Dict:
    """Evaluate specificity of generated TCRs.

    For each TCR:
    - Score against target peptide (positive)
    - Score against N decoy peptides (negative)
    - Compute AUROC: can we distinguish target from decoys?

    Returns dict with per-TCR and aggregate metrics.
    """
    if not tcrs:
        return {"auroc": 0.0, "mean_target_score": 0.0, "mean_decoy_score": 0.0}

    all_target_scores = []
    all_decoy_scores = []

    for tcr in tcrs:
        # Target score (using MC Dropout for more reliable estimate)
        target_score, target_conf = affinity_scorer.score(tcr, target_peptide)
        all_target_scores.append(target_score)

        # Decoy scores
        decoy_peps = decoy_scorer.sample_decoys(target_peptide, k=n_decoys)
        if decoy_peps:
            tcr_batch = [tcr] * len(decoy_peps)
            if hasattr(affinity_scorer, 'score_batch'):
                scores, _ = affinity_scorer.score_batch(tcr_batch, decoy_peps)
            else:
                scores = []
                for dp in decoy_peps:
                    s, _ = affinity_scorer.score(tcr, dp)
                    scores.append(s)
            all_decoy_scores.extend(scores)

    # Compute AUROC
    # Labels: 1 = target (positive), 0 = decoy (negative)
    # Good specificity = target scores > decoy scores = high AUROC
    n_pos = len(all_target_scores)
    n_neg = len(all_decoy_scores)

    if n_pos == 0 or n_neg == 0:
        return {"auroc": 0.5, "mean_target_score": 0.0, "mean_decoy_score": 0.0}

    labels = np.array([1] * n_pos + [0] * n_neg)
    scores = np.array(all_target_scores + all_decoy_scores)

    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5

    return {
        "auroc": float(auroc),
        "mean_target_score": float(np.mean(all_target_scores)),
        "mean_decoy_score": float(np.mean(all_decoy_scores)),
        "std_target_score": float(np.std(all_target_scores)),
        "std_decoy_score": float(np.std(all_decoy_scores)),
        "n_tcrs": n_pos,
        "n_decoys_per_tcr": n_decoys,
    }


def main():
    parser = argparse.ArgumentParser(description="TCRPPO v2 TCR Generation & Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--n_tcrs", type=int, default=50, help="TCRs to generate per target")
    parser.add_argument("--n_decoys", type=int, default=50, help="Decoys per TCR for AUROC")
    parser.add_argument("--output_dir", default="results/eval", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--targets", nargs="+", default=None, help="Specific targets to evaluate")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    targets = args.targets or list(EVAL_TARGETS.keys())

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Targets: {targets}")
    print(f"N TCRs per target: {args.n_tcrs}")
    print(f"N decoys per TCR: {args.n_decoys}")

    # Load components
    print("Loading ERGO scorer...")
    model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
    affinity_scorer = AffinityERGOScorer(
        model_file=model_file,
        ae_file=ERGO_AE_FILE,
        device=device,
        mc_samples=config.get("affinity_mc_samples", 10),
    )

    print("Loading ESM cache...")
    esm_cache = ESMCache(device=device, tcr_cache_size=config.get("esm_tcr_cache_size", 4096))

    print("Loading pMHC loader...")
    pmhc_loader = PMHCLoader(targets=targets)

    print("Loading TCR pool...")
    tcr_pool = TCRPool(
        l1_seeds_dir=config.get("l1_seeds_dir", "data/l1_seeds"),
        curriculum_schedule=config.get("curriculum_schedule"),
        seed=config.get("seed", 42),
    )
    decoy_lib_path = config.get("decoy_library_path", "/share/liuyutian/pMHC_decoy_library")
    tcr_pool.load_l0_from_decoy_d(decoy_lib_path, targets)

    print("Loading decoy scorer...")
    decoy_scorer = DecoyScorer(
        decoy_library_path=decoy_lib_path,
        targets=targets,
        tier_weights=config.get("decoy_tier_weights"),
        K=config.get("decoy_K", 32),
        tau=config.get("decoy_tau", 10.0),
        affinity_scorer=affinity_scorer,
        rng=np.random.default_rng(config.get("seed", 42)),
    )
    # Unlock all tiers for evaluation
    decoy_scorer.set_unlocked_tiers(["A", "B", "D", "C"])

    # Minimal reward manager (just for env compatibility)
    reward_manager = RewardManager(
        affinity_scorer=affinity_scorer,
        reward_mode="v1_ergo_only",
    )

    # Create single env for generation
    env = TCREditEnv(
        esm_cache=esm_cache,
        pmhc_loader=pmhc_loader,
        tcr_pool=tcr_pool,
        reward_manager=reward_manager,
        max_steps=config.get("max_steps_per_episode", MAX_STEPS_PER_EPISODE),
    )

    # Load policy
    print("Loading policy...")
    policy = load_policy(
        args.checkpoint,
        obs_dim=env.obs_dim,
        hidden_dim=config.get("hidden_dim", 512),
        device=device,
    )

    # Generate and evaluate per target
    all_results = {}
    aurocs = []

    for peptide in targets:
        print(f"\n{'='*60}")
        print(f"Target: {peptide}")
        print(f"{'='*60}")

        # Generate TCRs
        t0 = time.time()
        trajectories = generate_tcrs(policy, env, peptide, args.n_tcrs, device)
        gen_time = time.time() - t0
        print(f"  Generated {len(trajectories)} TCRs in {gen_time:.1f}s")

        # Extract final TCRs
        final_tcrs = [t["final_tcr"] for t in trajectories]
        unique_tcrs = list(set(final_tcrs))
        print(f"  Unique TCRs: {len(unique_tcrs)} / {len(final_tcrs)}")

        # Mean steps
        mean_steps = np.mean([t["n_steps"] for t in trajectories])
        mean_reward = np.mean([t["cumulative_reward"] for t in trajectories])
        print(f"  Mean steps: {mean_steps:.1f}, Mean reward: {mean_reward:.3f}")

        # Evaluate specificity
        t0 = time.time()
        spec_results = evaluate_specificity(
            unique_tcrs, peptide, affinity_scorer, decoy_scorer, args.n_decoys
        )
        eval_time = time.time() - t0
        print(f"  AUROC: {spec_results['auroc']:.4f}")
        print(f"  Target score: {spec_results['mean_target_score']:.4f} +/- {spec_results.get('std_target_score', 0):.4f}")
        print(f"  Decoy score: {spec_results['mean_decoy_score']:.4f} +/- {spec_results.get('std_decoy_score', 0):.4f}")
        print(f"  Eval time: {eval_time:.1f}s")

        all_results[peptide] = {
            "trajectories": trajectories,
            "specificity": spec_results,
            "n_unique": len(unique_tcrs),
            "mean_steps": float(mean_steps),
            "mean_reward": float(mean_reward),
        }
        aurocs.append(spec_results["auroc"])

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Target':<15} {'AUROC':>8} {'Target Score':>13} {'Decoy Score':>12} {'Unique':>7}")
    print("-" * 60)
    for peptide in targets:
        r = all_results[peptide]
        s = r["specificity"]
        print(f"{peptide:<15} {s['auroc']:>8.4f} {s['mean_target_score']:>13.4f} {s['mean_decoy_score']:>12.4f} {r['n_unique']:>7}")

    mean_auroc = float(np.mean(aurocs))
    print(f"\nMean AUROC: {mean_auroc:.4f}")
    print(f"v1 baseline: 0.4538")
    print(f"Delta: {mean_auroc - 0.4538:+.4f}")

    # Save results
    # Convert trajectories to serializable format
    save_results = {}
    for peptide, r in all_results.items():
        save_results[peptide] = {
            "specificity": r["specificity"],
            "n_unique": r["n_unique"],
            "mean_steps": r["mean_steps"],
            "mean_reward": r["mean_reward"],
            "generated_tcrs": [t["final_tcr"] for t in r["trajectories"]],
        }

    save_results["_summary"] = {
        "mean_auroc": mean_auroc,
        "v1_baseline": 0.4538,
        "delta": mean_auroc - 0.4538,
        "checkpoint": args.checkpoint,
        "n_tcrs": args.n_tcrs,
        "n_decoys": args.n_decoys,
    }

    output_file = os.path.join(args.output_dir, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
