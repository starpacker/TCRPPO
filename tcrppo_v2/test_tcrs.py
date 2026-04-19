"""Generate TCRs using trained policy and evaluate specificity.

Supports multi-scorer evaluation: ERGO, TCBind, NetTCR, tFold.

Usage:
    python tcrppo_v2/test_tcrs.py \
        --checkpoint output/v2_full_run1/checkpoints/final.pt \
        --config configs/default.yaml \
        --n_tcrs 20 \
        --n_decoys 50 \
        --scorers ergo,tcbind,nettcr,tfold \
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


def load_policy(checkpoint_path: str, obs_dim: int, hidden_dim: int, device: str, max_tcr_len: int = MAX_TCR_LEN) -> ActorCritic:
    """Load trained policy from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Use max_tcr_len from checkpoint config if available
    if "config" in ckpt:
        max_tcr_len = ckpt["config"].get("max_tcr_len", max_tcr_len)
    policy = ActorCritic(obs_dim=obs_dim, hidden_dim=hidden_dim, max_tcr_len=max_tcr_len).to(device)
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
    """Generate TCRs for a target peptide using the trained policy."""
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


def _score_tcrs_with_scorer(scorer, tcrs: List[str], peptide: str, scorer_name: str) -> List[float]:
    """Score a list of TCRs against a peptide using one scorer."""
    if hasattr(scorer, 'score_batch_fast'):
        peptides = [peptide] * len(tcrs)
        return scorer.score_batch_fast(tcrs, peptides)
    elif hasattr(scorer, 'score_batch'):
        peptides = [peptide] * len(tcrs)
        scores, _ = scorer.score_batch(tcrs, peptides)
        return scores
    else:
        scores = []
        for tcr in tcrs:
            s, _ = scorer.score(tcr, peptide)
            scores.append(s)
        return scores


def evaluate_specificity_multi(
    tcrs: List[str],
    target_peptide: str,
    scorers: Dict,
    decoy_scorer: DecoyScorer,
    n_decoys: int = 50,
) -> Dict[str, Dict]:
    """Evaluate specificity using multiple scorers.

    For each scorer independently: score TCRs against target and decoys, compute AUROC.
    """
    if not tcrs:
        return {name: {"auroc": 0.0, "mean_target_score": 0.0, "mean_decoy_score": 0.0}
                for name in scorers}

    # Sample decoys once (shared across all scorers)
    decoy_peps = decoy_scorer.sample_decoys(target_peptide, k=n_decoys)

    results = {}
    for scorer_name, scorer in scorers.items():
        # Score against target
        target_scores = _score_tcrs_with_scorer(scorer, tcrs, target_peptide, scorer_name)

        # Score against decoys
        all_decoy_scores = []
        if decoy_peps:
            for tcr in tcrs:
                decoy_scores = _score_tcrs_with_scorer(
                    scorer, [tcr] * len(decoy_peps), decoy_peps[0], scorer_name
                )
                # Actually score each decoy peptide individually
                tcr_batch = [tcr] * len(decoy_peps)
                if hasattr(scorer, 'score_batch_fast'):
                    decoy_scores = scorer.score_batch_fast(tcr_batch, decoy_peps)
                elif hasattr(scorer, 'score_batch'):
                    decoy_scores, _ = scorer.score_batch(tcr_batch, decoy_peps)
                else:
                    decoy_scores = []
                    for dp in decoy_peps:
                        s, _ = scorer.score(tcr, dp)
                        decoy_scores.append(s)
                all_decoy_scores.extend(decoy_scores)

        n_pos = len(target_scores)
        n_neg = len(all_decoy_scores)

        if n_pos == 0 or n_neg == 0:
            results[scorer_name] = {
                "auroc": 0.5, "mean_target_score": 0.0, "mean_decoy_score": 0.0
            }
            continue

        labels = np.array([1] * n_pos + [0] * n_neg)
        scores_arr = np.array(target_scores + all_decoy_scores)

        try:
            auroc = roc_auc_score(labels, scores_arr)
        except ValueError:
            auroc = 0.5

        results[scorer_name] = {
            "auroc": float(auroc),
            "mean_target_score": float(np.mean(target_scores)),
            "mean_decoy_score": float(np.mean(all_decoy_scores)),
            "std_target_score": float(np.std(target_scores)),
            "std_decoy_score": float(np.std(all_decoy_scores)),
            "n_tcrs": n_pos,
            "n_decoys_per_tcr": n_decoys,
        }

    return results


def load_scorers(scorer_names: List[str], device: str, config: dict) -> Dict:
    """Load requested scorers by name."""
    scorers = {}

    for name in scorer_names:
        name = name.strip().lower()
        if name == "ergo":
            model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
            scorers["ergo"] = AffinityERGOScorer(
                model_file=model_file,
                ae_file=ERGO_AE_FILE,
                device=device,
                mc_samples=config.get("affinity_mc_samples", 10),
            )
            print("  Loaded ERGO scorer")

        elif name == "tcbind":
            from tcrppo_v2.scorers.affinity_tcbind import AffinityTCBindScorer
            tcbind_weights = config.get(
                "tcbind_weights", "runs/binding_classifier_v2/best_model.pt"
            )
            scorers["tcbind"] = AffinityTCBindScorer(
                weights_path=tcbind_weights,
                device=device,
            )
            print(f"  Loaded TCBind scorer ({tcbind_weights})")

        elif name == "nettcr":
            from tcrppo_v2.scorers.affinity_nettcr import AffinityNetTCRScorer
            scorers["nettcr"] = AffinityNetTCRScorer(device=device)
            print("  Loaded NetTCR scorer")

        elif name == "tfold":
            from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer
            gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
            scorers["tfold"] = AffinityTFoldScorer(
                device=device,
                gpu_id=gpu_id,
                cache_only=True,  # eval uses cache only — no slow server calls
                cache_miss_score=0.5,
            )
            stats = scorers["tfold"].cache_stats
            print(f"  Loaded tFold scorer (cache_only, {stats['cache_size']} cached entries)")

        else:
            print(f"  WARNING: Unknown scorer '{name}', skipping")

    return scorers


def main():
    parser = argparse.ArgumentParser(description="TCRPPO v2 TCR Generation & Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--n_tcrs", type=int, default=20, help="TCRs to generate per target")
    parser.add_argument("--n_decoys", type=int, default=50, help="Decoys per TCR for AUROC")
    parser.add_argument("--output_dir", default="results/eval", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--targets", nargs="+", default=None, help="Specific targets to evaluate")
    parser.add_argument("--scorers", default="ergo,tcbind,nettcr,tfold",
                        help="Comma-separated scorers: ergo,tcbind,nettcr,tfold")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    targets = args.targets or list(EVAL_TARGETS.keys())
    scorer_names = [s.strip() for s in args.scorers.split(",") if s.strip()]

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Targets: {targets}")
    print(f"N TCRs per target: {args.n_tcrs}")
    print(f"N decoys per TCR: {args.n_decoys}")
    print(f"Scorers: {scorer_names}")

    # Load checkpoint config to determine encoder type, hidden_dim, and max_tcr_len
    ckpt_preview = torch.load(args.checkpoint, map_location="cpu")
    ckpt_config = ckpt_preview.get("config", {})
    max_tcr_len_from_ckpt = ckpt_config.get("max_tcr_len", MAX_TCR_LEN)
    encoder_type = ckpt_config.get("encoder", "esm2")
    encoder_dim = ckpt_config.get("encoder_dim", 256)
    hidden_dim_from_ckpt = ckpt_config.get("hidden_dim", config.get("hidden_dim", 512))
    del ckpt_preview

    # Load scorers
    print("Loading scorers...")
    scorers = load_scorers(scorer_names, device, config)

    if not scorers:
        print("ERROR: No scorers loaded successfully. Exiting.")
        sys.exit(1)

    # Use first scorer as the "primary" for reward manager compatibility
    primary_scorer_name = list(scorers.keys())[0]
    primary_scorer = scorers[primary_scorer_name]

    # State encoder: match what the checkpoint was trained with
    if encoder_type == "lightweight":
        from tcrppo_v2.utils.lightweight_encoder import LightweightEncoder
        print(f"Loading lightweight encoder (dim={encoder_dim})...")
        esm_cache = LightweightEncoder(
            device=device,
            encoder_output_dim=encoder_dim,
            tcr_cache_size=config.get("esm_tcr_cache_size", 4096),
        )
    else:
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
    # Use ERGO scorer for decoy sampling if available, otherwise primary scorer
    decoy_affinity = scorers.get("ergo", primary_scorer)
    decoy_scorer = DecoyScorer(
        decoy_library_path=decoy_lib_path,
        targets=targets,
        tier_weights=config.get("decoy_tier_weights"),
        K=config.get("decoy_K", 32),
        tau=config.get("decoy_tau", 10.0),
        affinity_scorer=decoy_affinity,
        rng=np.random.default_rng(config.get("seed", 42)),
    )
    # Unlock all tiers for evaluation
    decoy_scorer.set_unlocked_tiers(["A", "B", "D", "C"])

    # Minimal reward manager (just for env compatibility)
    reward_manager = RewardManager(
        affinity_scorer=primary_scorer,
        reward_mode="v1_ergo_only",
    )

    # Create single env for generation
    env = TCREditEnv(
        esm_cache=esm_cache,
        pmhc_loader=pmhc_loader,
        tcr_pool=tcr_pool,
        reward_manager=reward_manager,
        max_steps=config.get("max_steps_per_episode", MAX_STEPS_PER_EPISODE),
        max_tcr_len=max_tcr_len_from_ckpt,
    )

    # Load policy
    print(f"Loading policy (hidden_dim={hidden_dim_from_ckpt})...")
    policy = load_policy(
        args.checkpoint,
        obs_dim=env.obs_dim,
        hidden_dim=hidden_dim_from_ckpt,
        device=device,
    )

    # Generate and evaluate per target
    all_results = {}
    # aurocs[scorer_name] = list of per-target AUROCs
    aurocs_per_scorer = {name: [] for name in scorers}

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

        # Evaluate specificity with all scorers
        t0 = time.time()
        spec_results = evaluate_specificity_multi(
            unique_tcrs, peptide, scorers, decoy_scorer, args.n_decoys
        )
        eval_time = time.time() - t0

        # Print per-scorer results
        for scorer_name, sr in spec_results.items():
            print(f"  [{scorer_name:>7}] AUROC: {sr['auroc']:.4f}  "
                  f"Target: {sr['mean_target_score']:.4f}  "
                  f"Decoy: {sr['mean_decoy_score']:.4f}")
            aurocs_per_scorer[scorer_name].append(sr["auroc"])
        print(f"  Eval time: {eval_time:.1f}s")

        all_results[peptide] = {
            "trajectories": trajectories,
            "specificity": spec_results,
            "n_unique": len(unique_tcrs),
            "mean_steps": float(mean_steps),
            "mean_reward": float(mean_reward),
        }

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Header
    scorer_list = list(scorers.keys())
    header = f"{'Target':<15}"
    for sn in scorer_list:
        header += f" {sn:>8}"
    header += f" {'Unique':>7}"
    print(header)
    print("-" * (15 + 9 * len(scorer_list) + 8))

    for peptide in targets:
        r = all_results[peptide]
        line = f"{peptide:<15}"
        for sn in scorer_list:
            auroc = r["specificity"][sn]["auroc"]
            line += f" {auroc:>8.4f}"
        line += f" {r['n_unique']:>7}"
        print(line)

    # Mean AUROCs
    print("-" * (15 + 9 * len(scorer_list) + 8))
    mean_line = f"{'Mean AUROC':<15}"
    mean_aurocs = {}
    for sn in scorer_list:
        mean_val = float(np.mean(aurocs_per_scorer[sn]))
        mean_aurocs[sn] = mean_val
        mean_line += f" {mean_val:>8.4f}"
    print(mean_line)

    # v1 baseline comparison
    print(f"\nv1 baseline (ERGO): 0.4538")
    for sn in scorer_list:
        delta = mean_aurocs[sn] - 0.4538
        print(f"  {sn}: {mean_aurocs[sn]:.4f} (delta: {delta:+.4f})")

    # Save results
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
        "mean_auroc": mean_aurocs,
        "v1_baseline": 0.4538,
        "checkpoint": args.checkpoint,
        "n_tcrs": args.n_tcrs,
        "n_decoys": args.n_decoys,
        "scorers": scorer_names,
    }

    output_file = os.path.join(args.output_dir, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
