"""Generate TCRs using trained policy and evaluate specificity.

Supports multi-scorer evaluation: tFold (primary), ERGO, TCBind, NetTCR.

tFold V3.4 is the primary evaluation scorer due to superior accuracy:
  - Mean AUC=0.800 on tc-hard (vs NetTCR 0.601, ERGO 0.541)
  - Structure-aware: 735M tFold features + 1.57M cross-attention classifier
  - See evaluation/EVALUATION_METHODOLOGY.md for full justification

Usage:
    python tcrppo_v2/test_tcrs.py \
        --checkpoint output/v2_full_run1/checkpoints/final.pt \
        --config configs/default.yaml \
        --n_tcrs 20 \
        --n_decoys 50 \
        --scorers tfold,ergo,nettcr \
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

    Correct AUROC: for each TCR individually, compute AUROC where
    positive = target score, negatives = decoy scores. Then average
    across all TCRs. Also computes per-tier AUROC breakdown.
    """
    if not tcrs:
        return {name: {"auroc": 0.0, "mean_target_score": 0.0, "mean_decoy_score": 0.0}
                for name in scorers}

    # Sample decoys grouped by tier for per-tier AUROC (exclude C per user request)
    decoys_by_tier = decoy_scorer.sample_decoys_by_tier(
        target_peptide, k_per_tier=max(10, n_decoys // 3)
    )
    # Flatten all decoy peptides for overall scoring
    all_decoy_peps = []
    decoy_tier_labels = []  # parallel array: tier label per decoy
    for tier, peps in sorted(decoys_by_tier.items()):
        all_decoy_peps.extend(peps)
        decoy_tier_labels.extend([tier] * len(peps))

    if not all_decoy_peps:
        # Fallback to old sampling if per-tier failed
        all_decoy_peps = decoy_scorer.sample_decoys(target_peptide, k=n_decoys)
        decoy_tier_labels = ["?"] * len(all_decoy_peps)

    results = {}
    for scorer_name, scorer in scorers.items():
        # Score all TCRs against target peptide
        target_scores = _score_tcrs_with_scorer(scorer, tcrs, target_peptide, scorer_name)

        # Score all TCRs against all decoy peptides
        # Build flat batches: for each TCR, score against all decoys
        per_tcr_decoy_scores = []  # list of lists: [n_tcrs][n_decoys]
        for tcr in tcrs:
            tcr_batch = [tcr] * len(all_decoy_peps)
            if hasattr(scorer, 'score_batch_fast'):
                d_scores = scorer.score_batch_fast(tcr_batch, all_decoy_peps)
            elif hasattr(scorer, 'score_batch'):
                d_scores, _ = scorer.score_batch(tcr_batch, all_decoy_peps)
            else:
                d_scores = []
                for dp in all_decoy_peps:
                    s, _ = scorer.score(tcr, dp)
                    d_scores.append(s)
            per_tcr_decoy_scores.append(d_scores)

        n_tcrs_scored = len(target_scores)
        n_decoys_total = len(all_decoy_peps)

        if n_tcrs_scored == 0 or n_decoys_total == 0:
            results[scorer_name] = {
                "auroc": 0.5, "mean_target_score": 0.0, "mean_decoy_score": 0.0,
            }
            continue

        # --- Per-TCR AUROC (correct method) ---
        per_tcr_aurocs = []
        for i in range(n_tcrs_scored):
            pos = np.array([target_scores[i]])
            neg = np.array(per_tcr_decoy_scores[i])
            if len(neg) == 0:
                continue
            labels = np.array([1] + [0] * len(neg))
            scores_arr = np.concatenate([pos, neg])
            try:
                a = roc_auc_score(labels, scores_arr)
            except ValueError:
                a = 0.5
            per_tcr_aurocs.append(a)

        mean_auroc = float(np.mean(per_tcr_aurocs)) if per_tcr_aurocs else 0.5

        # --- Per-tier AUROC ---
        tier_aurocs = {}
        unique_tiers = sorted(set(decoy_tier_labels))
        for tier in unique_tiers:
            tier_indices = [j for j, t in enumerate(decoy_tier_labels) if t == tier]
            if not tier_indices:
                continue
            tier_per_tcr = []
            for i in range(n_tcrs_scored):
                pos = np.array([target_scores[i]])
                neg = np.array([per_tcr_decoy_scores[i][j] for j in tier_indices])
                if len(neg) == 0:
                    continue
                labels = np.array([1] + [0] * len(neg))
                scores_arr = np.concatenate([pos, neg])
                try:
                    a = roc_auc_score(labels, scores_arr)
                except ValueError:
                    a = 0.5
                tier_per_tcr.append(a)
            if tier_per_tcr:
                tier_aurocs[tier] = float(np.mean(tier_per_tcr))

        all_decoy_flat = [s for ds in per_tcr_decoy_scores for s in ds]

        # --- Ranking metrics (rank TCRs by target binding score, descending) ---
        target_arr = np.array(target_scores)
        auroc_arr = np.array(per_tcr_aurocs)
        ranked_idx = np.argsort(target_arr)[::-1]
        ranked_target = target_arr[ranked_idx]
        ranked_auroc = auroc_arr[ranked_idx]

        n = len(ranked_target)
        top1_target = float(ranked_target[0]) if n >= 1 else 0.0
        top3_target = float(np.mean(ranked_target[:min(3, n)])) if n >= 1 else 0.0
        top5_target = float(np.mean(ranked_target[:min(5, n)])) if n >= 1 else 0.0
        top1_auroc = float(ranked_auroc[0]) if n >= 1 else 0.5
        top3_auroc = float(np.mean(ranked_auroc[:min(3, n)])) if n >= 1 else 0.5
        top5_auroc = float(np.mean(ranked_auroc[:min(5, n)])) if n >= 1 else 0.5
        hit_rate_07 = float(np.mean(target_arr > 0.7)) if n >= 1 else 0.0
        top1_composite = float(ranked_target[0] * ranked_auroc[0]) if n >= 1 else 0.0
        k5 = min(5, n)
        top5_composite = float(np.mean(ranked_target[:k5] * ranked_auroc[:k5])) if n >= 1 else 0.0

        # Store per-TCR details for JSON (TCR sequence, target score, AUROC)
        per_tcr_details = []
        for i in range(n):
            idx = ranked_idx[i]
            per_tcr_details.append({
                "tcr": tcrs[idx],
                "target_score": float(target_arr[idx]),
                "auroc": float(auroc_arr[idx]),
                "composite": float(target_arr[idx] * auroc_arr[idx]),
                "rank": i + 1,
            })

        results[scorer_name] = {
            "auroc": mean_auroc,
            "mean_target_score": float(np.mean(target_scores)),
            "mean_decoy_score": float(np.mean(all_decoy_flat)) if all_decoy_flat else 0.0,
            "std_target_score": float(np.std(target_scores)),
            "std_decoy_score": float(np.std(all_decoy_flat)) if all_decoy_flat else 0.0,
            "n_tcrs": n_tcrs_scored,
            "n_decoys_per_tcr": n_decoys_total,
            "per_tier_auroc": tier_aurocs,
            # Ranking metrics
            "top1_target": top1_target,
            "top3_target": top3_target,
            "top5_target": top5_target,
            "top1_auroc": top1_auroc,
            "top3_auroc": top3_auroc,
            "top5_auroc": top5_auroc,
            "hit_rate_07": hit_rate_07,
            "top1_composite": top1_composite,
            "top5_composite": top5_composite,
            # Per-TCR ranked details
            "per_tcr_ranked": per_tcr_details,
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
            )
            stats = scorers["tfold"].cache_stats
            print(f"  Loaded tFold scorer ({stats['cache_size']} cached entries)")

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
    parser.add_argument("--scorers", default="tfold,ergo,nettcr",
                        help="Comma-separated scorers (default: tfold,ergo,nettcr). tFold is primary (AUC=0.800)")
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
    # Unlock tiers A/B/D for evaluation (exclude C per design choice)
    decoy_scorer.set_unlocked_tiers(["A", "B", "D"])

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
            tier_str = ""
            if "per_tier_auroc" in sr and sr["per_tier_auroc"]:
                tier_parts = [f"{t}:{v:.3f}" for t, v in sorted(sr["per_tier_auroc"].items())]
                tier_str = f"  [{', '.join(tier_parts)}]"
            print(f"  [{scorer_name:>7}] AUROC: {sr['auroc']:.4f}  "
                  f"Target: {sr['mean_target_score']:.4f}  "
                  f"Decoy: {sr['mean_decoy_score']:.4f}{tier_str}")
            # Ranking metrics line
            hit_pct = sr.get('hit_rate_07', 0) * 100
            print(f"  {' ':>9} Top1: {sr.get('top1_target', 0):.3f} (AUROC {sr.get('top1_auroc', 0):.3f})  "
                  f"Top5: {sr.get('top5_target', 0):.3f} (AUROC {sr.get('top5_auroc', 0):.3f})  "
                  f"Hit@0.7: {hit_pct:.0f}%  "
                  f"Comp: {sr.get('top1_composite', 0):.3f}")
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
    print(f"\n{'='*70}")
    print("SUMMARY — Overall AUROC")
    print(f"{'='*70}")

    # Collect all tier names that appeared
    all_tiers = set()
    for peptide in targets:
        for sn in scorers:
            tier_data = all_results[peptide]["specificity"][sn].get("per_tier_auroc", {})
            all_tiers.update(tier_data.keys())
    tier_list = sorted(all_tiers)

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

    # Per-tier AUROC breakdown
    if tier_list:
        print(f"\n{'='*70}")
        print("PER-TIER AUROC BREAKDOWN")
        print(f"{'='*70}")
        for sn in scorer_list:
            print(f"\n  Scorer: {sn}")
            tier_header = f"  {'Target':<15}"
            for tier in tier_list:
                tier_header += f" Tier{tier:>2}"
            print(tier_header)
            print(f"  {'-'*(15 + 7 * len(tier_list))}")

            tier_auroc_lists = {t: [] for t in tier_list}
            for peptide in targets:
                tier_data = all_results[peptide]["specificity"][sn].get("per_tier_auroc", {})
                line = f"  {peptide:<15}"
                for tier in tier_list:
                    val = tier_data.get(tier, float("nan"))
                    if np.isnan(val):
                        line += f" {'N/A':>6}"
                    else:
                        line += f" {val:>6.3f}"
                        tier_auroc_lists[tier].append(val)
                print(line)

            # Tier means
            print(f"  {'-'*(15 + 7 * len(tier_list))}")
            mean_tier_line = f"  {'Mean':<15}"
            for tier in tier_list:
                vals = tier_auroc_lists[tier]
                if vals:
                    mean_tier_line += f" {np.mean(vals):>6.3f}"
                else:
                    mean_tier_line += f" {'N/A':>6}"
            print(mean_tier_line)

    # Ranking metrics summary
    print(f"\n{'='*70}")
    print("RANKING METRICS — Top-K Binding + Specificity")
    print(f"{'='*70}")
    ranking_agg = {}  # scorer -> {metric_name -> [values across targets]}
    for sn in scorer_list:
        ranking_agg[sn] = {
            "top1_target": [], "top1_auroc": [],
            "top5_target": [], "top5_auroc": [],
            "hit_rate_07": [], "top1_composite": [],
        }
        print(f"\n  Scorer: {sn}")
        print(f"  {'Target':<15} Top1_Bind Top1_AUC Top5_Bind Top5_AUC Hit@0.7 Composite")
        print(f"  {'-'*79}")
        for peptide in targets:
            sr = all_results[peptide]["specificity"].get(sn, {})
            t1b = sr.get("top1_target", 0)
            t1a = sr.get("top1_auroc", 0)
            t5b = sr.get("top5_target", 0)
            t5a = sr.get("top5_auroc", 0)
            hr  = sr.get("hit_rate_07", 0)
            comp = sr.get("top1_composite", 0)
            print(f"  {peptide:<15} {t1b:>9.3f} {t1a:>8.3f} {t5b:>9.3f} {t5a:>8.3f} {hr:>7.1%} {comp:>9.3f}")
            ranking_agg[sn]["top1_target"].append(t1b)
            ranking_agg[sn]["top1_auroc"].append(t1a)
            ranking_agg[sn]["top5_target"].append(t5b)
            ranking_agg[sn]["top5_auroc"].append(t5a)
            ranking_agg[sn]["hit_rate_07"].append(hr)
            ranking_agg[sn]["top1_composite"].append(comp)
        print(f"  {'-'*79}")
        print(f"  {'Mean':<15} "
              f"{np.mean(ranking_agg[sn]['top1_target']):>9.3f} "
              f"{np.mean(ranking_agg[sn]['top1_auroc']):>8.3f} "
              f"{np.mean(ranking_agg[sn]['top5_target']):>9.3f} "
              f"{np.mean(ranking_agg[sn]['top5_auroc']):>8.3f} "
              f"{np.mean(ranking_agg[sn]['hit_rate_07']):>7.1%} "
              f"{np.mean(ranking_agg[sn]['top1_composite']):>9.3f}")

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

    # Build ranking summary across targets
    ranking_summary = {}
    for sn in scorer_list:
        ra = ranking_agg[sn]
        ranking_summary[sn] = {
            "mean_top1_target": float(np.mean(ra["top1_target"])),
            "mean_top5_target": float(np.mean(ra["top5_target"])),
            "mean_top1_auroc": float(np.mean(ra["top1_auroc"])),
            "mean_top5_auroc": float(np.mean(ra["top5_auroc"])),
            "mean_top1_composite": float(np.mean(ra["top1_composite"])),
            "mean_hit_rate": float(np.mean(ra["hit_rate_07"])),
        }

    save_results["_summary"] = {
        "mean_auroc": mean_aurocs,
        "ranking": ranking_summary,
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
