#!/usr/bin/env python3
"""Evaluate checkpoint target-vs-decoy tFold rewards.

This script is intentionally read-only with respect to training runs: it loads
checkpoint files, generates TCRs with reward computation disabled, then scores
the generated TCRs against target and decoy peptides with a specified tFold
feature-server socket.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcrppo_v2.data.decoy_sampler import DecoySampler
from tcrppo_v2.data.pmhc_loader import PMHCLoader
from tcrppo_v2.data.tcr_pool import TCRPool
from tcrppo_v2.env import TCREditEnv
from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.reward_manager import RewardManager
from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer
from tcrppo_v2.utils.constants import MAX_STEPS_PER_EPISODE, MAX_TCR_LEN
from tcrppo_v2.utils.esm_cache import ESMCache
from tcrppo_v2.utils.lightweight_encoder import LightweightEncoder


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path
    step: int


def read_targets(path: Path) -> List[str]:
    with path.open() as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def safe_torch_load(path: Path, map_location: str = "cpu") -> dict:
    return torch.load(str(path), map_location=map_location, weights_only=False)


def checkpoint_step(path: Path) -> int:
    try:
        return int(safe_torch_load(path, "cpu").get("global_step", 0))
    except Exception:
        match = re.search(r"(\d+)", path.stem)
        return int(match.group(1)) if match else 0


def snapshot_latest(src: Path, out_dir: Path, retries: int = 3) -> Path | None:
    if not src.exists():
        return None
    snap_dir = out_dir / "checkpoint_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        step = checkpoint_step(src)
        dst = snap_dir / f"latest_snapshot_step{step}.pt"
        try:
            shutil.copy2(src, dst)
            # Validate that the copy is a coherent torch checkpoint.
            _ = safe_torch_load(dst, "cpu")
            return dst
        except Exception as exc:
            print(f"[warn] latest snapshot attempt {attempt}/{retries} failed: {exc}", flush=True)
            time.sleep(2)
    return None


def discover_checkpoints(
    checkpoint_dir: Path,
    output_dir: Path,
    include_latest: bool,
    explicit: Sequence[str] | None,
) -> List[CheckpointSpec]:
    if explicit:
        paths = []
        for item in explicit:
            p = Path(item)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            paths.append(p)
    else:
        paths = sorted(
            checkpoint_dir.glob("milestone_*.pt"),
            key=lambda p: int(re.search(r"milestone_(\d+)", p.stem).group(1)),
        )
        if include_latest:
            latest = snapshot_latest(checkpoint_dir / "latest.pt", output_dir)
            if latest is not None:
                paths.append(latest)

    specs = []
    seen = set()
    for path in paths:
        if not path.exists():
            print(f"[warn] checkpoint missing, skipping: {path}", flush=True)
            continue
        step = checkpoint_step(path)
        label = path.stem
        if label.startswith("latest_snapshot"):
            label = f"latest_step{step}"
        key = (label, step, str(path))
        if key in seen:
            continue
        seen.add(key)
        specs.append(CheckpointSpec(label=label, path=path, step=step))
    return specs


def build_encoder(config: dict, device: str, esm_cache_path: Path):
    encoder_type = config.get("encoder", "esm2")
    if encoder_type == "lightweight":
        return LightweightEncoder(
            device=device,
            encoder_output_dim=config.get("encoder_dim", 256),
            tcr_cache_size=config.get("esm_tcr_cache_size", 4096),
            disk_cache_path=str(esm_cache_path),
        )
    return ESMCache(
        device=device,
        tcr_cache_size=config.get("esm_tcr_cache_size", 4096),
        disk_cache_path=str(esm_cache_path),
    )


def build_generation_env(config: dict, targets: List[str], device: str, esm_cache_path: Path) -> TCREditEnv:
    esm_cache = build_encoder(config, device, esm_cache_path)
    pmhc_loader = PMHCLoader(targets=targets)

    l1_dir = PROJECT_ROOT / "data" / "l1_seeds"
    tcr_pool = TCRPool(
        tcrdb_path=config.get("tcrdb_path", "/share/liuyutian/TCRPPO/data/tcrdb"),
        l1_seeds_dir=str(l1_dir) if l1_dir.is_dir() else None,
        l0_mutation_range=tuple(config.get("l0_mutation_range", (3, 5))),
        l1_top_k=config.get("l1_top_k", 500),
        curriculum_schedule=config.get("curriculum_schedule"),
        seed=config.get("seed", 42),
    )
    decoy_lib_path = config.get("decoy_library_path", "/share/liuyutian/pMHC_decoy_library")
    tcr_pool.load_l0_from_decoy_d(decoy_lib_path, targets)
    l0_tchard_dir = PROJECT_ROOT / "data" / "l0_seeds_tchard"
    if l0_tchard_dir.is_dir():
        tcr_pool.load_l0_from_dir(str(l0_tchard_dir))

    reward_manager = RewardManager(affinity_scorer=None, reward_mode="disabled")
    return TCREditEnv(
        esm_cache=esm_cache,
        pmhc_loader=pmhc_loader,
        tcr_pool=tcr_pool,
        reward_manager=reward_manager,
        max_steps=config.get("max_steps", config.get("max_steps_per_episode", MAX_STEPS_PER_EPISODE)),
        max_tcr_len=config.get("max_tcr_len", MAX_TCR_LEN),
        min_tcr_len=config.get("min_tcr_len", 8),
        reward_mode=config.get("reward_mode", "v2_full"),
        min_steps=config.get("min_steps", 0),
        min_steps_penalty=config.get("min_steps_penalty", 0.0),
        ban_stop=config.get("ban_stop", False),
        sub_only=config.get("sub_only", False),
        terminal_reward_only=config.get("terminal_reward_only", False),
    )


def load_policy(path: Path, obs_dim: int, hidden_dim: int, max_tcr_len: int, device: str) -> ActorCritic:
    ckpt = safe_torch_load(path, device)
    ckpt_config = ckpt.get("config", {})
    policy = ActorCritic(
        obs_dim=obs_dim,
        hidden_dim=ckpt_config.get("hidden_dim", hidden_dim),
        max_tcr_len=ckpt_config.get("max_tcr_len", max_tcr_len),
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


def generate_tcrs(
    policy: ActorCritic,
    env: TCREditEnv,
    target: str,
    n_tcrs: int,
    device: str,
) -> List[dict]:
    out = []
    for i in range(n_tcrs):
        obs = env.reset(peptide=target)
        trajectory = {
            "target": target,
            "sample_index": i,
            "initial_tcr": env.initial_tcr,
            "steps": [],
        }
        while not env.done:
            mask = env.get_action_mask()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_dict = {
                "op_mask": torch.as_tensor(mask["op_mask"], dtype=torch.bool, device=device).unsqueeze(0),
                "pos_mask": torch.as_tensor(mask["pos_mask"], dtype=torch.bool, device=device).unsqueeze(0),
            }
            with torch.no_grad():
                op, pos, tok, _ = policy(obs_tensor, mask_dict)
            action = (int(op[0]), int(pos[0]), int(tok[0]))
            obs, reward, done, info = env.step(action)
            trajectory["steps"].append(
                {
                    "action": info.get("action_name", ""),
                    "tcr": info.get("new_tcr", ""),
                    "reward": float(reward),
                }
            )
        trajectory["final_tcr"] = env.current_tcr
        trajectory["n_steps"] = env.step_count
        out.append(trajectory)
    return out


def choose_decoys(
    sampler: DecoySampler,
    target: str,
    n_decoys: int,
    tiers: Sequence[str],
) -> List[Tuple[str, str]]:
    if target not in sampler.decoys:
        sampler._load_target_decoys(target)  # Existing helper; keeps this script simple.
    pools = sampler.decoys.get(target, {})
    available = [tier for tier in tiers if pools.get(tier)]
    if not available:
        return []

    chosen: List[Tuple[str, str]] = []
    chosen_set = set()
    per_tier = int(np.ceil(n_decoys / len(available)))
    for tier in available:
        seqs = [s for s in pools[tier] if s and s not in chosen_set]
        if not seqs:
            continue
        k = min(per_tier, len(seqs), n_decoys - len(chosen))
        if k <= 0:
            break
        idx = sampler.rng.choice(len(seqs), size=k, replace=False)
        for i in idx:
            seq = seqs[int(i)]
            chosen.append((seq, tier))
            chosen_set.add(seq)

    if len(chosen) < n_decoys:
        fill = []
        for tier in available:
            for seq in pools[tier]:
                if seq and seq not in chosen_set:
                    fill.append((seq, tier))
        sampler.rng.shuffle(fill)
        chosen.extend(fill[: n_decoys - len(chosen)])

    return chosen[:n_decoys]


def batched_score(
    scorer: AffinityTFoldScorer,
    tcrs: Sequence[str],
    peptides: Sequence[str],
    batch_size: int,
) -> List[float]:
    scores: List[float] = []
    for start in range(0, len(tcrs), batch_size):
        end = start + batch_size
        batch_scores, _ = scorer.score_batch(list(tcrs[start:end]), list(peptides[start:end]))
        scores.extend(float(s) for s in batch_scores)
    return scores


def summarize_target(
    target_scores: List[float],
    decoy_scores_by_tcr: List[List[float]],
    decoy_tiers: List[str],
) -> dict:
    flat_decoys = [s for scores in decoy_scores_by_tcr for s in scores]
    per_tcr_aurocs = []
    for ts, ds in zip(target_scores, decoy_scores_by_tcr):
        if not ds:
            continue
        labels = np.array([1] + [0] * len(ds))
        scores = np.array([ts] + ds)
        try:
            per_tcr_aurocs.append(float(roc_auc_score(labels, scores)))
        except ValueError:
            per_tcr_aurocs.append(0.5)

    tier_means = {}
    for tier in sorted(set(decoy_tiers)):
        idx = [i for i, t in enumerate(decoy_tiers) if t == tier]
        vals = []
        for ds in decoy_scores_by_tcr:
            vals.extend(ds[i] for i in idx if i < len(ds))
        if vals:
            tier_means[tier] = float(np.mean(vals))

    target_mean = float(np.mean(target_scores)) if target_scores else float("nan")
    decoy_mean = float(np.mean(flat_decoys)) if flat_decoys else float("nan")
    return {
        "target_reward_mean": target_mean,
        "target_reward_std": float(np.std(target_scores)) if target_scores else float("nan"),
        "decoy_reward_mean": decoy_mean,
        "decoy_reward_std": float(np.std(flat_decoys)) if flat_decoys else float("nan"),
        "margin_target_minus_decoy": target_mean - decoy_mean,
        "auroc": float(np.mean(per_tcr_aurocs)) if per_tcr_aurocs else 0.5,
        "n_tcrs": len(target_scores),
        "n_decoys_per_tcr": len(decoy_tiers),
        "decoy_tier_means": tier_means,
    }


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(
    path: Path,
    run_name: str,
    checkpoint_specs: List[CheckpointSpec],
    targets: List[str],
    summary_rows: List[dict],
    per_target_rows: List[dict],
    args: argparse.Namespace,
) -> None:
    by_step = {int(r["step"]): r for r in summary_rows}
    lines = [
        f"# {run_name} checkpoint decoy-reward evaluation",
        "",
        f"- Generated TCRs per target: {args.n_tcrs}",
        f"- Decoys per target: {args.n_decoys}",
        f"- Targets: {len(targets)} from `{args.targets_file}`",
        f"- tFold socket: `{args.tfold_socket}`",
        f"- Score: raw tFold binding logit, higher means stronger predicted binding.",
        "",
        "## Checkpoint Summary",
        "",
        "| checkpoint | step | target mean | decoy mean | margin | AUROC | unique/TCR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in checkpoint_specs:
        row = by_step.get(spec.step)
        if not row:
            continue
        lines.append(
            "| {label} | {step} | {target:.4f} | {decoy:.4f} | {margin:.4f} | {auroc:.4f} | {uniq:.3f} |".format(
                label=spec.label,
                step=spec.step,
                target=float(row["target_reward_mean"]),
                decoy=float(row["decoy_reward_mean"]),
                margin=float(row["margin_target_minus_decoy"]),
                auroc=float(row["mean_auroc"]),
                uniq=float(row["unique_fraction"]),
            )
        )

    if summary_rows:
        best_margin = max(summary_rows, key=lambda r: float(r["margin_target_minus_decoy"]))
        best_target = max(summary_rows, key=lambda r: float(r["target_reward_mean"]))
        best_decoy = min(summary_rows, key=lambda r: float(r["decoy_reward_mean"]))
        lines.extend(
            [
                "",
                "## Quick Read",
                "",
                f"- Best margin: step {best_margin['step']} ({float(best_margin['margin_target_minus_decoy']):.4f}).",
                f"- Best target reward: step {best_target['step']} ({float(best_target['target_reward_mean']):.4f}).",
                f"- Lowest decoy reward: step {best_decoy['step']} ({float(best_decoy['decoy_reward_mean']):.4f}).",
                "",
                "Interpretation: the preferred checkpoint should keep target reward high while keeping decoy reward low; margin and AUROC summarize that tradeoff.",
            ]
        )

    latest_step = max(int(r["step"]) for r in summary_rows) if summary_rows else None
    if latest_step is not None:
        latest_rows = [r for r in per_target_rows if int(r["step"]) == latest_step]
        lines.extend(
            [
                "",
                f"## Latest Step {latest_step} Per-Target",
                "",
                "| target | target mean | decoy mean | margin | AUROC | decoy tiers |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in latest_rows:
            tier_counts = row.get("decoy_tier_counts", "")
            lines.append(
                "| {target} | {target_mean:.4f} | {decoy_mean:.4f} | {margin:.4f} | {auroc:.4f} | {tiers} |".format(
                    target=row["target"],
                    target_mean=float(row["target_reward_mean"]),
                    decoy_mean=float(row["decoy_reward_mean"]),
                    margin=float(row["margin_target_minus_decoy"]),
                    auroc=float(row["auroc"]),
                    tiers=tier_counts,
                )
            )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate target/decoy tFold rewards across PPO checkpoints.")
    parser.add_argument("--checkpoint-dir", default="output/test51c_amp_server_rerun_detached_trace11_delta/checkpoints")
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Optional explicit checkpoint paths.")
    parser.add_argument("--include-latest", action="store_true", help="Snapshot and include latest.pt.")
    parser.add_argument("--config", default="configs/test51c.yaml")
    parser.add_argument("--targets-file", default="data/tfold_excellent_peptides.txt")
    parser.add_argument("--output-dir", default="results/test51c_trace11_delta_decoy_rewards")
    parser.add_argument("--tfold-socket", default="/tmp/tfold_server_trace11_delta.sock")
    parser.add_argument("--tfold-cache-path", default="data/tfold_feature_cache.db")
    parser.add_argument("--n-tcrs", type=int, default=3)
    parser.add_argument("--n-decoys", type=int, default=12)
    parser.add_argument("--decoy-tiers", default="A,B,D,C")
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--max-subprocess-batch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    targets_file = Path(args.targets_file)
    if not targets_file.is_absolute():
        targets_file = PROJECT_ROOT / targets_file

    with config_path.open() as f:
        config = yaml.safe_load(f) or {}
    config["reward_mode"] = config.get("reward_mode", "v2_no_decoy_delta")
    config["ban_stop"] = True
    config["terminal_reward_only"] = config.get("terminal_reward_only", True)

    targets = read_targets(targets_file)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    checkpoint_specs = discover_checkpoints(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        include_latest=args.include_latest,
        explicit=args.checkpoints,
    )
    if not checkpoint_specs:
        raise SystemExit(f"No checkpoints found under {checkpoint_dir}")

    esm_cache_path = output_dir / "esm_cache_eval.db"
    print(f"Targets: {len(targets)}", flush=True)
    print("Checkpoints:", ", ".join(f"{c.label}@{c.step}" for c in checkpoint_specs), flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Using tFold socket: {args.tfold_socket}", flush=True)

    env = build_generation_env(config, targets, args.device, esm_cache_path)

    tfold_cache_path = Path(args.tfold_cache_path)
    if not tfold_cache_path.is_absolute():
        tfold_cache_path = PROJECT_ROOT / tfold_cache_path
    scorer = AffinityTFoldScorer(
        device=args.device,
        cache_path=str(tfold_cache_path),
        server_socket_path=args.tfold_socket,
        max_subprocess_batch=args.max_subprocess_batch,
        use_cache=True,
    )

    decoy_tiers = [t.strip() for t in args.decoy_tiers.split(",") if t.strip()]
    sampler = DecoySampler(
        decoy_library_path=config.get("decoy_library_path", "/share/liuyutian/pMHC_decoy_library"),
        targets=targets,
        seed=args.seed,
    )
    sampler.unlocked_tiers = set(decoy_tiers)

    all_pair_rows: List[dict] = []
    per_target_rows: List[dict] = []
    summary_rows: List[dict] = []
    generated_blob: Dict[str, dict] = {}

    for spec in checkpoint_specs:
        print(f"\n=== {spec.label} (step {spec.step}) ===", flush=True)
        env.set_global_step(spec.step)
        ckpt_preview = safe_torch_load(spec.path, "cpu")
        ckpt_cfg = ckpt_preview.get("config", {})
        hidden_dim = ckpt_cfg.get("hidden_dim", config.get("hidden_dim", 512))
        max_tcr_len = ckpt_cfg.get("max_tcr_len", config.get("max_tcr_len", MAX_TCR_LEN))
        del ckpt_preview
        policy = load_policy(spec.path, env.obs_dim, hidden_dim, max_tcr_len, args.device)

        ckpt_target_rows = []
        generated_blob[spec.label] = {"step": spec.step, "targets": {}}

        for target_idx, target in enumerate(targets, start=1):
            print(f"[{spec.label}] target {target_idx}/{len(targets)} {target}", flush=True)
            trajectories = generate_tcrs(policy, env, target, args.n_tcrs, args.device)
            tcrs = [t["final_tcr"] for t in trajectories]
            generated_blob[spec.label]["targets"][target] = trajectories

            decoys = choose_decoys(sampler, target, args.n_decoys, decoy_tiers)
            decoy_peptides = [d[0] for d in decoys]
            decoy_labels = [d[1] for d in decoys]
            if not decoys:
                print(f"[warn] no decoys for {target}, skipping target", flush=True)
                continue

            target_scores = batched_score(
                scorer,
                tcrs,
                [target] * len(tcrs),
                batch_size=args.score_batch_size,
            )

            decoy_batch_tcrs = []
            decoy_batch_peps = []
            decoy_batch_tcr_index = []
            decoy_batch_decoy_index = []
            for tcr_i, tcr in enumerate(tcrs):
                for decoy_i, decoy in enumerate(decoy_peptides):
                    decoy_batch_tcrs.append(tcr)
                    decoy_batch_peps.append(decoy)
                    decoy_batch_tcr_index.append(tcr_i)
                    decoy_batch_decoy_index.append(decoy_i)
            flat_decoy_scores = batched_score(
                scorer,
                decoy_batch_tcrs,
                decoy_batch_peps,
                batch_size=args.score_batch_size,
            )
            decoy_scores_by_tcr = [[] for _ in tcrs]
            for score, tcr_i in zip(flat_decoy_scores, decoy_batch_tcr_index):
                decoy_scores_by_tcr[tcr_i].append(score)

            target_summary = summarize_target(target_scores, decoy_scores_by_tcr, decoy_labels)
            tier_counts = {tier: decoy_labels.count(tier) for tier in sorted(set(decoy_labels))}
            per_target = {
                "checkpoint": spec.label,
                "step": spec.step,
                "target": target,
                **{k: v for k, v in target_summary.items() if k != "decoy_tier_means"},
                "n_unique_tcrs": len(set(tcrs)),
                "unique_fraction": len(set(tcrs)) / max(len(tcrs), 1),
                "decoy_tier_counts": json.dumps(tier_counts, sort_keys=True),
                "decoy_tier_means": json.dumps(target_summary["decoy_tier_means"], sort_keys=True),
            }
            per_target_rows.append(per_target)
            ckpt_target_rows.append(per_target)

            for i, (traj, score) in enumerate(zip(trajectories, target_scores)):
                all_pair_rows.append(
                    {
                        "checkpoint": spec.label,
                        "step": spec.step,
                        "target": target,
                        "tcr_index": i,
                        "tcr": traj["final_tcr"],
                        "initial_tcr": traj["initial_tcr"],
                        "n_steps": traj["n_steps"],
                        "scored_peptide": target,
                        "peptide_type": "target",
                        "decoy_tier": "",
                        "score": score,
                    }
                )
            for score, tcr_i, decoy_i in zip(flat_decoy_scores, decoy_batch_tcr_index, decoy_batch_decoy_index):
                traj = trajectories[tcr_i]
                all_pair_rows.append(
                    {
                        "checkpoint": spec.label,
                        "step": spec.step,
                        "target": target,
                        "tcr_index": tcr_i,
                        "tcr": traj["final_tcr"],
                        "initial_tcr": traj["initial_tcr"],
                        "n_steps": traj["n_steps"],
                        "scored_peptide": decoy_peptides[decoy_i],
                        "peptide_type": "decoy",
                        "decoy_tier": decoy_labels[decoy_i],
                        "score": score,
                    }
                )

        if ckpt_target_rows:
            summary = {
                "checkpoint": spec.label,
                "step": spec.step,
                "n_targets": len(ckpt_target_rows),
                "n_tcrs_per_target": args.n_tcrs,
                "n_decoys_per_target": args.n_decoys,
                "target_reward_mean": float(np.mean([r["target_reward_mean"] for r in ckpt_target_rows])),
                "target_reward_std_across_targets": float(np.std([r["target_reward_mean"] for r in ckpt_target_rows])),
                "decoy_reward_mean": float(np.mean([r["decoy_reward_mean"] for r in ckpt_target_rows])),
                "decoy_reward_std_across_targets": float(np.std([r["decoy_reward_mean"] for r in ckpt_target_rows])),
                "margin_target_minus_decoy": float(np.mean([r["margin_target_minus_decoy"] for r in ckpt_target_rows])),
                "mean_auroc": float(np.mean([r["auroc"] for r in ckpt_target_rows])),
                "unique_fraction": float(np.mean([r["unique_fraction"] for r in ckpt_target_rows])),
            }
            summary_rows.append(summary)
            print(
                "summary step={step} target={target:.4f} decoy={decoy:.4f} margin={margin:.4f} auroc={auroc:.4f}".format(
                    step=spec.step,
                    target=summary["target_reward_mean"],
                    decoy=summary["decoy_reward_mean"],
                    margin=summary["margin_target_minus_decoy"],
                    auroc=summary["mean_auroc"],
                ),
                flush=True,
            )

        del policy
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        write_csv(
            output_dir / "summary.csv",
            summary_rows,
            [
                "checkpoint", "step", "n_targets", "n_tcrs_per_target", "n_decoys_per_target",
                "target_reward_mean", "target_reward_std_across_targets",
                "decoy_reward_mean", "decoy_reward_std_across_targets",
                "margin_target_minus_decoy", "mean_auroc", "unique_fraction",
            ],
        )
        write_csv(
            output_dir / "per_target.csv",
            per_target_rows,
            [
                "checkpoint", "step", "target", "target_reward_mean", "target_reward_std",
                "decoy_reward_mean", "decoy_reward_std", "margin_target_minus_decoy",
                "auroc", "n_tcrs", "n_decoys_per_tcr", "n_unique_tcrs",
                "unique_fraction", "decoy_tier_counts", "decoy_tier_means",
            ],
        )
        write_csv(
            output_dir / "pair_scores.csv",
            all_pair_rows,
            [
                "checkpoint", "step", "target", "tcr_index", "tcr", "initial_tcr",
                "n_steps", "scored_peptide", "peptide_type", "decoy_tier", "score",
            ],
        )
        (output_dir / "generated_tcrs.json").write_text(json.dumps(generated_blob, indent=2))
        write_report(
            output_dir / "report.md",
            run_name=checkpoint_dir.parent.name,
            checkpoint_specs=checkpoint_specs,
            targets=targets,
            summary_rows=summary_rows,
            per_target_rows=per_target_rows,
            args=args,
        )

    print(f"\nDone. Report: {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
