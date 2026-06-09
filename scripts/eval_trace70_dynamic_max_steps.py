#!/usr/bin/env python
"""Step-wise max-steps probe for trace70 checkpoints.

The probe selects initial TCR/peptide pairs from trace70 tFoldScore logs by
logged initial affinity bands, loads one PPO checkpoint, runs a longer rollout,
and scores every intermediate TCR with tFold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_checkpoint_decoy_reward_tfold import (  # noqa: E402
    build_env,
    build_tcr_pool,
    load_config,
    load_policy,
    load_targets,
    make_mask_tensors,
    merged_checkpoint_config,
)
from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer  # noqa: E402
from tcrppo_v2.utils.constants import IDX_TO_AA  # noqa: E402
from tcrppo_v2.utils.esm_cache import ESMCache  # noqa: E402


TFOLD_RE = re.compile(
    r"affinity_logit=(?P<aff>-?\d+(?:\.\d+)?)\s+"
    r"conf=.*?\s+cdr3b=(?P<tcr>[A-Z]+)\s+peptide=(?P<pep>[A-Z]+)"
)


def finite_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def read_logged_pairs(log_path: str, allowed_peptides: set[str]) -> List[dict]:
    pairs: "OrderedDict[Tuple[str, str], dict]" = OrderedDict()
    with open(log_path) as f:
        for line in f:
            match = TFOLD_RE.search(line)
            if not match:
                continue
            peptide = match.group("pep")
            if peptide not in allowed_peptides:
                continue
            tcr = match.group("tcr")
            aff = float(match.group("aff"))
            key = (tcr, peptide)
            if key not in pairs:
                pairs[key] = {
                    "tcr": tcr,
                    "peptide": peptide,
                    "logged_affinity": aff,
                }
    return list(pairs.values())


def choose_candidates(
    pairs: List[dict],
    per_band: int,
    high_threshold: float,
    low_threshold: float,
) -> List[dict]:
    bands = {
        "high_gt_m0p5": [
            p for p in pairs
            if p["logged_affinity"] > high_threshold
        ],
        "mid_m5_to_m0p5": [
            p for p in pairs
            if low_threshold <= p["logged_affinity"] <= high_threshold
        ],
        "low_lt_m5": [
            p for p in pairs
            if p["logged_affinity"] < low_threshold
        ],
    }
    bands["high_gt_m0p5"].sort(key=lambda p: p["logged_affinity"], reverse=True)
    bands["mid_m5_to_m0p5"].sort(key=lambda p: abs(p["logged_affinity"] + 2.0))
    bands["low_lt_m5"].sort(key=lambda p: p["logged_affinity"])

    selected: List[dict] = []
    for band, band_pairs in bands.items():
        if len(band_pairs) < per_band:
            raise RuntimeError(
                f"Not enough candidates for {band}: need {per_band}, found {len(band_pairs)}"
            )
        for row in band_pairs[:per_band]:
            selected.append({**row, "band": band})
    return selected


def format_action(action: Tuple[int, int, int]) -> str:
    op, pos, tok = action
    if op == 0:
        return f"SUB:{pos}:{IDX_TO_AA.get(tok, tok)}"
    if op == 1:
        return f"INS:{pos}:{IDX_TO_AA.get(tok, tok)}"
    if op == 2:
        return f"DEL:{pos}"
    if op == 3:
        return "STOP"
    return f"op{op}:{pos}:{tok}"


@torch.no_grad()
def greedy_action(policy, obs: np.ndarray, mask: dict, device: str) -> Tuple[int, int, int]:
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    mask_dict = make_mask_tensors(mask, device)
    features = policy.backbone(obs_tensor)
    op_logits = policy.op_head(features).masked_fill(~mask_dict["op_mask"], float("-inf"))
    op = torch.argmax(op_logits, dim=-1)
    op_emb = policy.op_embed(op)
    pos_logits = policy.pos_head(torch.cat([features, op_emb], dim=-1))
    pos_logits = pos_logits.masked_fill(~mask_dict["pos_mask"], float("-inf"))
    pos = torch.argmax(pos_logits, dim=-1)
    pos_emb = policy.pos_embed(pos)
    tok_logits = policy.token_head(torch.cat([features, op_emb, pos_emb], dim=-1))
    tok = torch.argmax(tok_logits, dim=-1)
    return int(op[0]), int(pos[0]), int(tok[0])


@torch.no_grad()
def stochastic_action(policy, obs: np.ndarray, mask: dict, device: str) -> Tuple[int, int, int]:
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    mask_dict = make_mask_tensors(mask, device)
    op, pos, tok, _ = policy(obs_tensor, mask_dict)
    return int(op[0]), int(pos[0]), int(tok[0])


def iter_chunks(items: Sequence[Tuple[str, str]], size: int) -> Iterable[Sequence[Tuple[str, str]]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def score_pairs(
    scorer: AffinityTFoldScorer,
    pairs: Sequence[Tuple[str, str]],
    batch_size: int,
) -> Dict[Tuple[str, str], float]:
    unique_pairs = list(dict.fromkeys(pairs))
    scores: Dict[Tuple[str, str], float] = {}
    total = len(unique_pairs)
    for batch_idx, batch in enumerate(iter_chunks(unique_pairs, batch_size), start=1):
        print(f"[score] batch {batch_idx}: {len(batch)} pairs ({len(scores)}/{total} done)", flush=True)
        tcrs = [p[0] for p in batch]
        peps = [p[1] for p in batch]
        batch_scores, _ = scorer.score_batch(tcrs, peps)
        for pair, score in zip(batch, batch_scores):
            scores[pair] = float(score)
    return scores


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="output/trace70_gate_m1p5_from_trace61/checkpoints/latest.pt")
    parser.add_argument("--config", default="configs/trace70_gate_m1p5_from_trace61.yaml")
    parser.add_argument("--trace-log", default="logs/trace70_gate_m1p5_from_trace61_train.log")
    parser.add_argument("--targets-file", default="data/tfold_excellent_peptides.txt")
    parser.add_argument("--output-dir", default="results/trace70_dynamic_max_steps_probe")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--per-band", type=int, default=3)
    parser.add_argument("--high-threshold", type=float, default=-0.5)
    parser.add_argument("--low-threshold", type=float, default=-5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--esm-cache-path", default=None)
    parser.add_argument("--tfold-cache-path", default="data/tfold_feature_cache_trace70_gate_m1p5_from_trace61.db")
    parser.add_argument("--tfold-server-socket", default="/tmp/tfold_server_trace70_dynamic_max_steps_eval.sock")
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    targets = load_targets(args.targets_file)
    allowed_peptides = set(targets)
    logged_pairs = read_logged_pairs(args.trace_log, allowed_peptides)
    candidates = choose_candidates(
        logged_pairs,
        per_band=args.per_band,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
    )

    fallback_config = load_config(args.config)
    config = merged_checkpoint_config(args.checkpoint, fallback_config)
    config["max_steps"] = int(args.max_steps)
    config["max_steps_per_episode"] = int(args.max_steps)
    config["tfold_cache_path"] = args.tfold_cache_path
    config["tfold_server_socket"] = args.tfold_server_socket

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_step = int(ckpt.get("global_step", -1))
    del ckpt

    if args.esm_cache_path is None:
        args.esm_cache_path = os.path.join(args.output_dir, "esm_cache_eval.db")
    esm_cache = ESMCache(
        device=args.device,
        tcr_cache_size=int(config.get("esm_tcr_cache_size", 4096)),
        disk_cache_path=args.esm_cache_path,
    )
    tcr_pool = build_tcr_pool(config, targets, args.seed)
    env = build_env(config, targets, esm_cache, tcr_pool)
    env.set_global_step(checkpoint_step)
    policy = load_policy(args.checkpoint, obs_dim=env.obs_dim, config=config, device=args.device)

    action_fn = stochastic_action if args.stochastic else greedy_action
    trajectories: List[dict] = []
    all_score_pairs: List[Tuple[str, str]] = []

    print(
        f"[generate] checkpoint_step={checkpoint_step} candidates={len(candidates)} max_steps={args.max_steps}",
        flush=True,
    )
    for idx, cand in enumerate(candidates):
        seed_i = args.seed + idx * 1009
        random.seed(seed_i)
        np.random.seed(seed_i)
        torch.manual_seed(seed_i)

        obs = env.reset(peptide=cand["peptide"], init_tcr=cand["tcr"])
        states = [{
            "step": 0,
            "action": "INIT",
            "tcr": cand["tcr"],
        }]
        while not env.done:
            action = action_fn(policy, obs, env.get_action_mask(), args.device)
            obs, _, _, info = env.step(action)
            states.append({
                "step": int(info.get("step", env.step_count)),
                "action": format_action(action),
                "tcr": info.get("new_tcr", env.current_tcr),
            })

        trajectories.append({
            "candidate_idx": idx,
            "band": cand["band"],
            "peptide": cand["peptide"],
            "initial_tcr": cand["tcr"],
            "logged_initial_affinity": cand["logged_affinity"],
            "states": states,
        })
        all_score_pairs.extend((state["tcr"], cand["peptide"]) for state in states)

    print("[score] Loading tFold scorer", flush=True)
    scorer = AffinityTFoldScorer(
        device=args.device,
        cache_path=args.tfold_cache_path,
        server_socket_path=args.tfold_server_socket,
        max_subprocess_batch=args.extract_batch_size,
        use_cache=True,
    )
    t0 = time.time()
    score_map = score_pairs(scorer, all_score_pairs, args.score_batch_size)
    print(f"[score] completed in {time.time() - t0:.1f}s", flush=True)

    step_rows: List[dict] = []
    trajectory_rows: List[dict] = []
    for traj in trajectories:
        peptide = traj["peptide"]
        affinities = [score_map[(state["tcr"], peptide)] for state in traj["states"]]
        initial_affinity = affinities[0]
        best_affinity = max(affinities)
        best_step = int(np.argmax(affinities))
        affinity_at_4 = affinities[min(4, len(affinities) - 1)]
        affinity_at_8 = affinities[min(8, len(affinities) - 1)]
        affinity_at_max = affinities[-1]
        best_after_8 = max(affinities[9:]) if len(affinities) > 9 else float("nan")

        trajectory_rows.append({
            "candidate_idx": traj["candidate_idx"],
            "band": traj["band"],
            "peptide": peptide,
            "initial_tcr": traj["initial_tcr"],
            "logged_initial_affinity": traj["logged_initial_affinity"],
            "rescored_initial_affinity": initial_affinity,
            "best_affinity": best_affinity,
            "best_step": best_step,
            "affinity_step4": affinity_at_4,
            "affinity_step8": affinity_at_8,
            "affinity_step_max": affinity_at_max,
            "gain_best_minus_initial": best_affinity - initial_affinity,
            "gain_step8_minus_initial": affinity_at_8 - initial_affinity,
            "gain_stepmax_minus_initial": affinity_at_max - initial_affinity,
            "gain_best_after8_minus_step8": (
                best_after_8 - affinity_at_8 if np.isfinite(best_after_8) else float("nan")
            ),
            "did_improve_after8": int(np.isfinite(best_after_8) and best_after_8 > affinity_at_8),
        })

        running_best = -math.inf
        for state, affinity in zip(traj["states"], affinities):
            running_best = max(running_best, affinity)
            step_rows.append({
                "candidate_idx": traj["candidate_idx"],
                "band": traj["band"],
                "peptide": peptide,
                "initial_tcr": traj["initial_tcr"],
                "logged_initial_affinity": traj["logged_initial_affinity"],
                "step": state["step"],
                "action": state["action"],
                "tcr": state["tcr"],
                "affinity": affinity,
                "delta_from_initial": affinity - initial_affinity,
                "running_best_affinity": running_best,
                "is_running_best": int(abs(affinity - running_best) < 1e-9),
            })

    band_summary: List[dict] = []
    for band in sorted({row["band"] for row in trajectory_rows}):
        rows = [row for row in trajectory_rows if row["band"] == band]
        band_summary.append({
            "band": band,
            "n": len(rows),
            "mean_initial_affinity": finite_mean([r["rescored_initial_affinity"] for r in rows]),
            "mean_best_affinity": finite_mean([r["best_affinity"] for r in rows]),
            "mean_best_step": finite_mean([r["best_step"] for r in rows]),
            "mean_step4_affinity": finite_mean([r["affinity_step4"] for r in rows]),
            "mean_step8_affinity": finite_mean([r["affinity_step8"] for r in rows]),
            "mean_stepmax_affinity": finite_mean([r["affinity_step_max"] for r in rows]),
            "mean_gain_step8_minus_initial": finite_mean([r["gain_step8_minus_initial"] for r in rows]),
            "mean_gain_stepmax_minus_initial": finite_mean([r["gain_stepmax_minus_initial"] for r in rows]),
            "mean_gain_best_after8_minus_step8": finite_mean([r["gain_best_after8_minus_step8"] for r in rows]),
            "frac_improved_after8": finite_mean([r["did_improve_after8"] for r in rows]),
        })

    write_csv(os.path.join(args.output_dir, "stepwise_affinity.csv"), step_rows)
    write_csv(os.path.join(args.output_dir, "trajectory_summary.csv"), trajectory_rows)
    write_csv(os.path.join(args.output_dir, "band_summary.csv"), band_summary)

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "checkpoint_step": checkpoint_step,
        "n_logged_pairs": len(logged_pairs),
        "candidates": candidates,
        "tfold_cache_stats": scorer.cache_stats,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nBand summary:")
    for row in band_summary:
        print(
            f"  {row['band']:>16} n={row['n']} "
            f"init={row['mean_initial_affinity']:.3f} "
            f"best={row['mean_best_affinity']:.3f} "
            f"best_step={row['mean_best_step']:.1f} "
            f"step4={row['mean_step4_affinity']:.3f} "
            f"step8={row['mean_step8_affinity']:.3f} "
            f"step{args.max_steps}={row['mean_stepmax_affinity']:.3f} "
            f"after8_gain={row['mean_gain_best_after8_minus_step8']:.3f} "
            f"after8_frac={row['frac_improved_after8']:.2f}"
        )
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
