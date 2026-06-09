#!/usr/bin/env python
"""Evaluate checkpoint-specific target vs decoy tFold rewards.

This script is intentionally read-only with respect to PPO experiments: it loads
checkpoint files, generates TCRs without invoking the reward scorer, then scores
the resulting TCRs against target and decoy peptides using an existing tFold
feature server.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.data.decoy_sampler import DecoySampler
from tcrppo_v2.data.pmhc_loader import PMHCLoader
from tcrppo_v2.data.tcr_pool import TCRPool
from tcrppo_v2.env import TCREditEnv
from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.reward_manager import RewardManager
from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer
from tcrppo_v2.utils.constants import MAX_STEPS_PER_EPISODE, MAX_TCR_LEN
from tcrppo_v2.utils.esm_cache import ESMCache


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    label: str
    step: int


@dataclass(frozen=True)
class DecoyInfo:
    peptide: str
    tier: str


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def mean_or_nan(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def decoy_plan_from_mode(mode: str, n_decoys: int) -> Dict[str, int]:
    """Return the requested tier counts for a named eval mode."""
    if mode == "auto":
        return {}
    if mode in {"1", "a1"}:
        return {"A": 1}
    if mode in {"4", "abbd4"}:
        return {"A": 1, "B": 2, "D": 1}
    if mode in {"16", "a4b8d4"}:
        return {"A": 4, "B": 8, "D": 4}
    if mode == "custom":
        return {}
    raise ValueError(f"Unknown decoy mode: {mode}")


def parse_decoy_plan(text: str) -> Dict[str, int]:
    """Parse a comma-separated decoy plan such as A:1,B:2,D:1."""
    plan: Dict[str, int] = {}
    if not text:
        return plan
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid decoy plan item '{item}', expected TIER:COUNT")
        tier, count_text = item.split(":", 1)
        tier = tier.strip().upper()
        if tier not in {"A", "B", "C", "D"}:
            raise ValueError(f"Invalid decoy tier in plan: {tier}")
        count = int(count_text)
        if count < 0:
            raise ValueError(f"Negative decoy count in plan: {item}")
        if count:
            plan[tier] = plan.get(tier, 0) + count
    return plan


def format_decoy_plan(plan: Dict[str, int]) -> str:
    if not plan:
        return "auto"
    return "+".join(f"{tier}*{plan[tier]}" for tier in ("A", "B", "D", "C") if plan.get(tier, 0))


def load_targets(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def checkpoint_step_from_name(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"milestone_(\d+)\.pt$", base)
    if m:
        return int(m.group(1))
    return -1


def discover_checkpoints(checkpoint_dir: str, include_latest: bool = True) -> List[str]:
    paths = sorted(
        glob.glob(os.path.join(checkpoint_dir, "milestone_*.pt")),
        key=checkpoint_step_from_name,
    )
    latest = os.path.join(checkpoint_dir, "latest.pt")
    if include_latest and os.path.exists(latest):
        paths.append(latest)
    return paths


def inspect_checkpoints(paths: Sequence[str]) -> List[CheckpointInfo]:
    infos: List[CheckpointInfo] = []
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        step = int(ckpt.get("global_step", checkpoint_step_from_name(path)))
        label = os.path.splitext(os.path.basename(path))[0]
        infos.append(CheckpointInfo(path=path, label=label, step=step))
        del ckpt
    return infos


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config


def merged_checkpoint_config(checkpoint_path: str, fallback_config: dict) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(fallback_config)
    cfg.update(ckpt.get("config", {}) or {})
    del ckpt
    return cfg


def build_tcr_pool(config: dict, targets: List[str], seed: int) -> TCRPool:
    l1_dir = os.path.join(PROJECT_ROOT, "data", "l1_seeds")
    if not os.path.isdir(l1_dir):
        l1_dir = None

    pool = TCRPool(
        tcrdb_path=config.get("tcrdb_path", "/share/liuyutian/TCRPPO/data/tcrdb"),
        l1_seeds_dir=l1_dir,
        l0_mutation_range=tuple(config.get("l0_mutation_range", (3, 5))),
        l1_top_k=config.get("l1_top_k", 500),
        curriculum_schedule=config.get("curriculum_schedule"),
        seed=seed,
    )
    decoy_lib_path = config.get("decoy_library_path", "/share/liuyutian/pMHC_decoy_library")
    pool.load_l0_from_decoy_d(decoy_lib_path, targets)
    l0_tchard_dir = os.path.join(PROJECT_ROOT, "data", "l0_seeds_tchard")
    if os.path.isdir(l0_tchard_dir):
        pool.load_l0_from_dir(l0_tchard_dir)
    return pool


def precompute_initial_tcrs(
    config: dict,
    targets: List[str],
    n_tcrs: int,
    seed: int,
    initial_step: int,
    reward_mode: str,
) -> Dict[str, List[str]]:
    pool = build_tcr_pool(config, targets, seed)
    initials: Dict[str, List[str]] = {}
    for target in targets:
        initials[target] = [
            pool.sample_tcr(target, step=initial_step, reward_mode=reward_mode)[0]
            for _ in range(n_tcrs)
        ]
    return initials


def build_env(
    config: dict,
    targets: List[str],
    esm_cache: ESMCache,
    tcr_pool: TCRPool,
) -> TCREditEnv:
    pmhc_loader = PMHCLoader(
        targets=targets,
        decoy_library_path=config.get("decoy_library_path", "/share/liuyutian/pMHC_decoy_library"),
    )
    disabled_reward = RewardManager(affinity_scorer=None, reward_mode="disabled")
    env = TCREditEnv(
        esm_cache=esm_cache,
        pmhc_loader=pmhc_loader,
        tcr_pool=tcr_pool,
        reward_manager=disabled_reward,
        max_steps=int(config.get("max_steps", config.get("max_steps_per_episode", MAX_STEPS_PER_EPISODE))),
        max_tcr_len=int(config.get("max_tcr_len", MAX_TCR_LEN)),
        min_tcr_len=int(config.get("min_tcr_len", 8)),
        reward_mode=config.get("reward_mode", "v2_no_decoy_delta"),
        min_steps=int(config.get("min_steps", 0)),
        min_steps_penalty=float(config.get("min_steps_penalty", 0.0)),
        ban_stop=bool(config.get("ban_stop", False)),
        sub_only=bool(config.get("sub_only", False)),
        # During trace11_delta, terminal_reward_only keeps cumulative reward at
        # zero throughout the editable part of each episode. Keep that behavior
        # while disabling reward computation.
        terminal_reward_only=True,
    )
    return env


def load_policy(checkpoint_path: str, obs_dim: int, config: dict, device: str) -> ActorCritic:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("config", {}) or {}
    hidden_dim = int(ckpt_config.get("hidden_dim", config.get("hidden_dim", 512)))
    max_tcr_len = int(ckpt_config.get("max_tcr_len", config.get("max_tcr_len", MAX_TCR_LEN)))
    policy = ActorCritic(obs_dim=obs_dim, hidden_dim=hidden_dim, max_tcr_len=max_tcr_len).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    del ckpt
    return policy


def make_mask_tensors(mask: dict, device: str) -> Dict[str, torch.Tensor]:
    return {
        "op_mask": torch.BoolTensor(mask["op_mask"]).unsqueeze(0).to(device),
        "pos_mask": torch.BoolTensor(mask["pos_mask"]).unsqueeze(0).to(device),
    }


@torch.no_grad()
def sample_action(
    policy: ActorCritic,
    obs: np.ndarray,
    mask: dict,
    device: str,
    deterministic: bool,
) -> Tuple[int, int, int]:
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    mask_dict = make_mask_tensors(mask, device)
    if not deterministic:
        op, pos, tok, _ = policy(obs_tensor, mask_dict)
        return int(op[0]), int(pos[0]), int(tok[0])

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


def generate_tcrs_for_checkpoint(
    checkpoint: CheckpointInfo,
    config: dict,
    targets: List[str],
    initials: Dict[str, List[str]],
    esm_cache: ESMCache,
    device: str,
    deterministic: bool,
    seed: int,
) -> List[dict]:
    tcr_pool = build_tcr_pool(config, targets, seed)
    env = build_env(config, targets, esm_cache, tcr_pool)
    env.set_global_step(checkpoint.step)
    policy = load_policy(checkpoint.path, obs_dim=env.obs_dim, config=config, device=device)

    trajectories: List[dict] = []
    for target_idx, target in enumerate(targets):
        for sample_idx, init_tcr in enumerate(initials[target]):
            torch.manual_seed(seed + target_idx * 1000 + sample_idx)
            np.random.seed(seed + target_idx * 1000 + sample_idx)
            random.seed(seed + target_idx * 1000 + sample_idx)

            obs = env.reset(peptide=target, init_tcr=init_tcr)
            steps = []
            while not env.done:
                action = sample_action(policy, obs, env.get_action_mask(), device, deterministic)
                obs, reward, done, info = env.step(action)
                steps.append({
                    "action": info.get("action_name", ""),
                    "position": int(info.get("position", -1)),
                    "token": int(info.get("token", -1)),
                    "tcr": info.get("new_tcr", ""),
                })

            trajectories.append({
                "checkpoint": checkpoint.label,
                "step": checkpoint.step,
                "target": target,
                "sample_idx": sample_idx,
                "initial_tcr": init_tcr,
                "final_tcr": env.current_tcr,
                "n_steps": env.step_count,
                "steps": steps,
            })

    return trajectories


def sample_decoys(
    config: dict,
    targets: List[str],
    n_decoys: int,
    seed: int,
    decoy_plan: Optional[Dict[str, int]] = None,
    allow_c_fallback: bool = True,
) -> Dict[str, List[DecoyInfo]]:
    sampler = DecoySampler(
        decoy_library_path=config.get("decoy_library_path", "/share/liuyutian/pMHC_decoy_library"),
        targets=targets,
        seed=seed,
        decoy_a_min_count=int(config.get("decoy_a_min_count", 10)),
        decoy_d_max_count=int(config.get("decoy_d_max_count", 50)),
    )
    rng = np.random.default_rng(seed)
    selected: Dict[str, List[DecoyInfo]] = {}
    plan = decoy_plan or {}
    allowed_fill_tiers = ["A", "B", "D"] + (["C"] if allow_c_fallback else [])

    def clean_pool(tiers: dict, tier: str, target: str, chosen_peptides: set[str]) -> List[str]:
        return [
            pep
            for pep in tiers.get(tier, [])
            if pep and pep != target and pep not in chosen_peptides
        ]

    def draw_from_tier(
        tiers: dict,
        tier: str,
        count: int,
        target: str,
        chosen: List[DecoyInfo],
    ) -> int:
        if count <= 0:
            return 0
        chosen_peptides = {d.peptide for d in chosen}
        pool = clean_pool(tiers, tier, target, chosen_peptides)
        if not pool:
            return 0
        take = min(count, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        chosen.extend(DecoyInfo(pool[int(i)], tier) for i in idx)
        return take

    for target in targets:
        if target not in sampler.decoys:
            sampler._load_target_decoys(target)  # noqa: SLF001 - local eval utility
        tiers = sampler.decoys.get(target, {})
        chosen: List[DecoyInfo] = []
        if plan:
            for tier in ("A", "B", "D", "C"):
                draw_from_tier(tiers, tier, int(plan.get(tier, 0)), target, chosen)
            needed = n_decoys - len(chosen)
            if needed > 0:
                for tier in allowed_fill_tiers:
                    needed -= draw_from_tier(tiers, tier, needed, target, chosen)
                    if needed <= 0:
                        break
        else:
            hard_pool: List[DecoyInfo] = []
            for tier in ("A", "B", "D"):
                for pep in tiers.get(tier, []):
                    if pep and pep != target:
                        hard_pool.append(DecoyInfo(pep, tier))

            if hard_pool:
                idx = rng.choice(len(hard_pool), size=min(n_decoys, len(hard_pool)), replace=False)
                chosen.extend(hard_pool[int(i)] for i in idx)

            if len(chosen) < n_decoys and allow_c_fallback:
                c_pool = [pep for pep in tiers.get("C", sampler.tier_c_global) if pep and pep != target]
                if c_pool:
                    needed = n_decoys - len(chosen)
                    replace = len(c_pool) < needed
                    idx = rng.choice(len(c_pool), size=needed, replace=replace)
                    chosen.extend(DecoyInfo(c_pool[int(i)], "C") for i in idx)

        # Dedupe while preserving order.
        seen = set()
        deduped = []
        for decoy in chosen:
            if decoy.peptide in seen:
                continue
            seen.add(decoy.peptide)
            deduped.append(decoy)
        selected[target] = deduped[:n_decoys]

    return selected


def summarize_decoy_selection(decoys_by_target: Dict[str, List[DecoyInfo]]) -> Dict[str, object]:
    per_target = {}
    aggregate = {"A": 0, "B": 0, "C": 0, "D": 0}
    n_short = 0
    for target, decoys in decoys_by_target.items():
        counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for decoy in decoys:
            counts[decoy.tier] = counts.get(decoy.tier, 0) + 1
            aggregate[decoy.tier] = aggregate.get(decoy.tier, 0) + 1
        per_target[target] = {
            "n_decoys": len(decoys),
            "tier_counts": counts,
        }
    if per_target:
        expected = max(v["n_decoys"] for v in per_target.values())
        n_short = sum(1 for v in per_target.values() if v["n_decoys"] < expected)
    return {
        "aggregate_tier_counts": aggregate,
        "per_target": per_target,
        "n_targets_with_fewer_decoys_than_max": n_short,
    }


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


def evaluate_trajectories(
    trajectories: List[dict],
    decoys_by_target: Dict[str, List[DecoyInfo]],
    scorer: AffinityTFoldScorer,
    score_batch_size: int,
    topk_values: Sequence[int],
) -> Tuple[List[dict], List[dict]]:
    pairs: List[Tuple[str, str]] = []
    for traj in trajectories:
        tcr = traj["final_tcr"]
        target = traj["target"]
        pairs.append((tcr, target))
        for decoy in decoys_by_target.get(target, []):
            pairs.append((tcr, decoy.peptide))

    score_map = score_pairs(scorer, pairs, score_batch_size)

    tcr_rows: List[dict] = []
    pair_rows: List[dict] = []
    for traj in trajectories:
        target = traj["target"]
        tcr = traj["final_tcr"]
        target_score = score_map[(tcr, target)]
        decoy_infos = decoys_by_target.get(target, [])
        decoy_scores = [score_map[(tcr, d.peptide)] for d in decoy_infos]
        labels = np.array([1] + [0] * len(decoy_scores))
        vals = np.array([target_score] + decoy_scores)
        auroc = float(roc_auc_score(labels, vals)) if decoy_scores else float("nan")
        margin = target_score - float(np.mean(decoy_scores)) if decoy_scores else float("nan")
        sorted_decoys_desc = sorted(decoy_scores, reverse=True)
        topk_metrics = {}
        for k in topk_values:
            k_eff = min(int(k), len(sorted_decoys_desc))
            if k_eff <= 0:
                topk_mean = float("nan")
            else:
                topk_mean = float(np.mean(sorted_decoys_desc[:k_eff]))
            topk_metrics[f"decoy_top{k}_mean_reward"] = topk_mean
            topk_metrics[f"margin_target_minus_decoy_top{k}"] = target_score - topk_mean
        target_rank = 1 + int(np.sum(np.array(decoy_scores) > target_score))

        tcr_row = {
            **{k: traj[k] for k in ("checkpoint", "step", "target", "sample_idx", "initial_tcr", "final_tcr", "n_steps")},
            "target_reward": target_score,
            "target_prob": sigmoid(target_score),
            "mean_decoy_reward": mean_or_nan(decoy_scores),
            "mean_decoy_prob": mean_or_nan([sigmoid(x) for x in decoy_scores]),
            "max_decoy_reward": float(np.max(decoy_scores)) if decoy_scores else float("nan"),
            "min_decoy_reward": float(np.min(decoy_scores)) if decoy_scores else float("nan"),
            "margin_target_minus_decoy": margin,
            "margin_target_minus_decoy_max": target_score - float(np.max(decoy_scores)) if decoy_scores else float("nan"),
            "auroc": auroc,
            "target_rank_among_decoys": target_rank,
            "n_decoys": len(decoy_scores),
            "decoy_tiers": ",".join(d.tier for d in decoy_infos),
        }
        tcr_row.update(topk_metrics)
        tcr_rows.append(tcr_row)

        pair_rows.append({
            "checkpoint": traj["checkpoint"],
            "step": traj["step"],
            "target": target,
            "sample_idx": traj["sample_idx"],
            "tcr": tcr,
            "scored_peptide": target,
            "tier": "target",
            "is_target": 1,
            "reward": target_score,
            "prob": sigmoid(target_score),
        })
        for decoy, score in zip(decoy_infos, decoy_scores):
            pair_rows.append({
                "checkpoint": traj["checkpoint"],
                "step": traj["step"],
                "target": target,
                "sample_idx": traj["sample_idx"],
                "tcr": tcr,
                "scored_peptide": decoy.peptide,
                "tier": decoy.tier,
                "is_target": 0,
                "reward": score,
                "prob": sigmoid(score),
            })

    return tcr_rows, pair_rows


def aggregate(rows: List[dict], group_keys: Sequence[str]) -> List[dict]:
    groups: Dict[Tuple[object, ...], List[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    out: List[dict] = []
    for key, vals in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec.update({
            "n_tcrs": len(vals),
            "n_unique_tcrs": len({v["final_tcr"] for v in vals}),
            "mean_target_reward": mean_or_nan([v["target_reward"] for v in vals]),
            "mean_target_prob": mean_or_nan([v["target_prob"] for v in vals]),
            "mean_decoy_reward": mean_or_nan([v["mean_decoy_reward"] for v in vals]),
            "mean_decoy_prob": mean_or_nan([v["mean_decoy_prob"] for v in vals]),
            "mean_decoy_max_reward": mean_or_nan([v["max_decoy_reward"] for v in vals]),
            "mean_margin": mean_or_nan([v["margin_target_minus_decoy"] for v in vals]),
            "mean_margin_vs_decoy_max": mean_or_nan([v["margin_target_minus_decoy_max"] for v in vals]),
            "mean_auroc": mean_or_nan([v["auroc"] for v in vals]),
            "median_auroc": (
                float(np.median([v["auroc"] for v in vals if np.isfinite(float(v["auroc"]))]))
                if any(np.isfinite(float(v["auroc"])) for v in vals)
                else float("nan")
            ),
            "mean_steps": mean_or_nan([float(v["n_steps"]) for v in vals]),
        })
        topk_reward_keys = sorted(
            {k for row in vals for k in row if k.startswith("decoy_top") and k.endswith("_mean_reward")},
            key=lambda name: int(re.search(r"decoy_top(\d+)_", name).group(1)),
        )
        for reward_key in topk_reward_keys:
            k = re.search(r"decoy_top(\d+)_", reward_key).group(1)
            margin_key = f"margin_target_minus_decoy_top{k}"
            rec[f"mean_{reward_key}"] = mean_or_nan([v[reward_key] for v in vals])
            rec[f"mean_{margin_key}"] = mean_or_nan([v[margin_key] for v in vals])
        out.append(rec)
    return out


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float, ndigits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{float(x):.{ndigits}f}"


def write_report(
    path: str,
    run_name: str,
    checkpoint_infos: List[CheckpointInfo],
    checkpoint_summary: List[dict],
    target_summary: List[dict],
    args: argparse.Namespace,
) -> None:
    by_label = {row["checkpoint"]: row for row in checkpoint_summary}
    ordered = [by_label[ck.label] for ck in checkpoint_infos if ck.label in by_label]
    best_margin = max(ordered, key=lambda r: r["mean_margin"]) if ordered else None
    best_auroc = max(ordered, key=lambda r: r["mean_auroc"]) if ordered else None

    lines = [
        f"# tFold Decoy Reward Evaluation: {run_name}",
        "",
        "Higher target reward is better. Lower decoy reward is better. Rewards are raw tFold V3.4 binding logits, matching the training scorer convention used in the log.",
        "",
        "## Setup",
        f"- Checkpoints evaluated: {len(checkpoint_infos)}",
        f"- Targets: {args.targets_file}",
        f"- TCRs per target per checkpoint: {args.n_tcrs}",
        f"- Decoys per target: {args.n_decoys}",
        f"- Decoy mode: {args.decoy_mode}",
        f"- Requested decoy plan: {args.decoy_plan_resolved}",
        f"- Decoy source: pMHC_decoy_library tiers A/B/D"
        f"{' with tier C fallback' if args.allow_decoy_c_fallback else ' with no tier C fallback'}",
        f"- tFold server socket: {args.tfold_server_socket}",
        f"- tFold cache: {args.tfold_cache_path}",
        "",
    ]
    if getattr(args, "decoy_selection_summary", None):
        summary = args.decoy_selection_summary
        lines.append(
            "- Actual decoy tier counts: "
            + ", ".join(
                f"{tier}={summary['aggregate_tier_counts'].get(tier, 0)}"
                for tier in ("A", "B", "D", "C")
            )
        )
        n_short = int(summary.get("n_targets_with_fewer_decoys_than_max", 0))
        if n_short:
            lines.append(f"- Warning: {n_short} targets had fewer decoys than the maximum selected count.")
        lines.append("")
    if best_margin:
        lines.append(
            f"Best mean target-minus-decoy margin: **{best_margin['checkpoint']}** "
            f"(step {best_margin['step']}, margin {fmt(best_margin['mean_margin'])}, "
            f"target {fmt(best_margin['mean_target_reward'])}, decoy {fmt(best_margin['mean_decoy_reward'])})."
        )
    if best_auroc:
        lines.append(
            f"Best mean AUROC: **{best_auroc['checkpoint']}** "
            f"(step {best_auroc['step']}, AUROC {fmt(best_auroc['mean_auroc'])})."
        )
    lines.extend(["", "## Checkpoint Summary", ""])
    lines.append("| checkpoint | step | target reward | decoy reward | margin | AUROC | unique TCRs |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in ordered:
        lines.append(
            f"| {row['checkpoint']} | {row['step']} | {fmt(row['mean_target_reward'])} | "
            f"{fmt(row['mean_decoy_reward'])} | {fmt(row['mean_margin'])} | "
            f"{fmt(row['mean_auroc'])} | {row['n_unique_tcrs']} |"
        )

    lines.extend(["", "## Hard-Decoy Summary", ""])
    topk_cols = sorted(
        [k for k in ordered[0] if k.startswith("mean_decoy_top") and k.endswith("_mean_reward")] if ordered else [],
        key=lambda name: int(re.search(r"top(\d+)", name).group(1)),
    )
    header = "| checkpoint | step | decoy mean | decoy max | margin vs max |"
    sep = "|---|---:|---:|---:|---:|"
    for col in topk_cols:
        k = re.search(r"top(\d+)", col).group(1)
        header += f" decoy top{k} | margin vs top{k} |"
        sep += "---:|---:|"
    lines.append(header)
    lines.append(sep)
    for row in ordered:
        line = (
            f"| {row['checkpoint']} | {row['step']} | {fmt(row['mean_decoy_reward'])} | "
            f"{fmt(row['mean_decoy_max_reward'])} | {fmt(row['mean_margin_vs_decoy_max'])} |"
        )
        for col in topk_cols:
            k = re.search(r"top(\d+)", col).group(1)
            line += f" {fmt(row[col])} | {fmt(row[f'mean_margin_target_minus_decoy_top{k}'])} |"
        lines.append(line)

    lines.extend(["", "## Per-Target Snapshot At Best-Margin Checkpoint", ""])
    if best_margin:
        label = best_margin["checkpoint"]
        subset = [r for r in target_summary if r["checkpoint"] == label]
        subset = sorted(subset, key=lambda r: r["mean_margin"])
        lines.append(f"Checkpoint: **{label}**")
        lines.append("")
        lines.append("| target | target reward | decoy mean | decoy max | margin mean | margin max | AUROC |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in subset:
            lines.append(
                f"| {row['target']} | {fmt(row['mean_target_reward'])} | "
                f"{fmt(row['mean_decoy_reward'])} | {fmt(row['mean_decoy_max_reward'])} | "
                f"{fmt(row['mean_margin'])} | {fmt(row['mean_margin_vs_decoy_max'])} | "
                f"{fmt(row['mean_auroc'])} |"
            )

    lines.extend([
        "",
        "## Output Files",
        "- `summary_by_checkpoint.csv`: aggregate trend over training steps",
        "- `summary_by_target.csv`: per-target aggregate for each checkpoint",
        "- `tcr_level_results.csv`: generated TCR-level target/decoy statistics",
        "- `pair_scores.csv`: raw tFold score for every target or decoy pair",
        "- `generated_trajectories.json`: generation traces",
        "",
    ])
    with open(path, "w") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config", default="configs/test51c.yaml")
    parser.add_argument("--targets-file", default="data/tfold_excellent_peptides.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Explicit checkpoint paths. Default: all milestones plus latest.pt")
    parser.add_argument("--no-latest", action="store_true", help="Do not include latest.pt when auto-discovering checkpoints")
    parser.add_argument("--n-tcrs", type=int, default=1)
    parser.add_argument("--n-decoys", type=int, default=3)
    parser.add_argument(
        "--decoy-mode",
        choices=["auto", "1", "4", "16", "a1", "abbd4", "a4b8d4", "custom"],
        default="auto",
        help=(
            "Named decoy plan. 1/a1=A; 4/abbd4=A+B+B+D; "
            "16/a4b8d4=A*4+B*8+D*4; auto preserves legacy random A/B/D sampling."
        ),
    )
    parser.add_argument(
        "--decoy-plan",
        default="",
        help="Custom tier plan like A:1,B:2,D:1. Implies --decoy-mode custom.",
    )
    parser.add_argument(
        "--allow-decoy-c-fallback",
        action="store_true",
        help="Allow tier C only if the requested A/B/D plan cannot be filled.",
    )
    parser.add_argument(
        "--decoy-top-k",
        type=int,
        nargs="+",
        default=[3, 5],
        help="Hard-decoy top-k means to report. Uses highest tFold logits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-step", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy policy actions instead of categorical sampling")
    parser.add_argument("--esm-cache-path", default=None, help="Separate ESM cache for evaluation generation")
    parser.add_argument("--tfold-cache-path", default="data/tfold_feature_cache.db")
    parser.add_argument("--tfold-server-socket", default="/tmp/tfold_server_trace11_delta.sock")
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    config["tfold_cache_path"] = args.tfold_cache_path
    config["tfold_server_socket"] = args.tfold_server_socket
    targets = load_targets(args.targets_file)

    checkpoint_paths = args.checkpoints or discover_checkpoints(args.checkpoint_dir, include_latest=not args.no_latest)
    checkpoint_infos = inspect_checkpoints(checkpoint_paths)
    checkpoint_infos.sort(key=lambda ck: (ck.step, ck.label))

    print(f"Evaluating {len(checkpoint_infos)} checkpoints x {len(targets)} targets")
    print(f"Output: {args.output_dir}")
    print(f"tFold socket: {args.tfold_server_socket}")

    first_cfg = merged_checkpoint_config(checkpoint_infos[0].path, config)
    reward_mode = first_cfg.get("reward_mode", "v2_no_decoy_delta")
    initials = precompute_initial_tcrs(
        first_cfg,
        targets,
        n_tcrs=args.n_tcrs,
        seed=args.seed,
        initial_step=args.initial_step,
        reward_mode=reward_mode,
    )
    if args.decoy_plan:
        requested_plan = parse_decoy_plan(args.decoy_plan)
        args.decoy_mode = "custom"
    else:
        requested_plan = decoy_plan_from_mode(args.decoy_mode, args.n_decoys)
    if requested_plan:
        args.n_decoys = sum(requested_plan.values())
    args.decoy_plan_resolved = format_decoy_plan(requested_plan)

    decoys_by_target = sample_decoys(
        first_cfg,
        targets,
        n_decoys=args.n_decoys,
        seed=args.seed,
        decoy_plan=requested_plan,
        allow_c_fallback=args.allow_decoy_c_fallback,
    )
    args.decoy_selection_summary = summarize_decoy_selection(decoys_by_target)

    if args.esm_cache_path is None:
        args.esm_cache_path = os.path.join(args.output_dir, "esm_cache_eval.db")
    esm_cache = ESMCache(
        device=args.device,
        tcr_cache_size=int(first_cfg.get("esm_tcr_cache_size", 4096)),
        disk_cache_path=args.esm_cache_path,
    )

    all_trajectories: List[dict] = []
    for ck in checkpoint_infos:
        ck_cfg = merged_checkpoint_config(ck.path, config)
        print(f"\n[generate] {ck.label} step={ck.step}", flush=True)
        trajectories = generate_tcrs_for_checkpoint(
            checkpoint=ck,
            config=ck_cfg,
            targets=targets,
            initials=initials,
            esm_cache=esm_cache,
            device=args.device,
            deterministic=args.deterministic,
            seed=args.seed,
        )
        all_trajectories.extend(trajectories)

    with open(os.path.join(args.output_dir, "generated_trajectories.json"), "w") as f:
        json.dump(all_trajectories, f, indent=2)

    print("\n[score] Loading tFold scorer", flush=True)
    scorer = AffinityTFoldScorer(
        device=args.device,
        cache_path=args.tfold_cache_path,
        server_socket_path=args.tfold_server_socket,
        max_subprocess_batch=args.extract_batch_size,
        use_cache=True,
    )
    t0 = time.time()
    tcr_rows, pair_rows = evaluate_trajectories(
        all_trajectories,
        decoys_by_target,
        scorer,
        score_batch_size=args.score_batch_size,
        topk_values=args.decoy_top_k,
    )
    elapsed = time.time() - t0
    print(f"[score] completed in {elapsed:.1f}s", flush=True)

    checkpoint_summary = aggregate(tcr_rows, ["checkpoint", "step"])
    checkpoint_summary.sort(key=lambda r: int(r["step"]))
    target_summary = aggregate(tcr_rows, ["checkpoint", "step", "target"])
    target_summary.sort(key=lambda r: (int(r["step"]), str(r["target"])))

    write_csv(os.path.join(args.output_dir, "tcr_level_results.csv"), tcr_rows)
    write_csv(os.path.join(args.output_dir, "pair_scores.csv"), pair_rows)
    write_csv(os.path.join(args.output_dir, "summary_by_checkpoint.csv"), checkpoint_summary)
    write_csv(os.path.join(args.output_dir, "summary_by_target.csv"), target_summary)

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "checkpoints": [ck.__dict__ for ck in checkpoint_infos],
        "targets": targets,
        "decoys_by_target": {
            target: [decoy.__dict__ for decoy in decoys]
            for target, decoys in decoys_by_target.items()
        },
        "tfold_cache_stats": scorer.cache_stats,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    run_name = os.path.basename(os.path.dirname(args.checkpoint_dir.rstrip("/")))
    report_path = os.path.join(args.output_dir, "REPORT.md")
    write_report(report_path, run_name, checkpoint_infos, checkpoint_summary, target_summary, args)

    print("\nSummary:")
    for row in checkpoint_summary:
        print(
            f"  {row['checkpoint']:>18} step={row['step']:>7} "
            f"target={fmt(row['mean_target_reward'])} decoy={fmt(row['mean_decoy_reward'])} "
            f"margin={fmt(row['mean_margin'])} auroc={fmt(row['mean_auroc'])}"
        )
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
