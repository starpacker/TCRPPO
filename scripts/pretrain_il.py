#!/usr/bin/env python3
"""Behavior-cloning pretraining for TCRPPO edit policies."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tcrppo_v2.data.pmhc_loader import PMHCLoader
from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.utils.constants import MAX_TCR_LEN, MIN_TCR_LEN, NUM_OPS, OP_DEL, OP_INS, OP_STOP
from tcrppo_v2.utils.esm_cache import ESMCache


def load_config(path: str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def load_rows(path: Path) -> List[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_action_masks(
    rows: List[dict],
    *,
    max_tcr_len: int,
    min_tcr_len: int,
    ban_stop: bool,
    min_steps: int,
    sub_only: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    op_masks = np.ones((len(rows), NUM_OPS), dtype=bool)
    pos_masks = np.zeros((len(rows), max_tcr_len), dtype=bool)
    valid = np.ones(len(rows), dtype=bool)

    for i, row in enumerate(rows):
        seq_len = len(row["tcr"])
        step_idx = int(row.get("step_idx", 0))
        op = int(row["op"])
        pos = int(row["pos"])

        if seq_len >= max_tcr_len:
            op_masks[i, OP_INS] = False
        if seq_len <= min_tcr_len:
            op_masks[i, OP_DEL] = False
        if sub_only:
            op_masks[i, OP_INS] = False
            op_masks[i, OP_DEL] = False
        if step_idx == 0 or step_idx < min_steps or ban_stop:
            op_masks[i, OP_STOP] = False

        if seq_len <= 1:
            pos_masks[i, :seq_len] = True
        else:
            pos_masks[i, 1:seq_len] = True

        if op != OP_STOP and (pos < 0 or pos >= max_tcr_len or not pos_masks[i, pos]):
            valid[i] = False
        if not op_masks[i, op]:
            valid[i] = False

    return op_masks, pos_masks, valid


def build_obs_batch(
    rows: List[dict],
    esm_cache: ESMCache,
    pmhc_loader: PMHCLoader,
    *,
    max_steps: int,
    include_state_scalars: bool,
) -> torch.Tensor:
    tcrs = [row["tcr"] for row in rows]
    peptides = [row["peptide"] for row in rows]
    tcr_embs = esm_cache.encode_tcr_batch(tcrs)

    pmhc_embs = []
    for peptide in peptides:
        pmhc_embs.append(esm_cache.encode_pmhc(pmhc_loader.get_pmhc_string(peptide)))
    parts = [tcr_embs, torch.stack(pmhc_embs, dim=0)]

    if include_state_scalars:
        scalars = []
        for row in rows:
            step_idx = int(row.get("step_idx", 0))
            remaining = max(0.0, (max_steps - step_idx) / max_steps)
            scalars.append([remaining, 0.0])
        parts.append(torch.tensor(scalars, device=tcr_embs.device, dtype=tcr_embs.dtype))
    return torch.cat(parts, dim=-1)


def iterate_minibatches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


def save_checkpoint(
    *,
    path: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    global_step: int,
    il_metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "global_step": int(global_step),
            "il_metadata": il_metadata,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/trace62_multi_gates.yaml")
    parser.add_argument("--dataset", default="data/il/trace29_trace61_tchard_il.jsonl")
    parser.add_argument(
        "--base-checkpoint",
        default="output/test62_simple_target_gated_decoy_trace29_simple_target_gated_decoy/checkpoints/milestone_580000.pt",
        help="Base checkpoint to load. Set to empty string '' to train from scratch.",
    )
    parser.add_argument("--out", default="output/il_pretrain_trace29_trace61_tchard/checkpoints/latest.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--esm-cache-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0, help="Debug limit; 0 means all rows.")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.0, help="Validation split ratio (e.g., 0.1 for 10%)")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 disables early stopping)")
    parser.add_argument("--from-scratch", action="store_true", help="Train from random initialization (ignore base checkpoint)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(Path(args.dataset))
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    max_tcr_len = int(config.get("max_tcr_len", MAX_TCR_LEN))
    min_tcr_len = int(config.get("min_tcr_len", MIN_TCR_LEN))
    max_steps = int(config.get("max_steps", config.get("max_steps_per_episode", 8)))
    include_state_scalars = bool(config.get("include_state_scalars", True))

    op_masks, pos_masks, valid = make_action_masks(
        rows,
        max_tcr_len=max_tcr_len,
        min_tcr_len=min_tcr_len,
        ban_stop=bool(config.get("ban_stop", False)),
        min_steps=int(config.get("min_steps", 0)),
        sub_only=bool(config.get("sub_only", False)),
    )
    if not valid.all():
        bad = int((~valid).sum())
        rows = [row for row, keep in zip(rows, valid) if keep]
        op_masks = op_masks[valid]
        pos_masks = pos_masks[valid]
        print(f"Filtered {bad} invalid action rows; remaining={len(rows)}", flush=True)

    # Train/val split
    train_indices = np.arange(len(rows))
    val_indices = np.array([], dtype=np.int64)
    if args.val_split > 0:
        n_val = int(len(rows) * args.val_split)
        shuffled = np.arange(len(rows))
        rng.shuffle(shuffled)
        val_indices = shuffled[:n_val]
        train_indices = shuffled[n_val:]
        print(f"Train/val split: {len(train_indices)} train, {len(val_indices)} val", flush=True)

    targets_cfg = config.get("train_targets", "data/tfold_excellent_peptides.txt")
    if os.path.isfile(targets_cfg):
        targets = [line.strip() for line in Path(targets_cfg).read_text().splitlines() if line.strip()]
    else:
        targets = sorted({row["peptide"] for row in rows})
    pmhc_loader = PMHCLoader(targets=targets)

    esm_cache = ESMCache(
        device=device,
        tcr_cache_size=int(config.get("esm_tcr_cache_size", 4096)),
        disk_cache_path=args.esm_cache_path or config.get("esm_cache_path"),
    )
    obs_dim = esm_cache.output_dim * 2 + (2 if include_state_scalars else 0)
    policy = ActorCritic(
        obs_dim=obs_dim,
        hidden_dim=int(config.get("hidden_dim", 512)),
        max_tcr_len=max_tcr_len,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    base_step = 0
    if args.from_scratch:
        print("Training from scratch (random initialization)", flush=True)
    elif args.base_checkpoint and args.base_checkpoint.strip():
        ckpt = torch.load(args.base_checkpoint, map_location=device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
            except Exception as exc:
                print(f"Warning: could not load optimizer state: {exc}", flush=True)
        base_step = int(ckpt.get("global_step", 0))
        print(f"Loaded base checkpoint {args.base_checkpoint} at step {base_step}", flush=True)
    else:
        print("No base checkpoint specified, training from scratch", flush=True)

    out_path = Path(args.out)
    metadata = {
        "dataset": args.dataset,
        "base_checkpoint": args.base_checkpoint if not args.from_scratch else None,
        "from_scratch": args.from_scratch,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "n_rows": len(rows),
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "val_split": args.val_split,
        "patience": args.patience,
    }

    # Early stopping state
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    policy.train()
    for epoch in range(1, args.epochs + 1):
        # Training phase
        losses = []
        n_seen = 0
        for batch_idx in iterate_minibatches(train_indices, args.batch_size, rng):
            batch_rows = [rows[i] for i in batch_idx]
            obs = build_obs_batch(
                batch_rows,
                esm_cache,
                pmhc_loader,
                max_steps=max_steps,
                include_state_scalars=include_state_scalars,
            )
            obs = torch.nan_to_num(obs.float(), nan=0.0, posinf=1.0, neginf=-1.0)
            actions = (
                torch.tensor([int(row["op"]) for row in batch_rows], device=device, dtype=torch.long),
                torch.tensor([int(row["pos"]) for row in batch_rows], device=device, dtype=torch.long),
                torch.tensor([int(row["tok"]) for row in batch_rows], device=device, dtype=torch.long),
            )
            masks = {
                "op_mask": torch.tensor(op_masks[batch_idx], device=device, dtype=torch.bool),
                "pos_mask": torch.tensor(pos_masks[batch_idx], device=device, dtype=torch.bool),
            }
            weights = torch.tensor(
                [float(row.get("weight", 1.0)) for row in batch_rows],
                device=device,
                dtype=torch.float32,
            )

            log_probs, entropy, _values, _ = policy(obs, masks, actions=actions)
            loss = -(log_probs * weights).sum() / weights.sum().clamp_min(1e-6)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config.get("max_grad_norm", 0.5)))
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            n_seen += len(batch_idx)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        
        # Validation phase
        val_loss = None
        if len(val_indices) > 0:
            policy.eval()
            val_losses = []
            with torch.no_grad():
                for batch_idx in iterate_minibatches(val_indices, args.batch_size, rng):
                    batch_rows = [rows[i] for i in batch_idx]
                    obs = build_obs_batch(
                        batch_rows,
                        esm_cache,
                        pmhc_loader,
                        max_steps=max_steps,
                        include_state_scalars=include_state_scalars,
                    )
                    obs = torch.nan_to_num(obs.float(), nan=0.0, posinf=1.0, neginf=-1.0)
                    actions = (
                        torch.tensor([int(row["op"]) for row in batch_rows], device=device, dtype=torch.long),
                        torch.tensor([int(row["pos"]) for row in batch_rows], device=device, dtype=torch.long),
                        torch.tensor([int(row["tok"]) for row in batch_rows], device=device, dtype=torch.long),
                    )
                    masks = {
                        "op_mask": torch.tensor(op_masks[batch_idx], device=device, dtype=torch.bool),
                        "pos_mask": torch.tensor(pos_masks[batch_idx], device=device, dtype=torch.bool),
                    }
                    weights = torch.tensor(
                        [float(row.get("weight", 1.0)) for row in batch_rows],
                        device=device,
                        dtype=torch.float32,
                    )
                    log_probs, entropy, _values, _ = policy(obs, masks, actions=actions)
                    loss = -(log_probs * weights).sum() / weights.sum().clamp_min(1e-6)
                    val_losses.append(float(loss.cpu()))
            val_loss = float(np.mean(val_losses)) if val_losses else None
            policy.train()
        
        # Logging
        log_str = f"Epoch {epoch}/{args.epochs}: train_loss={mean_loss:.4f} rows={n_seen}"
        if val_loss is not None:
            log_str += f" | val_loss={val_loss:.4f}"
        print(log_str, flush=True)
        
        # Early stopping check
        if args.patience > 0 and val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best model
                save_checkpoint(
                    path=out_path.with_name("best.pt"),
                    policy=policy,
                    optimizer=optimizer,
                    config=config,
                    global_step=base_step,
                    il_metadata={**metadata, "epoch": epoch, "train_loss": mean_loss, "val_loss": val_loss, "best_epoch": True},
                )
                print(f"  → New best val_loss: {val_loss:.4f} (saved to best.pt)", flush=True)
            else:
                patience_counter += 1
                print(f"  → Val loss did not improve (patience: {patience_counter}/{args.patience})", flush=True)
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}", flush=True)
                    break
        
        if args.save_every_epoch:
            save_checkpoint(
                path=out_path.with_name(f"epoch_{epoch}.pt"),
                policy=policy,
                optimizer=optimizer,
                config=config,
                global_step=base_step,
                il_metadata={**metadata, "epoch": epoch, "train_loss": mean_loss, "val_loss": val_loss},
            )

    # Save final checkpoint
    final_metadata = {**metadata, "final_train_loss": mean_loss}
    if val_loss is not None:
        final_metadata["final_val_loss"] = val_loss
    if args.patience > 0 and len(val_indices) > 0:
        final_metadata["best_epoch"] = best_epoch
        final_metadata["best_val_loss"] = best_val_loss
    
    save_checkpoint(
        path=out_path,
        policy=policy,
        optimizer=optimizer,
        config=config,
        global_step=base_step,
        il_metadata=final_metadata,
    )
    print(f"Saved IL-pretrained checkpoint to {out_path}", flush=True)
    
    if args.patience > 0 and len(val_indices) > 0:
        print(f"Best model (epoch {best_epoch}, val_loss={best_val_loss:.4f}) saved to {out_path.with_name('best.pt')}", flush=True)


if __name__ == "__main__":
    main()
