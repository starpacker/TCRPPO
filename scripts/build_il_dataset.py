#!/usr/bin/env python3
"""Build imitation-learning demonstrations for TCRPPO edit policies.

The dataset is intentionally action-level JSONL, not pre-embedded tensors.  This
keeps it portable across policy variants and lets BC pretraining reuse the same
ESM observation path as PPO.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tcrppo_v2.utils.constants import (
    AA_TO_IDX,
    AMINO_ACIDS,
    MAX_STEPS_PER_EPISODE,
    MAX_TCR_LEN,
    MIN_TCR_LEN,
    OP_DEL,
    OP_INS,
    OP_SUB,
)


TRACE_SCORE_RE = re.compile(
    r"affinity_logit=([-\d.]+).*?cdr3b=([A-Z]+)\s+peptide=([A-Z]+)"
)
TRACE_EPISODE_RE = re.compile(r"^Episode\s+(\d+)\s+\|\s+Step\s+(\d+)\s+\|\s+R=([-\d.]+)")
TRACE_FIELD_RE = re.compile(r"(\w+)=([-\d.]+)")


@dataclass(frozen=True)
class ScoredTCR:
    peptide: str
    tcr: str
    affinity: Optional[float]
    source: str
    episode: Optional[int] = None
    step: Optional[int] = None
    init_tcr: Optional[str] = None
    init_affinity: Optional[float] = None
    reward: Optional[float] = None
    decoy_violation: Optional[float] = None
    decoy_affinity: Optional[float] = None


def is_valid_tcr(seq: object, min_len: int = MIN_TCR_LEN, max_len: int = MAX_TCR_LEN) -> bool:
    if not isinstance(seq, str):
        return False
    seq = seq.strip().upper()
    return min_len <= len(seq) <= max_len and seq.startswith("C") and all(aa in AA_TO_IDX for aa in seq)


def apply_action(seq: str, action: Tuple[int, int, int]) -> str:
    op, pos, tok = action
    aa = AMINO_ACIDS[tok % len(AMINO_ACIDS)]
    if op == OP_SUB:
        if pos <= 0 or pos >= len(seq):
            return seq
        chars = list(seq)
        chars[pos] = aa
        return "".join(chars)
    if op == OP_INS:
        if len(seq) >= MAX_TCR_LEN:
            return seq
        pos = min(max(pos, 1), len(seq))
        return seq[:pos] + aa + seq[pos:]
    if op == OP_DEL:
        if len(seq) <= MIN_TCR_LEN or pos <= 0 or pos >= len(seq):
            return seq
        return seq[:pos] + seq[pos + 1 :]
    raise ValueError(f"Unsupported edit op {op}")


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] if ca == cb else 1 + min(prev[j], cur[-1], prev[j - 1]))
        prev = cur
    return prev[-1]


def candidate_actions(seq: str, target: str) -> Iterable[Tuple[int, int, int]]:
    """Generate useful one-edit actions for moving seq toward target."""
    seq_len = len(seq)
    # Keep this intentionally narrow: thousands of paths are built per run, so
    # exhaustive 20AA substitution search is unnecessary and slow.
    for pos in range(1, seq_len):
        if pos < len(target) and seq[pos] != target[pos]:
            yield (OP_SUB, pos, AA_TO_IDX[target[pos]])

    if seq_len < MAX_TCR_LEN:
        for pos in range(1, seq_len + 1):
            if pos < len(target):
                yield (OP_INS, pos, AA_TO_IDX[target[pos]])
            if pos - 1 < len(target):
                yield (OP_INS, pos, AA_TO_IDX[target[pos - 1]])

    if seq_len > MIN_TCR_LEN:
        for pos in range(1, seq_len):
            yield (OP_DEL, pos, 0)


def greedy_edit_path(source: str, target: str, max_edits: int) -> Optional[List[Tuple[str, Tuple[int, int, int], str]]]:
    """Create a valid edit path from source to target within max_edits.

    The policy action space protects position 0, so paths that require changing
    the leading residue are rejected by construction.
    """
    source = source.strip().upper()
    target = target.strip().upper()
    if source == target:
        return []
    if not (is_valid_tcr(source) and is_valid_tcr(target)):
        return None
    if source[0] != target[0]:
        return None

    current = source
    path: List[Tuple[str, Tuple[int, int, int], str]] = []
    last_dist = levenshtein(current, target)
    for _ in range(max_edits):
        best = None
        best_key = None
        seen = set()
        for action in candidate_actions(current, target):
            if action in seen:
                continue
            seen.add(action)
            nxt = apply_action(current, action)
            if nxt == current or not is_valid_tcr(nxt):
                continue
            dist = levenshtein(nxt, target)
            op_priority = {OP_SUB: 0, OP_INS: 1, OP_DEL: 2}.get(action[0], 9)
            key = (dist, op_priority, action[1], action[2])
            if best_key is None or key < best_key:
                best_key = key
                best = (action, nxt)
        if best is None:
            return None
        action, nxt = best
        if best_key is None or best_key[0] > last_dist:
            return None
        path.append((current, action, nxt))
        current = nxt
        last_dist = best_key[0]
        if current == target:
            return path
    return path if current == target else None


def random_corrupt_path(
    seq: str,
    rng: random.Random,
    n_edits: int,
) -> Optional[Tuple[str, List[Tuple[str, Tuple[int, int, int], str]]]]:
    """Randomly damage seq and return the exact reverse repair path."""
    current = seq
    forward: List[Tuple[str, Tuple[int, int, int], str, Tuple[int, int, int]]] = []
    for _ in range(n_edits):
        valid_ops = [OP_SUB]
        if len(current) < MAX_TCR_LEN:
            valid_ops.append(OP_INS)
        if len(current) > MIN_TCR_LEN:
            valid_ops.append(OP_DEL)
        op = rng.choice(valid_ops)
        if op == OP_SUB:
            pos = rng.randrange(1, len(current))
            old = current[pos]
            aa = rng.choice([x for x in AMINO_ACIDS if x != old])
            action = (OP_SUB, pos, AA_TO_IDX[aa])
            inverse = (OP_SUB, pos, AA_TO_IDX[old])
        elif op == OP_INS:
            pos = rng.randrange(1, len(current) + 1)
            action = (OP_INS, pos, AA_TO_IDX[rng.choice(AMINO_ACIDS)])
            inverse = (OP_DEL, pos, 0)
        else:
            pos = rng.randrange(1, len(current))
            old = current[pos]
            action = (OP_DEL, pos, 0)
            inverse = (OP_INS, pos, AA_TO_IDX[old])
        nxt = apply_action(current, action)
        if not is_valid_tcr(nxt):
            return None
        forward.append((current, action, nxt, inverse))
        current = nxt
    if current == seq:
        return None

    reverse_path = []
    repair_current = current
    for before, _action, after, inverse in reversed(forward):
        if repair_current != after:
            return None
        repaired = apply_action(repair_current, inverse)
        if repaired != before:
            return None
        reverse_path.append((repair_current, inverse, repaired))
        repair_current = repaired
    return current, reverse_path


def find_window(scores: Sequence[dict], values: Sequence[float], end_before: Optional[int] = None) -> Optional[int]:
    rounded_values = [round(v, 4) for v in values]
    rounded_scores = [round(s["affinity"], 4) for s in scores]
    limit = len(scores) - len(values) + 1
    if end_before is not None:
        limit = min(limit, end_before - len(values) + 1)
    match = None
    for i in range(max(0, limit)):
        if rounded_scores[i : i + len(values)] == rounded_values:
            match = i
    return match


def flush_trace_block(
    *,
    scores: Sequence[dict],
    episodes: Sequence[dict],
    targets: set,
    source_name: str,
    out: List[ScoredTCR],
) -> None:
    if not episodes:
        return

    final_values = [ep["affinity"] for ep in episodes]
    final_start = find_window(scores, final_values)
    if final_start is None:
        return

    init_values = [ep.get("init_affinity") for ep in episodes]
    init_start = None
    if all(v is not None for v in init_values):
        init_start = find_window(scores, [float(v) for v in init_values], end_before=final_start)

    final_scores = scores[final_start : final_start + len(episodes)]
    init_scores = scores[init_start : init_start + len(episodes)] if init_start is not None else [None] * len(episodes)
    for ep, final_score, init_score in zip(episodes, final_scores, init_scores):
        peptide = final_score["peptide"]
        tcr = final_score["tcr"]
        if peptide not in targets or not is_valid_tcr(tcr):
            continue
        init_tcr = init_score["tcr"] if init_score and init_score["peptide"] == peptide else None
        if init_tcr is not None and not is_valid_tcr(init_tcr):
            init_tcr = None
        out.append(
            ScoredTCR(
                peptide=peptide,
                tcr=tcr,
                affinity=final_score["affinity"],
                source=source_name,
                episode=ep["episode"],
                step=ep["step"],
                init_tcr=init_tcr,
                init_affinity=ep["init_affinity"],
                reward=ep["reward"],
                decoy_violation=ep["decoy_violation"],
                decoy_affinity=ep["decoy_affinity"],
            )
        )


def load_trace_endpoints(log_paths: Sequence[Path], targets: set) -> List[ScoredTCR]:
    endpoints: List[ScoredTCR] = []
    for path in log_paths:
        scores: List[dict] = []
        episodes: List[dict] = []
        source_name = path.stem.replace("_train", "")
        with path.open(errors="replace") as handle:
            for line in handle:
                score_match = TRACE_SCORE_RE.search(line)
                if score_match:
                    if episodes:
                        flush_trace_block(
                            scores=scores,
                            episodes=episodes,
                            targets=targets,
                            source_name=source_name,
                            out=endpoints,
                        )
                        scores = []
                        episodes = []
                    scores.append(
                        {
                            "affinity": float(score_match.group(1)),
                            "tcr": score_match.group(2),
                            "peptide": score_match.group(3),
                        }
                    )
                    continue

                ep_match = TRACE_EPISODE_RE.search(line)
                if ep_match:
                    fields = {k: float(v) for k, v in TRACE_FIELD_RE.findall(line)}
                    if "A" not in fields:
                        continue
                    episodes.append(
                        {
                            "episode": int(ep_match.group(1)),
                            "step": int(ep_match.group(2)),
                            "reward": float(ep_match.group(3)),
                            "affinity": fields["A"],
                            "init_affinity": fields.get("InitA"),
                            "decoy_violation": fields.get("DecViol"),
                            "decoy_affinity": fields.get("DecA"),
                            "target_short": fields.get("TargetShort"),
                            "target_sat": fields.get("TargetSat"),
                        }
                    )
                    continue

                if episodes and not line.startswith("Episode "):
                    flush_trace_block(
                        scores=scores,
                        episodes=episodes,
                        targets=targets,
                        source_name=source_name,
                        out=endpoints,
                    )
                    scores = []
                    episodes = []
        if episodes:
            flush_trace_block(
                scores=scores,
                episodes=episodes,
                targets=targets,
                source_name=source_name,
                out=endpoints,
            )
    return endpoints


def load_tchard_endpoints(path: Path, targets: set, per_peptide: int, seed: int) -> List[ScoredTCR]:
    df = pd.read_csv(
        path,
        usecols=["cdr3.beta", "antigen.epitope", "label"],
        low_memory=False,
    )
    df = df[df["label"].astype(str).isin({"1", "1.0"})]
    df = df[df["antigen.epitope"].isin(targets)]
    df["cdr3.beta"] = df["cdr3.beta"].astype(str).str.strip().str.upper()
    df = df[df["cdr3.beta"].map(is_valid_tcr)]
    df = df.drop_duplicates(["antigen.epitope", "cdr3.beta"])
    out: List[ScoredTCR] = []
    for peptide, group in df.groupby("antigen.epitope"):
        group = group.sample(frac=1.0, random_state=seed)
        selected = group["cdr3.beta"] if per_peptide <= 0 else group["cdr3.beta"].head(per_peptide)
        for tcr in selected:
            out.append(ScoredTCR(peptide=peptide, tcr=tcr, affinity=None, source="tc-hard_positive"))
    return out


def top_trace_endpoints(
    endpoints: Sequence[ScoredTCR],
    per_source_peptide: int,
    min_affinity: Optional[float] = None,
) -> List[ScoredTCR]:
    grouped: Dict[Tuple[str, str], Dict[str, ScoredTCR]] = defaultdict(dict)
    for item in endpoints:
        if item.affinity is None:
            continue
        if min_affinity is not None and item.affinity < min_affinity:
            continue
        key = (item.source, item.peptide)
        existing = grouped[key].get(item.tcr)
        if existing is None or (item.affinity or -math.inf) > (existing.affinity or -math.inf):
            grouped[key][item.tcr] = item

    selected: List[ScoredTCR] = []
    for _key, by_tcr in grouped.items():
        items = sorted(by_tcr.values(), key=lambda x: x.affinity if x.affinity is not None else -math.inf, reverse=True)
        selected.extend(items[:per_source_peptide])
    return selected


def make_demo_rows(
    *,
    peptide: str,
    path: Sequence[Tuple[str, Tuple[int, int, int], str]],
    endpoint: ScoredTCR,
    method: str,
    episode_id: str,
    weight: float,
    max_steps: int,
) -> List[dict]:
    rows = []
    for step_idx, (state, action, nxt) in enumerate(path):
        rows.append(
            {
                "episode_id": episode_id,
                "method": method,
                "source": endpoint.source,
                "peptide": peptide,
                "tcr": state,
                "next_tcr": nxt,
                "endpoint_tcr": endpoint.tcr,
                "endpoint_affinity": endpoint.affinity,
                "init_tcr": endpoint.init_tcr,
                "init_affinity": endpoint.init_affinity,
                "reward": endpoint.reward,
                "decoy_violation": endpoint.decoy_violation,
                "decoy_affinity": endpoint.decoy_affinity,
                "trace_episode": endpoint.episode,
                "trace_step": endpoint.step,
                "step_idx": step_idx,
                "max_steps": max_steps,
                "op": int(action[0]),
                "pos": int(action[1]),
                "tok": int(action[2]),
                "weight": weight,
            }
        )
    return rows


def endpoint_weight(endpoint: ScoredTCR) -> float:
    if endpoint.affinity is None:
        return 0.7
    # Positive-affinity trace endpoints get more say, but keep the range tame.
    return float(max(0.5, min(2.0, 1.0 + endpoint.affinity / 2.0)))


def build_demos(
    *,
    endpoints: Sequence[ScoredTCR],
    max_steps: int,
    trace_replay_per_endpoint: int,
    corruptions_per_endpoint: int,
    corruption_min_edits: int,
    corruption_max_edits: int,
    rng: random.Random,
) -> List[dict]:
    rows: List[dict] = []
    seen_episodes = set()
    for endpoint_idx, endpoint in enumerate(endpoints):
        weight = endpoint_weight(endpoint)
        if endpoint.init_tcr and trace_replay_per_endpoint > 0:
            path = greedy_edit_path(endpoint.init_tcr, endpoint.tcr, max_steps)
            if path:
                episode_id = f"trace_replay:{endpoint.source}:{endpoint.episode}:{endpoint.peptide}:{endpoint.tcr}"
                if episode_id not in seen_episodes:
                    seen_episodes.add(episode_id)
                    rows.extend(
                        make_demo_rows(
                            peptide=endpoint.peptide,
                            path=path,
                            endpoint=endpoint,
                            method="trace_init_to_final",
                            episode_id=episode_id,
                            weight=weight,
                            max_steps=max_steps,
                        )
                    )

        for corruption_idx in range(corruptions_per_endpoint):
            n_edits = rng.randint(corruption_min_edits, corruption_max_edits)
            corrupted_payload = random_corrupt_path(endpoint.tcr, rng, n_edits=n_edits)
            if not corrupted_payload:
                continue
            corrupted, path = corrupted_payload
            if not path:
                continue
            episode_id = f"corrupt_reverse:{endpoint_idx}:{corruption_idx}:{endpoint.peptide}:{corrupted}>{endpoint.tcr}"
            if episode_id in seen_episodes:
                continue
            seen_episodes.add(episode_id)
            rows.extend(
                make_demo_rows(
                    peptide=endpoint.peptide,
                    path=path,
                    endpoint=endpoint,
                    method="corrupt_reverse",
                    episode_id=episode_id,
                    weight=weight,
                    max_steps=max_steps,
                )
            )
    return rows


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(rows: Sequence[dict], endpoints: Sequence[ScoredTCR]) -> dict:
    episodes = defaultdict(int)
    steps_by_method = defaultdict(int)
    steps_by_peptide = defaultdict(int)
    for row in rows:
        episodes[row["episode_id"]] += 1
        steps_by_method[row["method"]] += 1
        steps_by_peptide[row["peptide"]] += 1
    endpoints_by_source = defaultdict(int)
    endpoints_by_peptide = defaultdict(int)
    for endpoint in endpoints:
        endpoints_by_source[endpoint.source] += 1
        endpoints_by_peptide[endpoint.peptide] += 1
    return {
        "n_steps": len(rows),
        "n_episodes": len(episodes),
        "steps_by_method": dict(sorted(steps_by_method.items())),
        "steps_by_peptide": dict(sorted(steps_by_peptide.items())),
        "endpoints": len(endpoints),
        "endpoints_by_source": dict(sorted(endpoints_by_source.items())),
        "endpoints_by_peptide": dict(sorted(endpoints_by_peptide.items())),
        "episode_length_mean": (sum(episodes.values()) / len(episodes)) if episodes else 0.0,
        "episode_length_max": max(episodes.values()) if episodes else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="data/tfold_excellent_peptides.txt")
    parser.add_argument(
        "--trace-log",
        action="append",
        default=None,
        help="Trace train log to mine final TCR endpoints from. Can be repeated.",
    )
    parser.add_argument("--tc-hard", default="/share/liuyutian/TCRdata/tc-hard/ds.csv")
    parser.add_argument(
        "--trace-top-per-peptide",
        type=int,
        default=40,
        help="Top trace endpoints to keep per source and peptide.",
    )
    parser.add_argument(
        "--trace-min-affinity",
        type=float,
        default=None,
        help="Minimum final tFold affinity for trace endpoints. tc-hard positives are not filtered by this.",
    )
    parser.add_argument(
        "--tc-hard-per-peptide",
        type=int,
        default=80,
        help="tc-hard positive endpoints per peptide; <=0 keeps all valid positives.",
    )
    parser.add_argument("--corruptions-per-endpoint", type=int, default=2)
    parser.add_argument("--trace-replay-per-endpoint", type=int, default=1)
    parser.add_argument("--corruption-min-edits", type=int, default=2)
    parser.add_argument("--corruption-max-edits", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/il/trace29_trace61_tchard_il.jsonl")
    parser.add_argument("--summary-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trace_log is None:
        args.trace_log = [
            "logs/test62_simple_target_gated_decoy_trace29_simple_target_gated_decoy_train.log",
            "logs/trace61_dynamic_pool_train.log",
        ]
    rng = random.Random(args.seed)
    targets = {
        line.strip()
        for line in Path(args.targets).read_text().splitlines()
        if line.strip()
    }

    trace_logs = [Path(p) for p in args.trace_log if Path(p).exists()]
    trace_all = load_trace_endpoints(trace_logs, targets)
    trace_selected = top_trace_endpoints(trace_all, args.trace_top_per_peptide, args.trace_min_affinity)
    tchard_selected = load_tchard_endpoints(Path(args.tc_hard), targets, args.tc_hard_per_peptide, args.seed)
    endpoints = trace_selected + tchard_selected

    rows = build_demos(
        endpoints=endpoints,
        max_steps=args.max_steps,
        trace_replay_per_endpoint=args.trace_replay_per_endpoint,
        corruptions_per_endpoint=args.corruptions_per_endpoint,
        corruption_min_edits=args.corruption_min_edits,
        corruption_max_edits=args.corruption_max_edits,
        rng=rng,
    )

    out_path = Path(args.out)
    write_jsonl(out_path, rows)
    summary = summarize(rows, endpoints)
    summary.update(
        {
            "targets": sorted(targets),
            "trace_logs": [str(p) for p in trace_logs],
            "trace_endpoints_seen": len(trace_all),
            "trace_endpoints_selected": len(trace_selected),
            "tc_hard_endpoints_selected": len(tchard_selected),
            "output": str(out_path),
        }
    )
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
