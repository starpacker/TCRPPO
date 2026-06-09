#!/usr/bin/env python3
"""Current per-peptide max affinity analysis for trace29."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "logs/test62_simple_target_gated_decoy_trace29_simple_target_gated_decoy_train.log"
DEFAULT_TARGETS = ROOT / "data/tfold_excellent_peptides.txt"
DEFAULT_OUT = ROOT / "results/trace29_current_max_affinity"

TFOLD_RE = re.compile(
    r"^\[tFoldScore\]\s+ts=(?P<ts>.*?)\s+source=(?P<source>\S+).*?"
    r"affinity_logit=(?P<affinity>[-+]?\d+(?:\.\d+)?).*?"
    r"cdr3b=(?P<cdr3b>\S+)\s+peptide=(?P<peptide>\S+)\s+hla=(?P<hla>\S+)"
)
EPISODE_RE = re.compile(r"^Episode\s+(?P<episode>\d+)\s+\|\s+Step\s+(?P<step>\d+)")
STEP_RE = re.compile(r"^Step\s+(?P<step>[\d,]+)\s+\|\s+Eps:\s+(?P<episodes>\d+)")


@dataclass(frozen=True)
class ScoreRow:
    score_index: int
    line_no: int
    ts: str
    source: str
    affinity: float
    cdr3b: str
    peptide: str
    hla: str


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def load_targets(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def parse_log(path: Path) -> tuple[list[ScoreRow], dict[str, int | str | None]]:
    rows: list[ScoreRow] = []
    status: dict[str, int | str | None] = {
        "last_episode": None,
        "last_step": None,
        "last_score_ts": None,
    }

    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.startswith("[tFoldScore]"):
                match = TFOLD_RE.search(line)
                if not match:
                    continue
                rows.append(
                    ScoreRow(
                        score_index=len(rows) + 1,
                        line_no=line_no,
                        ts=match.group("ts"),
                        source=match.group("source"),
                        affinity=float(match.group("affinity")),
                        cdr3b=match.group("cdr3b"),
                        peptide=match.group("peptide"),
                        hla=match.group("hla"),
                    )
                )
                status["last_score_ts"] = match.group("ts")
                continue

            ep_match = EPISODE_RE.search(line)
            if ep_match:
                status["last_episode"] = int(ep_match.group("episode"))
                status["last_step"] = int(ep_match.group("step"))
                continue

            step_match = STEP_RE.search(line)
            if step_match:
                status["last_episode"] = int(step_match.group("episodes"))
                status["last_step"] = int(step_match.group("step").replace(",", ""))

    return rows, status


def summarize(rows_by_peptide: dict[str, list[ScoreRow]], targets: list[str]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for peptide in targets:
        rows = rows_by_peptide.get(peptide, [])
        values = [row.affinity for row in rows]
        if not values:
            summaries.append({"peptide": peptide, "n": 0})
            continue

        max_row = max(rows, key=lambda row: row.affinity)
        summaries.append(
            {
                "peptide": peptide,
                "n": len(rows),
                "max_affinity": max_row.affinity,
                "max_score_index": max_row.score_index,
                "max_line_no": max_row.line_no,
                "max_ts": max_row.ts,
                "max_cdr3b": max_row.cdr3b,
                "last_affinity": rows[-1].affinity,
                "mean": mean(values),
                "median": median(values),
                "p95": percentile(values, 0.95),
                "min": min(values),
                "count_gt_0": sum(value > 0 for value in values),
                "count_gt_neg1": sum(value > -1 for value in values),
                "count_gt_neg2": sum(value > -2 for value in values),
            }
        )
    return sorted(summaries, key=lambda item: item.get("max_affinity", float("-inf")), reverse=True)


def write_target_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    fieldnames = [
        "rank",
        "peptide",
        "n",
        "max_affinity",
        "max_ts",
        "max_score_index",
        "max_line_no",
        "max_cdr3b",
        "last_affinity",
        "mean",
        "median",
        "p95",
        "min",
        "count_gt_0",
        "count_gt_neg1",
        "count_gt_neg2",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, item in enumerate(summaries, start=1):
            row = {"rank": rank, **item}
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_all_peptide_csv(path: Path, rows_by_peptide: dict[str, list[ScoreRow]]) -> None:
    fieldnames = ["rank", "peptide", "n", "max_affinity", "max_ts", "max_cdr3b", "is_target_like"]
    all_rows = []
    for peptide, rows in rows_by_peptide.items():
        max_row = max(rows, key=lambda row: row.affinity)
        all_rows.append(
            {
                "peptide": peptide,
                "n": len(rows),
                "max_affinity": max_row.affinity,
                "max_ts": max_row.ts,
                "max_cdr3b": max_row.cdr3b,
            }
        )
    all_rows.sort(key=lambda row: row["max_affinity"], reverse=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(all_rows, start=1):
            writer.writerow({"rank": rank, **row, "is_target_like": ""})


def cumulative_max(values: list[float]) -> list[float]:
    out = []
    current = float("-inf")
    for value in values:
        current = max(current, value)
        out.append(current)
    return out


def plot_peptide(peptide: str, rows: list[ScoreRow], out_dir: Path) -> None:
    xs = [row.score_index for row in rows]
    ys = [row.affinity for row in rows]
    max_row = max(rows, key=lambda row: row.affinity)

    fig, (ax_traj, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
    )
    fig.suptitle(f"{peptide} affinity trajectory", fontsize=14, fontweight="bold")

    ax_traj.scatter(xs, ys, s=9, alpha=0.36, color="#2f6fbb", linewidths=0)
    ax_traj.plot(xs, cumulative_max(ys), color="#d94801", linewidth=1.8, label="cumulative max")
    ax_traj.scatter(
        [max_row.score_index],
        [max_row.affinity],
        s=80,
        color="#d94801",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
        label=f"max={max_row.affinity:.4f}",
    )
    ax_traj.axhline(0, color="#333333", linestyle="--", linewidth=1, alpha=0.7, label="0")
    ax_traj.axhline(-2, color="#777777", linestyle=":", linewidth=1, alpha=0.8, label="target gate -2")
    ax_traj.set_ylabel("tFold affinity logit")
    ax_traj.set_xlabel("global tFoldScore index")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.legend(loc="best", fontsize=9)
    ax_traj.text(
        0.01,
        0.02,
        f"n={len(rows)} | max cdr3b={max_row.cdr3b} | {max_row.ts}",
        transform=ax_traj.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )

    ax_hist.hist(ys, bins=40, color="#6aaed6", alpha=0.85, edgecolor="white")
    ax_hist.axvline(max_row.affinity, color="#d94801", linewidth=1.5)
    ax_hist.axvline(0, color="#333333", linestyle="--", linewidth=1, alpha=0.7)
    ax_hist.axvline(-2, color="#777777", linestyle=":", linewidth=1, alpha=0.8)
    ax_hist.set_xlabel("tFold affinity logit")
    ax_hist.set_ylabel("count")
    ax_hist.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / f"{peptide}_affinity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summaries: list[dict[str, object]], out_path: Path) -> None:
    ordered = list(reversed(summaries))
    peptides = [str(item["peptide"]) for item in ordered]
    max_values = [float(item["max_affinity"]) for item in ordered]
    p95_values = [float(item["p95"]) for item in ordered]

    fig, ax = plt.subplots(figsize=(10, 9))
    y = range(len(peptides))
    ax.barh(y, max_values, color="#d94801", alpha=0.78, label="max")
    ax.scatter(p95_values, y, color="#2f6fbb", s=35, label="p95", zorder=4)
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
    ax.axvline(-2, color="#777777", linestyle=":", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(peptides)
    ax.set_xlabel("tFold affinity logit")
    ax.set_title("Trace29 current max affinity by target peptide")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    summaries: list[dict[str, object]],
    all_rows: list[ScoreRow],
    target_rows: list[ScoreRow],
    status: dict[str, int | str | None],
    out_dir: Path,
) -> None:
    top = summaries[0]
    positive = [item for item in summaries if float(item["max_affinity"]) > 0]
    lines = [
        "# Trace29 Current Max Affinity",
        "",
        f"- Log snapshot last tFoldScore ts: `{status['last_score_ts']}`",
        f"- Last parsed step / episode: `{status['last_step']}` / `{status['last_episode']}`",
        f"- Total tFoldScore rows parsed: `{len(all_rows):,}`",
        f"- Target tFoldScore rows parsed: `{len(target_rows):,}`",
        f"- Target peptides analyzed: `{len(summaries)}`",
        f"- Best target peptide max: `{top['peptide']}` = `{float(top['max_affinity']):.4f}`",
        f"- Target peptides with max > 0: `{len(positive)}/{len(summaries)}`",
        "",
        "## Ranked Target Peptides",
        "",
        "| Rank | Peptide | n | Max | Last | Mean | p95 | >0 | Max CDR3B | Max Time |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, item in enumerate(summaries, start=1):
        lines.append(
            "| {rank} | {peptide} | {n} | {max_affinity:.4f} | {last_affinity:.4f} | "
            "{mean:.4f} | {p95:.4f} | {count_gt_0} | `{max_cdr3b}` | {max_ts} |".format(
                rank=rank,
                peptide=item["peptide"],
                n=item["n"],
                max_affinity=float(item["max_affinity"]),
                last_affinity=float(item["last_affinity"]),
                mean=float(item["mean"]),
                p95=float(item["p95"]),
                count_gt_0=item["count_gt_0"],
                max_cdr3b=item["max_cdr3b"],
                max_ts=item["max_ts"],
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary plot: `{out_dir / 'trace29_target_peptide_max_affinity.png'}`",
            f"- Per-peptide plots: `{out_dir / 'per_peptide'}`",
            f"- CSV: `{out_dir / 'trace29_current_max_affinity_by_target_peptide.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    targets = load_targets(args.targets)
    target_set = set(targets)
    rows, status = parse_log(args.log)
    target_rows = [row for row in rows if row.peptide in target_set]

    rows_by_target: dict[str, list[ScoreRow]] = defaultdict(list)
    rows_by_all_peptide: dict[str, list[ScoreRow]] = defaultdict(list)
    for row in rows:
        rows_by_all_peptide[row.peptide].append(row)
        if row.peptide in target_set:
            rows_by_target[row.peptide].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    peptide_plot_dir = args.out_dir / "per_peptide"
    peptide_plot_dir.mkdir(parents=True, exist_ok=True)

    summaries = summarize(rows_by_target, targets)
    write_target_csv(args.out_dir / "trace29_current_max_affinity_by_target_peptide.csv", summaries)
    write_all_peptide_csv(args.out_dir / "trace29_current_max_affinity_all_peptides.csv", rows_by_all_peptide)
    plot_summary(summaries, args.out_dir / "trace29_target_peptide_max_affinity.png")
    for peptide in targets:
        if rows_by_target.get(peptide):
            plot_peptide(peptide, rows_by_target[peptide], peptide_plot_dir)
    write_report(args.out_dir / "TRACE29_CURRENT_MAX_AFFINITY.md", summaries, rows, target_rows, status, args.out_dir)

    print(f"Parsed {len(rows):,} tFoldScore rows")
    print(f"Target rows: {len(target_rows):,} across {len(targets)} peptides")
    print(f"Last parsed step/episode: {status['last_step']} / {status['last_episode']}")
    print(f"Best target max: {summaries[0]['peptide']} = {float(summaries[0]['max_affinity']):.4f}")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
