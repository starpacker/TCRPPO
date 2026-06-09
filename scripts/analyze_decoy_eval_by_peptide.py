#!/usr/bin/env python
"""Analyze decoy-eval behavior per peptide and draw reward curves."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_TRACE11_RUN = Path("results/test51c_trace11_delta_decoy_reward_eval_n1d16_topk_dedicated_gpu6")
DEFAULT_STEP0_RUN = Path("results/test51c_trace11_delta_decoy_reward_eval_step0_n1d16_topk_newserver_gpu6")
DEFAULT_TRACE20_21_RUN = Path("results/trace20_trace21_latest_decoy_eval_mode4_noC_constructed")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_step0_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["checkpoint"] = out["checkpoint"].astype(str)
    return out


def combine_trace11(trace11_run: Path, step0_run: Path) -> pd.DataFrame:
    trained = read_csv(trace11_run / "summary_by_target.csv")
    step0 = normalize_step0_label(read_csv(step0_run / "summary_by_target.csv"))
    combined = pd.concat([step0, trained], ignore_index=True)
    combined = combined.sort_values(["target", "step", "checkpoint"]).reset_index(drop=True)
    return combined


def trend_slope(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return float("nan")
    xv = x[mask].astype(float)
    yv = y[mask].astype(float)
    denom = ((xv - xv.mean()) ** 2).sum()
    if denom == 0:
        return float("nan")
    return float(((xv - xv.mean()) * (yv - yv.mean())).sum() / denom)


def classify(row: pd.Series) -> str:
    latest_margin = float(row["latest_margin"])
    latest_auroc = float(row["latest_auroc"])
    best_auroc = float(row["best_auroc"])
    latest_target = float(row["latest_target_reward"])
    latest_decoy = float(row["latest_decoy_reward"])
    target_delta = float(row["target_reward_delta_step0_to_latest"])

    if latest_margin > 0.2 and latest_auroc >= 0.75:
        return "works_well"
    if latest_margin < -0.2 and latest_auroc <= 0.25:
        return "decoys_score_above_target"
    if target_delta < 0 and latest_target < latest_decoy:
        return "training_made_target_worse"
    if best_auroc >= 0.75 and latest_auroc < 0.55:
        return "unstable_regressed"
    if latest_auroc < 0.55 and abs(latest_margin) < 0.2:
        return "ambiguous_close_scores"
    return "mixed"


def summarize_trace11(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, g in df.groupby("target"):
        g = g.sort_values("step")
        first = g.iloc[0]
        latest = g.iloc[-1]
        best_margin = g.loc[g["mean_margin"].idxmax()]
        best_auroc = g.loc[g["mean_auroc"].idxmax()]
        worst_latest_gap = latest["mean_margin_vs_decoy_max"]
        rows.append(
            {
                "target": target,
                "n_steps": len(g),
                "step0_target_reward": first["mean_target_reward"],
                "latest_step": int(latest["step"]),
                "latest_target_reward": latest["mean_target_reward"],
                "latest_decoy_reward": latest["mean_decoy_reward"],
                "latest_decoy_max_reward": latest["mean_decoy_max_reward"],
                "latest_margin": latest["mean_margin"],
                "latest_margin_vs_decoy_max": worst_latest_gap,
                "latest_auroc": latest["mean_auroc"],
                "latest_top3_decoy": latest.get("mean_decoy_top3_mean_reward", float("nan")),
                "latest_margin_vs_top3": latest.get("mean_margin_target_minus_decoy_top3", float("nan")),
                "best_margin": best_margin["mean_margin"],
                "best_margin_step": int(best_margin["step"]),
                "best_auroc": best_auroc["mean_auroc"],
                "best_auroc_step": int(best_auroc["step"]),
                "target_reward_delta_step0_to_latest": latest["mean_target_reward"] - first["mean_target_reward"],
                "margin_delta_step0_to_latest": latest["mean_margin"] - first["mean_margin"],
                "auroc_delta_step0_to_latest": latest["mean_auroc"] - first["mean_auroc"],
                "target_reward_slope_per_100k": trend_slope(g["step"], g["mean_target_reward"]) * 100000,
                "margin_slope_per_100k": trend_slope(g["step"], g["mean_margin"]) * 100000,
            }
        )
    out = pd.DataFrame(rows)
    out["interpretation"] = out.apply(classify, axis=1)
    return out.sort_values(["latest_auroc", "latest_margin", "target"]).reset_index(drop=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def plot_peptide_curves(df: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "peptide_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for target, g in df.groupby("target"):
        g = g.sort_values("step")
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
        ax = axes[0]
        ax.plot(g["step"], g["mean_target_reward"], marker="o", label="target reward", color="#1f77b4")
        ax.plot(g["step"], g["mean_decoy_reward"], marker="o", label="decoy mean", color="#ff7f0e")
        ax.plot(g["step"], g["mean_decoy_max_reward"], marker="o", label="decoy max", color="#d62728")
        if "mean_decoy_top3_mean_reward" in g:
            ax.plot(g["step"], g["mean_decoy_top3_mean_reward"], marker="o", label="decoy top3", color="#9467bd")
        ax.set_ylabel("tFold reward logit")
        ax.set_title(target)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        ax2 = axes[1]
        ax2.axhline(0, color="#666666", linewidth=0.8, alpha=0.6)
        ax2.plot(g["step"], g["mean_margin"], marker="o", label="target - decoy mean", color="#2ca02c")
        ax2.plot(g["step"], g["mean_margin_vs_decoy_max"], marker="o", label="target - decoy max", color="#8c564b")
        ax2b = ax2.twinx()
        ax2b.plot(g["step"], g["mean_auroc"], marker="s", label="AUROC", color="#111111", linestyle="--")
        ax2.set_xlabel("checkpoint step")
        ax2.set_ylabel("margin")
        ax2b.set_ylabel("AUROC")
        ax2.set_ylim(
            min(-0.5, float(g[["mean_margin", "mean_margin_vs_decoy_max"]].min().min()) - 0.2),
            max(0.5, float(g[["mean_margin", "mean_margin_vs_decoy_max"]].max().max()) + 0.2),
        )
        ax2b.set_ylim(-0.02, 1.02)
        ax2.grid(True, alpha=0.25)
        lines, labels = ax2.get_legend_handles_labels()
        lines_b, labels_b = ax2b.get_legend_handles_labels()
        ax2.legend(lines + lines_b, labels + labels_b, loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(plots_dir / f"{safe_name(target)}.png", dpi=180)
        plt.close(fig)


def plot_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    for metric, filename, title in [
        ("mean_auroc", "heatmap_auroc.png", "Per-peptide AUROC"),
        ("mean_margin", "heatmap_margin.png", "Per-peptide target minus decoy mean"),
        ("mean_target_reward", "heatmap_target_reward.png", "Per-peptide target reward"),
    ]:
        pivot = df.pivot(index="target", columns="step", values=metric).sort_index()
        fig_w = max(8, 0.7 * len(pivot.columns))
        fig_h = max(8, 0.35 * len(pivot.index))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(x)) for x in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def tier_mix(trace11_run: Path) -> pd.DataFrame:
    pair_scores = read_csv(trace11_run / "pair_scores.csv")
    decoys = pair_scores[pair_scores["tier"] != "target"]
    mix = decoys.groupby(["target", "tier"]).size().unstack(fill_value=0)
    for tier in ["A", "B", "C", "D"]:
        if tier not in mix.columns:
            mix[tier] = 0
    mix = mix[["A", "B", "C", "D"]].reset_index()
    return mix.sort_values("target").reset_index(drop=True)


def write_markdown_report(
    output_dir: Path,
    summary: pd.DataFrame,
    trace20_21: pd.DataFrame | None,
    mix: pd.DataFrame,
) -> None:
    report = output_dir / "REPORT.md"
    cols = [
        "target",
        "latest_step",
        "latest_target_reward",
        "latest_decoy_reward",
        "latest_decoy_max_reward",
        "latest_margin",
        "latest_margin_vs_decoy_max",
        "latest_auroc",
        "best_margin",
        "best_margin_step",
        "best_auroc",
        "best_auroc_step",
        "target_reward_delta_step0_to_latest",
        "margin_delta_step0_to_latest",
        "interpretation",
    ]
    display = summary[cols].copy()
    numeric_cols = display.select_dtypes(include=["number"]).columns
    display[numeric_cols] = display[numeric_cols].round(3)

    counts = summary["interpretation"].value_counts().to_dict()
    lines = [
        "# Per-Peptide Decoy Reward Analysis",
        "",
        "This report uses the 16-decoy trace11 evaluation curve plus the untrained step-0 baseline.",
        "Curves are drawn per peptide under `peptide_curves/`.",
        "",
        "Important caveat: this historical 16-decoy run predates the no-C standard.",
        "Most peptides used A/B/D decoys only, but peptides with insufficient A/B/D at the time used tier C.",
        "Use the Trace20/Trace21 mode-4 no-C cross-check below when judging those peptides.",
        "",
        "## Interpretation Labels",
        "",
        "- `works_well`: latest margin is positive and AUROC is high.",
        "- `decoys_score_above_target`: selected decoys score above target; this can mean hard design or a tFold/decoy confound.",
        "- `training_made_target_worse`: target reward decreased from step 0 and is below decoy mean.",
        "- `unstable_regressed`: some checkpoint worked but latest regressed.",
        "- `ambiguous_close_scores`: target and decoys are very close, so tFold is not separating them cleanly.",
        "- `mixed`: no simple label from aggregate metrics.",
        "",
        "## Label Counts",
        "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- `{key}`: {value}")

    lines.extend([
        "",
        "## Peptide Table",
        "",
        dataframe_to_markdown(display),
        "",
        "## Heatmaps",
        "",
        "- `heatmap_auroc.png`",
        "- `heatmap_margin.png`",
        "- `heatmap_target_reward.png`",
        "",
        "## Historical 16-Decoy Tier Mix",
        "",
        dataframe_to_markdown(mix),
        "",
    ])

    if trace20_21 is not None and not trace20_21.empty:
        keep = [
            "checkpoint",
            "step",
            "target",
            "mean_target_reward",
            "mean_decoy_reward",
            "mean_decoy_max_reward",
            "mean_margin",
            "mean_margin_vs_decoy_max",
            "mean_auroc",
        ]
        t = trace20_21[keep].copy()
        t = t.sort_values(["target", "step"])
        t[t.select_dtypes(include=["number"]).columns] = t.select_dtypes(include=["number"]).round(3)
        lines.extend([
            "## Trace20/Trace21 Latest Mode-4 Cross-Check",
            "",
            "These rows use the separate 4-decoy no-C eval, so they are a cross-check rather than the same 16-decoy curve.",
            "",
            dataframe_to_markdown(t),
            "",
        ])

    report.write_text("\n".join(lines))


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Small dependency-free Markdown table writer."""
    headers = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([format_md_cell(row[c]) for c in df.columns])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_md_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return str(value).replace("|", "\\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace11-run", type=Path, default=DEFAULT_TRACE11_RUN)
    parser.add_argument("--step0-run", type=Path, default=DEFAULT_STEP0_RUN)
    parser.add_argument("--trace20-21-run", type=Path, default=DEFAULT_TRACE20_21_RUN)
    parser.add_argument("--output-dir", type=Path, default=Path("results/per_peptide_decoy_reward_analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    curve = combine_trace11(args.trace11_run, args.step0_run)
    summary = summarize_trace11(curve)
    curve.to_csv(args.output_dir / "trace11_step0_combined_by_target.csv", index=False)
    summary.to_csv(args.output_dir / "per_peptide_summary.csv", index=False)

    trace20_21 = None
    if args.trace20_21_run.exists():
        trace20_21 = read_csv(args.trace20_21_run / "summary_by_target.csv")
        trace20_21.to_csv(args.output_dir / "trace20_trace21_mode4_by_target.csv", index=False)

    plot_peptide_curves(curve, args.output_dir)
    plot_heatmaps(curve, args.output_dir)
    mix = tier_mix(args.trace11_run)
    mix.to_csv(args.output_dir / "trace11_16decoy_tier_mix.csv", index=False)
    write_markdown_report(args.output_dir, summary, trace20_21, mix)
    print(f"Wrote {args.output_dir / 'REPORT.md'}")
    print(f"Wrote {args.output_dir / 'per_peptide_summary.csv'}")
    print(f"Wrote {args.output_dir / 'peptide_curves'}")


if __name__ == "__main__":
    main()
