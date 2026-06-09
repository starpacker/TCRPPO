#!/usr/bin/env python
"""Evaluate tFold accuracy on tc-hard peptides and correlate with design metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer

TFOLD_PYTHON = "/home/liuyutian/server/miniconda3/envs/tfold/bin/python"
TC_HARD = Path("/share/liuyutian/TCRdata/tc-hard/ds.csv")
TC_HARD_VERIFIED = Path("/share/liuyutian/TCRdata/verified_tc_hard_format.csv")


def recv_msg(sock: socket.socket) -> bytes:
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("server closed before header")
        header += chunk
    length = struct.unpack(">I", header)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(length - len(data), 65536))
        if not chunk:
            raise ConnectionError("server closed during payload")
        data += chunk
    return data


def send_server_cmd(socket_path: str, cmd: str) -> dict:
    payload = json.dumps({"cmd": cmd}).encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect(socket_path)
    sock.sendall(struct.pack(">I", len(payload)) + payload)
    response = json.loads(recv_msg(sock).decode("utf-8"))
    sock.close()
    return response


def wait_ready(log_path: Path, socket_path: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if log_path.exists() and "\nREADY\n" in f"\n{log_path.read_text(errors='ignore')}\n":
            response = send_server_cmd(socket_path, "ping")
            if response.get("status") == "pong":
                return
        time.sleep(5)
    tail = log_path.read_text(errors="ignore")[-4000:] if log_path.exists() else ""
    raise TimeoutError(f"tFold server did not become READY within {timeout_s}s\n{tail}")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def load_targets(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text().splitlines() if x.strip() and not x.startswith("#")]


def sample_ground_truth(targets: list[str], per_label: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = ["cdr3.beta", "mhc.a", "antigen.epitope", "label", "negative.source", "license"]
    df = pd.read_csv(TC_HARD, usecols=usecols, low_memory=False)
    df = df.dropna(subset=["cdr3.beta", "mhc.a", "antigen.epitope", "label"])
    df = df[df["antigen.epitope"].isin(targets)].copy()
    df["source_dataset"] = "tc-hard/ds.csv"

    rows = []
    coverage_rows = []
    covered = set(df["antigen.epitope"].unique())
    for target in targets:
        g = df[df["antigen.epitope"] == target]
        if g.empty:
            continue
        for label in [1.0, 0.0]:
            h = g[g["label"] == label].drop_duplicates(["cdr3.beta", "antigen.epitope", "mhc.a"])
            n_avail = len(h)
            n = min(per_label, n_avail)
            coverage_rows.append({
                "target": target,
                "source_dataset": "tc-hard/ds.csv",
                "label": int(label),
                "available": n_avail,
                "sampled": n,
            })
            if n:
                rows.append(h.sample(n=n, random_state=seed + int(label)))

    missing = [t for t in targets if t not in covered]
    if missing and TC_HARD_VERIFIED.exists():
        vcols = ["cdr3.beta", "mhc.a", "antigen.epitope", "label", "source"]
        vf = pd.read_csv(TC_HARD_VERIFIED, usecols=vcols, low_memory=False)
        vf = vf.dropna(subset=["cdr3.beta", "mhc.a", "antigen.epitope", "label"])
        vf = vf[vf["antigen.epitope"].isin(missing)].copy()
        vf["source_dataset"] = "verified_tc_hard_format.csv"
        for target in missing:
            g = vf[vf["antigen.epitope"] == target]
            for label in [1.0, 0.0]:
                h = g[g["label"] == label].drop_duplicates(["cdr3.beta", "antigen.epitope", "mhc.a"])
                n_avail = len(h)
                n = min(per_label, n_avail)
                coverage_rows.append({
                    "target": target,
                    "source_dataset": "verified_tc_hard_format.csv",
                    "label": int(label),
                    "available": n_avail,
                    "sampled": n,
                })
                if n:
                    rows.append(h.sample(n=n, random_state=seed + 17 + int(label)))

    sample = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    sample = sample.rename(columns={"cdr3.beta": "cdr3b", "antigen.epitope": "target", "mhc.a": "hla"})
    sample["label"] = sample["label"].astype(int)
    sample = sample[["target", "cdr3b", "hla", "label", "source_dataset"]].copy()
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sample, pd.DataFrame(coverage_rows)


def best_threshold_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    thresholds = np.unique(scores)
    if len(thresholds) == 0:
        return {"best_threshold": float("nan"), "best_balanced_accuracy": float("nan"), "best_accuracy": float("nan")}
    candidates = np.r_[thresholds[0] - 1e-6, thresholds, thresholds[-1] + 1e-6]
    best = None
    for thr in candidates:
        pred = (scores >= thr).astype(int)
        bal = balanced_accuracy_score(labels, pred)
        acc = accuracy_score(labels, pred)
        rec = (bal, acc, float(thr))
        if best is None or rec > best:
            best = rec
    return {
        "best_threshold": best[2],
        "best_balanced_accuracy": best[0],
        "best_accuracy": best[1],
    }


def summarize_predictions(scored: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    global_metrics = best_threshold_metrics(scored["label"].to_numpy(), scored["tfold_reward"].to_numpy())
    global_thr = float(global_metrics["best_threshold"])
    rows = []
    for target, g in scored.groupby("target"):
        labels = g["label"].to_numpy()
        scores = g["tfold_reward"].to_numpy()
        n_pos = int(labels.sum())
        n_neg = int((labels == 0).sum())
        auroc = roc_auc_score(labels, scores) if n_pos and n_neg else float("nan")
        pred0 = (scores >= 0.0).astype(int)
        pred_global = (scores >= global_thr).astype(int)
        best = best_threshold_metrics(labels, scores) if n_pos and n_neg else {}
        rows.append({
            "target": target,
            "n": len(g),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "tc_hard_hlas": ",".join(sorted(map(str, g["hla"].dropna().unique()))),
            "mean_pos_reward": float(g[g["label"] == 1]["tfold_reward"].mean()) if n_pos else float("nan"),
            "mean_neg_reward": float(g[g["label"] == 0]["tfold_reward"].mean()) if n_neg else float("nan"),
            "pos_minus_neg_reward": (
                float(g[g["label"] == 1]["tfold_reward"].mean() - g[g["label"] == 0]["tfold_reward"].mean())
                if n_pos and n_neg else float("nan")
            ),
            "tfold_auroc": auroc,
            "acc_at_zero": accuracy_score(labels, pred0),
            "balanced_acc_at_zero": balanced_accuracy_score(labels, pred0),
            "acc_at_global_threshold": accuracy_score(labels, pred_global),
            "balanced_acc_at_global_threshold": balanced_accuracy_score(labels, pred_global),
            "best_threshold": best.get("best_threshold", float("nan")),
            "best_accuracy": best.get("best_accuracy", float("nan")),
            "best_balanced_accuracy": best.get("best_balanced_accuracy", float("nan")),
        })
    return pd.DataFrame(rows).sort_values("tfold_auroc"), global_thr


def score_sample(args: argparse.Namespace, sample: pd.DataFrame) -> pd.DataFrame:
    scorer = AffinityTFoldScorer(
        device=args.device,
        cache_path=args.tfold_cache_path,
        server_socket_path=args.tfold_server_socket,
        max_subprocess_batch=args.extract_batch_size,
        use_cache=True,
    )
    rows = []
    total = len(sample)
    for start in range(0, total, args.score_batch_size):
        batch = sample.iloc[start:start + args.score_batch_size]
        print(f"[score] batch {start // args.score_batch_size + 1}: {len(batch)} pairs ({start}/{total})", flush=True)
        scores, confidences = scorer.score_batch(
            batch["cdr3b"].tolist(),
            batch["target"].tolist(),
            hlas=batch["hla"].tolist(),
        )
        out = batch.copy()
        out["tfold_reward"] = scores
        out["confidence"] = confidences
        rows.append(out)
    scored = pd.concat(rows, ignore_index=True)
    return scored


def corr_records(df: pd.DataFrame, xcols: list[str], ycols: list[str]) -> pd.DataFrame:
    rows = []
    for x in xcols:
        for y in ycols:
            sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 3:
                continue
            rows.append({
                "x_tfold_metric": x,
                "y_design_metric": y,
                "n": len(sub),
                "pearson_r": pearsonr(sub[x], sub[y]).statistic,
                "pearson_p": pearsonr(sub[x], sub[y]).pvalue,
                "spearman_r": spearmanr(sub[x], sub[y]).statistic,
                "spearman_p": spearmanr(sub[x], sub[y]).pvalue,
            })
    return pd.DataFrame(rows)


def analyze_correlation(output_dir: Path, tfold_summary: pd.DataFrame, design_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    design = pd.read_csv(design_path)
    merged = tfold_summary.merge(design, on="target", how="inner", suffixes=("_tfold", "_design"))
    merged.to_csv(output_dir / "tfold_vs_design_by_peptide.csv", index=False)
    corr = corr_records(
        merged,
        xcols=[
            "tfold_auroc",
            "balanced_acc_at_global_threshold",
            "best_balanced_accuracy",
            "pos_minus_neg_reward",
        ],
        ycols=[
            "latest_auroc",
            "latest_margin",
            "latest_margin_vs_decoy_max",
            "best_auroc",
            "best_margin",
            "target_reward_delta_step0_to_latest",
        ],
    )
    corr.to_csv(output_dir / "correlations.csv", index=False)
    return merged, corr


def plot_correlation(output_dir: Path, merged: pd.DataFrame) -> None:
    plots = [
        ("tfold_auroc", "latest_auroc", "tfold_auroc_vs_design_latest_auroc.png"),
        ("tfold_auroc", "latest_margin", "tfold_auroc_vs_design_latest_margin.png"),
        ("best_balanced_accuracy", "latest_auroc", "tfold_best_balacc_vs_design_latest_auroc.png"),
        ("pos_minus_neg_reward", "latest_margin", "tfold_pos_neg_gap_vs_design_margin.png"),
    ]
    for x, y, name in plots:
        sub = merged[[x, y, "target"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 3:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.scatter(sub[x], sub[y], s=45)
        for _, row in sub.iterrows():
            ax.annotate(row["target"], (row[x], row[y]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="#999999", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=180)
        plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append("NA" if not np.isfinite(v) else f"{v:.3f}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    tfold_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    merged: pd.DataFrame,
    corr: pd.DataFrame,
    global_threshold: float,
    args: argparse.Namespace,
) -> None:
    view_cols = [
        "target",
        "n",
        "n_pos",
        "n_neg",
        "tfold_auroc",
        "balanced_acc_at_global_threshold",
        "best_balanced_accuracy",
        "pos_minus_neg_reward",
        "latest_auroc",
        "latest_margin",
        "interpretation",
    ]
    view = merged[view_cols].sort_values("tfold_auroc").copy()
    best_corr = corr.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).head(10) if not corr.empty else corr
    lines = [
        "# tFold tc-hard Accuracy vs Design Performance",
        "",
        f"- tc-hard sample per label per peptide: {args.per_label}",
        f"- global threshold selected by balanced accuracy on this sampled set: {global_threshold:.4f}",
        f"- scored pairs: {int(tfold_summary['n'].sum())}",
        f"- peptides with both positive and negative samples: {len(tfold_summary)}",
        "",
        "Accuracy is measured from tc-hard labels using raw tFold binding logits. AUROC is the primary metric because score calibration differs by peptide/HLA.",
        "",
        "## Per-Peptide Table",
        "",
        dataframe_to_markdown(view),
        "",
        "## Strongest Correlations",
        "",
        dataframe_to_markdown(best_corr) if not best_corr.empty else "No correlation rows.",
        "",
        "## Coverage",
        "",
        dataframe_to_markdown(coverage),
        "",
        "## Output Files",
        "",
        "- `tc_hard_sample.csv`",
        "- `tc_hard_tfold_scores.csv`",
        "- `tfold_accuracy_by_peptide.csv`",
        "- `tfold_vs_design_by_peptide.csv`",
        "- `correlations.csv`",
        "- scatter plot PNG files",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def run_with_server(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    socket_path = f"/tmp/tfold_server_eval_{args.run_name}.sock"
    server_log = PROJECT_ROOT / "logs" / f"tfold_server_eval_{args.run_name}.log"
    completion_log = PROJECT_ROOT / "logs" / f"tfold_completion_eval_{args.run_name}.log"
    for path in (Path(socket_path), Path(socket_path + ".pid")):
        if path.exists():
            path.unlink()

    server_cmd = [
        TFOLD_PYTHON,
        "scripts/tfold_feature_server.py",
        "--socket",
        socket_path,
        "--gpu",
        "0",
        "--use-amp-wrapper",
        "--chunk-size",
        "64",
        "--completion-log",
        str(completion_log),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with server_log.open("w") as log_f:
        server = subprocess.Popen(server_cmd, cwd=PROJECT_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    args.tfold_server_socket = socket_path
    try:
        print(f"[server] pid={server.pid} socket={socket_path}", flush=True)
        wait_ready(server_log, socket_path, args.server_timeout_s)
        print("[server] READY", flush=True)
        run_analysis(args)
    finally:
        print("[server] shutting down", flush=True)
        try:
            send_server_cmd(socket_path, "shutdown")
        except Exception as exc:
            print(f"[server] graceful shutdown failed: {exc}; terminating pid={server.pid}", flush=True)
            os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=30)
        print("[server] stopped", flush=True)


def run_analysis(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.targets_file)
    sample, coverage = sample_ground_truth(targets, args.per_label, args.seed)
    sample.to_csv(output_dir / "tc_hard_sample.csv", index=False)
    coverage.to_csv(output_dir / "tc_hard_sample_coverage.csv", index=False)
    print(f"[sample] {len(sample)} pairs from {sample['target'].nunique()} peptides", flush=True)

    scored = score_sample(args, sample)
    scored.to_csv(output_dir / "tc_hard_tfold_scores.csv", index=False)

    tfold_summary, global_thr = summarize_predictions(scored)
    tfold_summary.to_csv(output_dir / "tfold_accuracy_by_peptide.csv", index=False)
    merged, corr = analyze_correlation(output_dir, tfold_summary, args.design_summary)
    plot_correlation(output_dir, merged)
    write_report(output_dir, tfold_summary, coverage, merged, corr, global_thr, args)
    print(f"[done] {output_dir / 'REPORT.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="tfold_tc_hard_accuracy_correlation")
    parser.add_argument("--targets-file", type=Path, default=Path("data/tfold_excellent_peptides.txt"))
    parser.add_argument("--design-summary", type=Path, default=Path("results/per_peptide_decoy_reward_analysis/per_peptide_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tfold_tc_hard_accuracy_correlation"))
    parser.add_argument("--per-label", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--extract-batch-size", type=int, default=64)
    parser.add_argument("--tfold-cache-path", default="data/tfold_feature_cache.db")
    parser.add_argument("--tfold-server-socket", default="")
    parser.add_argument("--server-timeout-s", type=int, default=900)
    parser.add_argument("--no-start-server", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_start_server:
        if not args.tfold_server_socket:
            raise SystemExit("--tfold-server-socket is required with --no-start-server")
        run_analysis(args)
    else:
        run_with_server(args)


if __name__ == "__main__":
    main()

