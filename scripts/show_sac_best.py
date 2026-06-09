#!/usr/bin/env python3
"""Show current best TCRs for a SAC run.

Priority:
  1. best_tcr_by_peptide.json written by completed/newer runs
  2. checkpoint best_records_by_peptide if latest/final checkpoint exists
  3. parse logs_sac/<run>.log episode lines for the best terminal A observed
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional


EP_RE = re.compile(
    r"Episode\s+(?P<episode>\d+)\s+\|\s+Step\s+(?P<step>\d+)\s+\|\s+"
    r"R=(?P<reward>[-+0-9.eE]+)\s+\|\s+Len=(?P<length>\d+)\s+\|\s+"
    r"A=(?P<affinity>[-+0-9.eE]+)\s+\|\s+Peptide=(?P<peptide>[A-Z]+)\s+\|\s+"
    r"TCR=(?P<tcr>[A-Z]+)"
)


def load_json(path: Path):
    if path.exists():
        with path.open() as f:
            return json.load(f), f"json:{path}"
    return None, None


def load_checkpoint(path: Path):
    if not path.exists():
        return None, None
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    records = ckpt.get("best_records_by_peptide")
    if records:
        return records, f"checkpoint:{path}"
    record = ckpt.get("best_record")
    if record:
        return {record["peptide"]: record}, f"checkpoint:{path}"
    return None, None


def parse_log(path: Path):
    if not path.exists():
        return None, None
    best = {}
    with path.open(errors="replace") as f:
        for line in f:
            m = EP_RE.search(line)
            if not m:
                continue
            rec = m.groupdict()
            rec["episode"] = int(rec["episode"])
            rec["step"] = int(rec["step"])
            rec["length"] = int(rec["length"])
            rec["reward"] = float(rec["reward"])
            rec["affinity"] = float(rec["affinity"])
            pep = rec["peptide"]
            if pep not in best or rec["affinity"] > float(best[pep]["affinity"]):
                best[pep] = rec
    return (best, f"log-terminal-only:{path}") if best else (None, None)


def print_records(records, source: str, top_k: Optional[int]):
    rows = sorted(records.values(), key=lambda r: float(r.get("affinity", float("-inf"))), reverse=True)
    if top_k is not None:
        rows = rows[:top_k]
    print(f"source\t{source}")
    if source.startswith("log-terminal-only:"):
        print("warning\tparsed from episode logs only; step-wise in-episode best records require live JSON/checkpoint from updated trainer")
    print(f"n_targets\t{len(records)}")
    print("rank\tpeptide\taffinity\treward\tstep\tepisode_step/len\tTCR")
    for rank, rec in enumerate(rows, 1):
        step = rec.get("step", "")
        ep_step = rec.get("episode_step", rec.get("length", ""))
        reward = rec.get("reward", "")
        print(
            f"{rank}\t{rec.get('peptide', '')}\t{float(rec.get('affinity', 0.0)):.4f}\t"
            f"{float(reward):.4f}\t{step}\t{ep_step}\t{rec.get('tcr', '')}"
        )


def main():
    parser = argparse.ArgumentParser(description="Show current best TCRs for a SAC run")
    parser.add_argument("run_name", help="Run name under output_sac/logs_sac")
    parser.add_argument("--output-dir", default="output_sac")
    parser.add_argument("--log-dir", default="logs_sac")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.output_dir) / args.run_name
    paths = [
        run_dir / "best_tcr_by_peptide.json",
        run_dir / "best_tcr.json",
    ]
    for path in paths:
        data, source = load_json(path)
        if data:
            if "peptide" in data:
                data = {data["peptide"]: data}
            print_records(data, source, args.top_k)
            return

    for ckpt_name in ("latest.pt", "final.pt"):
        data, source = load_checkpoint(run_dir / "checkpoints" / ckpt_name)
        if data:
            print_records(data, source, args.top_k)
            return

    data, source = parse_log(Path(args.log_dir) / f"{args.run_name}.log")
    if data:
        print_records(data, source, args.top_k)
        return

    print(f"No best records found for run {args.run_name}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
