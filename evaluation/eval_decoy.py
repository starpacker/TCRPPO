#!/usr/bin/env python3
"""Test trained TCRPPO model against pMHC decoys for specificity evaluation.

For each target peptide:
  1. Run the trained PPO agent to generate optimized TCRs (re-using the
     inference loop from ``eval_model.py``).
  2. Score every generated TCR against the target peptide AND against every
     decoy peptide loaded from the sibling ``decoy_library/`` repo.
  3. Use MC Dropout (``ergo_uncertainty.mc_dropout_predict``) to also report
     prediction uncertainty (``ergo_std``) per (TCR, peptide) pair.

Layout assumption:
    parent/
    ├── TCRPPO/                 (this repo)
    │   └── evaluation/eval_decoy.py
    └── decoy_library/
        └── data/
            ├── candidate_targets.json
            ├── decoy_c/decoy_library.json
            ├── decoy_a/decoy_a_results.json   (optional, per-target)
            ├── decoy_b/final_ranked_decoys.json (optional)
            └── decoy_d/<TARGET>/decoy_d_results.csv (optional, per-target)

The default ``--decoy_library_root`` is ``../decoy_library`` relative to the
TCRPPO repo root, matching this layout.

Output:
    A single CSV per run with one row per (target_peptide, generated_tcr,
    scored_peptide) triple. Columns:
        target_pep, target_hla, target_source_protein,
        decoy_pep, decoy_hla, decoy_source_protein, decoy_source_tier,
        decoy_evidence_level, decoy_origin_target,
        init_tcr, final_tcr, edit_dist, gmm_likelihood,
        ergo_mean, ergo_std, is_target_pep

Use ``eval_decoy_metrics.py`` to compute summary statistics from this CSV.

Example:
    python evaluation/eval_decoy.py \\
        --model_path output/ae_mcpas_mcpas_0.5_0.0_0.9_256_None/ppo_tcr \\
        --ergo_model ae_mcpas \\
        --num_tcrs_per_target 50 \\
        --n_mc_samples 20 \\
        --out_csv evaluation/results/decoy/eval_decoy_ae_mcpas.csv
"""
import argparse
import csv
import io
import json
import os
import sys
import tempfile
import time

import numpy as np

# --- Path setup -----------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_CODE_DIR = os.path.join(_REPO, "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

# --- Reuse helpers from the existing eval suite ---------------------------
from eval_utils import ERGO_MODELS, ensure_dir, find_model_checkpoint  # noqa: E402
from ergo_uncertainty import mc_dropout_predict_chunked  # noqa: E402


# ============================================================================
#  Target peptide loading
# ============================================================================
DEFAULT_TARGETS_FALLBACK = [
    # Used only if candidate_targets.json cannot be located. Mirrors the 12
    # entries in decoy_library/data/candidate_targets.json verbatim so the
    # script remains usable even without the sibling repo's metadata file.
    {"sequence": "GILGFVFTL", "hla_allele": "HLA-A*02:01", "source_protein": "Influenza M1"},
    {"sequence": "KVAELVHFL", "hla_allele": "HLA-A*02:01", "source_protein": "Variant"},
    {"sequence": "NLVPMVATV", "hla_allele": "HLA-A*02:01", "source_protein": "CMV pp65 (UL83)"},
    {"sequence": "GLCTLVAML", "hla_allele": "HLA-A*02:01", "source_protein": "EBV BMLF1"},
    {"sequence": "CLGGLLTMV", "hla_allele": "HLA-A*02:01", "source_protein": "EBV LMP2"},
    {"sequence": "YLQPRTFLL", "hla_allele": "HLA-A*02:01", "source_protein": "SARS-CoV-2 Spike"},
    {"sequence": "FLPSDFFPSV", "hla_allele": "HLA-A*02:01", "source_protein": "HBV Core"},
    {"sequence": "YMLDLQPET", "hla_allele": "HLA-A*02:01", "source_protein": "HPV16 E7"},
    {"sequence": "SLYNTVATL", "hla_allele": "HLA-A*02:01", "source_protein": "HIV-1 Gag p17"},
    {"sequence": "SLLMWITQC", "hla_allele": "HLA-A*02:01", "source_protein": "NY-ESO-1"},
    {"sequence": "RMFPNAPYL", "hla_allele": "HLA-A*02:01", "source_protein": "WT1"},
    {"sequence": "EVDPIGHLY", "hla_allele": "HLA-A*01:01", "source_protein": "MAGE-A3"},
]


def load_target_peptides(decoy_library_root, override_file=None):
    """Return a list of target peptide dicts.

    Each dict has keys: ``sequence``, ``hla_allele``, ``source_protein``,
    plus any extra metadata available.

    Resolution order:
      1. ``override_file`` (if provided): a JSON file with the same schema as
         ``candidate_targets.json``, OR a plain text file with one peptide per
         line (in which case HLA / source_protein are set to "unknown").
      2. ``<decoy_library_root>/data/candidate_targets.json`` (preferred).
      3. ``DEFAULT_TARGETS_FALLBACK`` (built-in).
    """
    if override_file:
        if override_file.endswith(".json"):
            with io.open(override_file, encoding="utf-8") as f:
                blob = json.load(f)
            return _flatten_candidate_targets(blob)
        # Plain text fallback
        with io.open(override_file, encoding="utf-8") as f:
            seqs = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        return [{"sequence": s, "hla_allele": "unknown", "source_protein": "unknown"}
                for s in seqs]

    cand_path = os.path.join(decoy_library_root, "data", "candidate_targets.json")
    if os.path.isfile(cand_path):
        with io.open(cand_path, encoding="utf-8") as f:
            blob = json.load(f)
        return _flatten_candidate_targets(blob)

    print("[warn] candidate_targets.json not found at {}; using built-in fallback".format(cand_path))
    return list(DEFAULT_TARGETS_FALLBACK)


def _flatten_candidate_targets(blob):
    """Flatten ``candidate_targets.json`` into a list of dicts."""
    out = []
    existing = blob.get("existing_targets") or []
    for s in existing:
        if isinstance(s, str):
            out.append({"sequence": s, "hla_allele": "unknown", "source_protein": "existing"})
        else:
            out.append(_target_dict(s))
    proposed = blob.get("proposed_targets") or []
    for entry in proposed:
        out.append(_target_dict(entry))
    return out


def _target_dict(entry):
    return {
        "sequence": entry.get("sequence", ""),
        "hla_allele": entry.get("hla_allele", "unknown"),
        "source_protein": entry.get("source_protein", "unknown"),
        "category": entry.get("category", ""),
        "therapeutic_context": entry.get("therapeutic_context", ""),
    }


# ============================================================================
#  Decoy peptide loading
# ============================================================================
def _empty_decoy(extra=None):
    base = {
        "peptide": "",
        "hla": "unknown",
        "source_protein": "",
        "tier": "",
        "evidence_level": "",
        "origin_target": "",
    }
    if extra:
        base.update(extra)
    return base


def load_decoys(decoy_library_root, target_sequences,
                include_a=True, include_b=True, include_c=True, include_d=True):
    """Load and merge decoys from all four Decoy A/B/C/D tiers.

    For tiers A, B, D the per-target generation may or may not be present:
    - Decoy A: ``data/decoy_a/decoy_a_results.json`` (single file, holds whatever
      target was last run for; we filter by ``target_sequence`` if available).
    - Decoy B: ``data/decoy_b/final_ranked_decoys.json`` (same).
    - Decoy D: ``data/decoy_d/<TARGET>/decoy_d_results.csv`` (per-target subdir).
    Decoy C is always global.

    Returns:
        list of decoy dicts. Each dict has keys: peptide, hla, source_protein,
        tier (one of A/B/C/D), evidence_level, origin_target.

    Decoys are deduped by ``(peptide, hla)``.
    """
    out = []

    if include_c:
        out.extend(_load_decoy_c(decoy_library_root))
    if include_a:
        out.extend(_load_decoy_a(decoy_library_root))
    if include_b:
        out.extend(_load_decoy_b(decoy_library_root))
    if include_d:
        for tgt in target_sequences:
            out.extend(_load_decoy_d(decoy_library_root, tgt))

    # Dedupe by (peptide, hla)
    seen = {}
    for d in out:
        key = (d["peptide"], d["hla"])
        if not d["peptide"]:
            continue
        if "X" in d["peptide"]:
            # ERGO/AE encoder cannot handle the unknown-residue token "X"
            continue
        if key not in seen:
            seen[key] = d
    return list(seen.values())


def _load_decoy_c(root):
    path = os.path.join(root, "data", "decoy_c", "decoy_library.json")
    if not os.path.isfile(path):
        print("[warn] Decoy C not found: {}".format(path))
        return []
    with io.open(path, encoding="utf-8") as f:
        blob = json.load(f)
    out = []
    for e in blob.get("entries", []):
        pi = e.get("peptide_info") or {}
        rp = e.get("risk_profile") or {}
        dc = e.get("discovery_context") or {}
        out.append({
            "peptide": pi.get("decoy_sequence", ""),
            "hla": pi.get("hla_allele", "unknown"),
            "source_protein": pi.get("source_protein", ""),
            "tier": "C",
            "evidence_level": rp.get("evidence_level", ""),
            "origin_target": dc.get("original_target_sequence", ""),
        })
    print("[load] Decoy C: {} entries".format(len(out)))
    return out


def _load_decoy_a(root):
    base_dir = os.path.join(root, "data", "decoy_a")
    if not os.path.isdir(base_dir):
        return []
    out = []
    for d in os.listdir(base_dir):
        path = os.path.join(base_dir, d, "decoy_a_results.json")
        if not os.path.isfile(path):
            if d == "decoy_a_results.json":
                path = os.path.join(base_dir, d)
            else:
                continue
        with io.open(path, encoding="utf-8") as f:
            blob = json.load(f)
        items = blob if isinstance(blob, list) else blob.get("results", [])
        for e in items:
            out.append({
                "peptide": e.get("sequence", ""),
                "hla": e.get("hla_allele", "unknown"),
                "source_protein": ", ".join((e.get("source_proteins") or [])[:3]),
                "tier": "A",
                "evidence_level": "",
                "origin_target": e.get("target_sequence", ""),
            })
    print("[load] Decoy A: {} entries".format(len(out)))
    return out


def _load_decoy_b(root):
    base_dir = os.path.join(root, "data", "decoy_b")
    if not os.path.isdir(base_dir):
        return []
    out = []
    for d in os.listdir(base_dir):
        path = os.path.join(base_dir, d, "final_ranked_decoys.json")
        if not os.path.isfile(path):
            path = os.path.join(base_dir, d, "decoy_b_results.json")
            if not os.path.isfile(path):
                if d in ("final_ranked_decoys.json", "decoy_b_results.json"):
                    path = os.path.join(base_dir, d)
                else:
                    continue
        with io.open(path, encoding="utf-8") as f:
            blob = json.load(f)
        items = blob if isinstance(blob, list) else blob.get("results", [])
        for e in items:
            out.append({
                "peptide": e.get("sequence", ""),
                "hla": e.get("hla_allele", "unknown"),
                "source_protein": ", ".join((e.get("source_proteins") or [])[:3]),
                "tier": "B",
                "evidence_level": "",
                "origin_target": e.get("target_sequence", ""),
            })
    print("[load] Decoy B: {} entries".format(len(out)))
    return out


def _load_decoy_d(root, target_sequence):
    csv_path = os.path.join(root, "data", "decoy_d", target_sequence,
                            "decoy_d_results.csv")
    if not os.path.isfile(csv_path):
        return []
    out = []
    with io.open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get("sequence") or ""
            if not seq:
                continue
            out.append({
                "peptide": seq,
                "hla": row.get("hla_allele", "unknown"),
                "source_protein": "",
                "tier": "D",
                "evidence_level": "",
                "origin_target": target_sequence,
            })
    print("[load] Decoy D ({}): {} entries".format(target_sequence, len(out)))
    return out


# ============================================================================
#  TCRPPO inference (delegates to eval_model.run_inference)
# ============================================================================
def run_ppo_inference(args, target_peptides):
    """Run the trained PPO agent on the given target peptides.

    We write the peptide list to a temp file (because TCREnv loads peptides
    from disk at construction time) and then call ``eval_model.run_inference``
    unmodified. Returns a dict ``{target_peptide: [record, ...]}``.

    Each record has the keys: init_tcr, final_tcr, ergo_score, edit_dist,
    gmm_likelihood (deterministic, computed during the rollout — these are
    NOT the MC dropout scores; those are computed downstream).
    """
    from eval_model import run_inference  # imported here to avoid path issues

    # Write target peptides to a tempfile
    fd, peptide_file = tempfile.mkstemp(suffix="_eval_decoy_targets.txt", text=True)
    os.close(fd)
    with io.open(peptide_file, "w", encoding="utf-8") as f:
        for tp in target_peptides:
            f.write(tp + "\n")

    # Build the namespace eval_model.run_inference expects.
    ns = argparse.Namespace(
        model_path=args.model_path,
        ergo_model=args.ergo_model,
        peptide_file=peptide_file,
        tcr_file=args.tcr_file,
        num_tcrs=args.num_tcrs_per_target,
        num_envs=args.num_envs,
        rollout=args.rollout,
        max_step=args.max_step,
        beta=args.beta,
        score_stop_criteria=args.score_stop_criteria,
        gmm_stop_criteria=args.gmm_stop_criteria,
        hour=args.hour,
        max_size=args.max_size,
    )
    try:
        results, _ = run_inference(ns)
    finally:
        try:
            os.remove(peptide_file)
        except OSError:
            pass
    return results


# ============================================================================
#  Decoy scoring with MC dropout
# ============================================================================
def score_against_decoys(reward_model, generated_tcrs, target_peptide,
                          decoys, n_mc_samples, ergo_chunk_size):
    """For each generated TCR, compute MC-dropout ERGO scores against the
    target peptide AND every decoy peptide.

    Returns a tuple ``(mean_grid, std_grid, all_meta)`` where:
        mean_grid: ndarray of shape (n_tcrs, n_peps)
        std_grid:  ndarray of shape (n_tcrs, n_peps)
        all_meta:  list of length n_peps. Index 0 is the target (tier="TARGET"),
                   indices 1..n are the decoys.
    """
    n_tcrs = len(generated_tcrs)

    # Construct the per-peptide metadata list. Index 0 = target.
    target_meta = {
        "peptide": target_peptide,
        "hla": "TARGET",  # filled in upstream where target HLA is known
        "source_protein": "",
        "tier": "TARGET",
        "evidence_level": "",
        "origin_target": target_peptide,
    }
    all_meta = [target_meta] + list(decoys)
    n_peps = len(all_meta)

    # Flatten into (TCR, peptide) pairs
    tcrs_flat = [tcr for tcr in generated_tcrs for _ in range(n_peps)]
    peps_flat = [m["peptide"] for _ in range(n_tcrs) for m in all_meta]

    print("  scoring {} TCRs x {} peptides = {} pairs (n_mc_samples={})".format(
        n_tcrs, n_peps, len(tcrs_flat), n_mc_samples))
    t0 = time.time()
    mean, std = mc_dropout_predict_chunked(
        reward_model, tcrs_flat, peps_flat,
        n_samples=n_mc_samples, chunk_size=ergo_chunk_size,
    )
    elapsed = time.time() - t0
    print("  MC dropout scoring done in {:.1f}s ({:.0f} pairs/s)".format(
        elapsed, len(tcrs_flat) / max(elapsed, 1e-6)))

    mean_grid = np.asarray(mean).reshape(n_tcrs, n_peps)
    std_grid = np.asarray(std).reshape(n_tcrs, n_peps)
    return mean_grid, std_grid, all_meta


# ============================================================================
#  CSV writing
# ============================================================================
CSV_FIELDS = [
    "target_pep", "target_hla", "target_source_protein",
    "decoy_pep", "decoy_hla", "decoy_source_protein", "decoy_source_tier",
    "decoy_evidence_level", "decoy_origin_target",
    "init_tcr", "final_tcr", "edit_dist", "gmm_likelihood",
    "ergo_mean", "ergo_std", "is_target_pep",
]


def write_results_row(writer, target_meta, decoy_meta, record, ergo_mean, ergo_std,
                       is_target):
    writer.writerow({
        "target_pep": target_meta["sequence"],
        "target_hla": target_meta["hla_allele"],
        "target_source_protein": target_meta["source_protein"],
        "decoy_pep": decoy_meta["peptide"],
        "decoy_hla": (target_meta["hla_allele"] if is_target else decoy_meta["hla"]),
        "decoy_source_protein": (target_meta["source_protein"] if is_target
                                  else decoy_meta["source_protein"]),
        "decoy_source_tier": "TARGET" if is_target else decoy_meta["tier"],
        "decoy_evidence_level": "" if is_target else decoy_meta["evidence_level"],
        "decoy_origin_target": "" if is_target else decoy_meta["origin_target"],
        "init_tcr": record.get("init_tcr", ""),
        "final_tcr": record.get("final_tcr", ""),
        "edit_dist": "{:.4f}".format(float(record.get("edit_dist", 0.0))),
        "gmm_likelihood": "{:.4f}".format(float(record.get("gmm_likelihood", 0.0))),
        "ergo_mean": "{:.6f}".format(float(ergo_mean)),
        "ergo_std": "{:.6f}".format(float(ergo_std)),
        "is_target_pep": int(bool(is_target)),
    })


# ============================================================================
#  Reward model loader
# ============================================================================
def load_reward_model(args):
    """Load a Reward instance with the user-selected ERGO checkpoint."""
    from eval_model import resolve_ergo_model  # local import after sys.path setup
    from reward import Reward

    ergo_path = resolve_ergo_model(args.ergo_model)
    print("[load] Reward model with ERGO = {}".format(ergo_path))
    return Reward(args.beta, args.gmm_stop_criteria, ergo_model_file=ergo_path)


# ============================================================================
#  Main
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="TCRPPO decoy specificity evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model + ERGO
    p.add_argument("--model_path", type=str, default=None,
                   help="Path to trained PPO checkpoint (without .zip). Auto-detected if omitted.")
    p.add_argument("--ergo_model", type=str, default="ae_mcpas",
                   help="ERGO model: 'ae_mcpas', 'ae_vdjdb', or path to .pt file")

    # Decoy library
    p.add_argument("--decoy_library_root", type=str,
                   default=os.path.normpath(os.path.join(_REPO, "..", "decoy_library")),
                   help="Path to the sibling decoy_library/ repo (default: ../decoy_library)")
    p.add_argument("--target_peptides_file", type=str, default=None,
                   help="Override target list (JSON like candidate_targets.json or txt with one peptide per line)")
    p.add_argument("--max_targets", type=int, default=0,
                   help="If > 0, truncate target list to this many entries")

    # Decoy tier toggles
    p.add_argument("--no_decoy_a", action="store_true")
    p.add_argument("--no_decoy_b", action="store_true")
    p.add_argument("--no_decoy_c", action="store_true")
    p.add_argument("--no_decoy_d", action="store_true")

    # Inference settings
    p.add_argument("--tcr_file", type=str, default=None,
                   help="Test TCR pool (default: data/tcrdb/test_uniq_tcr_seqs.txt)")
    p.add_argument("--num_tcrs_per_target", type=int, default=50,
                   help="Number of TCRs to optimize per target peptide")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--rollout", type=int, default=1)
    p.add_argument("--max_step", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--score_stop_criteria", type=float, default=0.9)
    p.add_argument("--gmm_stop_criteria", type=float, default=1.2577)
    p.add_argument("--hour", type=float, default=10)
    p.add_argument("--max_size", type=int, default=50000)

    # MC Dropout
    p.add_argument("--n_mc_samples", type=int, default=20,
                   help="Number of MC dropout forward passes for ERGO uncertainty")
    p.add_argument("--ergo_chunk_size", type=int, default=4096,
                   help="Chunk size for MC dropout scoring (lower if OOM)")

    # Output
    p.add_argument("--out_csv", type=str, default=None,
                   help="Output CSV path (default: evaluation/results/decoy/eval_decoy_<ergo>.csv)")

    # Debug
    p.add_argument("--dry_run", action="store_true",
                   help="Load decoys + targets and print stats, but skip PPO inference and scoring")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Resolve model path ----------------------------------------------
    if args.model_path is None and not args.dry_run:
        args.model_path = find_model_checkpoint(ergo_key=args.ergo_model)
        if args.model_path is None:
            print("ERROR: --model_path not given and no checkpoint auto-detected under output/.")
            sys.exit(1)
        print("[load] Auto-detected model: {}".format(args.model_path))

    # --- Resolve decoy library root --------------------------------------
    if not os.path.isdir(args.decoy_library_root):
        print("ERROR: decoy_library_root not found: {}".format(args.decoy_library_root))
        print("       Pass --decoy_library_root or place decoy_library/ next to TCRPPO/")
        sys.exit(1)

    # --- Load targets and decoys -----------------------------------------
    targets = load_target_peptides(args.decoy_library_root, args.target_peptides_file)
    if args.max_targets and args.max_targets > 0:
        targets = targets[:args.max_targets]
    target_seqs = [t["sequence"] for t in targets if t["sequence"]]
    print("[load] Targets: {}".format(len(target_seqs)))
    for t in targets:
        print("  - {} ({}, {})".format(t["sequence"], t["hla_allele"], t["source_protein"]))

    decoys = load_decoys(
        args.decoy_library_root,
        target_sequences=target_seqs,
        include_a=not args.no_decoy_a,
        include_b=not args.no_decoy_b,
        include_c=not args.no_decoy_c,
        include_d=not args.no_decoy_d,
    )
    print("[load] Total unique decoys (after dedupe): {}".format(len(decoys)))
    tier_counts = {}
    for d in decoys:
        tier_counts[d["tier"]] = tier_counts.get(d["tier"], 0) + 1
    print("       by tier: {}".format(tier_counts))

    if args.dry_run:
        print("[dry_run] Skipping inference and scoring.")
        return

    # --- Output path -----------------------------------------------------
    if args.out_csv is None:
        ergo_tag = args.ergo_model if args.ergo_model in ERGO_MODELS else "custom"
        out_dir = os.path.join(_HERE, "results", "decoy")
        ensure_dir(out_dir)
        args.out_csv = os.path.join(out_dir, "eval_decoy_{}.csv".format(ergo_tag))
    else:
        ensure_dir(os.path.dirname(os.path.abspath(args.out_csv)))

    # --- Run TCRPPO inference on the target peptides ---------------------
    print("\n[run] Running TCRPPO inference on {} targets...".format(len(target_seqs)))
    inference_results = run_ppo_inference(args, target_seqs)

    n_total_records = sum(len(v) for v in inference_results.values())
    print("[run] Generated {} TCR records across {} targets".format(
        n_total_records, len(inference_results)))
    if n_total_records == 0:
        print("ERROR: PPO inference produced no records. Check --model_path and ERGO model.")
        sys.exit(2)

    # --- Reward model for MC scoring -------------------------------------
    print("\n[run] Loading Reward + ERGO for MC dropout scoring...")
    reward_model = load_reward_model(args)

    # --- Score and write CSV ---------------------------------------------
    print("\n[run] Writing CSV: {}".format(args.out_csv))
    with io.open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for ti, target_meta in enumerate(targets):
            target_seq = target_meta["sequence"]
            records = inference_results.get(target_seq, [])
            if not records:
                print("[skip] target {} ({}/{}) — no PPO records".format(
                    target_seq, ti + 1, len(targets)))
                continue

            print("\n[target {}/{}] {} ({})".format(
                ti + 1, len(targets), target_seq, target_meta["hla_allele"]))
            print("  generated TCRs: {}".format(len(records)))

            generated_tcrs = [r.get("final_tcr", "") for r in records]
            # Filter out empty / X-containing TCRs that ERGO can't handle
            keep_idx = [i for i, t in enumerate(generated_tcrs) if t and "X" not in t]
            if len(keep_idx) < len(generated_tcrs):
                print("  filtered {} invalid TCRs".format(len(generated_tcrs) - len(keep_idx)))
            records = [records[i] for i in keep_idx]
            generated_tcrs = [generated_tcrs[i] for i in keep_idx]
            if not generated_tcrs:
                continue

            mean_grid, std_grid, all_meta = score_against_decoys(
                reward_model, generated_tcrs, target_seq, decoys,
                n_mc_samples=args.n_mc_samples,
                ergo_chunk_size=args.ergo_chunk_size,
            )

            # Row 0 of all_meta is the target itself
            for i, rec in enumerate(records):
                for j, dmeta in enumerate(all_meta):
                    is_target = (j == 0)
                    write_results_row(writer, target_meta, dmeta, rec,
                                      mean_grid[i, j], std_grid[i, j], is_target)
            f.flush()

    print("\n[done] Results written to: {}".format(args.out_csv))
    print("       Analyze with: python evaluation/eval_decoy_metrics.py --csv {}".format(args.out_csv))


if __name__ == "__main__":
    main()
