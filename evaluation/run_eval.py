#!/usr/bin/env python3
"""Main entry point for TCRPPO evaluation pipeline.

Orchestrates the full evaluation workflow:
  1. (Optional) Analyze training logs
  2. Run model inference on test data
  3. Compute metrics and paper comparison
  4. Generate visualizations

Usage:
    # Full pipeline (inference + metrics + plots)
    python run_eval.py --mode full \
        --ergo_model ae_mcpas \
        --num_tcrs 1000 --num_envs 4

    # Metrics + plots only (from existing result file)
    python run_eval.py --mode analyze --result results/eval_ae_mcpas_n1000.txt

    # Training log analysis only
    python run_eval.py --mode training --log training.log

    # All steps for both models
    python run_eval.py --mode full --ergo_model ae_mcpas ae_vdjdb --num_tcrs 500
"""
import argparse
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)

from eval_utils import (
    ensure_dir, find_model_checkpoint, ERGO_MODELS, PEPTIDE_FILES,
)


def run_training_analysis(args):
    """Step 1: Analyze training logs."""
    from eval_training import main as training_main

    print("\n" + "=" * 60)
    print("STEP 1: Training Log Analysis")
    print("=" * 60)

    if not args.log:
        print("  Skipped (no --log provided)")
        return

    out_dir = os.path.join(EVAL_DIR, "figures", "training")
    sys.argv = ["eval_training.py", "--log", args.log, "--out_dir", out_dir]
    training_main()


def run_model_inference(args, ergo_key):
    """Step 2: Run model on test data."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Model Inference ({ergo_key})")
    print("=" * 60)

    model_path = args.model_path or find_model_checkpoint(ergo_key=ergo_key)
    if model_path is None:
        print("  ERROR: No model checkpoint found. Specify --model_path.")
        return None
    print(f"  Using model: {model_path}")

    peptide_key = ergo_key if ergo_key in PEPTIDE_FILES else "ae_mcpas"
    out_file = os.path.join(EVAL_DIR, "results", f"eval_{ergo_key}_n{args.num_tcrs}.txt")

    if os.path.exists(out_file) and not args.force:
        print(f"  Result file already exists: {out_file}")
        print(f"  Use --force to re-run inference.")
        return out_file

    inference_args = [
        "eval_model.py",
        "--model_path", model_path,
        "--ergo_model", ergo_key,
        "--peptide_file", peptide_key,
        "--num_tcrs", str(args.num_tcrs),
        "--num_envs", str(args.num_envs),
        "--rollout", str(args.rollout),
        "--max_step", str(args.max_step),
        "--out", out_file,
    ]
    if args.tcr_file:
        inference_args.extend(["--tcr_file", args.tcr_file])

    sys.argv = inference_args
    from eval_model import main as model_main
    model_main()

    return out_file


def run_metrics_analysis(result_files, labels):
    """Step 3: Compute metrics."""
    print("\n" + "=" * 60)
    print("STEP 3: Metrics Computation & Paper Comparison")
    print("=" * 60)

    out_dir = os.path.join(EVAL_DIR, "results")
    metric_args = ["eval_metrics.py"]
    for rf in result_files:
        metric_args.extend(["--result", rf])
    if labels:
        metric_args.append("--labels")
        metric_args.extend(labels)
    metric_args.extend(["--out_dir", out_dir])

    sys.argv = metric_args
    from eval_metrics import main as metrics_main
    metrics_main()


def run_visualizations(result_files, labels):
    """Step 4: Generate plots."""
    print("\n" + "=" * 60)
    print("STEP 4: Visualization")
    print("=" * 60)

    for rf, label in zip(result_files, labels):
        out_dir = os.path.join(EVAL_DIR, "figures", label)
        print(f"\n  Generating plots for: {label}")
        sys.argv = ["visualize.py", "--result", rf, "--out_dir", out_dir]
        from visualize import main as viz_main
        viz_main()


def main():
    parser = argparse.ArgumentParser(
        description="TCRPPO Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline for ae_mcpas model
  python run_eval.py --mode full --ergo_model ae_mcpas --num_tcrs 1000

  # Analyze existing results
  python run_eval.py --mode analyze --result results/eval_ae_mcpas_n1000.txt

  # Training log only
  python run_eval.py --mode training --log ../logs/training.log

  # Compare two models
  python run_eval.py --mode full --ergo_model ae_mcpas ae_vdjdb --num_tcrs 500
        """,
    )
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "analyze", "training", "visualize"],
                        help="Evaluation mode")
    parser.add_argument("--ergo_model", type=str, nargs="+", default=["ae_mcpas"],
                        help="ERGO model(s): ae_mcpas, ae_vdjdb, or path(s)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained PPO model")
    parser.add_argument("--result", type=str, nargs="*", default=None,
                        help="Existing result file(s) for analyze/visualize mode")
    parser.add_argument("--log", type=str, default=None,
                        help="Training log file for training mode")
    parser.add_argument("--tcr_file", type=str, default=None)
    parser.add_argument("--num_tcrs", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--rollout", type=int, default=1)
    parser.add_argument("--max_step", type=int, default=8)
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if results exist")
    args = parser.parse_args()

    # Ensure output directories
    ensure_dir(os.path.join(EVAL_DIR, "results"))
    ensure_dir(os.path.join(EVAL_DIR, "figures"))

    if args.mode == "training":
        run_training_analysis(args)
        return

    if args.mode in ("analyze", "visualize"):
        if not args.result:
            print("ERROR: --result required for analyze/visualize mode.")
            sys.exit(1)
        result_files = args.result
        labels = [os.path.splitext(os.path.basename(f))[0] for f in result_files]

        if args.mode == "analyze":
            run_metrics_analysis(result_files, labels)
            run_visualizations(result_files, labels)
        else:
            run_visualizations(result_files, labels)
        return

    # Full mode
    if args.log:
        run_training_analysis(args)

    result_files = []
    labels = []
    for ergo_key in args.ergo_model:
        rf = run_model_inference(args, ergo_key)
        if rf:
            result_files.append(rf)
            labels.append(ergo_key)

    if result_files:
        run_metrics_analysis(result_files, labels)
        run_visualizations(result_files, labels)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results:  {os.path.join(EVAL_DIR, 'results')}/")
    print(f"Figures:  {os.path.join(EVAL_DIR, 'figures')}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
