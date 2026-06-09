#!/usr/bin/env python3
"""
Upload TCRPPO v2 checkpoints and results to Huggingface
"""
from huggingface_hub import HfApi, upload_folder, upload_file
from pathlib import Path
import json

# Configuration
TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"  # Replace with your actual token
REPO_ID = "starpacker/tcrppo-v2"

api = HfApi()

# Priority checkpoints to upload
CHECKPOINTS = [
    ("output/trace104_triple_constraint/checkpoints/milestone_5000000.pt", "checkpoints/trace104_5M.pt"),
    ("output/trace104_triple_constraint/checkpoints/latest.pt", "checkpoints/trace104_latest.pt"),
    ("output/trace98_finetune/checkpoints/milestone_200000.pt", "checkpoints/trace98_200K.pt"),
    ("output/trace99_finetune_nat5_from_trace61/checkpoints/milestone_800000.pt", "checkpoints/trace99_800K.pt"),
    ("output/trace61_fp32_restart/checkpoints/latest.pt", "checkpoints/trace61_baseline.pt"),
]

# Results to upload
RESULTS = [
    "all_traces_qualifying.json",
    "logs/alive_traces_affinity_summary_v2.csv",
    "logs/alive_traces_summary.csv",
    "docs/tcrppo_v2_report.html",
    "TCRPPO_V2_SUMMARY.md",
]

def main():
    print(f"Uploading to {REPO_ID}...")

    # Upload checkpoints
    print("\n📦 Uploading checkpoints...")
    for local_path, hf_path in CHECKPOINTS:
        if Path(local_path).exists():
            print(f"  {local_path} -> {hf_path}")
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=hf_path,
                    repo_id=REPO_ID,
                    token=TOKEN,
                )
                print(f"    ✅ Done")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
        else:
            print(f"  ⚠️  Not found: {local_path}")

    # Upload results
    print("\n📊 Uploading results...")
    for file_path in RESULTS:
        if Path(file_path).exists():
            print(f"  {file_path}")
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path,
                    repo_id=REPO_ID,
                    token=TOKEN,
                )
                print(f"    ✅ Done")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
        else:
            print(f"  ⚠️  Not found: {file_path}")

    print(f"\n✅ Upload complete! View at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
