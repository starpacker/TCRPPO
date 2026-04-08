"""Shared utilities for TCRPPO evaluation."""
import os
import sys
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVAL_DIR)
CODE_DIR = os.path.join(REPO_ROOT, "code")
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Ensure code/ is importable
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AMINO_ACIDS = list("ARNDCEQGHILKMFPSTWYV")

PEPTIDE_FILES = {
    "ae_mcpas": os.path.join(DATA_DIR, "test_peptides", "ae_mcpas_test_peptides.txt"),
    "ae_vdjdb": os.path.join(DATA_DIR, "test_peptides", "ae_vdjdb_test_peptides.txt"),
}

ERGO_MODELS = {
    "ae_mcpas": os.path.join(CODE_DIR, "ERGO", "models", "ae_mcpas1.pt"),
    "ae_vdjdb": os.path.join(CODE_DIR, "ERGO", "models", "ae_vdjdb1.pt"),
}

TEST_TCR_FILE = os.path.join(DATA_DIR, "tcrdb", "test_uniq_tcr_seqs.txt")


def load_peptides(path):
    """Load peptide list from a text file (one peptide per line)."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_tcrs(path, max_len=27):
    """Load TCR sequences from a text file, filtering by max length."""
    tcrs = []
    with open(path, "r") as f:
        for line in f:
            seq = line.strip()
            if seq and len(seq) <= max_len:
                tcrs.append(seq)
    return tcrs


# ---------------------------------------------------------------------------
# Result file parsing
# ---------------------------------------------------------------------------
def parse_result_file(result_file):
    """Parse a TCRPPO test output file.

    Expected format per line:
        <peptide> <init_tcr> <final_tcr> <ergo_score> <edit_dist> <gmm_likelihood>

    Returns:
        dict[str, list[dict]]: peptide -> list of result dicts
    """
    peptide_results = defaultdict(list)
    with open(result_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            peptide_results[parts[0]].append({
                "init_tcr": parts[1],
                "final_tcr": parts[2],
                "ergo_score": float(parts[3]),
                "edit_dist": float(parts[4]),
                "gmm_likelihood": float(parts[5]),
            })
    return dict(peptide_results)


def flatten_results(peptide_results):
    """Flatten peptide_results dict into a single list of records with peptide key."""
    rows = []
    for pep, records in peptide_results.items():
        for r in records:
            rows.append({"peptide": pep, **r})
    return rows


# ---------------------------------------------------------------------------
# Training log parsing
# ---------------------------------------------------------------------------
def parse_training_log(log_file):
    """Parse stdout training log for PPO metrics.

    Looks for lines of the form:
        key: value;
    produced by on_policy_algorithm.py / ppo.py during training.

    Also looks for SB3-style log blocks:
        | key              | value   |

    Returns:
        dict with keys: episodes (list of episode info dicts),
                        ppo_updates (list of update metric dicts)
    """
    episodes = []
    ppo_updates = []
    current_episode = {}
    current_update = {}

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_episode:
                    episodes.append(current_episode)
                    current_episode = {}
                continue

            # SB3-style table rows: | key | value |
            if line.startswith("|") and "|" in line[1:]:
                parts = [p.strip() for p in line.split("|")]
                parts = [p for p in parts if p and p != "-" * len(p)]
                if len(parts) == 2:
                    key, val = parts
                    key = key.replace("/", "_").replace(" ", "_")
                    if key.startswith("-"):
                        continue
                    try:
                        val = float(val)
                        current_update[key] = val
                    except ValueError:
                        pass
                continue

            # Separator lines in SB3 output
            if line.startswith("+") or line.startswith("-"):
                if current_update:
                    ppo_updates.append(current_update)
                    current_update = {}
                continue

            # Episode info: "key: value; key: value; ..."
            if ":" in line and ";" in line:
                for segment in line.split(";"):
                    segment = segment.strip()
                    if ":" not in segment:
                        continue
                    key, val = segment.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    try:
                        current_episode[key] = float(val)
                    except ValueError:
                        current_episode[key] = val

    # Flush remaining
    if current_episode:
        episodes.append(current_episode)
    if current_update:
        ppo_updates.append(current_update)

    return {"episodes": episodes, "ppo_updates": ppo_updates}


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------
def levenshtein_distance(s1, s2):
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def normalized_edit_distance(s1, s2):
    """Edit distance normalized by the length of the longer string."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return levenshtein_distance(s1, s2) / max_len


# ---------------------------------------------------------------------------
# Auto-detect model checkpoint
# ---------------------------------------------------------------------------
def find_model_checkpoint(output_dir=None, ergo_key=None):
    """Attempt to locate a ppo_tcr.zip checkpoint under output/.

    When *ergo_key* is given (e.g. "ae_mcpas" or "ae_vdjdb"), prefer the
    subdirectory whose name contains that key so we don't accidentally load
    a sanity-check or wrong-dataset model.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Collect every ppo_tcr.zip we can find
    candidates = []
    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            if fname == "ppo_tcr.zip":
                candidates.append(os.path.join(root, fname.replace(".zip", "")))

    if not candidates:
        return None

    # If an ergo_key hint is provided, try to match the right directory
    if ergo_key:
        # Build a prefix like "ae_mcpas_" or "ae_vdjdb_" to match the
        # training output directory naming convention
        prefix = ergo_key + "_"
        matched = [c for c in candidates if prefix in os.path.basename(os.path.dirname(c))]
        # Exclude sanity_check / test_run directories
        matched = [c for c in matched
                    if "sanity" not in c.lower() and "test_run" not in c.lower()]
        if matched:
            # Pick the largest file (most trained) if multiple remain
            matched.sort(key=lambda p: os.path.getsize(p + ".zip"), reverse=True)
            return matched[0]

    # Fallback: exclude sanity/test dirs, pick largest
    filtered = [c for c in candidates
                if "sanity" not in c.lower() and "test_run" not in c.lower()]
    if filtered:
        filtered.sort(key=lambda p: os.path.getsize(p + ".zip"), reverse=True)
        return filtered[0]

    # Last resort: return first candidate
    return candidates[0]


def find_all_checkpoints(output_dir=None):
    """Find all rl_model_*_steps.zip checkpoints, sorted by step number."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    checkpoints = []
    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            if fname.startswith("rl_model_") and fname.endswith("_steps.zip"):
                try:
                    step_str = fname.replace("rl_model_", "").replace("_steps.zip", "")
                    step = int(step_str)
                    path = os.path.join(root, fname.replace(".zip", ""))
                    checkpoints.append((step, path))
                except ValueError:
                    pass
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path
