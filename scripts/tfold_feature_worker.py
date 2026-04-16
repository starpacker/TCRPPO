#!/usr/bin/env python
"""tFold feature extraction worker — runs in the `tfold` conda env.

This script is called as a subprocess by AffinityTFoldScorer from the
tcrppo_v2 env.  It reads a batch of (cdr3b, peptide, hla) requests
from a JSON input file, runs the full tFold-TCR-pMHC pipeline to extract
V3.4 classifier features, and writes the results as .pt tensors.

Usage:
    /path/to/tfold/python scripts/tfold_feature_worker.py \
        --input  /tmp/tfold_batch_input.json \
        --output /tmp/tfold_batch_output.pt \
        --gpu 0

Input JSON format:
[
  {"cdr3b": "CASSIRSSYEQYF", "peptide": "GILGFVFTL", "hla": "HLA-A*02:01"},
  ...
]

Output .pt format:
{
  "features": [  # list of dicts, one per input sample
    {
      "sfea_cdr3b": Tensor[Lb, 192],
      "sfea_cdr3a": Tensor[La, 192],
      "sfea_pep": Tensor[Lp, 192],
      "ca_cdr3b": Tensor[Lb, 3],
      "ca_cdr3a": Tensor[La, 3],
      "ca_pep": Tensor[Lp, 3],
      "pfea_cdr3b_pep": Tensor[Lb, Lp, 128],
      "pfea_cdr3a_pep": Tensor[La, Lp, 128],
      "v33_feat": Tensor[448],
      "len_cdr3b": int,
      "len_cdr3a": int,
      "len_pep": int,
    },
    ...
  ],
  "errors": [null, "error msg", ...]  # null = success
}
"""

import argparse
import json
import os
import sys
import traceback
from typing import Dict, List, Optional

import torch

# Add tFold root to path
TFOLD_ROOT = "/share/liuyutian/tfold"
sys.path.insert(0, TFOLD_ROOT)

import importlib

# Import tFold components
fe = importlib.import_module("TCR_PMHC_pred.4_16.feature_extraction")
dp = importlib.import_module("TCR_PMHC_pred.4_16.data_pipeline")
hla_mod = importlib.import_module("TCR_PMHC_pred.4_16.hla_mapping")


# ---------------------------------------------------------------------------
# Template V-region scaffolds
# ---------------------------------------------------------------------------

# Default V-region scaffolds for CDR3 splicing.
# These are representative V-region sequences from TRBV7-2 (beta) and
# TRAV12-2 (alpha) — the most common V-genes in the McPAS targets.
# CDR3 starts at position 92 (beta) / 90 (alpha) in the V-region.

_DEFAULT_VREGION_BETA_SCAFFOLD = (
    # TRBV7-2 V-region framework (positions 0-91), CDR3 placeholder starts at 92
    "MGTSLLCWMALCLLGADHADTGVSQNPRHNITKRGQNVTFRCDPISEHNRLYWYRQTLGQGPEFLT"
    "YFQNEAQLEKSRLLSDRFSAERPKGSFSTLEIQRTEQGDSAMYL"
)
_CDR3_POS_BETA = 92  # CDR3 starts here in V-region

_DEFAULT_VREGION_ALPHA_SCAFFOLD = (
    # TRAV12-2 V-region framework, with a default CDR3α
    "MAMLLGASVLILWLQPDWVNSQQKNDDQQVKQNSPSLSVQEGRISILNCDYTNSMFDYFLWYKKYP"
    "ASGPELISLIYLQGFNPKESGIATLYEQPTAASATGLTSANTKSQTSVEFQLS"
)
_DEFAULT_CDR3_ALPHA = "CAVNFGNEKLTF"  # common CDR3α for TRAV12-2


def splice_cdr3_into_scaffold(
    scaffold: str, cdr3: str, cdr3_pos: int, fgxg_suffix: str = "FGXG"
) -> str:
    """Splice a CDR3 sequence into a V-region scaffold at the CDR3 position.

    The V-region = framework(0:cdr3_pos) + CDR3 + FG.G motif.
    """
    framework = scaffold[:cdr3_pos]
    # Add CDR3 + the conserved FGXG motif after CDR3
    vregion = framework + cdr3 + fgxg_suffix
    return vregion


def build_chains_from_cdr3b(
    cdr3b: str,
    peptide: str,
    hla: str,
    hla_lookup: Dict,
) -> Optional[List[Dict]]:
    """Build tFold-compatible chain list from CDR3beta + target info.

    Uses template V-region scaffolds and resolves HLA to full sequence.
    """
    # 1. Build TCRbeta V-region by splicing CDR3 into scaffold
    vregion_beta = splice_cdr3_into_scaffold(
        _DEFAULT_VREGION_BETA_SCAFFOLD, cdr3b, _CDR3_POS_BETA, "FGSG"
    )

    # 2. Use default TCRalpha V-region with default CDR3alpha
    vregion_alpha = splice_cdr3_into_scaffold(
        _DEFAULT_VREGION_ALPHA_SCAFFOLD, _DEFAULT_CDR3_ALPHA, 90, "FGKG"
    )

    # 3. Resolve HLA to full MHC sequences
    mhc_info = hla_lookup.get(hla)
    if mhc_info is None:
        return None
    else:
        mhc_full = mhc_info.get("M", "")
        b2m = mhc_info.get("N", "")

    if not mhc_full:
        return None

    # Clean sequences (remove non-standard AAs)
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    clean = lambda s: "".join(c for c in s if c in standard_aa)

    chains = [
        {"id": "B", "sequence": clean(vregion_beta)},
        {"id": "A", "sequence": clean(vregion_alpha)},
        {"id": "M", "sequence": clean(mhc_full)},
    ]
    if b2m:
        chains.append({"id": "N", "sequence": clean(b2m)})
    chains.append({"id": "P", "sequence": clean(peptide)})

    return chains


def extract_v34_features(
    predictor, chains: List[Dict], device: str
) -> Optional[Dict[str, torch.Tensor]]:
    """Run full tFold pipeline and extract V3.4 classifier features."""
    try:
        raw = fe.extract_features_from_chains(
            predictor, chains, device=device, chunk_size=16
        )
        structured = fe.extract_structured_features(raw)
        if structured is None:
            return None

        # Pad to fixed sizes (same as StructuredDataset._pad_features)
        MAX_CDR3, MAX_PEP = 25, 20

        def pad_2d(t, max_len):
            L, D = t.shape
            if L >= max_len:
                return t[:max_len]
            return torch.cat([t, torch.zeros(max_len - L, D)], dim=0)

        def pad_3d(t, max_r, max_c):
            R, C, D = t.shape
            out = torch.zeros(max_r, max_c, D)
            r, c = min(R, max_r), min(C, max_c)
            out[:r, :c, :] = t[:r, :c, :]
            return out

        Lb = structured["sfea_cdr3b"].shape[0]
        La = structured["sfea_cdr3a"].shape[0]
        Lp = structured["sfea_pep"].shape[0]

        return {
            "sfea_cdr3b": pad_2d(structured["sfea_cdr3b"], MAX_CDR3),
            "sfea_cdr3a": pad_2d(structured["sfea_cdr3a"], MAX_CDR3),
            "sfea_pep": pad_2d(structured["sfea_pep"], MAX_PEP),
            "ca_cdr3b": pad_2d(structured["ca_cdr3b"], MAX_CDR3),
            "ca_cdr3a": pad_2d(structured["ca_cdr3a"], MAX_CDR3),
            "ca_pep": pad_2d(structured["ca_pep"], MAX_PEP),
            "pfea_cdr3b_pep": pad_3d(structured["pfea_cdr3b_pep"], MAX_CDR3, MAX_PEP),
            "pfea_cdr3a_pep": pad_3d(structured["pfea_cdr3a_pep"], MAX_CDR3, MAX_PEP),
            "v33_feat": structured["v33_feat"],
            "len_cdr3b": min(Lb, MAX_CDR3),
            "len_cdr3a": min(La, MAX_CDR3),
            "len_pep": min(Lp, MAX_PEP),
        }
    except Exception:
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="tFold V3.4 feature extraction worker")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Load inputs
    with open(args.input) as f:
        requests = json.load(f)
    print(f"Processing {len(requests)} samples on {device}")

    # Load tFold predictor
    print("Loading tFold predictor...")
    predictor = fe.load_tfold_predictor(
        ppi_path=os.path.join(TFOLD_ROOT, "checkpoints", "esm_ppi_650m_tcr.pth"),
        trunk_path=os.path.join(TFOLD_ROOT, "checkpoints", "tfold_tcr_pmhc_trunk.pth"),
        device=device,
    )
    print("tFold predictor loaded")

    # Build HLA lookup using the IPD-IMGT/HLA database
    print("Building HLA lookup from IPD-IMGT/HLA database...")
    hla_seq_lookup = hla_mod.build_hla_lookup()
    print(f"HLA sequence lookup: {len(hla_seq_lookup)} entries")

    unique_hlas = list(set(r["hla"] for r in requests))
    hla_lookup = {}
    for hla in unique_hlas:
        try:
            mhc_chains = hla_mod.resolve_mhc_chains(hla, hla_seq_lookup)
            if mhc_chains:
                hla_lookup[hla] = mhc_chains
        except Exception:
            pass
    print(f"Resolved {len(hla_lookup)}/{len(unique_hlas)} HLA alleles")

    # Process each sample
    features_list = []
    errors_list = []

    for i, req in enumerate(requests):
        try:
            chains = build_chains_from_cdr3b(
                req["cdr3b"], req["peptide"], req["hla"], hla_lookup
            )
            if chains is None:
                features_list.append(None)
                errors_list.append("Failed to build chains (HLA resolution failed)")
                continue

            feats = extract_v34_features(predictor, chains, device)
            if feats is None:
                features_list.append(None)
                errors_list.append("Feature extraction failed")
            else:
                features_list.append(feats)
                errors_list.append(None)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(requests)}")
        except Exception as e:
            features_list.append(None)
            errors_list.append(str(e))

    # Save output
    output = {"features": features_list, "errors": errors_list}
    torch.save(output, args.output)
    n_ok = sum(1 for e in errors_list if e is None)
    print(f"Done: {n_ok}/{len(requests)} successful, saved to {args.output}")


if __name__ == "__main__":
    main()
