#!/usr/bin/env python3
"""Check perplexity and z-scores of CCC-ending TCRs from trace61/trace70."""

import json
import numpy as np
import torch
import esm

# Load stats
with open("data/cdr3_ppl_stats.json") as f:
    stats = json.load(f)
mean_ppl = stats["mean_ppl"]
std_ppl = stats["std_ppl"]

print(f"CDR3 PPL Stats: mean={mean_ppl:.4f}, std={std_ppl:.4f}")
print(f"Threshold z-score: 2.0")
print(f"Penalty triggers when |z| > 2.0, i.e., ppl > {mean_ppl + 2*std_ppl:.4f} or < {mean_ppl - 2*std_ppl:.4f}")
print()

# Load ESM-2
print("Loading ESM-2...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.cuda()
model.eval()
batch_converter = alphabet.get_batch_converter()

def compute_ppl(seq):
    """Compute pseudo-perplexity for a sequence."""
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.cuda()

    with torch.no_grad():
        output = model(tokens, repr_layers=[], return_contacts=False)
        logits = output["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)

        seq_len = len(seq)
        total_nll = 0.0
        for pos in range(1, seq_len + 1):
            true_token = tokens[0, pos]
            total_nll -= log_probs[0, pos, true_token].item()

        ppl = np.exp(total_nll / seq_len) if seq_len > 0 else float("inf")
    return ppl

def zscore(ppl):
    return (ppl - mean_ppl) / (std_ppl + 1e-8)

def penalty(z):
    excess = abs(z) - 2.0
    if excess <= 0:
        return 0.0
    return -excess

# Test sequences from trace61 logs
test_seqs = [
    # CCC endings
    "CASSIDHNSGNAQYFCCC",
    "CSAHDLAPRAQAEQFCCC",
    "CAIRASGGAEEQFYYCCC",
    "CSARPAPILYNEQYFCCC",
    "CSDELYRGGTDAQYYCCC",
    # CCCC endings
    "CASSLGAGIMDQYYCCCC",
    # YYYCCC ending
    "CSVYLARHEETAYYYCCC",
    # Normal endings (for comparison)
    "CASSLGAGIMDEQYF",
    "CSVEDYRGGTDTQYF",
    "CASSPDPEDTNTGELFF",
    "CASRASGGEEQFF",
    # Extreme test cases
    "CASSCCCCCCCCCQYF",  # Many consecutive C
    "CASSAAAAAAAAQYF",   # Many consecutive A
    "CASSGGGGGGGGQYF",   # Many consecutive G
]

print(f"{'Sequence':<25} {'PPL':>8} {'Z-score':>8} {'Penalty':>8} {'Triggers?'}")
print("-" * 70)

for seq in test_seqs:
    ppl = compute_ppl(seq)
    z = zscore(ppl)
    pen = penalty(z)
    triggers = "YES" if pen < 0 else "NO"
    print(f"{seq:<25} {ppl:>8.4f} {z:>8.4f} {pen:>8.4f} {triggers}")
