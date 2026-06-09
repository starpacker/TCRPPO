#!/usr/bin/env python3
"""Quick test: is tFold feature extraction deterministic?"""
import sys, os, logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
logging.disable(logging.CRITICAL)

sys.path.insert(0, '/share/liuyutian/tfold')
import torch
from tfold.deploy import TCRpMHCPredictor
import tfold.model

ppi_path = tfold.model.esm_ppi_650m_tcr()
trunk_path = tfold.model.tfold_tcr_pmhc_trunk()
model = TCRpMHCPredictor(str(ppi_path), str(trunk_path))
model.to('cuda')
model.eval()

for af2_mod in [
    model.model.ligand_model.net['af2_smod'],
    model.model.receptor_model.net['af2_smod'],
    model.model.docking_model.net['af2_smod'],
]:
    af2_mod.tmsc_pred = False
    if 'ptm' in af2_mod.net:
        del af2_mod.net['ptm']

from tfold.deploy.utils.chain_utils import build_chain
chains = [
    build_chain('B', 'CASSIRSSYEQYF'),
    build_chain('P', 'GILGFVFTL'),
]

print('=== tFold Determinism Test ===', flush=True)
print('Running same input 5 times through model...', flush=True)

results = []
for i in range(5):
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        inputs, outputs = model.infer(chains, chunk_size=64)
    sfea = outputs['sfea_lig'].cpu().float()
    results.append(sfea)
    print(f'  Run {i+1}: sfea_sum={sfea.sum().item():.6f}  sfea_max={sfea.max().item():.6f}', flush=True)

print(flush=True)
print('Pairwise differences:', flush=True)
for i in range(len(results)):
    for j in range(i+1, len(results)):
        diff = (results[i] - results[j]).abs()
        print(f'  Run{i+1} vs Run{j+1}: max_abs={diff.max().item():.8f}  mean_abs={diff.mean().item():.8f}', flush=True)

# Verdict
max_diff = max(
    (results[i] - results[j]).abs().max().item()
    for i in range(len(results)) for j in range(i+1, len(results))
)
print(flush=True)
if max_diff < 1e-6:
    print(f'VERDICT: DETERMINISTIC (max diff = {max_diff:.2e})', flush=True)
elif max_diff < 1e-3:
    print(f'VERDICT: NEAR-DETERMINISTIC (max diff = {max_diff:.2e}, likely bf16 rounding)', flush=True)
else:
    print(f'VERDICT: NON-DETERMINISTIC (max diff = {max_diff:.2e})', flush=True)
