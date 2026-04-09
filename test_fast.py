import sys, os
sys.path.append("/share/liuyutian/TCRPPO/code/ERGO")
import ae_utils as ae
import time
import torch
import numpy as np

tcrs = ["CASSYVGNTGELFF"] * 32768
peps = ["GILGFVFTL"] * 32768
signs = [0] * 32768
tcr_atox = ae.tcr_atox if hasattr(ae, 'tcr_atox') else {amino: index for index, amino in enumerate([letter for letter in 'ARNDCEQGHILKMFPSTWYV'] + ['X'])}
pep_atox = {amino: index for index, amino in enumerate(['PAD'] + [letter for letter in 'ARNDCEQGHILKMFPSTWYV'])}

t0 = time.time()
print("Starting get_full_batches...")
test_batches = ae.get_full_batches(tcrs, peps, signs, tcr_atox, pep_atox, 4096, 28)
t1 = time.time()
print(f"Finished in {t1-t0:.2f}s")

print("Pre-transfer to GPU...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_batches = []
for b in test_batches:
    t, p, l, s = b
    if not isinstance(t, torch.Tensor): t = torch.tensor(t)
    if not isinstance(p, torch.Tensor): p = torch.tensor(p)
    if not isinstance(l, torch.Tensor): l = torch.tensor(l)
    gpu_batches.append((t.to(device), p.to(device), l.to(device), s))
print("Done transfer!")

from ERGO_models import AutoencoderLSTMClassifier
import torch

with open('/share/liuyutian/TCRPPO/code/ERGO/models/ae_mcpas1.pt', 'rb') as f:
    model_data = torch.load(f, map_location=device)
params = model_data['params'].copy()
params.pop('lr', None)
params.pop('weight_decay', None)
params.pop('wd', None)
model = AutoencoderLSTMClassifier(**params)
model.load_state_dict(model_data['model_state_dict'])
model.to(device)

def _predict_no_eval_reset(model, batches, device):
    all_probs = []
    index = 0
    batch_size = 0
    pep_lens = None
    for batch in batches:
        tcrs, padded_peps, pep_lens, batch_signs = batch
        if not isinstance(tcrs, torch.Tensor):
            tcrs = torch.tensor(tcrs)
        tcrs = tcrs.to(device, non_blocking=True)
        padded_peps = padded_peps.to(device, non_blocking=True)
        pep_lens = pep_lens.to(device, non_blocking=True)
        
        with torch.no_grad():
            probs = model(tcrs, padded_peps, pep_lens)
        all_probs.append(probs.detach().squeeze(-1))
        batch_size = len(tcrs)
        index += batch_size

    preds = torch.cat(all_probs)
    if pep_lens is not None and len(pep_lens) > 0:
        border = pep_lens[-1]
        if not any(k != border for k in pep_lens[border:]):
            index -= batch_size - border
            preds = preds[:index]
            
    return preds.cpu().numpy().tolist()

t2 = time.time()
print("Starting predict x20...")
for i in range(20):
    tA = time.time()
    _predict_no_eval_reset(model, gpu_batches, device)
    tB = time.time()
    print(f"Pass {i} took {tB-tA:.2f}s")
t3 = time.time()
print(f"Predict x20 finished in {t3-t2:.2f}s")
