import sys, os
sys.path.append("/share/liuyutian/TCRPPO/code/ERGO")
import ae_utils as ae
import time
import torch

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
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from ERGO_models import AutoencoderLSTMClassifier
import pickle
with open('/share/liuyutian/TCRPPO/code/ERGO/models/ae_mcpas1.pt', 'rb') as f:
    model_data = pickle.load(f)
model = AutoencoderLSTMClassifier(**model_data['params'])
model.load_state_dict(model_data['state_dict'])
model.to(device)
t2 = time.time()
print("Starting predict x20...")
for _ in range(20):
    ae.predict(model, test_batches, device)
t3 = time.time()
print(f"Predict x20 finished in {t3-t2:.2f}s")
