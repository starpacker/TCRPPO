"""MC Dropout uncertainty estimation for the ERGO TCR-pMHC binding model.

The ERGO `AutoencoderLSTMClassifier` (code/ERGO/ERGO_models.py) contains three
``nn.Dropout`` modules in the forward path:
  - 2 inside ``PaddingAutoencoder.encoder`` (lines 85, 88), p=0.1
  - 1 on the classifier MLP head (line 138), p=0.1

We re-enable these at inference time and run multiple stochastic forward passes
to estimate predictive uncertainty (Gal & Ghahramani 2016).

Complication: ``code/ERGO/ae_utils.predict`` calls ``model.eval()`` at the very
start of every call, which would silently disable our re-enabled dropout. We
work around this by monkey-patching ``ae_utils.predict`` with a variant that
omits the ``model.eval()`` line for the duration of the MC sampling, then we
restore the original.

Public API:
    enable_dropout(model)             — switch all nn.Dropout modules to train()
    disable_dropout(model)            — restore them to eval()
    mc_dropout_predict(reward, tcrs, peptides, n_samples=20) -> (mean, std)
        — main entry point used by eval_decoy.py
"""
import sys
import os
import contextlib

import numpy as np
import torch
import torch.nn as nn

# Make ae_utils importable. Mirrors the path setup in code/reward.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_ERGO_DIR = os.path.join(_REPO, "code", "ERGO")
if _ERGO_DIR not in sys.path:
    sys.path.insert(0, _ERGO_DIR)

import ae_utils as ae  # noqa: E402


def enable_dropout(model):
    """Set all ``nn.Dropout`` submodules to train() mode (other modules untouched)."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            n += 1
    return n


def disable_dropout(model):
    """Set all ``nn.Dropout`` submodules back to eval() mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()


def _predict_no_eval_reset(model, batches, device):
    """Re-implementation of ``ae_utils.predict`` that does NOT call model.eval().
    Optimized for speed: keeps tensors on GPU until the very end.
    """
    all_probs = []
    index = 0
    batch_size = 0
    pep_lens = None
    for batch in batches:
        tcrs, padded_peps, pep_lens, batch_signs = batch
        # Avoid redundant torch.tensor if already a tensor
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

    # Fast GPU concat
    preds = torch.cat(all_probs)
    
    # Mirror the trailing-batch trimming logic from ae_utils.predict
    if pep_lens is not None and len(pep_lens) > 0:
        border = pep_lens[-1]
        if not any(k != border for k in pep_lens[border:]):
            index -= batch_size - border
            preds = preds[:index]
            
    return preds.cpu().numpy().tolist()


@contextlib.contextmanager
def _patched_ae_predict():
    """Temporarily replace ``ae_utils.predict`` with the no-eval-reset variant."""
    orig = ae.predict
    ae.predict = _predict_no_eval_reset
    try:
        yield
    finally:
        ae.predict = orig


def mc_dropout_predict(reward_model, tcrs, peptides, n_samples=20):
    """Run MC Dropout on the ERGO model wrapped by ``reward_model``.

    Args:
        reward_model: an instance of ``reward.Reward`` (already loaded with
            an ERGO checkpoint).
        tcrs: list[str] of TCR CDR3β sequences.
        peptides: list[str] of peptide sequences (same length as ``tcrs``).
        n_samples: number of stochastic forward passes.

    Returns:
        (mean, std) — two ``np.ndarray`` of shape ``(len(tcrs),)``.

    Notes:
        - The ``reward_model.ergo_model`` is left in eval() mode after the call.
        - We deepcopy the input lists because some downstream code mutates them.
    """
    if len(tcrs) != len(peptides):
        raise ValueError(
            "tcrs and peptides must have the same length, got {} vs {}".format(
                len(tcrs), len(peptides)))
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    model = reward_model.ergo_model
    n_drop = enable_dropout(model)
    if n_drop == 0:
        # Defensive: model has no dropout layers; return deterministic preds
        # with std=0 instead of crashing.
        preds = np.asarray(reward_model.get_ergo_reward(list(tcrs), list(peptides)),
                           dtype=np.float64)
        return preds, np.zeros_like(preds)

    samples = []
    try:
        import time
        import reward
        with _patched_ae_predict():
            # Build dataset ONCE!
            tcrs_copy = list(tcrs)
            peps_copy = list(peptides)
            signs = [0] * len(tcrs_copy)
            batch_size = min(len(tcrs_copy), 4096)
            
            t0 = time.time()
            if "ae" in reward_model.ergo_model_file:
                test_batches = ae.get_full_batches(
                    tcrs_copy, peps_copy, signs, 
                    reward.tcr_atox, reward.pep_atox, 
                    batch_size, reward_model.max_len
                )
                
                # Pre-transfer to GPU for huge speedup in the 20x loop
                gpu_batches = []
                for b in test_batches:
                    t, p, l, s = b
                    if not isinstance(t, torch.Tensor): t = torch.tensor(t)
                    if not isinstance(p, torch.Tensor): p = torch.tensor(p)
                    if not isinstance(l, torch.Tensor): l = torch.tensor(l)
                    gpu_batches.append((t.to(reward.device), p.to(reward.device), l.to(reward.device), s))
                    
                t1 = time.time()
                for i_samp in range(n_samples):
                    t2 = time.time()
                    preds = ae.predict(model, gpu_batches, reward.device)
                    t3 = time.time()
                    samples.append(np.asarray(preds, dtype=np.float64))
            else:
                _tcrs, _peps = reward.lstm.convert_data(tcrs_copy, peps_copy, reward.amino_to_ix)    
                test_batches = reward.lstm.get_full_batches(_tcrs, _peps, signs, batch_size, reward.amino_to_ix)
                for i_samp in range(n_samples):
                    preds = reward.lstm.predict(model, test_batches, reward.device)
                    samples.append(np.asarray(preds, dtype=np.float64))
    finally:
        disable_dropout(model)

    stacked = np.stack(samples, axis=0)  # (n_samples, n_pairs)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return mean, std


def mc_dropout_predict_chunked(reward_model, tcrs, peptides, n_samples=20,
                                chunk_size=4096):
    """Same as ``mc_dropout_predict`` but processes (TCR, peptide) pairs in
    fixed-size chunks to avoid OOM on large inputs.

    The chunking happens BEFORE the MC sampling loop — each chunk gets its own
    full set of ``n_samples`` forward passes (this matches mathematical MC
    dropout: each prediction is independent).
    """
    n = len(tcrs)
    if n != len(peptides):
        raise ValueError("tcrs/peptides length mismatch")

    means = np.empty(n, dtype=np.float64)
    stds = np.empty(n, dtype=np.float64)
    
    import sys
    print(f"    Starting MC Dropout scoring for {n} pairs in chunks of {chunk_size}...")
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        m, s = mc_dropout_predict(
            reward_model, tcrs[start:end], peptides[start:end], n_samples=n_samples
        )
        means[start:end] = m
        stds[start:end] = s
        
        sys.stdout.write(f"\r    Processed {end}/{n} pairs...")
        sys.stdout.flush()
        
    print() # New line after progress bar
    return means, stds
