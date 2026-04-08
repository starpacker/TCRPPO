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

    Identical body except the leading ``model.eval()`` line is removed so that
    dropout state set by ``enable_dropout`` is preserved across the call.
    """
    # NB: model.eval() intentionally NOT called here.
    preds = []
    index = 0
    batch_size = 0
    pep_lens = None
    for batch in batches:
        tcrs, padded_peps, pep_lens, batch_signs = batch
        tcrs = torch.tensor(tcrs).to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        with torch.no_grad():
            probs = model(tcrs, padded_peps, pep_lens)
        preds.extend([t[0] for t in probs.cpu().data.tolist()])
        batch_size = len(tcrs)
        index += batch_size

    # Mirror the trailing-batch trimming logic from ae_utils.predict
    if pep_lens is not None and len(pep_lens) > 0:
        border = pep_lens[-1]
        if not any(k != border for k in pep_lens[border:]):
            index -= batch_size - border
            preds = preds[:index]
    return preds


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
        with _patched_ae_predict():
            for _ in range(n_samples):
                preds = reward_model.get_ergo_reward(list(tcrs), list(peptides))
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
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        m, s = mc_dropout_predict(
            reward_model, tcrs[start:end], peptides[start:end], n_samples=n_samples
        )
        means[start:end] = m
        stds[start:end] = s
    return means, stds
