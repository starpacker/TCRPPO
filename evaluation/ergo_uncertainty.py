"""MC Dropout uncertainty estimation for the ERGO TCR-pMHC binding model.

The ERGO `AutoencoderLSTMClassifier` (code/ERGO/ERGO_models.py) contains three
``nn.Dropout`` modules in the forward path:
  - 2 inside ``PaddingAutoencoder.encoder`` (lines 85, 88), p=0.1
  - 1 on the classifier MLP head (line 138), p=0.1

We re-enable these at inference time and run multiple stochastic forward passes
to estimate predictive uncertainty (Gal & Ghahramani 2016).

Two complications had to be worked around:

1.  ``code/ERGO/ae_utils.predict`` calls ``model.eval()`` at the very start of
    every call, which would silently disable our re-enabled dropout. We provide
    a re-implementation (``_predict_no_eval_reset``) that omits the ``model.eval()``
    line and is used for the duration of the MC sampling loop.

2.  ``ae_utils.predict``'s trailing-batch trim logic is fragile when the actual
    sample count differs from the padded batch size. We bypass it entirely by
    requesting exactly ``expected_n`` outputs back.

Performance: the original implementation rebuilt batches and shuttled tensors
CPU→GPU on every MC sample. We now build the batches once, push them to the
GPU once, and only repeat the forward pass inside the inner loop. On A800 80GB
this takes per-pair throughput from ~5/s to ~1900/s.

Public API:
    enable_dropout(model)             — switch all nn.Dropout modules to train()
    disable_dropout(model)            — restore them to eval()
    mc_dropout_predict(reward, tcrs, peptides, n_samples=20) -> (mean, std)
    mc_dropout_predict_chunked(...)   — same as above, but processes in chunks
"""
import sys
import os

import numpy as np
import torch
import torch.nn as nn

# Make ae_utils importable. Mirrors the path setup in code/reward.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_ERGO_DIR = os.path.join(_REPO, "code", "ERGO")
_CODE_DIR = os.path.join(_REPO, "code")
if _ERGO_DIR not in sys.path:
    sys.path.insert(0, _ERGO_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

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


def _predict_no_eval_reset(model, gpu_batches, device, expected_n=None):
    """Run forward passes on pre-built GPU batches without resetting eval mode.

    Differences from ``ae_utils.predict``:
      * Does NOT call ``model.eval()`` (which would silently disable our
        re-enabled dropout).
      * Assumes ``gpu_batches`` are already on the right device.
      * Uses ``torch.cat`` on the GPU and a single ``.cpu()`` at the end
        instead of ``preds.extend([t[0] for t in probs.cpu().data.tolist()])``
        per batch.
      * Bypasses the original trailing-batch trim logic in favour of an
        explicit ``expected_n`` truncation, which is robust to padding.
    """
    all_probs = []
    for batch in gpu_batches:
        tcrs, padded_peps, pep_lens, _signs = batch
        with torch.no_grad():
            probs = model(tcrs, padded_peps, pep_lens)
        all_probs.append(probs.detach().squeeze(-1))

    preds = torch.cat(all_probs).cpu().numpy().tolist()
    if expected_n is not None:
        return preds[:expected_n]
    return preds


def _build_gpu_batches(tcrs, peps, max_len, batch_size, device):
    """Tokenise + pad ``tcrs``/``peps`` and push the resulting tensors to ``device``.

    Returns the batches as a list of tuples ``(tcrs, padded_peps, pep_lens, signs)``
    where the first three elements live on ``device``.
    """
    import reward  # local import to avoid a circular import at module load time
    signs = [0] * len(tcrs)
    batches = ae.get_full_batches(
        tcrs, peps, signs, reward.tcr_atox, reward.pep_atox, batch_size, max_len)
    gpu_batches = []
    for batch in batches:
        t, p, l, s = batch
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p)
        if not isinstance(l, torch.Tensor):
            l = torch.tensor(l)
        gpu_batches.append((
            t.to(device, non_blocking=True),
            p.to(device, non_blocking=True),
            l.to(device, non_blocking=True),
            s,
        ))
    return gpu_batches


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
        - We always make a defensive copy of the input lists.
        - For the AE branch, batches are built once and reused across all
          ``n_samples`` forward passes (the slow part of MC dropout used to be
          batch construction + CPU→GPU transfer, not the forward pass itself).
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

    import reward  # local import; reward.py imports torch + sets up device
    tcrs_copy = list(tcrs)
    peps_copy = list(peptides)
    expected_n = len(tcrs_copy)
    batch_size = min(expected_n, 4096) if expected_n > 0 else 1

    samples = []
    try:
        if "ae" in reward_model.ergo_model_file:
            gpu_batches = _build_gpu_batches(
                tcrs_copy, peps_copy,
                max_len=reward_model.max_len,
                batch_size=batch_size,
                device=reward.device,
            )
            for _ in range(n_samples):
                preds = _predict_no_eval_reset(
                    model, gpu_batches, reward.device, expected_n=expected_n)
                samples.append(np.asarray(preds, dtype=np.float64))
        else:
            # LSTM branch — kept for completeness even though we don't ship
            # an LSTM ERGO checkpoint.
            _tcrs, _peps = reward.lstm.convert_data(
                tcrs_copy, peps_copy, reward.amino_to_ix)
            test_batches = reward.lstm.get_full_batches(
                _tcrs, _peps, [0] * expected_n, batch_size, reward.amino_to_ix)
            for _ in range(n_samples):
                preds = reward.lstm.predict(model, test_batches, reward.device)
                samples.append(np.asarray(preds[:expected_n], dtype=np.float64))
    finally:
        disable_dropout(model)

    stacked = np.stack(samples, axis=0)  # (n_samples, n_pairs)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return mean, std


def mc_dropout_predict_chunked(reward_model, tcrs, peptides, n_samples=20,
                                chunk_size=4096, verbose=True):
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

    if verbose:
        print("    MC Dropout: {} pairs in chunks of {}".format(n, chunk_size))

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        m, s = mc_dropout_predict(
            reward_model, tcrs[start:end], peptides[start:end], n_samples=n_samples
        )
        means[start:end] = m
        stds[start:end] = s

        if verbose:
            sys.stdout.write("\r    progress: {}/{} pairs".format(end, n))
            sys.stdout.flush()

    if verbose:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return means, stds
