#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test tFold AMP wrapper integration with tcrppo_v2 on a specific GPU.

This script verifies:
1. TFoldAMPWrapper can be imported and initialized
2. Feature extraction works correctly
3. AMP provides expected speedup (target: 8.54×)
"""

import os
import sys
import time
from pathlib import Path

import torch

# Set GPU before importing anything
os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # Use GPU 4 to avoid conflict with test51c

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tcrppo_v2.inference_optimization.tfold_amp_wrapper import TFoldAMPWrapper


def test_amp_wrapper():
    """Test tFold AMP wrapper with a sample TCR-pMHC complex."""

    # Sample TCR-pMHC complex (GILGFVFTL from test51c)
    chains = [
        {
            "id": "B",
            "sequence": "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSLAPGTTNEKLFFGSGTQLSVLEDLNKVFPPEVAVFEPSEAEISHTQKATLVCLATGFYPDHVELSWWVNGKEVHSGVCTDPQPLKEQPALNDSRYALSSRLRVSATFWQNPRNHFRCQVQFYGLSENDEWTQDRAKPVTQIVSAEAWGRADCGFTSESYQQGVLSATILYEILLGKATLYAVLVSALVLMAMVKRKDF"
        },
        {
            "id": "A",
            "sequence": "AQKVTQAQPSVSVSPGQTARITCSGDALPGQSIYWYQQALGQGPQFIFQYYAKESDSDMRGGISGLTVDLKNIQPEDSGLYQCAASRDSSGNTGKLVFGKGTKLTVNPNIQNPDPAVYQLRDSKSSDKSVCLFTDFDSQTNVSQSKDSDVYITDKTVLDMRSMDFKSNSAVAWSNKSDFACANAFNNSIIPEDTFFPSPESSCDVKLVEKSFETDTNLNFQNLSVIGFRILLLKVAGFNLLMTLRLWSS"
        },
        {
            "id": "P",
            "sequence": "GILGFVFTL"
        },
        {
            "id": "M",
            "sequence": "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWE"
        },
        {
            "id": "N",
            "sequence": "GPHSLRYFVTAVSRPGLGEPRYMEVGYVDDTEFVRFDSDAENPRYEPRARWMEQEGPEYWERETQKAKGQEQWFRVSLRNLLGYYNQSAGGSHTLQQMSGCDLGPDGRLLRGHDQYAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGECVEWLRRYLKNGNATLLRTDSPKAHVTHHSRPEDKVTLRCWALGFYPADITLTWQLNGEELIQDMELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPVTLRWE"
        },
    ]

    print("=" * 70)
    print("tFold AMP Wrapper Integration Test")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    print()

    # Test 1: Without AMP (FP32 baseline)
    print("[1/2] Testing WITHOUT AMP (FP32 baseline)")
    print("-" * 70)

    wrapper_fp32 = TFoldAMPWrapper(device="cuda", use_amp=False)

    # Warmup
    print("  Warmup...")
    _ = wrapper_fp32.extract_features(chains)

    # Benchmark
    times_fp32 = []
    n_runs = 5
    for i in range(n_runs):
        start = time.time()
        features_fp32 = wrapper_fp32.extract_features(chains)
        elapsed = time.time() - start
        times_fp32.append(elapsed)
        print(f"  Run {i+1}/{n_runs}: {elapsed:.3f}s")

    mean_fp32 = sum(times_fp32) / len(times_fp32)
    print(f"  Mean: {mean_fp32:.3f}s")

    if features_fp32:
        print(f"\n  Features extracted:")
        print(f"    raw_sfea: {features_fp32['raw_sfea'].shape}")
        print(f"    ca_coords: {features_fp32['ca_coords'].shape}")
        if features_fp32['pfea_cdr3b_pep'] is not None:
            print(f"    pfea_cdr3b_pep: {features_fp32['pfea_cdr3b_pep'].shape}")
        if features_fp32['pfea_cdr3a_pep'] is not None:
            print(f"    pfea_cdr3a_pep: {features_fp32['pfea_cdr3a_pep'].shape}")

    print()

    # Test 2: With AMP
    print("[2/2] Testing WITH AMP (FP16/BF16)")
    print("-" * 70)

    wrapper_amp = TFoldAMPWrapper(device="cuda", use_amp=True)

    # Warmup
    print("  Warmup...")
    _ = wrapper_amp.extract_features(chains)

    # Benchmark
    times_amp = []
    for i in range(n_runs):
        start = time.time()
        features_amp = wrapper_amp.extract_features(chains)
        elapsed = time.time() - start
        times_amp.append(elapsed)
        print(f"  Run {i+1}/{n_runs}: {elapsed:.3f}s")

    mean_amp = sum(times_amp) / len(times_amp)
    print(f"  Mean: {mean_amp:.3f}s")

    if features_amp:
        print(f"\n  Features extracted:")
        print(f"    raw_sfea: {features_amp['raw_sfea'].shape}")
        print(f"    ca_coords: {features_amp['ca_coords'].shape}")
        if features_amp['pfea_cdr3b_pep'] is not None:
            print(f"    pfea_cdr3b_pep: {features_amp['pfea_cdr3b_pep'].shape}")
        if features_amp['pfea_cdr3a_pep'] is not None:
            print(f"    pfea_cdr3a_pep: {features_amp['pfea_cdr3a_pep'].shape}")

    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"FP32 (baseline):  {mean_fp32:.3f}s")
    print(f"AMP (optimized):  {mean_amp:.3f}s")
    print(f"Speedup:          {mean_fp32/mean_amp:.2f}×")
    print(f"Target speedup:   8.54×")
    print()

    if mean_fp32 / mean_amp >= 5.0:
        print("✓ SUCCESS: Achieved significant speedup (>5×)")
        status = "SUCCESS"
    elif mean_fp32 / mean_amp >= 2.0:
        print("⚠ PARTIAL: Achieved moderate speedup (2-5×)")
        status = "PARTIAL"
    else:
        print("✗ FAILED: Speedup below 2×")
        status = "FAILED"

    print("=" * 70)

    return {
        "fp32_mean": mean_fp32,
        "amp_mean": mean_amp,
        "speedup": mean_fp32 / mean_amp,
        "status": status
    }


if __name__ == "__main__":
    results = test_amp_wrapper()
