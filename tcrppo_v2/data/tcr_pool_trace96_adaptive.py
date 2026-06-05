"""
Adaptive band selection for trace96: DIRECT curriculum strategy.

Key idea: Sample from HARDER bands as performance IMPROVES (curriculum learning).
- When mean affinity ∈ [-4, -2], sample from < -4 (easier seeds for warmup)
- When mean affinity ∈ [-2, -1], sample from [-4, -2] (medium difficulty)
- When mean affinity ∈ [-1, 0], sample from [-2, -1] (harder seeds)
- When mean affinity ∈ [0, +∞], sample from [-1, 0] (hardest seeds)

This provides a natural curriculum that adjusts dynamically to model capability.
"""

import sys
import os

# Add tcrppo_v2 to path
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.data.tcr_pool import TCRPool
from collections import deque
from typing import Dict, List, Optional
import numpy as np


# Monkey-patch TCRPool with adaptive band selection
original_init = TCRPool.__init__

def patched_init(self, *args, **kwargs):
    """Enhanced init with recent affinity tracking for trace96."""
    original_init(self, *args, **kwargs)
    
    # Track recent final affinities per peptide
    self.recent_affinities: Dict[str, deque] = {}
    self.recent_affinity_window = kwargs.get('online_pool_recent_window', 50)
    
    # Cache for adaptive band selection
    self.cached_bands: Dict[str, dict] = {}
    self.band_update_counters: Dict[str, int] = {}
    self.band_update_interval = 20  # Update band every N episodes
    
    # Define adaptive sampling bands (DIRECT curriculum relationship)
    # When performance is WEAK (low mean A), sample from EASY seeds (high affinity)
    # When performance is STRONG (high mean A), sample from HARD seeds (low affinity)
    self.adaptive_bands = [
        # (mean_A_min, mean_A_max) -> (sample_min, sample_max)
        {"mean_min": 0.0, "mean_max": 10.0, "sample_min": -1.0, "sample_max": 0.0, "name": "expert→hardest"},
        {"mean_min": -1.0, "mean_max": 0.0, "sample_min": -2.0, "sample_max": -1.0, "name": "good→hard"},
        {"mean_min": -2.0, "mean_max": -1.0, "sample_min": -4.0, "sample_max": -2.0, "name": "medium→medium"},
        {"mean_min": -4.0, "mean_max": -2.0, "sample_min": -10.0, "sample_max": -4.0, "name": "weak→easy"},
        {"mean_min": -10.0, "mean_max": -4.0, "sample_min": -10.0, "sample_max": -2.0, "name": "veryWeak→veryEasy"},
    ]
    
    print(f"[Trace96 Adaptive Bands] Initialized with {len(self.adaptive_bands)} bands, window={self.recent_affinity_window}")

TCRPool.__init__ = patched_init


def record_episode_affinity(self, target: str, final_affinity: float):
    """Record a final affinity for adaptive band selection."""
    if target not in self.recent_affinities:
        self.recent_affinities[target] = deque(maxlen=self.recent_affinity_window)
        self.band_update_counters[target] = 0
    
    self.recent_affinities[target].append(float(final_affinity))
    self.band_update_counters[target] += 1
    
    # Update cached band every N episodes
    if self.band_update_counters[target] >= self.band_update_interval:
        old_band = self.cached_bands.get(target, {}).get("name", "none")
        self.cached_bands[target] = self._compute_adaptive_band(target)
        new_band = self.cached_bands[target]["name"]
        if old_band != new_band:
            mean_aff = np.mean(list(self.recent_affinities[target]))
            print(f"[Trace96 Band Update] {target}: mean_A={mean_aff:.2f} | {old_band} → {new_band}")
        self.band_update_counters[target] = 0

TCRPool.record_episode_affinity = record_episode_affinity


def _compute_adaptive_band(self, target: str) -> dict:
    """
    Compute the appropriate sampling band based on recent performance.
    DIRECT curriculum: better performance (higher A) → harder seeds (lower A).
    """
    if target not in self.recent_affinities or len(self.recent_affinities[target]) == 0:
        # Default: easiest band for new peptides (warmup)
        return self.adaptive_bands[-1]
    
    # Calculate mean of recent final affinities
    recent = list(self.recent_affinities[target])
    mean_affinity = np.mean(recent)
    
    # Find the appropriate sampling band based on mean performance
    for band in self.adaptive_bands:
        if band["mean_min"] <= mean_affinity < band["mean_max"]:
            return band
    
    # Fallback: if mean is very high, use hardest band
    if mean_affinity >= self.adaptive_bands[0]["mean_max"]:
        return self.adaptive_bands[0]
    
    # Fallback: if mean is very low, use easiest band
    return self.adaptive_bands[-1]

TCRPool._compute_adaptive_band = _compute_adaptive_band


def get_adaptive_band(self, target: str) -> Optional[dict]:
    """Get cached adaptive band for a target peptide."""
    if target not in self.cached_bands:
        # First time seeing this target, compute initial band
        self.cached_bands[target] = self._compute_adaptive_band(target)
    return self.cached_bands[target]

TCRPool.get_adaptive_band = get_adaptive_band


# Override sample_from_online_pool to use adaptive bands
original_sample = TCRPool.sample_from_online_pool

def adaptive_sample_from_online_pool(self, target: str, k: int = 1) -> List[str]:
    """Sample from online pool using adaptive bands if enabled."""
    # Check if adaptive bands are enabled
    if not getattr(self, 'adaptive_bands', None):
        return original_sample(self, target, k)
    
    band = self.get_adaptive_band(target)
    
    # Filter pool by band
    pool_key = (target, self.online_pool_max_per_target)
    if pool_key not in self.online_pool:
        return []
    
    pool_data = self.online_pool[pool_key]
    
    # Filter by band affinity range
    filtered = [
        (seq, aff, dec, ham) 
        for seq, aff, dec, ham in pool_data 
        if band["sample_min"] <= aff < band["sample_max"]
    ]
    
    if len(filtered) == 0:
        # Fallback to full pool if band is empty
        filtered = pool_data
    
    # Sample k sequences
    if len(filtered) <= k:
        return [seq for seq, _, _, _ in filtered]
    
    # Sample without replacement
    indices = np.random.choice(len(filtered), size=k, replace=False)
    return [filtered[i][0] for i in indices]

TCRPool.sample_from_online_pool = adaptive_sample_from_online_pool


print("[Trace96 Adaptive] TCRPool patched with adaptive curriculum bands")
