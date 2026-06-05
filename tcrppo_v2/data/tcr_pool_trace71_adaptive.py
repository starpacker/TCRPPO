"""
Patch for TCRPool with ADAPTIVE band selection for trace71.

Key idea: Sample from LOWER bands when performance is good, to maintain challenge.
- If mean final affinity is high (e.g., -1.0), sample from lower bands (e.g., [-4,-2])
- This ensures the model always has room to improve and doesn't overfit to easy seeds.

Band selection strategy:
- Mean A >= -0.5  → sample from [-4, -2]  (hardest)
- Mean A in [-1.5, -0.5) → sample from [-3, -1.5]
- Mean A in [-2.5, -1.5) → sample from [-2.5, -1.0]
- Mean A < -2.5  → sample from [-2, 0]  (easier, for warmup)
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
    """Enhanced init with recent affinity tracking."""
    original_init(self, *args, **kwargs)
    
    # Track recent final affinities per peptide
    self.recent_affinities: Dict[str, deque] = {}
    self.recent_affinity_window = 50  # Last N episodes per peptide
    
    # Cache for adaptive band selection
    self.cached_bands: Dict[str, dict] = {}
    self.band_update_counters: Dict[str, int] = {}
    self.band_update_interval = 50  # Update band every N episodes
    
    # Define adaptive sampling bands (INVERSE relationship with performance)
    # When performance is GOOD (high mean A), sample from HARD seeds (low affinity)
    self.adaptive_bands = [
        # (mean_A_min, mean_A_max) -> (sample_min, sample_max)
        {"mean_min": -0.5, "mean_max": 10.0, "sample_min": -4.0, "sample_max": -2.0, "name": "expert→hard"},
        {"mean_min": -1.5, "mean_max": -0.5, "sample_min": -3.0, "sample_max": -1.5, "name": "good→medium"},
        {"mean_min": -2.5, "mean_max": -1.5, "sample_min": -2.5, "sample_max": -1.0, "name": "medium→easy"},
        {"mean_min": -10.0, "mean_max": -2.5, "sample_min": -2.0, "sample_max": 0.0, "name": "weak→warmup"},
    ]

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
        self.cached_bands[target] = self._compute_adaptive_band(target)
        self.band_update_counters[target] = 0

TCRPool.record_episode_affinity = record_episode_affinity


def _compute_adaptive_band(self, target: str) -> Optional[dict]:
    """
    Compute the appropriate sampling band based on recent performance.
    INVERSE relationship: better performance → sample harder seeds.
    """
    if target not in self.recent_affinities or len(self.recent_affinities[target]) == 0:
        # Default: warmup band for new peptides
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
    
    # Fallback: if mean is very low, use warmup band
    return self.adaptive_bands[-1]

TCRPool._compute_adaptive_band = _compute_adaptive_band


def get_adaptive_band(self, target: str) -> Optional[dict]:
    """Get the appropriate sampling band (cached, updated every N episodes)."""
    # Return cached band if available
    if target in self.cached_bands:
        return self.cached_bands[target]
    
    # Otherwise compute and cache
    band = self._compute_adaptive_band(target)
    self.cached_bands[target] = band
    return band

TCRPool.get_adaptive_band = get_adaptive_band


# Override _online_candidates to use adaptive band selection
original_online_candidates = TCRPool._online_candidates

def patched_online_candidates(self, target: str, step: int) -> List[dict]:
    """Return online seeds that match the adaptively selected band."""
    pool = self.online_seeds.get(target, [])
    if not pool:
        return []
    
    # Use adaptive band selection
    band = self.get_adaptive_band(target)
    if band is None:
        return pool
    
    sample_min = float(band.get("sample_min", -1e9))
    sample_max = float(band.get("sample_max", 1e9))
    
    candidates = [
        item for item in pool
        if sample_min <= float(item.get("affinity", -1e9)) < sample_max
    ]
    
    # If the current band is empty, fall back to all available seeds
    if not candidates:
        return pool
    
    return candidates

TCRPool._online_candidates = patched_online_candidates


print("✓ TCRPool patched with ADAPTIVE band selection for trace71")
print("  Strategy: Better performance → Sample harder seeds")
print("  Bands:")
print("    Mean A >= -0.5:  sample from [-4.0, -2.0]  (expert→hard)")
print("    Mean A in [-1.5, -0.5): sample from [-3.0, -1.5]  (good→medium)")
print("    Mean A in [-2.5, -1.5): sample from [-2.5, -1.0]  (medium→easy)")
print("    Mean A < -2.5:   sample from [-2.0, 0.0]   (weak→warmup)")
print("  Window: last 50 episodes per peptide")
