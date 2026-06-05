"""
Patch for TCRPool to support dynamic affinity-based band selection for trace61.

This adds:
1. Per-peptide recent affinity tracking (last N episodes)
2. Dynamic band selection based on recent performance
3. 4-band system: [-4,-2], [-2,-1], [-1,0], [0,0.6]
"""

import sys
import os
import json

# Add tcrppo_v2 to path
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.data.tcr_pool import TCRPool
from collections import deque
from typing import Dict, List, Optional
import numpy as np


# Monkey-patch TCRPool with dynamic band selection
original_init = TCRPool.__init__

def patched_init(self, *args, **kwargs):
    """Enhanced init with recent affinity tracking."""
    original_init(self, *args, **kwargs)
    
    # Track recent final affinities per peptide (for dynamic band selection)
    self.recent_affinities: Dict[str, deque] = {}
    self.recent_affinity_window = 50  # Last N episodes per peptide
    
    # Cache for dynamic band selection (update every 50 episodes)
    self.cached_bands: Dict[str, dict] = {}
    self.band_update_counters: Dict[str, int] = {}
    self.band_update_interval = 50  # Update band every N episodes
    
    # Define 4 affinity bands
    self.affinity_bands = [
        {"min": -4.0, "max": -2.0, "name": "band1"},
        {"min": -2.0, "max": -1.0, "name": "band2"},
        {"min": -1.0, "max": 0.0, "name": "band3"},
        {"min": 0.0, "max": 0.6, "name": "band4"},
    ]

    snapshot_path = getattr(self, "online_pool_snapshot_path", None)
    dynamic_mode = getattr(self, "online_pool_dynamic_mode", "band")
    if snapshot_path and os.path.exists(snapshot_path):
        try:
            with open(snapshot_path) as handle:
                snapshot = json.load(handle)
            for target, payload in snapshot.get("targets", {}).items():
                entries = payload.get("entries", [])
                affinities = [
                    float(item.get("affinity", -1e9))
                    for item in entries
                    if item.get("affinity") is not None
                ]
                if dynamic_mode == "below_mean" and affinities:
                    # Seed the recent-performance estimate from the better half
                    # of the prior online pool, so resumed runs immediately
                    # sample slightly below the current frontier.
                    affinities = sorted(affinities, reverse=True)
                    frontier = affinities[: max(1, len(affinities) // 2)]
                    self.recent_affinities[target] = deque(
                        frontier[: self.recent_affinity_window],
                        maxlen=self.recent_affinity_window,
                    )
                    self.band_update_counters[target] = 0
                    self.cached_bands[target] = self._compute_band(target)
                elif dynamic_mode != "below_mean":
                    cached_band = payload.get("cached_dynamic_band")
                    if cached_band:
                        self.cached_bands[target] = cached_band
        except Exception:
            pass

TCRPool.__init__ = patched_init


def record_episode_affinity(self, target: str, final_affinity: float):
    """Record a final affinity for dynamic band selection."""
    if target not in self.recent_affinities:
        self.recent_affinities[target] = deque(maxlen=self.recent_affinity_window)
        self.band_update_counters[target] = 0
    
    self.recent_affinities[target].append(float(final_affinity))
    self.band_update_counters[target] += 1
    
    # Update cached band every N episodes
    if self.band_update_counters[target] >= self.band_update_interval:
        self.cached_bands[target] = self._compute_band(target)
        self.band_update_counters[target] = 0

TCRPool.record_episode_affinity = record_episode_affinity


def _compute_band(self, target: str) -> Optional[dict]:
    """Compute the appropriate band based on recent performance."""
    if target not in self.recent_affinities or len(self.recent_affinities[target]) == 0:
        return self.affinity_bands[0]
    
    # Calculate mean of recent affinities
    recent = list(self.recent_affinities[target])
    mean_affinity = np.mean(recent)

    if getattr(self, "online_pool_dynamic_mode", "band") == "below_mean":
        below = float(getattr(self, "online_pool_dynamic_below", 0.7))
        above = float(getattr(self, "online_pool_dynamic_above", 0.0))
        lo = max(-4.0, mean_affinity - below)
        hi = min(0.6, mean_affinity + above)
        if hi <= lo:
            hi = min(0.6, lo + 0.2)
        return {"min": lo, "max": hi, "name": "below_mean"}
    
    # Find the band that contains this mean affinity
    for band in self.affinity_bands:
        if band["min"] <= mean_affinity < band["max"]:
            return band
    
    # If above all bands, use highest band
    if mean_affinity >= self.affinity_bands[-1]["max"]:
        return self.affinity_bands[-1]
    
    # If below all bands, use lowest band
    return self.affinity_bands[0]

TCRPool._compute_band = _compute_band


def get_dynamic_band(self, target: str) -> Optional[dict]:
    """Get the appropriate band (cached, updated every N episodes)."""
    # Return cached band if available
    if target in self.cached_bands:
        return self.cached_bands[target]
    
    # Otherwise compute and cache
    band = self._compute_band(target)
    self.cached_bands[target] = band
    return band

TCRPool.get_dynamic_band = get_dynamic_band


# Override _online_candidates to use dynamic band selection
original_online_candidates = TCRPool._online_candidates

def patched_online_candidates(self, target: str, step: int) -> List[dict]:
    """Return online seeds that match the dynamically selected band."""
    pool = self.online_seeds.get(target, [])
    if not pool:
        return []

    elite_ratio = float(getattr(self, "online_pool_elite_ratio", 0.0))
    elite_min_affinity = float(getattr(self, "online_pool_elite_min_affinity", -0.5))
    if elite_ratio > 0.0 and self.rng.random() < elite_ratio:
        elite = [
            item for item in pool
            if float(item.get("affinity", -1e9)) >= elite_min_affinity
        ]
        if elite:
            return elite
    
    # Use dynamic band selection instead of step-based bands
    band = self.get_dynamic_band(target)
    if band is None:
        return pool
    
    min_aff = float(band.get("min", -1e9))
    max_aff = float(band.get("max", 1e9))
    candidates = [
        item for item in pool
        if min_aff <= float(item.get("affinity", -1e9)) < max_aff
    ]
    
    # If the current band is empty, fall back to the closest lower band
    if candidates:
        return candidates
    
    below = [
        item for item in pool
        if float(item.get("affinity", -1e9)) < min_aff
    ]
    if below:
        return below
    return pool

TCRPool._online_candidates = patched_online_candidates


print("✓ TCRPool patched with dynamic band selection for trace61")
print("  Bands: [-4,-2], [-2,-1], [-1,0], [0,0.6]")
print("  Window: last 50 episodes per peptide")
