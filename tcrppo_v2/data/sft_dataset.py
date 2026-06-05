"""
SFT Dataset with stratified sampling.

Loads trajectories and provides batches with balanced affinity bins.
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Tuple
import random

OP_NAME_TO_IDX = {'SUB': 0, 'INS': 1, 'DEL': 2, 'STOP': 3}


class SFTDataset(Dataset):
    """Dataset of edit trajectories for supervised fine-tuning."""

    def __init__(self, trajectories_path: str, max_steps: int = 8):
        self.max_steps = max_steps
        self.trajectories = self._load_trajectories(trajectories_path)
        self.affinity_bins = self._create_bins()

    @property
    def bins(self):
        return self.affinity_bins

    def _load_trajectories(self, path: str) -> List[Dict]:
        """Load trajectories from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return data['trajectories']

    def _get_affinity(self, traj: Dict) -> float:
        """Get affinity from trajectory (handles both field names)."""
        return traj.get('final_affinity', traj.get('affinity', -10.0))

    def _normalize_actions(self, actions: List[Dict]) -> List[Dict]:
        """Normalize action format to {op_type: int, position: int, token: str}."""
        normalized = []
        for a in actions:
            if 'op_type' in a:
                normalized.append(a)
            else:
                op_str = a.get('op', 'STOP')
                normalized.append({
                    'op_type': OP_NAME_TO_IDX.get(op_str, 3),
                    'position': a.get('pos', a.get('position', 0)),
                    'token': a.get('token', None),
                })
        return normalized

    def _create_bins(self) -> Dict[str, List[int]]:
        """Create affinity bins for stratified sampling."""
        bins = {'high': [], 'medium': [], 'low': []}

        for idx, traj in enumerate(self.trajectories):
            A = self._get_affinity(traj)
            if A >= 0:
                bins['high'].append(idx)
            elif A >= -2:
                bins['medium'].append(idx)
            elif A >= -4:
                bins['low'].append(idx)

        return bins

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict:
        traj = self.trajectories[idx]

        A = self._get_affinity(traj)
        if A >= 0:
            bin_label = 'high'
        elif A >= -2:
            bin_label = 'medium'
        else:
            bin_label = 'low'

        return {
            'init_tcr': traj['init_tcr'],
            'final_tcr': traj['final_tcr'],
            'peptide': traj['peptide'],
            'actions': self._normalize_actions(traj['actions']),
            'final_affinity': A,
            'bin': bin_label,
        }

    def get_bin_indices(self, bin_name: str) -> List[int]:
        """Get indices for a specific bin."""
        return self.bins[bin_name]

    def print_stats(self):
        """Print dataset statistics."""
        print(f"Total trajectories: {len(self.trajectories)}")
        print(f"Bin sizes:")
        print(f"  High (A≥0):      {len(self.bins['high'])}")
        print(f"  Medium (-2≤A<0): {len(self.bins['medium'])}")
        print(f"  Low (-4≤A<-2):   {len(self.bins['low'])}")


class StratifiedBatchSampler(Sampler):
    """Sampler that balances across non-empty affinity bins."""

    def __init__(self, dataset: SFTDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Only use non-empty bins
        all_bins = {
            'high': dataset.get_bin_indices('high'),
            'medium': dataset.get_bin_indices('medium'),
            'low': dataset.get_bin_indices('low')
        }
        self.bins = {k: v for k, v in all_bins.items() if len(v) > 0}
        self.bin_names = list(self.bins.keys())
        n_bins = len(self.bin_names)

        if n_bins == 0:
            self.samples_per_bin = 0
            self.num_batches = 0
            return

        self.samples_per_bin = batch_size // n_bins
        min_bin_size = min(len(indices) for indices in self.bins.values())
        self.num_batches = max(1, min_bin_size // self.samples_per_bin)

    def __iter__(self):
        """Generate batch indices with stratified sampling."""
        # Shuffle bins if needed
        bin_indices = {}
        for bin_name, indices in self.bins.items():
            if self.shuffle:
                shuffled = indices.copy()
                random.shuffle(shuffled)
                bin_indices[bin_name] = shuffled
            else:
                bin_indices[bin_name] = indices

        # Generate batches
        for batch_idx in range(self.num_batches):
            batch = []

            # Sample from each bin
            for bin_name in self.bin_names:
                start = batch_idx * self.samples_per_bin
                end = start + self.samples_per_bin
                batch.extend(bin_indices[bin_name][start:end])

            # Shuffle within batch
            if self.shuffle:
                random.shuffle(batch)

            yield batch

    def __len__(self):
        return self.num_batches


def collate_sft_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for SFT DataLoader.

    Returns:
        {
            'init_tcrs': List[str],
            'final_tcrs': List[str],
            'peptides': List[str],
            'actions': List[List[Dict]],  # Batch of action sequences
            'final_affinities': torch.Tensor,
            'bins': List[str]
        }
    """
    return {
        'init_tcrs': [item['init_tcr'] for item in batch],
        'final_tcrs': [item['final_tcr'] for item in batch],
        'peptides': [item['peptide'] for item in batch],
        'actions': [item['actions'] for item in batch],
        'final_affinities': torch.tensor([item['final_affinity'] for item in batch], dtype=torch.float32),
        'bins': [item['bin'] for item in batch]
    }


if __name__ == "__main__":
    # Test dataset
    dataset = SFTDataset("/share/liuyutian/tcrppo_v2/data/sft_trajectories.json")
    dataset.print_stats()

    # Test sampler
    sampler = StratifiedBatchSampler(dataset, batch_size=64, shuffle=True)
    print(f"\nNumber of batches: {len(sampler)}")

    # Test first batch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_sft_batch)

    for batch in loader:
        print(f"\nFirst batch:")
        print(f"  Batch size: {len(batch['init_tcrs'])}")
        print(f"  Bins: {batch['bins'][:10]}...")
        print(f"  Affinity range: [{batch['final_affinities'].min():.2f}, {batch['final_affinities'].max():.2f}]")
        break
