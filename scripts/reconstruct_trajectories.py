#!/usr/bin/env python3
"""
Reconstruct edit trajectories from TCR pairs.

Converts (InitTCR, FinalTCR) pairs into sequences of edit actions:
- SUB (substitute), INS (insert), DEL (delete), STOP

Uses two strategies:
1. Shortest path (Levenshtein) for high/low quality
2. Random paths for medium quality (data augmentation)
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict
from enum import IntEnum
import argparse


class OpType(IntEnum):
    """Edit operation types."""
    SUB = 0
    INS = 1
    DEL = 2
    STOP = 3


@dataclass
class Action:
    """Single edit action."""
    op_type: int  # OpType
    position: int
    token: str  # AA character or empty for DEL/STOP


@dataclass
class Trajectory:
    """Complete edit trajectory."""
    init_tcr: str
    final_tcr: str
    peptide: str
    hla: str
    init_affinity: float
    final_affinity: float
    delta_affinity: float
    actions: List[Dict]  # List of {op_type, position, token}
    source_log: str
    episode_id: int
    reconstruction_method: str  # 'shortest' or 'random'


class LevenshteinReconstructor:
    """Reconstruct shortest edit path using dynamic programming."""

    AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self):
        pass

    def reconstruct(self, s1: str, s2: str, max_steps: int = 8) -> List[Action]:
        """
        Compute shortest edit path from s1 to s2.

        Returns:
            List of Actions (excluding final STOP)
        """
        # Compute edit distance matrix
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # delete
                        dp[i][j - 1],      # insert
                        dp[i - 1][j - 1]   # substitute
                    )

        # Backtrack to get actions
        actions = []
        i, j = m, n
        current = s1

        while i > 0 or j > 0:
            if len(actions) >= max_steps:
                break

            if i == 0:
                # Only insertions left
                actions.append(Action(OpType.INS, j - 1, s2[j - 1]))
                j -= 1
            elif j == 0:
                # Only deletions left
                actions.append(Action(OpType.DEL, i - 1, ""))
                i -= 1
            elif s1[i - 1] == s2[j - 1]:
                # Match, no action needed
                i -= 1
                j -= 1
            else:
                # Choose operation with minimum cost
                costs = [
                    (dp[i - 1][j], 'DEL', i - 1, ""),
                    (dp[i][j - 1], 'INS', j - 1, s2[j - 1]),
                    (dp[i - 1][j - 1], 'SUB', i - 1, s2[j - 1])
                ]
                min_cost, op, pos, token = min(costs, key=lambda x: x[0])

                if op == 'DEL':
                    actions.append(Action(OpType.DEL, pos, ""))
                    i -= 1
                elif op == 'INS':
                    actions.append(Action(OpType.INS, pos, token))
                    j -= 1
                else:  # SUB
                    actions.append(Action(OpType.SUB, pos, token))
                    i -= 1
                    j -= 1

        # Reverse (we backtracked)
        actions.reverse()

        return actions


class RandomPathReconstructor:
    """Generate random valid edit paths for data augmentation."""

    AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps

    def reconstruct(self, s1: str, s2: str) -> List[Action]:
        """
        Generate a random valid path from s1 to s2.

        Strategy:
        1. Start with s1
        2. Apply random edits that move toward s2
        3. Stop when we reach s2 or hit max_steps
        """
        actions = []
        current = s1
        target = s2

        for step in range(self.max_steps):
            if current == target:
                break

            # Find differences
            diffs = self._find_differences(current, target)
            if not diffs:
                break

            # Pick random difference to fix
            diff_type, pos = random.choice(diffs)

            if diff_type == 'mismatch':
                # Substitute
                if pos < len(target):
                    action = Action(OpType.SUB, pos, target[pos])
                    current = current[:pos] + target[pos] + current[pos + 1:]
                    actions.append(action)

            elif diff_type == 'extra':
                # Delete
                if pos < len(current):
                    action = Action(OpType.DEL, pos, "")
                    current = current[:pos] + current[pos + 1:]
                    actions.append(action)

            elif diff_type == 'missing':
                # Insert
                if pos < len(target):
                    action = Action(OpType.INS, pos, target[pos])
                    current = current[:pos] + target[pos] + current[pos:]
                    actions.append(action)

        return actions

    def _find_differences(self, current: str, target: str) -> List[Tuple[str, int]]:
        """Find all positions where current differs from target."""
        diffs = []

        # Simple approach: find mismatches, extras, and missing
        min_len = min(len(current), len(target))

        # Check mismatches in overlapping region
        for i in range(min_len):
            if current[i] != target[i]:
                diffs.append(('mismatch', i))

        # Extra characters in current
        if len(current) > len(target):
            for i in range(len(target), len(current)):
                diffs.append(('extra', i))

        # Missing characters (need to insert)
        if len(target) > len(current):
            for j in range(len(current), len(target)):
                diffs.append(('missing', j))

        return diffs


class TrajectoryBuilder:
    """Build trajectories from TCR pairs."""

    def __init__(self, max_steps: int = 8):
        self.shortest_reconstructor = LevenshteinReconstructor()
        self.random_reconstructor = RandomPathReconstructor(max_steps)
        self.max_steps = max_steps

    def build_trajectories(self, pairs: List[Dict], augment_medium: bool = True) -> List[Trajectory]:
        """
        Build trajectories from pairs with augmentation strategy:
        - High (A≥0): 1x shortest path
        - Medium (-2≤A<0): 2x random paths (augmentation)
        - Low (-4≤A<-2): 1x shortest path
        """
        trajectories = []

        for pair in pairs:
            A = pair['final_affinity']

            if A >= 0:
                # High quality: shortest path only
                traj = self._build_single(pair, method='shortest')
                if traj:
                    trajectories.append(traj)

            elif A >= -2:
                # Medium quality: 2x random paths for augmentation
                if augment_medium:
                    for _ in range(2):
                        traj = self._build_single(pair, method='random')
                        if traj:
                            trajectories.append(traj)
                else:
                    traj = self._build_single(pair, method='shortest')
                    if traj:
                        trajectories.append(traj)

            elif A >= -4:
                # Low quality: shortest path only
                traj = self._build_single(pair, method='shortest')
                if traj:
                    trajectories.append(traj)

        return trajectories

    def _build_single(self, pair: Dict, method: str) -> Trajectory:
        """Build single trajectory."""
        s1 = pair['init_tcr']
        s2 = pair['final_tcr']

        # Reconstruct actions
        if method == 'shortest':
            actions = self.shortest_reconstructor.reconstruct(s1, s2, self.max_steps)
        else:  # random
            actions = self.random_reconstructor.reconstruct(s1, s2)

        # Verify trajectory is valid
        if not self._verify_trajectory(s1, s2, actions):
            return None

        # Convert actions to dict format
        action_dicts = [
            {'op_type': int(a.op_type), 'position': a.position, 'token': a.token}
            for a in actions
        ]

        return Trajectory(
            init_tcr=pair['init_tcr'],
            final_tcr=pair['final_tcr'],
            peptide=pair['peptide'],
            hla=pair['hla'],
            init_affinity=pair['init_affinity'],
            final_affinity=pair['final_affinity'],
            delta_affinity=pair['delta_affinity'],
            actions=action_dicts,
            source_log=pair['source_log'],
            episode_id=pair['episode_id'],
            reconstruction_method=method
        )

    def _verify_trajectory(self, s1: str, s2: str, actions: List[Action]) -> bool:
        """Verify that applying actions to s1 produces s2."""
        current = s1

        for action in actions:
            if action.op_type == OpType.SUB:
                if action.position >= len(current):
                    return False
                current = current[:action.position] + action.token + current[action.position + 1:]
            elif action.op_type == OpType.INS:
                if action.position > len(current):
                    return False
                current = current[:action.position] + action.token + current[action.position:]
            elif action.op_type == OpType.DEL:
                if action.position >= len(current):
                    return False
                current = current[:action.position] + current[action.position + 1:]

        return current == s2


def main():
    parser = argparse.ArgumentParser(description="Reconstruct edit trajectories")
    parser.add_argument("--input", type=str, default="/share/liuyutian/tcrppo_v2/data/sft_raw_pairs.json",
                        help="Input pairs JSON")
    parser.add_argument("--output", type=str, default="/share/liuyutian/tcrppo_v2/data/sft_trajectories.json",
                        help="Output trajectories JSON")
    parser.add_argument("--max_steps", type=int, default=8,
                        help="Maximum edit steps per trajectory")
    parser.add_argument("--augment_medium", action="store_true", default=True,
                        help="Augment medium-quality pairs with random paths")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load pairs
    print(f"Loading pairs from {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)
    pairs = data['pairs']
    print(f"Loaded {len(pairs)} pairs")

    # Build trajectories
    print(f"\nBuilding trajectories (max_steps={args.max_steps}, augment_medium={args.augment_medium})")
    builder = TrajectoryBuilder(max_steps=args.max_steps)
    trajectories = builder.build_trajectories(pairs, augment_medium=args.augment_medium)
    print(f"Built {len(trajectories)} trajectories")

    # Statistics
    by_method = {'shortest': 0, 'random': 0}
    by_bin = {'high': 0, 'medium': 0, 'low': 0}
    for traj in trajectories:
        by_method[traj.reconstruction_method] += 1
        A = traj.final_affinity
        if A >= 0:
            by_bin['high'] += 1
        elif A >= -2:
            by_bin['medium'] += 1
        else:
            by_bin['low'] += 1

    print(f"\n=== Trajectory Statistics ===")
    print(f"By method: shortest={by_method['shortest']}, random={by_method['random']}")
    print(f"By bin: high={by_bin['high']}, medium={by_bin['medium']}, low={by_bin['low']}")

    # Save
    print(f"\nSaving to {args.output}")
    output_path = Path(args.output)
    output_data = {
        'metadata': {
            'total_trajectories': len(trajectories),
            'max_steps': args.max_steps,
            'augment_medium': args.augment_medium,
            'by_method': by_method,
            'by_bin': by_bin,
            'seed': args.seed
        },
        'trajectories': [asdict(t) for t in trajectories]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved {len(trajectories)} trajectories")
    print(f"✓ File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
