#!/usr/bin/env python3
"""
Extract SFT training data from training logs.

This script parses tFoldScore records from 157 training logs and matches
InitTCR ↔ FinalTCR pairs via (peptide, affinity_logit) correlation.

Output: data/sft_raw_pairs.json with stratified sampling:
- A≥0: keep all (~2.6K)
- -2≤A<0: sample 20K
- -4≤A<-2: sample 20K
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
import argparse


@dataclass
class TFoldRecord:
    """Single tFoldScore record."""
    timestamp: str
    affinity_logit: float
    cdr3b: str
    peptide: str
    hla: str
    source_log: str
    line_number: int


@dataclass
class TCRPair:
    """Matched InitTCR → FinalTCR pair."""
    init_tcr: str
    final_tcr: str
    peptide: str
    hla: str
    init_affinity: float
    final_affinity: float
    delta_affinity: float
    source_log: str
    episode_id: int  # Estimated episode number


class LogParser:
    """Parse tFoldScore records from training logs."""

    TFOLD_PATTERN = re.compile(
        r'\[tFoldScore\] ts=(?P<timestamp>[\d\-: ]+) '
        r'source=\S+ path_ms=[\d.]+ classify_ms=[\d.]+ end_to_end_ms=[\d.]+ '
        r'affinity_logit=(?P<affinity>-?[\d.]+) conf=[\d.]+ '
        r'cdr3b=(?P<cdr3b>[A-Z]+) peptide=(?P<peptide>[A-Z]+) hla=(?P<hla>HLA-[A-Z0-9*:]+)'
    )

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.records: List[TFoldRecord] = []

    def parse_all_logs(self) -> List[TFoldRecord]:
        """Parse all training logs in directory."""
        log_files = sorted(self.log_dir.glob("trace*.log"))
        print(f"Found {len(log_files)} log files")

        for log_file in log_files:
            self.parse_log(log_file)

        print(f"Extracted {len(self.records)} tFoldScore records")
        return self.records

    def parse_log(self, log_path: Path):
        """Parse single log file."""
        log_name = log_path.name
        line_num = 0

        try:
            with open(log_path, 'r') as f:
                for line in f:
                    line_num += 1
                    if '[tFoldScore]' not in line:
                        continue

                    match = self.TFOLD_PATTERN.search(line)
                    if match:
                        record = TFoldRecord(
                            timestamp=match.group('timestamp'),
                            affinity_logit=float(match.group('affinity')),
                            cdr3b=match.group('cdr3b'),
                            peptide=match.group('peptide'),
                            hla=match.group('hla'),
                            source_log=log_name,
                            line_number=line_num
                        )
                        self.records.append(record)
        except Exception as e:
            print(f"Error parsing {log_name}: {e}")


class PairMatcher:
    """Match InitTCR ↔ FinalTCR pairs from tFoldScore batches."""

    def __init__(self, records: List[TFoldRecord], n_envs: int = 8):
        self.records = records
        self.n_envs = n_envs
        self.pairs: List[TCRPair] = []

    def match_pairs(self) -> List[TCRPair]:
        """
        Match Init → Final pairs using batch structure.

        Strategy:
        - tFoldScore records come in batches of n_envs (usually 8)
        - First batch = InitTCR, second batch = FinalTCR
        - Match via (peptide, affinity_logit) proximity
        """
        # Group records by log file
        by_log = defaultdict(list)
        for rec in self.records:
            by_log[rec.source_log].append(rec)

        print(f"Processing {len(by_log)} log files for pair matching")

        for log_name, log_records in by_log.items():
            self._match_log_pairs(log_name, log_records)

        print(f"Matched {len(self.pairs)} TCR pairs")
        return self.pairs

    def _match_log_pairs(self, log_name: str, records: List[TFoldRecord]):
        """Match pairs within a single log file."""
        # Group into batches (consecutive n_envs records)
        batches = []
        for i in range(0, len(records), self.n_envs):
            batch = records[i:i + self.n_envs]
            if len(batch) == self.n_envs:
                batches.append(batch)

        # Match consecutive batch pairs (Init → Final)
        for i in range(0, len(batches) - 1, 2):
            init_batch = batches[i]
            final_batch = batches[i + 1]

            # Match by peptide (should be same within episode)
            init_by_pep = {rec.peptide: rec for rec in init_batch}
            final_by_pep = {rec.peptide: rec for rec in final_batch}

            for peptide in init_by_pep:
                if peptide in final_by_pep:
                    init_rec = init_by_pep[peptide]
                    final_rec = final_by_pep[peptide]

                    # Only keep if TCRs are different (actual editing happened)
                    if init_rec.cdr3b != final_rec.cdr3b:
                        pair = TCRPair(
                            init_tcr=init_rec.cdr3b,
                            final_tcr=final_rec.cdr3b,
                            peptide=peptide,
                            hla=init_rec.hla,
                            init_affinity=init_rec.affinity_logit,
                            final_affinity=final_rec.affinity_logit,
                            delta_affinity=final_rec.affinity_logit - init_rec.affinity_logit,
                            source_log=log_name,
                            episode_id=i // 2  # Approximate episode number
                        )
                        self.pairs.append(pair)


class StratifiedSampler:
    """Stratified sampling across affinity bins."""

    def __init__(self, pairs: List[TCRPair]):
        self.pairs = pairs
        self.bins = {
            'high': [],    # A ≥ 0
            'medium': [],  # -2 ≤ A < 0
            'low': []      # -4 ≤ A < -2
        }
        self._bin_pairs()

    def _bin_pairs(self):
        """Assign pairs to affinity bins based on final affinity."""
        for pair in self.pairs:
            A = pair.final_affinity
            if A >= 0:
                self.bins['high'].append(pair)
            elif A >= -2:
                self.bins['medium'].append(pair)
            elif A >= -4:
                self.bins['low'].append(pair)
            # Discard A < -4

    def sample(self, n_high: int = None, n_medium: int = 20000, n_low: int = 20000) -> List[TCRPair]:
        """
        Sample from bins with stratification.

        Args:
            n_high: Keep all if None
            n_medium: Target count for medium bin
            n_low: Target count for low bin
        """
        sampled = []

        # High bin: keep all
        high_samples = self.bins['high']
        if n_high is not None:
            high_samples = random.sample(high_samples, min(n_high, len(high_samples)))
        sampled.extend(high_samples)

        # Medium bin: sample n_medium
        medium_samples = self.bins['medium']
        if len(medium_samples) > n_medium:
            medium_samples = random.sample(medium_samples, n_medium)
        sampled.extend(medium_samples)

        # Low bin: sample n_low
        low_samples = self.bins['low']
        if len(low_samples) > n_low:
            low_samples = random.sample(low_samples, n_low)
        sampled.extend(low_samples)

        print(f"Sampled: high={len(high_samples)}, medium={len(medium_samples)}, low={len(low_samples)}")
        print(f"Total sampled pairs: {len(sampled)}")

        return sampled

    def print_stats(self):
        """Print bin statistics."""
        print("\n=== Affinity Bin Statistics ===")
        print(f"High (A≥0):      {len(self.bins['high']):6d} pairs")
        print(f"Medium (-2≤A<0): {len(self.bins['medium']):6d} pairs")
        print(f"Low (-4≤A<-2):   {len(self.bins['low']):6d} pairs")
        print(f"Total:           {sum(len(b) for b in self.bins.values()):6d} pairs")


def main():
    parser = argparse.ArgumentParser(description="Extract SFT data from training logs")
    parser.add_argument("--log_dir", type=str, default="/share/liuyutian/tcrppo_v2/logs",
                        help="Directory containing training logs")
    parser.add_argument("--output", type=str, default="/share/liuyutian/tcrppo_v2/data/sft_raw_pairs.json",
                        help="Output JSON file")
    parser.add_argument("--n_medium", type=int, default=20000,
                        help="Sample size for medium bin (-2≤A<0)")
    parser.add_argument("--n_low", type=int, default=20000,
                        help="Sample size for low bin (-4≤A<-2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Step 1: Parse all logs
    print("=== Step 1: Parsing logs ===")
    parser_obj = LogParser(Path(args.log_dir))
    records = parser_obj.parse_all_logs()

    # Step 2: Match Init ↔ Final pairs
    print("\n=== Step 2: Matching pairs ===")
    matcher = PairMatcher(records)
    pairs = matcher.match_pairs()

    # Step 3: Stratified sampling
    print("\n=== Step 3: Stratified sampling ===")
    sampler = StratifiedSampler(pairs)
    sampler.print_stats()
    sampled_pairs = sampler.sample(n_medium=args.n_medium, n_low=args.n_low)

    # Step 4: Save to JSON
    print(f"\n=== Step 4: Saving to {args.output} ===")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'metadata': {
            'total_records': len(records),
            'total_pairs': len(pairs),
            'sampled_pairs': len(sampled_pairs),
            'bins': {
                'high': len(sampler.bins['high']),
                'medium': len(sampler.bins['medium']),
                'low': len(sampler.bins['low'])
            },
            'seed': args.seed
        },
        'pairs': [asdict(p) for p in sampled_pairs]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved {len(sampled_pairs)} pairs to {output_path}")
    print(f"✓ File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
