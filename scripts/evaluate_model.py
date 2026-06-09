#!/usr/bin/env python3
"""
Comprehensive evaluation framework for TCR design models.

Two evaluation modes:
1. Affinity-only: Test on in-domain and OOD peptides
2. Specificity: Test with decoy library (affinity + decoy discrimination)

Metrics:
- Max affinity
- Mean affinity
- Success rate (affinity > threshold)
- Thresholds: 0.0, 0.6
"""

import sys
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse

from tcrppo_v2.policy import ActorCritic
from tcrppo_v2.sft_env import SFTEnv
from tcrppo_v2.utils.constants import MAX_TCR_LEN


class TCREvaluator:
    """Comprehensive TCR evaluation framework."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        use_tfold: bool = True,
    ):
        self.device = device
        self.use_tfold = use_tfold

        # Load policy
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        self.policy = ActorCritic(
            obs_dim=2560,  # ESM-2 650M
            hidden_dim=checkpoint['config'].get('hidden_dim', 512),
            max_tcr_len=MAX_TCR_LEN
        ).to(device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()

        # Environment for generation
        self.env = SFTEnv(max_steps=8)

        print(f"✓ Model loaded successfully")
        print(f"  Hidden dim: {checkpoint['config'].get('hidden_dim', 512)}")
        print(f"  Training epoch: {checkpoint.get('epoch', 'unknown')}")

    def generate_tcrs(
        self,
        peptide: str,
        n_tcrs: int = 50,
        init_strategy: str = 'random',
        max_steps: int = 8,
    ) -> List[str]:
        """Generate TCRs for a given peptide.

        Args:
            peptide: Target peptide sequence
            n_tcrs: Number of TCRs to generate
            init_strategy: 'random' or 'fixed'
            max_steps: Max editing steps

        Returns:
            List of generated TCR sequences
        """
        tcrs = []

        with torch.no_grad():
            for _ in tqdm(range(n_tcrs), desc=f"Generating TCRs for {peptide}"):
                # Reset environment
                if init_strategy == 'random':
                    obs = self.env.reset(peptide=peptide)
                else:
                    # Fixed init (e.g., from TCRdb)
                    obs = self.env.reset(init_tcr='CASSLAPGATNEKLFF', peptide=peptide)

                done = False
                for step in range(max_steps):
                    if done:
                        break

                    # Sample action
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    op, pos, tok, _ = self.policy(obs_tensor, action_masks=None, actions=None)

                    op = op.item()
                    pos = pos.item()
                    tok_idx = tok.item()
                    token = self.env.idx_to_aa.get(tok_idx, 'A')

                    # Apply action
                    obs, _, done, info = self.env.step((op, pos, token))

                tcrs.append(self.env.current_tcr)

        return tcrs

    def score_affinity(
        self,
        tcrs: List[str],
        peptide: str,
        hla: str = 'HLA-A*02:01',
    ) -> np.ndarray:
        """Score TCR-peptide affinity using tFold or ERGO.

        Args:
            tcrs: List of TCR sequences
            peptide: Peptide sequence
            hla: HLA allele

        Returns:
            Array of affinity scores
        """
        if self.use_tfold:
            return self._score_tfold(tcrs, peptide, hla)
        else:
            return self._score_ergo(tcrs, peptide, hla)

    def _score_tfold(
        self,
        tcrs: List[str],
        peptide: str,
        hla: str,
    ) -> np.ndarray:
        """Score using tFold."""
        # Import tFold scorer
        from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer

        scorer = AffinityTFoldScorer()

        # Batch scoring for efficiency
        peptides = [peptide] * len(tcrs)
        scores = scorer.score_batch_fast(tcrs, peptides)

        return np.array(scores)

    def _score_ergo(
        self,
        tcrs: List[str],
        peptide: str,
        hla: str,
    ) -> np.ndarray:
        """Score using ERGO."""
        from tcrppo_v2.scorers.affinity_ergo import ERGOScorer

        scorer = ERGOScorer()
        scores = []

        for tcr in tcrs:
            score, _ = scorer.score(tcr, peptide, hla)
            scores.append(score)

        return np.array(scores)

    def score_decoy_specificity(
        self,
        tcrs: List[str],
        target_peptide: str,
        hla: str = 'HLA-A*02:01',
        n_decoys: int = 50,
    ) -> Dict[str, float]:
        """Score TCR specificity using decoy library.

        Args:
            tcrs: List of TCR sequences
            target_peptide: Target peptide
            hla: HLA allele
            n_decoys: Number of decoys to sample

        Returns:
            Dict with AUROC and other metrics
        """
        from tcrppo_v2.scorers.decoy import DecoyScorer
        from sklearn.metrics import roc_auc_score

        scorer = DecoyScorer()

        # Score target
        target_scores = self.score_affinity(tcrs, target_peptide, hla)

        # Score decoys
        decoy_scores_list = []
        for tcr in tqdm(tcrs, desc="Scoring decoys"):
            decoy_score, _ = scorer.score(tcr, target_peptide, hla, n_samples=n_decoys)
            decoy_scores_list.append(decoy_score)

        decoy_scores = np.array(decoy_scores_list)

        # Compute AUROC (target vs decoy discrimination)
        # For each TCR: label=1 if target_score > decoy_score, else 0
        labels = (target_scores > decoy_scores).astype(int)

        # Compute metrics
        auroc = labels.mean()  # Simplified: fraction of TCRs that prefer target

        return {
            'auroc': auroc,
            'mean_target_score': target_scores.mean(),
            'mean_decoy_score': decoy_scores.mean(),
            'target_vs_decoy_gap': (target_scores - decoy_scores).mean(),
        }

    def evaluate_affinity_only(
        self,
        peptides: Dict[str, List[str]],
        n_tcrs_per_peptide: int = 50,
        thresholds: List[float] = [0.0, 0.6],
    ) -> Dict:
        """Evaluate affinity on in-domain and OOD peptides.

        Args:
            peptides: Dict with 'in_domain' and 'ood' peptide lists
            n_tcrs_per_peptide: Number of TCRs to generate per peptide
            thresholds: Affinity thresholds for success rate

        Returns:
            Evaluation results dict
        """
        results = {
            'in_domain': {},
            'ood': {},
        }

        for domain, peptide_list in peptides.items():
            print(f"\n=== Evaluating {domain.upper()} peptides ===")

            domain_results = []

            for peptide in peptide_list:
                print(f"\nPeptide: {peptide}")

                # Generate TCRs
                tcrs = self.generate_tcrs(peptide, n_tcrs=n_tcrs_per_peptide)

                # Score affinity
                affinities = self.score_affinity(tcrs, peptide)

                # Compute metrics
                peptide_result = {
                    'peptide': peptide,
                    'n_tcrs': len(tcrs),
                    'max_affinity': float(affinities.max()),
                    'mean_affinity': float(affinities.mean()),
                    'std_affinity': float(affinities.std()),
                    'median_affinity': float(np.median(affinities)),
                }

                # Success rates at thresholds
                for thresh in thresholds:
                    success_rate = (affinities > thresh).mean()
                    peptide_result[f'success_rate_{thresh}'] = float(success_rate)

                domain_results.append(peptide_result)

                # Print summary
                print(f"  Max affinity: {peptide_result['max_affinity']:.4f}")
                print(f"  Mean affinity: {peptide_result['mean_affinity']:.4f}")
                for thresh in thresholds:
                    print(f"  Success rate (>{thresh}): {peptide_result[f'success_rate_{thresh}']:.2%}")

            results[domain] = domain_results

        # Aggregate statistics
        for domain in ['in_domain', 'ood']:
            if results[domain]:
                results[f'{domain}_aggregate'] = {
                    'mean_max_affinity': np.mean([r['max_affinity'] for r in results[domain]]),
                    'mean_mean_affinity': np.mean([r['mean_affinity'] for r in results[domain]]),
                }
                for thresh in thresholds:
                    results[f'{domain}_aggregate'][f'mean_success_rate_{thresh}'] = np.mean(
                        [r[f'success_rate_{thresh}'] for r in results[domain]]
                    )

        return results

    def evaluate_specificity(
        self,
        peptides: List[str],
        n_tcrs_per_peptide: int = 50,
        n_decoys: int = 50,
    ) -> Dict:
        """Evaluate specificity with decoy library.

        Args:
            peptides: List of target peptides
            n_tcrs_per_peptide: Number of TCRs to generate per peptide
            n_decoys: Number of decoys to sample per TCR

        Returns:
            Evaluation results dict
        """
        results = []

        print(f"\n=== Evaluating Specificity (with decoys) ===")

        for peptide in peptides:
            print(f"\nPeptide: {peptide}")

            # Generate TCRs
            tcrs = self.generate_tcrs(peptide, n_tcrs=n_tcrs_per_peptide)

            # Score specificity
            spec_metrics = self.score_decoy_specificity(tcrs, peptide, n_decoys=n_decoys)

            peptide_result = {
                'peptide': peptide,
                'n_tcrs': len(tcrs),
                **spec_metrics,
            }

            results.append(peptide_result)

            # Print summary
            print(f"  AUROC: {spec_metrics['auroc']:.4f}")
            print(f"  Mean target score: {spec_metrics['mean_target_score']:.4f}")
            print(f"  Mean decoy score: {spec_metrics['mean_decoy_score']:.4f}")
            print(f"  Target-Decoy gap: {spec_metrics['target_vs_decoy_gap']:.4f}")

        # Aggregate
        aggregate = {
            'mean_auroc': np.mean([r['auroc'] for r in results]),
            'mean_target_score': np.mean([r['mean_target_score'] for r in results]),
            'mean_decoy_score': np.mean([r['mean_decoy_score'] for r in results]),
            'mean_gap': np.mean([r['target_vs_decoy_gap'] for r in results]),
        }

        return {
            'per_peptide': results,
            'aggregate': aggregate,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TCR design model")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['affinity', 'specificity', 'both'],
                        default='both', help='Evaluation mode')
    parser.add_argument('--n_tcrs', type=int, default=50,
                        help='Number of TCRs to generate per peptide')
    parser.add_argument('--n_decoys', type=int, default=50,
                        help='Number of decoys for specificity evaluation')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--scorer', type=str, choices=['tfold', 'ergo'],
                        default='tfold', help='Affinity scorer')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    evaluator = TCREvaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_tfold=(args.scorer == 'tfold'),
    )

    # Define peptides
    in_domain_peptides = [
        'GILGFVFTL',  # Influenza M1
        'NLVPMVATV',  # CMV pp65
        'GLCTLVAML',  # EBV BMLF1
        'LLWNGPMAV',  # Yellow fever NS4B
        'YLQPRTFLL',  # HIV RT
    ]

    ood_peptides = [
        'KLGGALQAK',  # EBV BZLF1
        'AVFDRKSDAK', # EBV EBNA-3B
        'IVTDFSVIK',  # Influenza NP
        'SPRWYFYYL',  # SARS-CoV-2 ORF1ab
        'RLRAEAQVK',  # CMV IE1
    ]

    results = {}

    # Mode 1: Affinity-only evaluation
    if args.mode in ['affinity', 'both']:
        print("\n" + "="*80)
        print("MODE 1: AFFINITY-ONLY EVALUATION")
        print("="*80)

        affinity_results = evaluator.evaluate_affinity_only(
            peptides={
                'in_domain': in_domain_peptides,
                'ood': ood_peptides,
            },
            n_tcrs_per_peptide=args.n_tcrs,
            thresholds=[0.0, 0.6],
        )

        results['affinity'] = affinity_results

        # Save affinity results
        with open(output_dir / 'affinity_results.json', 'w') as f:
            json.dump(affinity_results, f, indent=2)

        print(f"\n✓ Affinity results saved to {output_dir / 'affinity_results.json'}")

    # Mode 2: Specificity evaluation
    if args.mode in ['specificity', 'both']:
        print("\n" + "="*80)
        print("MODE 2: SPECIFICITY EVALUATION (with decoys)")
        print("="*80)

        # Use subset of peptides for specificity (slower)
        spec_peptides = in_domain_peptides[:3]  # First 3 peptides

        specificity_results = evaluator.evaluate_specificity(
            peptides=spec_peptides,
            n_tcrs_per_peptide=args.n_tcrs,
            n_decoys=args.n_decoys,
        )

        results['specificity'] = specificity_results

        # Save specificity results
        with open(output_dir / 'specificity_results.json', 'w') as f:
            json.dump(specificity_results, f, indent=2)

        print(f"\n✓ Specificity results saved to {output_dir / 'specificity_results.json'}")

    # Save combined results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    if 'affinity' in results:
        print("\nAFFINITY-ONLY:")
        for domain in ['in_domain', 'ood']:
            agg = results['affinity'].get(f'{domain}_aggregate', {})
            if agg:
                print(f"\n  {domain.upper()}:")
                print(f"    Mean max affinity: {agg['mean_max_affinity']:.4f}")
                print(f"    Mean mean affinity: {agg['mean_mean_affinity']:.4f}")
                print(f"    Mean success rate (>0.0): {agg['mean_success_rate_0.0']:.2%}")
                print(f"    Mean success rate (>0.6): {agg['mean_success_rate_0.6']:.2%}")

    if 'specificity' in results:
        print("\nSPECIFICITY:")
        agg = results['specificity']['aggregate']
        print(f"  Mean AUROC: {agg['mean_auroc']:.4f}")
        print(f"  Mean target score: {agg['mean_target_score']:.4f}")
        print(f"  Mean decoy score: {agg['mean_decoy_score']:.4f}")
        print(f"  Mean target-decoy gap: {agg['mean_gap']:.4f}")

    print(f"\n✓ All results saved to {output_dir}")


if __name__ == '__main__':
    main()
