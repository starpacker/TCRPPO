"""Example: Creating an ensemble scorer with NetTCR, ERGO, and DeepAIR.

This script demonstrates how to initialize and use the three-scorer ensemble
for TCR-peptide binding prediction.
"""

import sys
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.scorers.affinity_nettcr import AffinityNetTCRScorer
from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
from tcrppo_v2.scorers.affinity_deepair import AffinityDeepAIRScorer
from tcrppo_v2.scorers.affinity_ensemble import EnsembleAffinityScorer
from tcrppo_v2.utils.constants import ERGO_MODEL_DIR
import os


def create_ensemble_scorer(
    use_nettcr: bool = True,
    use_ergo: bool = True,
    use_deepair: bool = True,
    weights: list = None,
    device: str = "cuda"
):
    """Create an ensemble scorer with selected models.

    Args:
        use_nettcr: Include NetTCR-2.0 in ensemble
        use_ergo: Include ERGO in ensemble
        use_deepair: Include DeepAIR in ensemble
        weights: Custom weights for each scorer (must match number of enabled scorers)
        device: Device for GPU-based scorers (ERGO, DeepAIR)

    Returns:
        EnsembleAffinityScorer instance
    """
    scorers = []
    scorer_names = []

    if use_nettcr:
        print("Initializing NetTCR-2.0...")
        nettcr = AffinityNetTCRScorer(device='cpu')  # NetTCR uses TensorFlow on CPU
        scorers.append(nettcr)
        scorer_names.append('NetTCR')

    if use_ergo:
        print("Initializing ERGO...")
        ergo_model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
        ergo = AffinityERGOScorer(
            model_file=ergo_model_file,
            device=device,
            mc_samples=1  # Use 1 for fast scoring, increase for uncertainty estimation
        )
        scorers.append(ergo)
        scorer_names.append('ERGO')

    if use_deepair:
        print("Initializing DeepAIR...")
        deepair = AffinityDeepAIRScorer(device=device)
        scorers.append(deepair)
        scorer_names.append('DeepAIR')

    if not scorers:
        raise ValueError("At least one scorer must be enabled")

    # Create ensemble
    print(f"\nCreating ensemble with: {', '.join(scorer_names)}")
    ensemble = EnsembleAffinityScorer(scorers=scorers, weights=weights)

    return ensemble


def example_usage():
    """Example usage of the ensemble scorer."""
    print("=" * 80)
    print("Ensemble Scorer Example")
    print("=" * 80)

    # Example 1: Equal-weighted ensemble of all three scorers
    print("\n--- Example 1: Equal-weighted ensemble ---")
    ensemble = create_ensemble_scorer(
        use_nettcr=True,
        use_ergo=True,
        use_deepair=True,
        weights=None,  # Equal weights
        device='cuda'
    )

    # Test on a single TCR-peptide pair
    tcr = "CASSIRSSYEQYF"
    peptide = "GILGFVFTL"

    score, confidence = ensemble.score(tcr, peptide)
    print(f"\nSingle prediction:")
    print(f"  TCR: {tcr}")
    print(f"  Peptide: {peptide}")
    print(f"  Score: {score:.4f}")
    print(f"  Confidence: {confidence:.4f}")

    # Example 2: Custom-weighted ensemble (favor ERGO)
    print("\n--- Example 2: Custom-weighted ensemble (ERGO-heavy) ---")
    ensemble_custom = create_ensemble_scorer(
        use_nettcr=True,
        use_ergo=True,
        use_deepair=True,
        weights=[0.2, 0.6, 0.2],  # NetTCR: 20%, ERGO: 60%, DeepAIR: 20%
        device='cuda'
    )

    score_custom, confidence_custom = ensemble_custom.score(tcr, peptide)
    print(f"\nCustom-weighted prediction:")
    print(f"  Score: {score_custom:.4f}")
    print(f"  Confidence: {confidence_custom:.4f}")

    # Example 3: Batch prediction
    print("\n--- Example 3: Batch prediction ---")
    tcrs = [
        "CASSIRSSYEQYF",
        "CASSSRSSYEQYF",
        "CASSLIYPGELFF",
    ]
    peptides = ["GILGFVFTL"] * 3

    scores, confidences = ensemble.score_batch(tcrs, peptides)
    print(f"\nBatch predictions:")
    for i, (tcr, score, conf) in enumerate(zip(tcrs, scores, confidences)):
        print(f"  {i+1}. {tcr}: score={score:.4f}, conf={conf:.4f}")

    # Example 4: Two-scorer ensemble (NetTCR + ERGO only)
    print("\n--- Example 4: Two-scorer ensemble (NetTCR + ERGO) ---")
    ensemble_two = create_ensemble_scorer(
        use_nettcr=True,
        use_ergo=True,
        use_deepair=False,
        weights=None,
        device='cuda'
    )

    score_two, _ = ensemble_two.score(tcr, peptide)
    print(f"\nTwo-scorer prediction: {score_two:.4f}")

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == '__main__':
    example_usage()
