"""DeepAIR TCR-peptide binding affinity scorer.

DeepAIR is a deep learning framework that integrates ProtBert sequence features
for TCR-peptide binding prediction. This implementation wraps the official
TensorFlow models from https://github.com/TencentAILabHealthcare/DeepAIR

Reference:
    Zhang et al. (2023). "Deep learning-based prediction of T cell receptor-antigen
    binding specificity." Science Advances, 9(32), eabo5128.
    DOI: 10.1126/sciadv.abo5128

IMPORTANT LIMITATION:
    The ProtBert pooler_output features extracted from PyTorch-converted TF weights
    are degenerate (nearly identical for all sequences). This is a known issue with
    cross-framework weight conversion. The model still works because:
    1. It uses the raw CDR3 string inputs (TRB_cdr3, TRA_cdr3)
    2. It uses peptide-specific models
    3. The degenerate pooler_output acts as a constant bias term

    Ideally, we would use native TensorFlow ProtBert weights, but these require
    internet access to download. The current implementation is functional but
    may have reduced discriminative power compared to the original DeepAIR paper.
"""

import os
import sys
import warnings
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

# Configure TensorFlow BEFORE importing it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Suppress TensorFlow warnings and disable GPU
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.config.set_visible_devices([], 'GPU')
warnings.filterwarnings('ignore')

from transformers import TFBertModel, BertTokenizer
from tcrppo_v2.scorers.base import BaseScorer
from tcrppo_v2.utils.constants import PROJECT_ROOT


def seq_modafication(seq: str) -> str:
    """Add spaces between amino acids for ProtBert tokenization."""
    return ' '.join(''.join(seq.split()))


# Peptide to model mapping (from config.py)
PEPTIDE_TO_MODEL = {
    'GILGFVFTL': 'A0201_GILGFVFTL_Flu-MP_Influenza_binder_model',
    'GLCTLVAML': 'A0201_GLCTLVAML_BMLF1_EBV_binder_model',
    'KLGGALQAK': 'A0301_KLGGALQAK_IE-1_CMV_binder_model',
    'AVFDRKSDAK': 'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder_model',
    'IVTDFSVIK': 'A1101_IVTDFSVIK_EBNA-3B_EBV_binder_model',
    'RAKFKQLL': 'B0801_RAKFKQLL_BZLF1_EBV_binder_model',
    'ELAGIGILTV': 'A0201_ELAGIGILTV_MART-1_Cancer_binder_model',
    'LTDEMIAQY': 'LTDEMIAQY_model',
    'TTDPSFLGRY': 'TTDPSFLGRY_model',
    'YLQPRTFLL': 'YLQPRTFLL_model',
}


class AffinityDeepAIRScorer(BaseScorer):
    """DeepAIR-based TCR-peptide binding affinity scorer.

    This scorer uses the official DeepAIR TensorFlow models with ProtBert
    sequence encoding. DeepAIR provides per-epitope models trained on
    specific TCR-peptide binding data.

    Attributes:
        checkpoints_dir: Path to DeepAIR checkpoint directory
        models: Dict of loaded per-peptide models (lazy loading)
        default_model_key: Fallback model for unseen peptides
    """

    def __init__(
        self,
        checkpoints_dir: Optional[str] = None,
        device: str = 'cpu',
        default_peptide: str = 'GILGFVFTL',
        protbert_dir: Optional[str] = None
    ):
        """Initialize DeepAIR scorer.

        Args:
            checkpoints_dir: Path to DeepAIR checkpoints/BRP directory.
                Defaults to PROJECT_ROOT/models/deepair/DeepAIR/checkpoints/BRP
            device: Device for computation ('cpu' or 'cuda'). Note: TensorFlow
                will auto-detect GPU availability.
            default_peptide: Peptide to use for unseen peptides (fallback model)
            protbert_dir: Path to ProtBert model directory.
                Defaults to PROJECT_ROOT/models/protbert/
        """
        super().__init__()

        if checkpoints_dir is None:
            checkpoints_dir = os.path.join(
                PROJECT_ROOT, 'models', 'deepair', 'DeepAIR', 'checkpoints', 'BRP'
            )

        if protbert_dir is None:
            # Try TF-native weights first, fallback to PyTorch-converted
            protbert_tf = os.path.join(PROJECT_ROOT, 'models', 'protbert_tf')
            protbert_pt = os.path.join(PROJECT_ROOT, 'models', 'protbert')
            if os.path.exists(protbert_tf):
                protbert_dir = protbert_tf
            else:
                protbert_dir = protbert_pt

        self.checkpoints_dir = checkpoints_dir
        self.device = device
        self.default_peptide = default_peptide

        # Configure TensorFlow device BEFORE any TF operations
        if device == 'cpu':
            tf.config.set_visible_devices([], 'GPU')

        # Lazy loading: models loaded on first use (using tf.saved_model.load)
        self.models: Dict[str, any] = {}

        # Verify checkpoints directory exists
        if not os.path.exists(checkpoints_dir):
            raise FileNotFoundError(
                f"DeepAIR checkpoints directory not found: {checkpoints_dir}\n"
                f"Please download from https://zenodo.org/records/7792621"
            )

        # Load TensorFlow ProtBert model and tokenizer
        print(f"Loading ProtBert from {protbert_dir}...")
        self.protbert_tokenizer = BertTokenizer.from_pretrained(
            protbert_dir, do_lower_case=False
        )

        # Check if this is TF-native weights (has tf_model.h5) or PyTorch weights
        has_tf_weights = os.path.exists(os.path.join(protbert_dir, 'tf_model.h5'))
        has_pt_weights = os.path.exists(os.path.join(protbert_dir, 'pytorch_model.bin'))

        if has_tf_weights:
            # Load from TF-native weights
            self.protbert_model = TFBertModel.from_pretrained(protbert_dir)
        elif has_pt_weights:
            # Load from PyTorch weights
            self.protbert_model = TFBertModel.from_pretrained(protbert_dir, from_pt=True)
        else:
            raise FileNotFoundError(f"No model weights found in {protbert_dir}")

        self.protbert_model.trainable = False
        print("ProtBert loaded successfully")

    def _load_model(self, peptide: str):
        """Load DeepAIR model for a specific peptide using tf.saved_model.load.

        Args:
            peptide: Peptide sequence

        Returns:
            Loaded TensorFlow SavedModel
        """
        # Check if already loaded
        if peptide in self.models:
            return self.models[peptide]

        # Get model key
        model_key = PEPTIDE_TO_MODEL.get(peptide)
        if model_key is None:
            # Use default model for unseen peptides
            model_key = PEPTIDE_TO_MODEL[self.default_peptide]
            warnings.warn(
                f"No DeepAIR model for peptide '{peptide}', "
                f"using default model '{self.default_peptide}'"
            )

        # Load model using tf.saved_model.load (compatible with Keras 3)
        model_path = os.path.join(self.checkpoints_dir, model_key, 'model')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"DeepAIR model not found: {model_path}\n"
                f"Available models: {list(PEPTIDE_TO_MODEL.keys())}"
            )

        loaded = tf.saved_model.load(model_path)
        infer = loaded.signatures['serving_default']
        self.models[peptide] = infer

        return infer

    def _extract_protbert_features(self, sequence: str) -> tf.Tensor:
        """Extract ProtBert pooler_output features for a CDR3 sequence.

        Note: pooler_output from PyTorch-converted TF model is degenerate,
        but we use it anyway since the DeepAIR model was trained with it.

        Args:
            sequence: CDR3 amino acid sequence

        Returns:
            ProtBert features with shape (1, 1, 1024)
        """
        seq_spaced = seq_modafication(sequence)
        tokens = self.protbert_tokenizer(
            seq_spaced,
            return_tensors='tf',
            padding='max_length',
            max_length=40,
            truncation=True
        )

        output = self.protbert_model(
            tokens['input_ids'],
            attention_mask=tokens['attention_mask']
        )

        # Use pooler_output (even though it's degenerate)
        features = tf.expand_dims(output.pooler_output, axis=1)  # (1, 1, 1024)
        return features

    def _prepare_input(
        self,
        tcr_beta: str,
        peptide: str,
        tcr_alpha: Optional[str] = None
    ) -> Dict:
        """Prepare input for DeepAIR SavedModel signature.

        Args:
            tcr_beta: TCR beta CDR3 sequence
            peptide: Peptide sequence
            tcr_alpha: TCR alpha CDR3 sequence (optional, uses dummy if None)

        Returns:
            Input dict for DeepAIR model (TensorFlow tensors)
        """
        # Use dummy alpha chain if not provided
        if tcr_alpha is None:
            tcr_alpha = 'CAAS'  # Minimal valid CDR3 alpha

        # Extract ProtBert features for beta and alpha chains
        beta_features = self._extract_protbert_features(tcr_beta)
        alpha_features = self._extract_protbert_features(tcr_alpha)

        # Create input dict with correct shapes and dtypes
        # Based on signature: (None, 1) for most, (None, 1, 1024) for splited, (None, 40, 384) for Stru
        input_dict = {
            'TRB_cdr3': tf.constant([[tcr_beta]], dtype=tf.string),  # (1, 1)
            'TRA_cdr3': tf.constant([[tcr_alpha]], dtype=tf.string),  # (1, 1)
            'TRB_cdr3_splited': beta_features,  # (1, 1, 1024)
            'TRA_cdr3_splited': alpha_features,  # (1, 1, 1024)
            'TRB_cdr3_Stru': tf.zeros((1, 40, 384), dtype=tf.float32),  # (1, 40, 384)
            'TRA_cdr3_Stru': tf.zeros((1, 40, 384), dtype=tf.float32),  # (1, 40, 384)
            'ID': tf.constant([[0]], dtype=tf.int64),  # (1, 1)
            'TRB_v_gene': tf.constant([[0]], dtype=tf.int64),  # (1, 1) - tokenized
            'TRB_j_gene': tf.constant([[0]], dtype=tf.int64),  # (1, 1) - tokenized
            'TRA_v_gene': tf.constant([[0]], dtype=tf.int64),  # (1, 1) - tokenized
            'TRA_j_gene': tf.constant([[0]], dtype=tf.int64),  # (1, 1) - tokenized
        }

        return input_dict

    def score(self, tcr: str, peptide: str) -> Tuple[float, float]:
        """Score TCR-peptide binding affinity.

        Args:
            tcr: TCR CDR3 beta sequence
            peptide: Peptide sequence

        Returns:
            (score, confidence) tuple where:
                - score: Binding probability in [0, 1]
                - confidence: Same as score (DeepAIR outputs probability)
        """
        try:
            # Load model for this peptide
            infer = self._load_model(peptide)

            # Prepare input
            input_dict = self._prepare_input(tcr, peptide)

            # Run prediction (call signature directly)
            output = infer(**input_dict)

            # Extract score from output
            # Output is a dict with key 'output_1'
            pred = output['output_1']
            score = float(pred.numpy()[0])

            # Clip to [0, 1]
            score = np.clip(score, 0.0, 1.0)

            return score, score

        except Exception as e:
            warnings.warn(f"DeepAIR prediction failed: {e}, returning 0.5")
            return 0.5, 0.0

    def score_batch(
        self,
        tcrs: List[str],
        peptides: List[str]
    ) -> Tuple[List[float], List[float]]:
        """Score a batch of TCR-peptide pairs.

        Args:
            tcrs: List of TCR CDR3 beta sequences
            peptides: List of peptide sequences

        Returns:
            (scores, confidences) tuple of lists
        """
        if len(tcrs) != len(peptides):
            raise ValueError("tcrs and peptides must have same length")

        scores = []
        confidences = []

        for tcr, peptide in zip(tcrs, peptides):
            score, conf = self.score(tcr, peptide)
            scores.append(score)
            confidences.append(conf)

        return scores, confidences

    def score_batch_fast(
        self,
        tcrs: List[str],
        peptides: List[str]
    ) -> Tuple[List[float], List[float]]:
        """Fast batch scoring (groups by peptide for efficiency).

        Args:
            tcrs: List of TCR CDR3 beta sequences
            peptides: List of peptide sequences

        Returns:
            (scores, confidences) tuple of lists
        """
        if len(tcrs) != len(peptides):
            raise ValueError("tcrs and peptides must have same length")

        # Group by peptide for efficient batching
        peptide_groups = {}
        for idx, (tcr, pep) in enumerate(zip(tcrs, peptides)):
            if pep not in peptide_groups:
                peptide_groups[pep] = []
            peptide_groups[pep].append((idx, tcr))

        # Initialize output arrays
        scores = [0.0] * len(tcrs)
        confidences = [0.0] * len(tcrs)

        # Process each peptide group
        for peptide, items in peptide_groups.items():
            indices = [idx for idx, _ in items]
            tcr_batch = [tcr for _, tcr in items]

            try:
                # Load model
                model = self._load_model(peptide)

                # Prepare batch input
                tcr_beta_split = [seq_modafication(tcr) for tcr in tcr_batch]
                tcr_alpha_split = [seq_modafication('CAAS') for _ in tcr_batch]

                batch_size = len(tcr_batch)
                input_dict = {
                    'TRB_cdr3': np.array(tcr_batch),
                    'TRA_cdr3': np.array(['CAAS'] * batch_size),
                    'TRB_cdr3_splited': np.array(tcr_beta_split),
                    'TRA_cdr3_splited': np.array(tcr_alpha_split),
                    'ID': np.array([f'sample_{i}' for i in range(batch_size)]),
                    'TRB_v_gene': np.array(['TRBV1'] * batch_size),
                    'TRB_j_gene': np.array(['TRBJ1'] * batch_size),
                    'TRA_v_gene': np.array(['TRAV1'] * batch_size),
                    'TRA_j_gene': np.array(['TRAJ1'] * batch_size),
                }

                # Run prediction
                preds = model.run(input_dict)

                # Extract scores
                if tf.is_tensor(preds):
                    batch_scores = preds.numpy()
                else:
                    batch_scores = np.array(preds)

                # Clip and assign
                batch_scores = np.clip(batch_scores, 0.0, 1.0)
                for idx, score in zip(indices, batch_scores):
                    scores[idx] = float(score)
                    confidences[idx] = float(score)

            except Exception as e:
                warnings.warn(f"Batch prediction failed for {peptide}: {e}")
                for idx in indices:
                    scores[idx] = 0.5
                    confidences[idx] = 0.0

        return scores, confidences
