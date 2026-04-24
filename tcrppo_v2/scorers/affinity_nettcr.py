"""NetTCR-2.0 binding affinity scorer conforming to BaseScorer interface.

Self-contained NetTCR scorer that builds and trains the model directly.
Uses BLOSUM50 encoding and the one-chain (beta) CNN architecture.
"""

import os
from typing import List, Tuple

import numpy as np

from tcrppo_v2.scorers.base import BaseScorer
from tcrppo_v2.utils.constants import PROJECT_ROOT


# BLOSUM50 encoding (20 amino acids) — copied from NetTCR-2.0/utils.py
BLOSUM50_20AA = {
    'A': np.array((5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0)),
    'R': np.array((-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3)),
    'N': np.array((-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3)),
    'D': np.array((-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4)),
    'C': np.array((-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1)),
    'Q': np.array((-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3)),
    'E': np.array((-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3)),
    'G': np.array((0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4)),
    'H': np.array((-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4)),
    'I': np.array((-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4)),
    'L': np.array((-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1)),
    'K': np.array((-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3)),
    'M': np.array((-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1)),
    'F': np.array((-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1)),
    'P': np.array((-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3)),
    'S': np.array((1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2)),
    'T': np.array((0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0)),
    'W': np.array((-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3)),
    'Y': np.array((-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1)),
    'V': np.array((0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5)),
}

# Maximum lengths for padding
MAX_CDR3_LEN = 30
MAX_PEP_LEN = 12  # Extended from 9 to handle longer peptides

# NetTCR-2.0 data directory
NETTCR_DATA_DIR = "/share/liuyutian/NetTCR-2.0/data"
NETTCR_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "data", "nettcr_model.weights.h5")


def encode_sequences(sequences: List[str], max_len: int) -> np.ndarray:
    """Encode amino acid sequences using BLOSUM50 with zero-padding."""
    n_seqs = len(sequences)
    n_features = 20
    encoded = np.zeros((n_seqs, max_len, n_features), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_len]):
            if aa in BLOSUM50_20AA:
                encoded[i, j] = BLOSUM50_20AA[aa]

    return encoded


class AffinityNetTCRScorer(BaseScorer):
    """NetTCR-2.0 CNN binding predictor as training reward scorer."""

    def __init__(
        self,
        model_path: str = None,
        device: str = "cpu",
        batch_size: int = 512,
        tf_gpu_id: int = None,  # If set, use this GPU for TensorFlow
    ):
        # Set TF environment to minimize noise and memory usage
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        # Configure TensorFlow GPU usage
        if tf_gpu_id is not None:
            # Use specified GPU for TensorFlow
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and tf_gpu_id < len(gpus):
                try:
                    tf.config.set_visible_devices([gpus[tf_gpu_id]], 'GPU')
                    tf.config.experimental.set_memory_growth(gpus[tf_gpu_id], True)
                    print(f"  NetTCR using GPU {tf_gpu_id}")
                except RuntimeError as e:
                    print(f"  NetTCR GPU setup failed: {e}, falling back to CPU")
                    tf.config.set_visible_devices([], 'GPU')
            else:
                print(f"  NetTCR: GPU {tf_gpu_id} not available, using CPU")
                tf.config.set_visible_devices([], 'GPU')
        else:
            # CPU-only mode: hide all GPUs from TensorFlow
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')

        self._device = device
        self._batch_size = batch_size

        # Resolve model path
        if model_path is None:
            model_path = NETTCR_WEIGHTS_PATH

        # Build and load the model
        self._model = self._build_model()

        if os.path.exists(model_path):
            self._model.load_weights(model_path)
            print(f"  NetTCR scorer loaded (weights: {model_path})")
        else:
            # Train from scratch using NetTCR-2.0 training data
            print(f"  NetTCR weights not found at {model_path}, training from scratch...")
            self._train_model(model_path)

    def _build_model(self):
        """Build the NetTCR one-chain (beta) architecture."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.layers import (
            Input, Conv1D, GlobalMaxPooling1D, Dense, concatenate
        )
        from tensorflow.keras.models import Model

        cdr_in = Input(shape=(MAX_CDR3_LEN, 20))
        pep_in = Input(shape=(MAX_PEP_LEN, 20))

        # Peptide tower (multi-kernel CNN)
        pep_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid',
                           kernel_initializer='glorot_normal')(pep_in)
        pep_pool1 = GlobalMaxPooling1D()(pep_conv1)
        pep_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid',
                           kernel_initializer='glorot_normal')(pep_in)
        pep_pool3 = GlobalMaxPooling1D()(pep_conv3)
        pep_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid',
                           kernel_initializer='glorot_normal')(pep_in)
        pep_pool5 = GlobalMaxPooling1D()(pep_conv5)
        pep_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid',
                           kernel_initializer='glorot_normal')(pep_in)
        pep_pool7 = GlobalMaxPooling1D()(pep_conv7)
        pep_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid',
                           kernel_initializer='glorot_normal')(pep_in)
        pep_pool9 = GlobalMaxPooling1D()(pep_conv9)

        # CDR3β tower (multi-kernel CNN)
        cdr_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid',
                           kernel_initializer='glorot_normal')(cdr_in)
        cdr_pool1 = GlobalMaxPooling1D()(cdr_conv1)
        cdr_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid',
                           kernel_initializer='glorot_normal')(cdr_in)
        cdr_pool3 = GlobalMaxPooling1D()(cdr_conv3)
        cdr_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid',
                           kernel_initializer='glorot_normal')(cdr_in)
        cdr_pool5 = GlobalMaxPooling1D()(cdr_conv5)
        cdr_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid',
                           kernel_initializer='glorot_normal')(cdr_in)
        cdr_pool7 = GlobalMaxPooling1D()(cdr_conv7)
        cdr_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid',
                           kernel_initializer='glorot_normal')(cdr_in)
        cdr_pool9 = GlobalMaxPooling1D()(cdr_conv9)

        pep_cat = concatenate([pep_pool1, pep_pool3, pep_pool5, pep_pool7, pep_pool9])
        cdr_cat = concatenate([cdr_pool1, cdr_pool3, cdr_pool5, cdr_pool7, cdr_pool9])

        cat = concatenate([pep_cat, cdr_cat], axis=1)
        dense = Dense(32, activation='sigmoid')(cat)
        out = Dense(1, activation='sigmoid')(dense)

        model = Model(inputs=[cdr_in, pep_in], outputs=[out])
        model.compile(loss="binary_crossentropy",
                      optimizer='adam')
        return model

    def _train_model(self, save_path: str) -> None:
        """Train the model on NetTCR-2.0 beta-chain training data."""
        import pandas as pd
        from tensorflow.keras.callbacks import EarlyStopping

        train_file = os.path.join(NETTCR_DATA_DIR, "train_beta_90.csv")
        if not os.path.exists(train_file):
            raise FileNotFoundError(
                f"NetTCR training data not found at {train_file}. "
                f"Please ensure /share/liuyutian/NetTCR-2.0/data/ is accessible."
            )

        train_data = pd.read_csv(train_file)

        pep_train = encode_sequences(train_data.peptide.tolist(), MAX_PEP_LEN)
        cdr_train = encode_sequences(train_data.CDR3b.tolist(), MAX_CDR3_LEN)
        y_train = np.array(train_data.binder, dtype=np.float32)

        early_stop = EarlyStopping(
            monitor='loss', min_delta=0, patience=10,
            verbose=0, mode='min', restore_best_weights=True
        )

        print(f"    Training on {len(y_train)} samples...")
        self._model.fit(
            [cdr_train, pep_train], y_train,
            epochs=100, batch_size=128, verbose=0, callbacks=[early_stop]
        )

        # Save weights
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._model.save_weights(save_path)
        print(f"    NetTCR model saved to {save_path}")

    def _predict_batch(self, tcrs: List[str], peptides: List[str]) -> np.ndarray:
        """Run prediction on a batch of TCR-peptide pairs."""
        cdr_enc = encode_sequences(tcrs, MAX_CDR3_LEN)
        pep_enc = encode_sequences(peptides, MAX_PEP_LEN)
        preds = self._model.predict(
            [cdr_enc, pep_enc], batch_size=self._batch_size, verbose=0
        )
        return np.ravel(preds)

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score a single TCR-peptide pair. Returns (score, confidence=1.0)."""
        scores = self._predict_batch([tcr], [peptide])
        return float(scores[0]), 1.0

    def score_batch(self, tcrs: list, peptides: list, **kwargs) -> Tuple[list, list]:
        """Score a batch of TCR-peptide pairs."""
        scores = self._predict_batch(tcrs, peptides)
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        confidences = [1.0] * len(scores)
        return list(scores), confidences

    def score_batch_fast(self, tcrs: list, peptides: list) -> List[float]:
        """Fast batch scoring for training (no uncertainty)."""
        scores = self._predict_batch(tcrs, peptides)
        if isinstance(scores, np.ndarray):
            return scores.tolist()
        return list(scores)
