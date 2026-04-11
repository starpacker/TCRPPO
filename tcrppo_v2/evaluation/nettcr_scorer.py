"""NetTCR-2.0 CDR3b-only scorer for cross-model validation.

Trains a CNN model (NetTCR architecture) on tc-hard data,
then uses it as an independent binding predictor for Tier 2 evaluation.

Architecture: Multi-kernel CNN on BLOSUM50-encoded CDR3b + peptide sequences,
concatenated, fed through dense layers -> binding probability.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

# Add NetTCR repo to path for utils
NETTCR_DIR = "/share/liuyutian/NetTCR-2.0"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BLOSUM50 encoding (20 standard amino acids)
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

MAX_CDR3_LEN = 30
MAX_PEP_LEN = 15  # Extended from original 9 to handle longer peptides


def encode_sequences(seqs: List[str], max_len: int) -> np.ndarray:
    """BLOSUM50-encode a list of AA sequences with zero-padding."""
    n = len(seqs)
    encoded = np.zeros((n, max_len, 20), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq[:max_len]):
            if aa in BLOSUM50_20AA:
                encoded[i, j] = BLOSUM50_20AA[aa]
    return encoded


def build_nettcr_beta_model(max_pep_len: int = MAX_PEP_LEN,
                             max_cdr3_len: int = MAX_CDR3_LEN) -> "keras.Model":
    """Build NetTCR single-chain (CDR3b) model with flexible peptide length."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow import keras
    from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, concatenate

    cdr_in = Input(shape=(max_cdr3_len, 20), name="cdr3b")
    pep_in = Input(shape=(max_pep_len, 20), name="peptide")

    # Multi-kernel CNN for peptide
    pep_pools = []
    for k in [1, 3, 5, 7, 9]:
        conv = Conv1D(16, k, padding='same', activation='sigmoid',
                      kernel_initializer='glorot_normal')(pep_in)
        pool = GlobalMaxPooling1D()(conv)
        pep_pools.append(pool)

    # Multi-kernel CNN for CDR3b
    cdr_pools = []
    for k in [1, 3, 5, 7, 9]:
        conv = Conv1D(16, k, padding='same', activation='sigmoid',
                      kernel_initializer='glorot_normal')(cdr_in)
        pool = GlobalMaxPooling1D()(conv)
        cdr_pools.append(pool)

    pep_cat = concatenate(pep_pools)
    cdr_cat = concatenate(cdr_pools)
    cat = concatenate([pep_cat, cdr_cat], axis=1)

    dense = Dense(32, activation='sigmoid')(cat)
    out = Dense(1, activation='sigmoid')(dense)

    model = keras.Model(inputs=[cdr_in, pep_in], outputs=[out])
    return model


class NetTCRScorer:
    """NetTCR-2.0 CDR3b scorer for cross-model validation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        train_data_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        epochs: int = 50,
        batch_size: int = 128,
    ):
        """Initialize NetTCR scorer.

        Args:
            model_path: Path to saved model weights. If None, trains from scratch.
            train_data_path: Path to training CSV (CDR3b, peptide, binder).
            test_data_path: Path to test CSV for validation during training.
            epochs: Training epochs if training from scratch.
            batch_size: Batch size for training/inference.
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.batch_size = batch_size
        self.model = build_nettcr_beta_model()

        default_model_path = os.path.join(PROJECT_ROOT, "data", "nettcr_model.weights.h5")

        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
            print(f"NetTCR: Loaded weights from {model_path}")
        elif os.path.exists(default_model_path):
            self.model.load_weights(default_model_path)
            print(f"NetTCR: Loaded weights from {default_model_path}")
        else:
            # Train from scratch
            if train_data_path is None:
                train_data_path = os.path.join(PROJECT_ROOT, "data", "nettcr_train.csv")
            if test_data_path is None:
                test_data_path = os.path.join(PROJECT_ROOT, "data", "nettcr_test.csv")

            if not os.path.exists(train_data_path):
                raise FileNotFoundError(
                    f"No model weights and no training data at {train_data_path}. "
                    "Run data preparation first."
                )

            self._train(train_data_path, test_data_path, epochs, default_model_path)

    def _train(self, train_path: str, test_path: str, epochs: int,
               save_path: str) -> None:
        """Train NetTCR model on CDR3b + peptide data."""
        from tensorflow.keras.optimizers import Adam
        from keras.callbacks import EarlyStopping

        print(f"NetTCR: Training from {train_path}")
        train_df = pd.read_csv(train_path)
        print(f"  Train: {len(train_df)} rows")

        cdr3b_train = encode_sequences(train_df["CDR3b"].tolist(), MAX_CDR3_LEN)
        pep_train = encode_sequences(train_df["peptide"].tolist(), MAX_PEP_LEN)
        y_train = train_df["binder"].values.astype(np.float32)

        self.model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001))

        callbacks = [
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
        ]

        # Add validation data if available
        val_data = None
        if test_path and os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            cdr3b_test = encode_sequences(test_df["CDR3b"].tolist(), MAX_CDR3_LEN)
            pep_test = encode_sequences(test_df["peptide"].tolist(), MAX_PEP_LEN)
            y_test = test_df["binder"].values.astype(np.float32)
            val_data = ([cdr3b_test, pep_test], y_test)
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10,
                              restore_best_weights=True, verbose=1)
            ]
            print(f"  Test: {len(test_df)} rows")

        history = self.model.fit(
            [cdr3b_train, pep_train], y_train,
            epochs=epochs, batch_size=self.batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate
        if val_data is not None:
            from sklearn.metrics import roc_auc_score
            preds = self.model.predict(val_data[0], verbose=0).ravel()
            auc = roc_auc_score(y_test, preds)
            print(f"  Validation AUC: {auc:.4f}")

        # Save weights
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save_weights(save_path)
        print(f"  Saved weights to {save_path}")

    def score_batch(
        self,
        cdr3b_seqs: List[str],
        peptides: List[str],
    ) -> np.ndarray:
        """Score a batch of CDR3b-peptide pairs.

        Args:
            cdr3b_seqs: List of CDR3b sequences.
            peptides: List of peptide sequences (same length as cdr3b_seqs).

        Returns:
            Array of binding probabilities [0, 1].
        """
        assert len(cdr3b_seqs) == len(peptides)
        cdr3b_enc = encode_sequences(cdr3b_seqs, MAX_CDR3_LEN)
        pep_enc = encode_sequences(peptides, MAX_PEP_LEN)
        preds = self.model.predict([cdr3b_enc, pep_enc], batch_size=self.batch_size, verbose=0)
        return preds.ravel()

    def score(self, cdr3b: str, peptide: str) -> float:
        """Score a single CDR3b-peptide pair."""
        return float(self.score_batch([cdr3b], [peptide])[0])

    def compute_auroc(
        self,
        cdr3b_seqs: List[str],
        target_peptide: str,
        decoy_peptides: List[str],
    ) -> float:
        """Compute AUROC: target vs decoy peptides for given TCRs.

        For each TCR: score against target (positive) and each decoy (negative).
        Returns AUROC across all pairs.
        """
        from sklearn.metrics import roc_auc_score

        all_cdr3b = []
        all_pep = []
        all_labels = []

        for tcr in cdr3b_seqs:
            # Target = positive
            all_cdr3b.append(tcr)
            all_pep.append(target_peptide)
            all_labels.append(1)
            # Decoys = negative
            for decoy in decoy_peptides:
                all_cdr3b.append(tcr)
                all_pep.append(decoy)
                all_labels.append(0)

        scores = self.score_batch(all_cdr3b, all_pep)
        try:
            return float(roc_auc_score(all_labels, scores))
        except ValueError:
            return 0.5  # Degenerate case
