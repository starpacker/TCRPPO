#!/usr/bin/env python
"""Convert NetTCR TensorFlow weights to PyTorch format."""

import os
import sys
import numpy as np
import torch

# Temporarily set CUDA_VISIBLE_DEVICES to empty to avoid GPU conflicts during conversion
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras

# Import PyTorch model
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')
from tcrppo_v2.scorers.affinity_nettcr_pytorch import NetTCRModel, MAX_CDR3_LEN, MAX_PEP_LEN


def build_tf_model():
    """Build the original TensorFlow NetTCR model."""
    from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, concatenate
    from tensorflow.keras.models import Model

    cdr_in = Input(shape=(MAX_CDR3_LEN, 20))
    pep_in = Input(shape=(MAX_PEP_LEN, 20))

    # Peptide tower
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

    # CDR3 tower
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

    # Concatenate
    concat = concatenate([pep_pool1, pep_pool3, pep_pool5, pep_pool7, pep_pool9,
                         cdr_pool1, cdr_pool3, cdr_pool5, cdr_pool7, cdr_pool9])

    # Dense layers
    dense1 = Dense(32, activation='sigmoid', kernel_initializer='glorot_normal')(concat)
    output = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')(dense1)

    model = Model(inputs=[cdr_in, pep_in], outputs=output)
    return model


def convert_weights():
    """Convert TensorFlow weights to PyTorch format."""

    # Load TensorFlow model
    print("Loading TensorFlow model...")
    tf_model = build_tf_model()
    tf_weights_path = "/share/liuyutian/tcrppo_v2/data/nettcr_model.weights.h5"

    if not os.path.exists(tf_weights_path):
        print(f"ERROR: TensorFlow weights not found at {tf_weights_path}")
        return False

    tf_model.load_weights(tf_weights_path)
    print(f"  Loaded TF weights from {tf_weights_path}")

    # Create PyTorch model
    print("Creating PyTorch model...")
    pt_model = NetTCRModel()

    # Get TF layer weights
    tf_layers = {layer.name: layer for layer in tf_model.layers}

    # Map TF layers to PyTorch layers
    # TF Conv1D weights: (kernel_size, in_channels, out_channels)
    # PyTorch Conv1d weights: (out_channels, in_channels, kernel_size)

    layer_mapping = [
        # Peptide tower
        ('conv1d', 'pep_conv1'),
        ('conv1d_1', 'pep_conv3'),
        ('conv1d_2', 'pep_conv5'),
        ('conv1d_3', 'pep_conv7'),
        ('conv1d_4', 'pep_conv9'),
        # CDR3 tower
        ('conv1d_5', 'cdr_conv1'),
        ('conv1d_6', 'cdr_conv3'),
        ('conv1d_7', 'cdr_conv5'),
        ('conv1d_8', 'cdr_conv7'),
        ('conv1d_9', 'cdr_conv9'),
        # Dense layers
        ('dense', 'fc1'),
        ('dense_1', 'fc2'),
    ]

    pt_state_dict = {}

    for tf_name, pt_name in layer_mapping:
        if tf_name not in tf_layers:
            print(f"  WARNING: TF layer {tf_name} not found")
            continue

        tf_layer = tf_layers[tf_name]
        weights = tf_layer.get_weights()

        if len(weights) == 0:
            print(f"  WARNING: No weights in {tf_name}")
            continue

        if 'conv' in pt_name:
            # Conv layer: kernel + bias
            kernel, bias = weights
            # TF: (kernel_size, in_channels, out_channels)
            # PT: (out_channels, in_channels, kernel_size)
            kernel_pt = np.transpose(kernel, (2, 1, 0))
            pt_state_dict[f'{pt_name}.weight'] = torch.from_numpy(kernel_pt)
            pt_state_dict[f'{pt_name}.bias'] = torch.from_numpy(bias)
            print(f"  Converted {tf_name} -> {pt_name}: kernel {kernel.shape} -> {kernel_pt.shape}")

        elif 'fc' in pt_name:
            # Dense layer: weight + bias
            weight, bias = weights
            # TF: (in_features, out_features)
            # PT: (out_features, in_features)
            weight_pt = np.transpose(weight, (1, 0))
            pt_state_dict[f'{pt_name}.weight'] = torch.from_numpy(weight_pt)
            pt_state_dict[f'{pt_name}.bias'] = torch.from_numpy(bias)
            print(f"  Converted {tf_name} -> {pt_name}: weight {weight.shape} -> {weight_pt.shape}")

    # Load into PyTorch model
    pt_model.load_state_dict(pt_state_dict)

    # Save PyTorch weights
    output_path = "/share/liuyutian/tcrppo_v2/data/nettcr_pytorch.pt"
    torch.save(pt_state_dict, output_path)
    print(f"\nSaved PyTorch weights to {output_path}")

    # Verify conversion with a test prediction
    print("\nVerifying conversion...")
    test_cdr = "CASSLAPGATNEKLFF"
    test_pep = "GILGFVFTL"

    # TF prediction
    from tcrppo_v2.scorers.affinity_nettcr import encode_sequences as tf_encode
    cdr_enc_tf = tf_encode([test_cdr], MAX_CDR3_LEN)
    pep_enc_tf = tf_encode([test_pep], MAX_PEP_LEN)
    tf_pred = tf_model.predict([cdr_enc_tf, pep_enc_tf], verbose=0)[0][0]

    # PyTorch prediction
    from tcrppo_v2.scorers.affinity_nettcr_pytorch import encode_sequences as pt_encode
    cdr_enc_pt = pt_encode([test_cdr], MAX_CDR3_LEN)
    pep_enc_pt = pt_encode([test_pep], MAX_PEP_LEN)
    pt_model.eval()
    with torch.no_grad():
        pt_pred = pt_model(cdr_enc_pt, pep_enc_pt).item()

    print(f"  TF prediction: {tf_pred:.6f}")
    print(f"  PT prediction: {pt_pred:.6f}")
    print(f"  Difference: {abs(tf_pred - pt_pred):.6f}")

    if abs(tf_pred - pt_pred) < 0.01:
        print("\n✓ Conversion successful!")
        return True
    else:
        print("\n✗ Conversion failed - predictions don't match!")
        return False


if __name__ == "__main__":
    success = convert_weights()
    sys.exit(0 if success else 1)
