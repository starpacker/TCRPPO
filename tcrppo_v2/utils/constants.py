"""Constants extracted from v1 config.py and data_utils.py."""

import os

# Amino acid alphabet (standard 20)
AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")
NUM_AMINO_ACIDS = len(AMINO_ACIDS)

# Mapping dicts
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

# ERGO uses 1-indexed with 0 as PAD
ERGO_AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
ERGO_IDX_TO_AA = {i + 1: aa for i, aa in enumerate(AMINO_ACIDS)}

# ERGO AE specific: 21-class (20 AA + X terminator), 0-indexed
ERGO_TCR_ATOX = {aa: i for i, aa in enumerate(AMINO_ACIDS + ["X"])}
ERGO_PEP_ATOX = {aa: i for i, aa in enumerate(["PAD"] + AMINO_ACIDS)}

# Sequence length limits
MAX_TCR_LEN = 27
MIN_TCR_LEN = 8
MAX_PEP_LEN = 25
ERGO_MAX_LEN = 28  # ERGO pads to 28
HLA_PSEUDOSEQ_LEN = 34

# PPO environment
MAX_STEPS_PER_EPISODE = 8

# Action space
OP_SUB = 0
OP_INS = 1
OP_DEL = 2
OP_STOP = 3
NUM_OPS = 4

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERGO_DIR = os.path.join(PROJECT_ROOT, "tcrppo_v2", "ERGO")
ERGO_AE_FILE = os.path.join(ERGO_DIR, "TCR_Autoencoder", "tcr_ae_dim_100.pt")
ERGO_MODEL_DIR = os.path.join(ERGO_DIR, "models")

# External data paths
DECOY_LIBRARY_PATH = "/share/liuyutian/pMHC_decoy_library"
TCRDB_PATH = "/share/liuyutian/TCRPPO/data/tcrdb"
TEST_PEPTIDES_PATH = "/share/liuyutian/TCRPPO/data/test_peptides"
V1_CHECKPOINT_PATH = "/share/liuyutian/TCRPPO/output"
