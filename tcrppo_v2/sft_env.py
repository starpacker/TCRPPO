"""Simplified environment for SFT training (no reward computation needed)."""

import numpy as np
from typing import Tuple, Dict, Optional

from tcrppo_v2.utils.constants import (
    AMINO_ACIDS, AA_TO_IDX, IDX_TO_AA,
    MAX_TCR_LEN, MIN_TCR_LEN, MAX_STEPS_PER_EPISODE,
    NUM_AMINO_ACIDS,
    OP_SUB, OP_INS, OP_DEL, OP_STOP,
)


class SFTEnv:
    """Simplified TCR editing environment for SFT training.

    Only handles sequence editing logic, no reward computation.
    Encodes current TCR + peptide as one-hot into a fixed-size observation.
    """

    # obs layout: [tcr_onehot (max_tcr_len * 20), pep_onehot (max_pep_len * 20),
    #              tcr_len_frac, step_frac, padding_to_2560]
    MAX_PEP_LEN = 25

    def __init__(
        self,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        max_tcr_len: int = MAX_TCR_LEN,
        min_tcr_len: int = MIN_TCR_LEN,
    ):
        self.max_steps = max_steps
        self.max_tcr_len = max_tcr_len
        self.min_tcr_len = min_tcr_len
        self.obs_dim = 2560  # keep compatible with policy obs_dim

        # AA alphabet
        self.aa_alphabet = AMINO_ACIDS
        self.aa_to_idx = AA_TO_IDX
        self.idx_to_aa = IDX_TO_AA

        # State
        self.current_tcr = ""
        self.current_peptide = ""
        self.current_hla = "HLA-A*02:01"
        self.step_count = 0

    def _encode_obs(self) -> np.ndarray:
        """Encode current TCR + peptide as one-hot observation."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # TCR one-hot: max_tcr_len * 20 = 25 * 20 = 500
        tcr_offset = 0
        for i, aa in enumerate(self.current_tcr[:self.max_tcr_len]):
            idx = self.aa_to_idx.get(aa, 0)
            obs[tcr_offset + i * NUM_AMINO_ACIDS + idx] = 1.0

        # Peptide one-hot: max_pep_len * 20 = 25 * 20 = 500
        pep_offset = self.max_tcr_len * NUM_AMINO_ACIDS
        for i, aa in enumerate(self.current_peptide[:self.MAX_PEP_LEN]):
            idx = self.aa_to_idx.get(aa, 0)
            obs[pep_offset + i * NUM_AMINO_ACIDS + idx] = 1.0

        # Scalar features
        scalar_offset = pep_offset + self.MAX_PEP_LEN * NUM_AMINO_ACIDS
        obs[scalar_offset] = len(self.current_tcr) / self.max_tcr_len  # TCR length fraction
        obs[scalar_offset + 1] = self.step_count / self.max_steps  # step fraction

        return obs

    def reset(
        self,
        init_tcr: Optional[str] = None,
        peptide: Optional[str] = None,
        hla: Optional[str] = None,
    ) -> np.ndarray:
        """Reset environment with given TCR/peptide/HLA."""
        if init_tcr is None:
            length = np.random.randint(self.min_tcr_len, self.max_tcr_len + 1)
            self.current_tcr = ''.join(np.random.choice(self.aa_alphabet, size=length))
        else:
            self.current_tcr = init_tcr

        self.current_peptide = peptide or "GILGFVFTL"
        self.current_hla = hla or "HLA-A*02:01"
        self.step_count = 0

        return self._encode_obs()

    def step(self, action: Tuple[int, int, str]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply action and return next observation.

        Args:
            action: (op_type, position, token)

        Returns:
            (obs, reward, done, info)
        """
        op_type, position, token = action
        self.step_count += 1

        # Apply action
        if op_type == OP_SUB:
            if 0 <= position < len(self.current_tcr):
                tcr_list = list(self.current_tcr)
                tcr_list[position] = token
                self.current_tcr = ''.join(tcr_list)
        elif op_type == OP_INS:
            if 0 <= position <= len(self.current_tcr):
                self.current_tcr = self.current_tcr[:position] + token + self.current_tcr[position:]
        elif op_type == OP_DEL:
            if 0 <= position < len(self.current_tcr):
                self.current_tcr = self.current_tcr[:position] + self.current_tcr[position+1:]
        elif op_type == OP_STOP:
            pass  # No-op

        # Clip length
        if len(self.current_tcr) > self.max_tcr_len:
            self.current_tcr = self.current_tcr[:self.max_tcr_len]
        if len(self.current_tcr) < self.min_tcr_len:
            # Pad with random AAs
            padding = ''.join(np.random.choice(self.aa_alphabet,
                                               size=self.min_tcr_len - len(self.current_tcr)))
            self.current_tcr += padding

        # Check done
        done = (op_type == OP_STOP) or (self.step_count >= self.max_steps)

        reward = 0.0
        info = {
            'tcr': self.current_tcr,
            'peptide': self.current_peptide,
            'affinity': 0.0,
        }

        return self._encode_obs(), reward, done, info

    def get_action_mask(self) -> Dict[str, np.ndarray]:
        """Get action mask for current state."""
        tcr_len = len(self.current_tcr)

        # Op mask: all ops allowed except STOP at step 0
        op_mask = np.ones(4, dtype=bool)
        if self.step_count == 0:
            op_mask[OP_STOP] = False

        # Position mask: valid positions for current TCR length
        pos_mask = np.zeros(self.max_tcr_len, dtype=bool)
        pos_mask[:tcr_len] = True  # SUB/DEL positions
        if tcr_len < self.max_tcr_len:
            pos_mask[tcr_len] = True  # INS position

        # Token mask: all tokens allowed (simplified, no per-position masking)
        token_mask = np.ones((self.max_tcr_len, len(self.aa_alphabet)), dtype=bool)

        return {
            'op_mask': op_mask,
            'pos_mask': pos_mask,
            'token_mask': token_mask,
        }
