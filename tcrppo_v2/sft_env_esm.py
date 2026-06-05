"""SFT environment with real ESM-2 embeddings."""

import numpy as np
import torch
from typing import Tuple, Dict, Optional

from tcrppo_v2.utils.constants import (
    AMINO_ACIDS, AA_TO_IDX, IDX_TO_AA,
    MAX_TCR_LEN, MIN_TCR_LEN, MAX_STEPS_PER_EPISODE,
    OP_SUB, OP_INS, OP_DEL, OP_STOP,
)


class SFTEnvESM:
    """TCR editing environment for SFT training with real ESM-2 embeddings.

    Uses precomputed ESM-2 embeddings for TCR and peptide sequences.
    Falls back to on-the-fly encoding for sequences not in cache.
    """

    def __init__(
        self,
        esm_cache_path: str = 'data/esm2_embeddings.pt',
        max_steps: int = MAX_STEPS_PER_EPISODE,
        max_tcr_len: int = MAX_TCR_LEN,
        min_tcr_len: int = MIN_TCR_LEN,
        device: str = 'cuda',
    ):
        self.max_steps = max_steps
        self.max_tcr_len = max_tcr_len
        self.min_tcr_len = min_tcr_len
        self.device = device

        # AA alphabet
        self.aa_alphabet = AMINO_ACIDS
        self.aa_to_idx = AA_TO_IDX
        self.idx_to_aa = IDX_TO_AA

        # Load ESM-2 embedding cache
        print(f"Loading ESM-2 embeddings from {esm_cache_path}...")
        self.esm_cache = torch.load(esm_cache_path, map_location='cpu')
        print(f"  Loaded {len(self.esm_cache)} cached embeddings")

        # ESM-2 model for on-the-fly encoding (lazy load)
        self._esm_model = None
        self._esm_alphabet = None

        # State
        self.current_tcr = ""
        self.current_peptide = ""
        self.current_hla = "HLA-A*02:01"
        self.step_count = 0

    def _get_esm_model(self):
        """Lazy load ESM-2 model for on-the-fly encoding."""
        if self._esm_model is None:
            print("Loading ESM-2 model for on-the-fly encoding...")
            import esm
            self._esm_model, self._esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self._esm_model = self._esm_model.to(self.device)
            self._esm_model.eval()
            print("  ESM-2 model loaded")
        return self._esm_model, self._esm_alphabet

    def _encode_sequence(self, seq: str) -> torch.Tensor:
        """Get ESM-2 embedding for a sequence (cached or on-the-fly).

        Returns:
            Tensor of shape [1280] (mean-pooled embedding)
        """
        # Check cache first
        if seq in self.esm_cache:
            return self.esm_cache[seq].float()  # [1280]

        # On-the-fly encoding
        model, alphabet = self._get_esm_model()
        batch_converter = alphabet.get_batch_converter()

        data = [("seq", seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_repr = results["representations"][33]  # [1, L+2, 1280]

        # Extract per-residue embeddings (skip BOS and EOS)
        seq_len = len(seq)
        emb = token_repr[0, 1:seq_len+1, :].cpu()  # [seq_len, 1280]
        mean_emb = emb.mean(dim=0)  # [1280]

        # Cache for future use
        self.esm_cache[seq] = mean_emb.half()

        return mean_emb

    def _get_observation(self) -> np.ndarray:
        """Get observation by concatenating TCR and peptide ESM-2 embeddings.

        Returns:
            np.ndarray of shape [2560] (1280 + 1280)
        """
        tcr_emb = self._encode_sequence(self.current_tcr)  # [1280]
        pep_emb = self._encode_sequence(self.current_peptide)  # [1280]

        obs = torch.cat([tcr_emb, pep_emb], dim=0)  # [2560]
        return obs.numpy().astype(np.float32)

    def reset(
        self,
        init_tcr: Optional[str] = None,
        peptide: Optional[str] = None,
        hla: Optional[str] = None,
    ) -> np.ndarray:
        """Reset environment with given TCR/peptide/HLA.

        Returns real ESM-2 observation.
        """
        if init_tcr is None:
            # Random TCR
            length = np.random.randint(self.min_tcr_len, self.max_tcr_len + 1)
            self.current_tcr = ''.join(np.random.choice(list(self.aa_alphabet), size=length))
        else:
            self.current_tcr = init_tcr

        self.current_peptide = peptide or "GILGFVFTL"
        self.current_hla = hla or "HLA-A*02:01"
        self.step_count = 0

        return self._get_observation()

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
            padding = ''.join(np.random.choice(list(self.aa_alphabet),
                                               size=self.min_tcr_len - len(self.current_tcr)))
            self.current_tcr += padding

        # Check done
        done = (op_type == OP_STOP) or (self.step_count >= self.max_steps)

        # Dummy reward and info
        reward = 0.0
        info = {
            'tcr': self.current_tcr,
            'peptide': self.current_peptide,
            'affinity': 0.0,  # Dummy
        }

        # Return real ESM-2 observation
        obs = self._get_observation()
        return obs, reward, done, info

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

        # Token mask: all tokens allowed
        token_mask = np.ones((self.max_tcr_len, len(self.aa_alphabet)), dtype=bool)

        return {
            'op_mask': op_mask,
            'pos_mask': pos_mask,
            'token_mask': token_mask,
        }
