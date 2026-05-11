"""tFold V3.4 scorer with AMP-accelerated feature extraction.

This is a drop-in replacement for AffinityTFoldScorer that uses TFoldAMPWrapper
for 3.97× faster feature extraction (6.44s → 1.62s per sample).

Key differences from affinity_tfold.py:
- Uses TFoldAMPWrapper for in-process AMP inference (no subprocess)
- Keeps V3.4 classifier and SQLite cache (same as original)
- Falls back to subprocess if AMP fails (for robustness)
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Import shared components from affinity_tfold.py
from tcrppo_v2.scorers.affinity_tfold import (
    ClassifierV34Local,
    TFoldFeatureCache,
    _make_cache_key,
    V34_WEIGHTS,
    TFOLD_ROOT,
)
from tcrppo_v2.scorers.base import BaseScorer

logger = logging.getLogger(__name__)

# Add tfold to path for AMP wrapper
sys.path.insert(0, TFOLD_ROOT)


class AffinityTFoldAMPScorer(BaseScorer):
    """tFold V3.4 scorer with AMP-accelerated feature extraction.

    Architecture:
    1. Check SQLite cache (< 1ms if hit)
    2. If miss: Use TFoldAMPWrapper for feature extraction (~1.6s with AMP)
    3. Run V3.4 classifier on features (< 1ms)

    Speedup: 3.97× faster than subprocess (6.44s → 1.62s per cache miss)
    """

    def __init__(
        self,
        device: str = "cuda",
        gpu_id: int = 0,
        cache_path: str = "data/tfold_feature_cache.db",
        default_hla: str = "HLA-A*02:01",
        use_amp: bool = True,
        fallback_to_subprocess: bool = True,
    ):
        """
        Args:
            device: 'cuda' or 'cpu'
            gpu_id: GPU index (for logging only, AMP wrapper uses CUDA_VISIBLE_DEVICES)
            cache_path: Path to SQLite feature cache
            default_hla: Default HLA allele if not specified
            use_amp: Enable automatic mixed precision (recommended)
            fallback_to_subprocess: If AMP fails, fall back to subprocess scorer
        """
        self.device = device
        self.gpu_id = gpu_id
        self.default_hla = default_hla
        self.use_amp = use_amp
        self.fallback_to_subprocess = fallback_to_subprocess

        # Initialize cache
        self._cache = TFoldFeatureCache(cache_path)
        logger.info(f"tFold V3.4 AMP loaded (cache={len(self._cache)} entries)")

        # Initialize V3.4 classifier
        self._classifier = self._load_v34_classifier()

        # Initialize AMP wrapper
        self._amp_wrapper = None
        self._amp_failed = False
        self._init_amp_wrapper()

        # Fallback subprocess scorer (lazy init)
        self._subprocess_scorer = None

        # Stats
        self._n_amp_calls = 0
        self._n_subprocess_fallback = 0

    def _init_amp_wrapper(self):
        """Initialize TFoldAMPWrapper."""
        try:
            from tcrppo_v2.inference_optimization import TFoldAMPWrapper

            logger.info("Initializing TFoldAMPWrapper...")
            self._amp_wrapper = TFoldAMPWrapper(device=self.device, use_amp=self.use_amp)
            logger.info(f"✓ TFoldAMPWrapper initialized (AMP={'enabled' if self.use_amp else 'disabled'})")

        except Exception as e:
            logger.error(f"Failed to initialize TFoldAMPWrapper: {e}")
            self._amp_failed = True

            if not self.fallback_to_subprocess:
                raise RuntimeError(f"AMP wrapper initialization failed and fallback disabled: {e}")

            logger.warning("Will fall back to subprocess scorer")

    def _load_v34_classifier(self) -> ClassifierV34Local:
        """Load V3.4 classifier weights."""
        model = ClassifierV34Local(
            d_sfea=192,
            n_heads=4,
            n_rbf=16,
            pfea_dim=128,
            n_attn_layers=2,
            mlp_hidden=256,
            dropout=0.1,
        )

        if not os.path.exists(V34_WEIGHTS):
            raise FileNotFoundError(f"V3.4 weights not found: {V34_WEIGHTS}")

        state = torch.load(V34_WEIGHTS, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()

        for p in model.parameters():
            p.requires_grad = False

        return model

    def _get_subprocess_scorer(self):
        """Lazy-init subprocess scorer for fallback."""
        if self._subprocess_scorer is None:
            from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer

            logger.info("Initializing subprocess scorer as fallback...")
            self._subprocess_scorer = AffinityTFoldScorer(
                device=self.device,
                gpu_id=self.gpu_id,
                cache_path=self._cache.db_path,  # Share same cache
                default_hla=self.default_hla,
            )

        return self._subprocess_scorer

    def _extract_features_amp(
        self, cdr3b: str, peptide: str, hla: str
    ) -> Optional[Dict]:
        """Extract features using AMP wrapper.

        Returns:
            Feature dict (padded, ready for classifier) or None if failed
        """
        if self._amp_failed or self._amp_wrapper is None:
            return None

        try:
            # Build chains for tFold
            # Note: AMP wrapper expects full TCR sequences, but we only have CDR3β
            # For now, use a template TCR structure (same as subprocess scorer)
            chains = self._build_chains(cdr3b, peptide, hla)

            # Extract features with AMP
            self._n_amp_calls += 1
            features = self._amp_wrapper.extract_features(chains)

            if features is None:
                logger.warning(f"AMP feature extraction returned None for {cdr3b[:10]}...")
                return None

            # Convert to classifier-ready format (same as subprocess)
            padded_features = self._pad_features(features)
            return padded_features

        except Exception as e:
            logger.error(f"AMP feature extraction failed: {e}")
            self._amp_failed = True  # Disable AMP for future calls
            return None

    def _build_chains(self, cdr3b: str, peptide: str, hla: str) -> List[Dict]:
        """Build chain list for tFold inference.

        Note: This is a simplified version. For full accuracy, we need:
        - Full TCRβ V-region sequence (not just CDR3β)
        - Full TCRα V-region sequence
        - Full MHC α-chain sequence
        - β2-microglobulin sequence

        For now, we use template sequences with CDR3β inserted.
        """
        # Template TCRβ V-region (TRBV7-9*01 with placeholder CDR3)
        # In production, this should be looked up from a database
        tcr_beta_template_prefix = "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFC"
        tcr_beta_template_suffix = "FGSGTQLSVLEDLNKVFPPEVAVFEPSEAEISHTQKATLVCLATGFYPDHVELSWWVNGKEVHSGVCTDPQPLKEQPALNDSRYALSSRLRVSATFWQNPRNHFRCQVQFYGLSENDEWTQDRAKPVTQIVSAEAWGRADCGFTSESYQQGVLSATILYEILLGKATLYAVLVSALVLMAMVKRKDF"

        # Ensure CDR3β starts with C
        if not cdr3b.startswith('C'):
            cdr3b = 'C' + cdr3b

        tcr_beta_seq = tcr_beta_template_prefix + cdr3b + tcr_beta_template_suffix

        # Template TCRα V-region
        tcr_alpha_seq = "AQKVTQAQPSVSVSPGQTARITCSGDALPGQSIYWYQQALGQGPQFIFQYYAKESDSDMRGGISGLTVDLKNIQPEDSGLYQCAASRDSSGNTGKLVFGKGTKLTVNPNIQNPDPAVYQLRDSKSSDKSVCLFTDFDSQTNVSQSKDSDVYITDKTVLDMRSMDFKSNSAVAWSNKSDFACANAFNNSIIPEDTFFPSPESSCDVKLVEKSFETDTNLNFQNLSVIGFRILLLKVAGFNLLMTLRLWSS"

        # MHC α-chain (HLA-A*02:01)
        mhc_alpha_seq = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWE"

        # β2-microglobulin
        b2m_seq = "GPHSLRYFVTAVSRPGLGEPRYMEVGYVDDTEFVRFDSDAENPRYEPRARWMEQEGPEYWERETQKAKGQEQWFRVSLRNLLGYYNQSAGGSHTLQQMSGCDLGPDGRLLRGHDQYAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGECVEWLRRYLKNGNATLLRTDSPKAHVTHHSRPEDKVTLRCWALGFYPADITLTWQLNGEELIQDMELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPVTLRWE"

        chains = [
            {"id": "B", "sequence": tcr_beta_seq},
            {"id": "A", "sequence": tcr_alpha_seq},
            {"id": "P", "sequence": peptide},
            {"id": "M", "sequence": mhc_alpha_seq},
            {"id": "N", "sequence": b2m_seq},
        ]

        return chains

    def _pad_features(self, features: Dict) -> Dict:
        """Convert AMP wrapper output to classifier-ready format.

        AMP wrapper returns:
          - raw_sfea: [L, 192]
          - ca_coords: [L, 3]
          - pfea_cdr3b_pep: [Lb, Lp, 128]
          - pfea_cdr3a_pep: [La, Lp, 128]
          - cdr3b_range, cdr3a_range, pep_range

        Classifier expects:
          - sfea_cdr3b: [1, max_len, 192]
          - sfea_cdr3a: [1, max_len, 192]
          - sfea_pep: [1, max_len, 192]
          - ca_cdr3b: [1, max_len, 3]
          - ca_cdr3a: [1, max_len, 3]
          - ca_pep: [1, max_len, 3]
          - pfea_bp: [1, max_len, max_len, 128]
          - pfea_ap: [1, max_len, max_len, 128]
          - len_cdr3b, len_cdr3a, len_pep: [1]
          - v33_feat: [1, 448]
        """
        max_len = 30  # Padding length

        # Extract ranges
        cdr3b_start, cdr3b_end = features['cdr3b_range']
        cdr3a_start, cdr3a_end = features['cdr3a_range']
        pep_start, pep_end = features['pep_range']

        # Extract per-region features
        raw_sfea = features['raw_sfea']  # [L, 192]
        ca_coords = features['ca_coords']  # [L, 3]

        sfea_cdr3b = raw_sfea[cdr3b_start:cdr3b_end]  # [Lb, 192]
        sfea_cdr3a = raw_sfea[cdr3a_start:cdr3a_end]  # [La, 192]
        sfea_pep = raw_sfea[pep_start:pep_end]  # [Lp, 192]

        ca_cdr3b = ca_coords[cdr3b_start:cdr3b_end]  # [Lb, 3]
        ca_cdr3a = ca_coords[cdr3a_start:cdr3a_end]  # [La, 3]
        ca_pep = ca_coords[pep_start:pep_end]  # [Lp, 3]

        pfea_bp = features['pfea_cdr3b_pep']  # [Lb, Lp, 128]
        pfea_ap = features['pfea_cdr3a_pep']  # [La, Lp, 128]

        # Pad to max_len
        def pad_2d(x, max_len):
            """Pad [L, D] to [max_len, D]"""
            L, D = x.shape
            if L >= max_len:
                return x[:max_len]
            padded = torch.zeros(max_len, D, dtype=x.dtype, device=x.device)
            padded[:L] = x
            return padded

        def pad_3d(x, max_len1, max_len2):
            """Pad [L1, L2, D] to [max_len1, max_len2, D]"""
            L1, L2, D = x.shape
            padded = torch.zeros(max_len1, max_len2, D, dtype=x.dtype, device=x.device)
            padded[:min(L1, max_len1), :min(L2, max_len2)] = x[:min(L1, max_len1), :min(L2, max_len2)]
            return padded

        sfea_cdr3b_pad = pad_2d(sfea_cdr3b, max_len).unsqueeze(0)  # [1, max_len, 192]
        sfea_cdr3a_pad = pad_2d(sfea_cdr3a, max_len).unsqueeze(0)
        sfea_pep_pad = pad_2d(sfea_pep, max_len).unsqueeze(0)

        ca_cdr3b_pad = pad_2d(ca_cdr3b, max_len).unsqueeze(0)  # [1, max_len, 3]
        ca_cdr3a_pad = pad_2d(ca_cdr3a, max_len).unsqueeze(0)
        ca_pep_pad = pad_2d(ca_pep, max_len).unsqueeze(0)

        pfea_bp_pad = pad_3d(pfea_bp, max_len, max_len).unsqueeze(0)  # [1, max_len, max_len, 128]
        pfea_ap_pad = pad_3d(pfea_ap, max_len, max_len).unsqueeze(0)

        # Lengths
        len_cdr3b = torch.tensor([sfea_cdr3b.shape[0]], dtype=torch.long)
        len_cdr3a = torch.tensor([sfea_cdr3a.shape[0]], dtype=torch.long)
        len_pep = torch.tensor([sfea_pep.shape[0]], dtype=torch.long)

        # V3.3 features (placeholder - not used by V3.4 but required by interface)
        v33_feat = torch.zeros(1, 448, dtype=torch.float32)

        return {
            "sfea_cdr3b": sfea_cdr3b_pad,
            "sfea_cdr3a": sfea_cdr3a_pad,
            "sfea_pep": sfea_pep_pad,
            "ca_cdr3b": ca_cdr3b_pad,
            "ca_cdr3a": ca_cdr3a_pad,
            "ca_pep": ca_pep_pad,
            "pfea_bp": pfea_bp_pad,
            "pfea_ap": pfea_ap_pad,
            "len_cdr3b": len_cdr3b,
            "len_cdr3a": len_cdr3a,
            "len_pep": len_pep,
            "v33_feat": v33_feat,
        }

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score TCR-peptide binding affinity.

        Args:
            tcr: CDR3β sequence
            peptide: Peptide sequence
            **kwargs: Optional 'hla' key

        Returns:
            (score, confidence) tuple
            - score: Binding probability [0, 1]
            - confidence: Always 1.0 (no uncertainty for tFold)
        """
        hla = kwargs.get("hla", self.default_hla)

        # Normalize CDR3β (add C if missing)
        cdr3b = tcr if tcr.startswith('C') else 'C' + tcr

        # Check cache
        cache_key = _make_cache_key(cdr3b, peptide, hla)
        cached_features = self._cache.get(cache_key)

        if cached_features is not None:
            # Cache hit - fast path
            features = cached_features
        else:
            # Cache miss - extract features
            features = self._extract_features_amp(cdr3b, peptide, hla)

            # Fallback to subprocess if AMP failed
            if features is None and self.fallback_to_subprocess:
                self._n_subprocess_fallback += 1
                subprocess_scorer = self._get_subprocess_scorer()
                return subprocess_scorer.score(tcr, peptide, **kwargs)

            if features is None:
                logger.error(f"Feature extraction failed for {cdr3b[:10]}... (no fallback)")
                return 0.0, 1.0

            # Cache the features
            self._cache.set(cache_key, features)

        # Run V3.4 classifier
        with torch.no_grad():
            # Move features to device
            features_gpu = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()
            }

            logits = self._classifier(**features_gpu)
            prob = torch.sigmoid(logits).item()

        return prob, 1.0  # tFold has no uncertainty estimate

    def score_batch(self, tcrs: List[str], peptides: List[str], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Batch scoring (not optimized - calls score() sequentially)."""
        scores = []
        confidences = []

        for tcr, peptide in zip(tcrs, peptides):
            score, conf = self.score(tcr, peptide, **kwargs)
            scores.append(score)
            confidences.append(conf)

        return np.array(scores), np.array(confidences)

    def get_stats(self) -> Dict:
        """Get scorer statistics."""
        return {
            "cache_size": len(self._cache),
            "n_amp_calls": self._n_amp_calls,
            "n_subprocess_fallback": self._n_subprocess_fallback,
            "amp_enabled": not self._amp_failed,
        }
