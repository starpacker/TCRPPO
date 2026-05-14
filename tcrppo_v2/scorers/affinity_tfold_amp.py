"""tFold V3.4 scorer with in-process AMP-accelerated feature extraction.

Drop-in replacement for AffinityTFoldScorer. Instead of calling tFold via
subprocess/socket, loads the tFold predictor in-process and runs inference
with AMP (automatic mixed precision) for ~4x speedup on cache misses.

Uses the same feature_extraction API as tfold_feature_server.py:
  fe.load_tfold_predictor() -> fe.extract_features_from_chains() ->
  fe.extract_structured_features() -> pad -> classify
"""

import importlib
import logging
import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tcrppo_v2.scorers.affinity_tfold import (
    ClassifierV34Local,
    TFoldFeatureCache,
    _make_cache_key,
    V34_WEIGHTS,
    TFOLD_ROOT,
)
from tcrppo_v2.scorers.base import BaseScorer

logger = logging.getLogger(__name__)

# Add tFold to path
sys.path.insert(0, TFOLD_ROOT)

# Standard amino acids (for cleaning)
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# V-region scaffolds — same as tfold_feature_server.py / tfold_feature_worker.py
_DEFAULT_VREGION_BETA_SCAFFOLD = (
    "MGTSLLCWMALCLLGADHADTGVSQNPRHNITKRGQNVTFRCDPISEHNRLYWYRQTLGQGPEFLT"
    "YFQNEAQLEKSRLLSDRFSAERPKGSFSTLEIQRTEQGDSAMYL"
)
_CDR3_POS_BETA = 92

_DEFAULT_VREGION_ALPHA_SCAFFOLD = (
    "MAMLLGASVLILWLQPDWVNSQQKNDDQQVKQNSPSLSVQEGRISILNCDYTNSMFDYFLWYKKYP"
    "ASGPELISLIYLQGFNPKESGIATLYEQPTAASATGLTSANTKSQTSVEFQLS"
)
_DEFAULT_CDR3_ALPHA = "CAVNFGNEKLTF"


def _splice_cdr3_into_scaffold(scaffold: str, cdr3: str, cdr3_pos: int,
                                fgxg_suffix: str = "FGXG") -> str:
    """Splice a CDR3 sequence into a V-region scaffold."""
    framework = scaffold[:cdr3_pos]
    return framework + cdr3 + fgxg_suffix


def _clean_seq(s: str) -> str:
    """Remove non-standard amino acids."""
    return "".join(c for c in s if c in STANDARD_AA)


class AffinityTFoldAMPScorer(BaseScorer):
    """tFold V3.4 scorer with in-process AMP-accelerated feature extraction.

    Uses the same tFold pipeline as the subprocess server but runs in-process
    with AMP for ~4x speedup on cache misses.
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
        self.device = device
        self.gpu_id = gpu_id
        self.default_hla = default_hla
        self.use_amp = use_amp and (device == "cuda")
        self.fallback_to_subprocess = fallback_to_subprocess
        self.cache_path = cache_path

        # Initialize cache (shared with subprocess scorer)
        self._cache = TFoldFeatureCache(cache_path)
        logger.info(f"tFold V3.4 AMP scorer (cache at {cache_path})")

        # Initialize V3.4 classifier
        self._classifier = self._load_v34_classifier()

        # Initialize tFold predictor in-process
        self._predictor = None
        self._hla_lookup = {}
        self._amp_failed = False
        self._init_predictor()

        # Import feature extraction module
        self._fe = importlib.import_module("TCR_PMHC_pred.4_16.feature_extraction")

        # Fallback subprocess scorer (lazy init)
        self._subprocess_scorer = None

        # Stats
        self._n_amp_calls = 0
        self._n_subprocess_fallback = 0
        self._n_scored = 0
        # Failed extractions must be worse than any plausible low binder logit.
        self._default_raw_score = -20.0

    def _init_predictor(self):
        """Load tFold predictor in-process via TFoldAMPWrapper."""
        try:
            from tcrppo_v2.inference_optimization.tfold_amp_wrapper import TFoldAMPWrapper
            hla_mod = importlib.import_module("TCR_PMHC_pred.4_16.hla_mapping")

            ppi_path = os.path.join(TFOLD_ROOT, "checkpoints", "esm_ppi_650m_tcr.pth")
            trunk_path = os.path.join(TFOLD_ROOT, "checkpoints", "tfold_tcr_pmhc_trunk.pth")

            logger.info(f"Loading tFold predictor on {self.device}...")
            self._predictor = TFoldAMPWrapper(
                ppi_path=ppi_path,
                trunk_path=trunk_path,
                device=self.device,
                use_amp=self.use_amp,
            )
            logger.info("tFold predictor loaded via TFoldAMPWrapper")

            # Build HLA lookup
            logger.info("Building HLA lookup...")
            self._hla_seq_lookup = hla_mod.build_hla_lookup()
            # Pre-resolve default HLA
            mhc_chains = hla_mod.resolve_mhc_chains(self.default_hla, self._hla_seq_lookup)
            if mhc_chains:
                self._hla_lookup[self.default_hla] = mhc_chains
            self._hla_mod = hla_mod
            logger.info(f"HLA lookup ready ({len(self._hla_seq_lookup)} entries)")

        except Exception as e:
            logger.error(f"Failed to initialize tFold predictor: {e}")
            traceback.print_exc()
            self._amp_failed = True
            if not self.fallback_to_subprocess:
                raise RuntimeError(f"tFold predictor init failed and fallback disabled: {e}")
            logger.warning("Will fall back to subprocess scorer")

    def _load_v34_classifier(self) -> ClassifierV34Local:
        """Load V3.4 classifier weights."""
        if not os.path.exists(V34_WEIGHTS):
            raise FileNotFoundError(f"V3.4 weights not found: {V34_WEIGHTS}")

        ckpt = torch.load(V34_WEIGHTS, map_location="cpu", weights_only=False)
        config = ckpt.get("model_config", {})

        model = ClassifierV34Local(
            d_sfea=config.get("d_sfea", 192),
            n_heads=config.get("n_heads", 4),
            n_rbf=config.get("n_rbf", 16),
            pfea_dim=config.get("pfea_dim", 128),
            n_attn_layers=config.get("n_attn_layers", 2),
            mlp_hidden=config.get("mlp_hidden", 256),
            dropout=config.get("dropout", 0.1),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def _resolve_hla(self, hla: str) -> Optional[Dict]:
        """Resolve HLA allele to MHC chain sequences, with caching."""
        if hla in self._hla_lookup:
            return self._hla_lookup[hla]
        try:
            mhc_chains = self._hla_mod.resolve_mhc_chains(hla, self._hla_seq_lookup)
            if mhc_chains:
                self._hla_lookup[hla] = mhc_chains
                return mhc_chains
        except Exception:
            pass
        return None

    def _build_chains(self, cdr3b: str, peptide: str, hla: str) -> Optional[List[Dict]]:
        """Build tFold chain list — same method as tfold_feature_server.py."""
        # Splice CDR3b into V-region scaffold
        vregion_beta = _splice_cdr3_into_scaffold(
            _DEFAULT_VREGION_BETA_SCAFFOLD, cdr3b, _CDR3_POS_BETA, "FGSG"
        )
        vregion_alpha = _splice_cdr3_into_scaffold(
            _DEFAULT_VREGION_ALPHA_SCAFFOLD, _DEFAULT_CDR3_ALPHA, 90, "FGKG"
        )

        # Resolve HLA
        mhc_info = self._resolve_hla(hla)
        if mhc_info is None:
            logger.warning(f"Failed to resolve HLA: {hla}")
            return None

        mhc_full = mhc_info.get("M", "")
        b2m = mhc_info.get("N", "")
        if not mhc_full:
            return None

        chains = [
            {"id": "B", "sequence": _clean_seq(vregion_beta)},
            {"id": "A", "sequence": _clean_seq(vregion_alpha)},
            {"id": "M", "sequence": _clean_seq(mhc_full)},
        ]
        if b2m:
            chains.append({"id": "N", "sequence": _clean_seq(b2m)})
        chains.append({"id": "P", "sequence": _clean_seq(peptide)})

        return chains

    def _extract_features_amp(self, cdr3b: str, peptide: str, hla: str) -> Optional[Dict]:
        """Extract features using in-process tFold with AMP Wrapper."""
        if self._amp_failed or self._predictor is None:
            return None

        try:
            chains = self._build_chains(cdr3b, peptide, hla)
            if chains is None:
                return None

            self._n_amp_calls += 1

            # Run tFold inference
            out = self._predictor.extract_features(chains)
            if out is None:
                logger.warning(f"tFold AMP extraction failed for {cdr3b[:10]}...")
                return None
                
            raw_sfea = out["raw_sfea"].float()
            ca_coords = out["ca_coords"].float()

            # Helper to slice
            def _slice(tensor, rng):
                if rng is None:
                    return torch.zeros((0, tensor.shape[-1]), dtype=tensor.dtype)
                return tensor[rng[0]:rng[1]]

            sfea_cdr3b = _slice(raw_sfea, out["cdr3b_range"])
            sfea_cdr3a = _slice(raw_sfea, out["cdr3a_range"])
            sfea_pep = _slice(raw_sfea, out["pep_range"])
            
            ca_cdr3b = _slice(ca_coords, out["cdr3b_range"])
            ca_cdr3a = _slice(ca_coords, out["cdr3a_range"])
            ca_pep = _slice(ca_coords, out["pep_range"])

            if sfea_cdr3b.shape[0] == 0 or sfea_cdr3a.shape[0] == 0 or sfea_pep.shape[0] == 0:
                logger.warning(f"Interface slicing failed for {cdr3b[:10]}...")
                return None

            pfea_cdr3b_pep = (
                out["pfea_cdr3b_pep"].float()
                if out["pfea_cdr3b_pep"] is not None
                else torch.zeros((sfea_cdr3b.shape[0], sfea_pep.shape[0], 128), dtype=torch.float32)
            )
            pfea_cdr3a_pep = (
                out["pfea_cdr3a_pep"].float()
                if out["pfea_cdr3a_pep"] is not None
                else torch.zeros((sfea_cdr3a.shape[0], sfea_pep.shape[0], 128), dtype=torch.float32)
            )

            # Match TCR_PMHC_pred.4_16.feature_extraction.extract_structured_features().
            interface_sfea = torch.cat([sfea_cdr3b, sfea_cdr3a, sfea_pep], dim=0)
            sfea_pooled = interface_sfea.mean(dim=0)
            pfea_b_pool = pfea_cdr3b_pep.mean(dim=(0, 1))
            pfea_a_pool = pfea_cdr3a_pep.mean(dim=(0, 1))
            v33_feat = torch.cat([sfea_pooled, pfea_b_pool, pfea_a_pool], dim=0)

            structured = {
                "sfea_cdr3b": sfea_cdr3b,
                "sfea_cdr3a": sfea_cdr3a,
                "sfea_pep": sfea_pep,
                "ca_cdr3b": ca_cdr3b,
                "ca_cdr3a": ca_cdr3a,
                "ca_pep": ca_pep,
                "pfea_cdr3b_pep": pfea_cdr3b_pep,
                "pfea_cdr3a_pep": pfea_cdr3a_pep,
                "v33_feat": v33_feat,
            }

            padded = self._pad_structured(structured)
            tensor_values = [v for v in padded.values() if isinstance(v, torch.Tensor)]
            if any(not torch.isfinite(v).all() for v in tensor_values):
                logger.warning(f"Non-finite AMP features detected for {cdr3b[:10]}...")
                return None
            return padded

        except Exception as e:
            logger.error(f"AMP feature extraction failed: {e}")
            traceback.print_exc()
            return None

    def _pad_structured(self, structured: Dict) -> Dict:
        """Pad structured features to fixed sizes for classifier.

        Same padding as tfold_feature_server.py / tfold_feature_worker.py.
        """
        MAX_CDR3, MAX_PEP = 25, 20

        def pad_2d(t, max_len):
            L, D = t.shape
            if L >= max_len:
                return t[:max_len]
            return torch.cat([t, torch.zeros(max_len - L, D)], dim=0)

        def pad_3d(t, max_r, max_c):
            R, C, D = t.shape
            out = torch.zeros(max_r, max_c, D)
            r, c = min(R, max_r), min(C, max_c)
            out[:r, :c, :] = t[:r, :c, :]
            return out

        Lb = structured["sfea_cdr3b"].shape[0]
        La = structured["sfea_cdr3a"].shape[0]
        Lp = structured["sfea_pep"].shape[0]

        return {
            "sfea_cdr3b": pad_2d(structured["sfea_cdr3b"], MAX_CDR3),
            "sfea_cdr3a": pad_2d(structured["sfea_cdr3a"], MAX_CDR3),
            "sfea_pep": pad_2d(structured["sfea_pep"], MAX_PEP),
            "ca_cdr3b": pad_2d(structured["ca_cdr3b"], MAX_CDR3),
            "ca_cdr3a": pad_2d(structured["ca_cdr3a"], MAX_CDR3),
            "ca_pep": pad_2d(structured["ca_pep"], MAX_PEP),
            "pfea_cdr3b_pep": pad_3d(structured["pfea_cdr3b_pep"], MAX_CDR3, MAX_PEP),
            "pfea_cdr3a_pep": pad_3d(structured["pfea_cdr3a_pep"], MAX_CDR3, MAX_PEP),
            "v33_feat": structured["v33_feat"],
            "len_cdr3b": min(Lb, MAX_CDR3),
            "len_cdr3a": min(La, MAX_CDR3),
            "len_pep": min(Lp, MAX_PEP),
        }

    def _get_subprocess_scorer(self):
        """Lazy-init subprocess scorer for fallback."""
        if self._subprocess_scorer is None:
            from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer

            logger.info("Initializing subprocess scorer as fallback...")
            self._subprocess_scorer = AffinityTFoldScorer(
                device=self.device,
                gpu_id=self.gpu_id,
                cache_path=self.cache_path,
                default_hla=self.default_hla,
            )
        return self._subprocess_scorer

    def score(self, tcr: str, peptide: str, **kwargs) -> Tuple[float, float]:
        """Score TCR-peptide binding affinity.

        Returns:
            (binding_logit, confidence) where binding_logit is -gate_logit.
        """
        # Normalize CDR3b
        cdr3b = tcr if tcr.startswith('C') else 'C' + tcr
        hla = kwargs.get("hla", self.default_hla)

        features = self._get_features_batch([cdr3b], [peptide], [hla])[0]
        if features is None:
            if self.fallback_to_subprocess:
                self._n_subprocess_fallback += 1
                subprocess_scorer = self._get_subprocess_scorer()
                return subprocess_scorer.score(tcr, peptide, **kwargs)
            logger.error(f"Feature extraction failed for {cdr3b[:10]}... (no fallback)")
            return self._default_raw_score, 0.0

        score = float(self._classify_batch([features])[0].item())
        if not np.isfinite(score):
            if self.fallback_to_subprocess:
                self._n_subprocess_fallback += 1
                subprocess_scorer = self._get_subprocess_scorer()
                return subprocess_scorer.score(tcr, peptide, **kwargs)
            logger.error(f"Non-finite tFold AMP score for {cdr3b[:10]}...")
            return self._default_raw_score, 0.0

        self._n_scored += 1
        return score, 1.0

    def _get_features_batch(
        self,
        cdr3bs: List[str],
        peptides: List[str],
        hlas: Optional[List[str]] = None,
    ) -> List[Optional[Dict]]:
        """Get features for a batch using cache-first lookups."""
        if hlas is None:
            hlas = [self.default_hla] * len(cdr3bs)

        keys = [_make_cache_key(c, p, h) for c, p, h in zip(cdr3bs, peptides, hlas)]
        cached = self._cache.get_batch(keys)

        results: List[Optional[Dict]] = [None] * len(keys)
        misses: List[Tuple[int, str, str, str]] = []
        for i, key in enumerate(keys):
            if key in cached:
                results[i] = cached[key]
            else:
                misses.append((i, cdr3bs[i], peptides[i], hlas[i]))

        to_cache: List[Tuple[str, Dict]] = []
        for idx, cdr3b, peptide, hla in misses:
            feats = self._extract_features_amp(cdr3b, peptide, hla)
            if feats is None and self.fallback_to_subprocess:
                self._n_subprocess_fallback += 1
                subprocess_scorer = self._get_subprocess_scorer()
                feats = subprocess_scorer._get_features_batch([cdr3b], [peptide], [hla])[0]
            if feats is not None:
                results[idx] = feats
                to_cache.append((keys[idx], feats))

        if to_cache:
            self._cache.put_batch(to_cache)

        return results

    @torch.no_grad()
    def _classify_batch(self, features_list: List[Dict]) -> torch.Tensor:
        """Run the V3.4 classifier on a batch of cached features."""
        if not features_list:
            return torch.tensor([], device=self.device)

        batch = {}
        tensor_keys = [
            "sfea_cdr3b", "sfea_cdr3a", "sfea_pep",
            "ca_cdr3b", "ca_cdr3a", "ca_pep",
            "pfea_cdr3b_pep", "pfea_cdr3a_pep", "v33_feat",
        ]
        int_keys = ["len_cdr3b", "len_cdr3a", "len_pep"]

        for key in tensor_keys:
            batch[key] = torch.stack([f[key] for f in features_list]).to(self.device)
        for key in int_keys:
            batch[key] = torch.tensor(
                [f[key] for f in features_list], dtype=torch.long, device=self.device
            )

        for key, value in batch.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                logger.warning("Non-finite tensor detected in AMP tFold classifier input: %s", key)
                batch[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        logits = self._classifier(batch)
        if not torch.isfinite(logits).all():
            logger.warning("Non-finite logits from AMP tFold classifier; replacing with zeros")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        # Classifier gate logits are larger for non-binding; negate them so
        # higher affinity_logit means stronger binding without sigmoid compression.
        return -logits

    def score_batch(self, tcrs: List[str], peptides: List[str], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Batch scoring with shared cache lookup and batched classifier inference."""
        hlas = kwargs.get("hlas")
        if hlas is None:
            default_hla = kwargs.get("hla", self.default_hla)
            hlas = [default_hla] * len(tcrs)

        norm_tcrs = [t if t.startswith("C") else "C" + t for t in tcrs]
        features = self._get_features_batch(norm_tcrs, peptides, hlas)

        default_raw = self._default_raw_score
        scores = np.full(len(tcrs), default_raw, dtype=np.float32)
        confidences = np.zeros(len(tcrs), dtype=np.float32)

        valid_features: List[Dict] = []
        valid_indices: List[int] = []
        for i, feat in enumerate(features):
            if feat is not None:
                valid_features.append(feat)
                valid_indices.append(i)

        if valid_features:
            binding_scores = self._classify_batch(valid_features)
            for j, orig_idx in enumerate(valid_indices):
                score = float(binding_scores[j].item())
                if np.isfinite(score):
                    scores[orig_idx] = score
                    confidences[orig_idx] = 1.0

        self._n_scored += len(tcrs)
        return scores, confidences

    def score_batch_fast(self, tcrs: List[str], peptides: List[str]) -> List[float]:
        """Fast batch scoring for RL reward, using pre-sigmoid binding logits."""
        scores, _ = self.score_batch(tcrs, peptides)
        return [float(s) for s in scores]

    def get_stats(self) -> Dict:
        """Get scorer statistics."""
        return {
            "total_calls": self._n_amp_calls + self._n_subprocess_fallback,
            "n_amp_calls": self._n_amp_calls,
            "n_subprocess_fallback": self._n_subprocess_fallback,
            "amp_enabled": not self._amp_failed,
        }
