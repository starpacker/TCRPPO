#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tFold AMP 推理包装器 - 单样本实时推理加速

使用自动混合精度（AMP）加速 tFold 特征提取，适合 RL 训练场景。

Usage:
  from tfold_amp_wrapper import TFoldAMPWrapper

  # 初始化（只需一次）
  wrapper = TFoldAMPWrapper(device='cuda', use_amp=True)

  # 推理（每个新 TCR）
  chains = [
      {"id": "B", "sequence": "TCR_BETA_VREGION..."},
      {"id": "A", "sequence": "TCR_ALPHA_VREGION..."},
      {"id": "P", "sequence": "PEPTIDE"},
      {"id": "M", "sequence": "MHC_ALPHA"},
      {"id": "N", "sequence": "B2M"},
  ]

  features = wrapper.extract_features(chains)
  # Returns: dict with sfea, pfea, ca_coords, etc.
"""

import logging
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Add tFold root
TFOLD_ROOT = PROJECT_ROOT.parent / "tfold"
if str(TFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(TFOLD_ROOT))

logger = logging.getLogger("tfold_amp_wrapper")


class TFoldAMPWrapper:
    """tFold 推理包装器，支持 AMP 加速。"""

    def __init__(
        self,
        ppi_path: Optional[str] = None,
        trunk_path: Optional[str] = None,
        device: str = "cuda",
        use_amp: bool = True,
        chunk_size: Optional[int] = 64,
        receptor_cache_size: int = 32,
        amp_dtype: Optional[str] = None,
        fallback_to_fp32: bool = True,
    ):
        """
        Args:
            ppi_path: ESM-PPI 模型路径（None = 自动下载）
            trunk_path: tFold trunk 路径（None = 自动下载）
            device: 'cuda' or 'cpu'
            use_amp: 是否启用自动混合精度（推荐 True）
        """
        self.device = device
        self.use_amp = use_amp and str(device).startswith("cuda")
        self.chunk_size = chunk_size
        self.receptor_cache_size = receptor_cache_size
        self.fallback_to_fp32 = fallback_to_fp32
        self._receptor_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._stats = {
            "calls": 0,
            "receptor_cache_hits": 0,
            "receptor_cache_misses": 0,
            "fp32_fallbacks": 0,
        }
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)
        self.amp_dtype_name = self._dtype_name(self.amp_dtype)

        if str(self.device).startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        if self.use_amp:
            logger.info("AMP enabled (automatic mixed precision, dtype=%s)", self.amp_dtype_name)
        else:
            logger.info("AMP disabled (FP32 mode)")

        # Load tFold model
        self._load_model(ppi_path, trunk_path)

    def _load_model(self, ppi_path: Optional[str], trunk_path: Optional[str]):
        """加载 tFold 模型。"""
        try:
            import tfold
            from tfold.deploy import TCRpMHCPredictor
        except ImportError as e:
            raise ImportError(
                f"Cannot import tfold: {e}\n"
                "Please install dependencies: pip install ml_collections termcolor requests"
            )

        checkpoint_dir = PROJECT_ROOT / "checkpoints"

        # ESM-PPI model
        if ppi_path is None:
            ppi_path = checkpoint_dir / "esm_ppi_650m_tcr.pth"
            if not ppi_path.exists():
                logger.info("Downloading ESM-PPI model...")
                ppi_path = tfold.model.esm_ppi_650m_tcr()

        # tFold trunk
        if trunk_path is None:
            trunk_path = checkpoint_dir / "tfold_tcr_pmhc_trunk.pth"
            if not trunk_path.exists():
                logger.info("Downloading tFold trunk...")
                trunk_path = tfold.model.tfold_tcr_pmhc_trunk()

        logger.info(f"Loading tFold model...")
        logger.info(f"  PPI:   {ppi_path}")
        logger.info(f"  Trunk: {trunk_path}")

        self.model = TCRpMHCPredictor(str(ppi_path), str(trunk_path))
        self.model.to(self.device)
        self.model.eval()
        self._disable_ptm_heads()

        logger.info("✓ tFold model loaded")

    def _disable_ptm_heads(self) -> None:
        """Disable pTM/ipTM heads that are not needed for feature extraction.

        The AMP path only needs structure features for the downstream V3.4
        classifier. The PTM head is numerically fragile under mixed precision
        and can crash before the structural outputs are returned.
        """
        af2_modules = [
            self.model.model.ligand_model.net["af2_smod"],
            self.model.model.receptor_model.net["af2_smod"],
            self.model.model.docking_model.net["af2_smod"],
        ]
        for module in af2_modules:
            module.tmsc_pred = False
            if "ptm" in module.net:
                del module.net["ptm"]
        logger.info("Disabled tFold PTM heads for feature-only inference")

    @torch.no_grad()
    def infer(self, chains: List[Dict]) -> tuple:
        """
        运行 tFold 推理（带 AMP 加速）。

        Args:
            chains: List of dicts with 'id' and 'sequence' keys

        Returns:
            (inputs, outputs) tuple from tFold
        """
        if self.use_amp:
            with self._autocast_ctx(self.amp_dtype):
                inputs, outputs = self.model.infer(chains, chunk_size=self.chunk_size)
        else:
            inputs, outputs = self.model.infer(chains, chunk_size=self.chunk_size)

        return inputs, outputs

    def extract_features(self, chains: List[Dict]) -> Optional[Dict]:
        """Extract a .pt-compatible feature dict with AMP and receptor caching."""
        t0 = time.time()
        attempt_errors = []
        attempts = self._precision_attempts()

        for attempt_idx, (precision_mode, amp_dtype) in enumerate(attempts):
            try:
                features = self._forward_feature_dict(
                    chains,
                    precision_mode=precision_mode,
                    amp_dtype=amp_dtype,
                )
                features["_meta"] = {
                    "amp_enabled": precision_mode != "fp32",
                    "amp_dtype": self._dtype_name(amp_dtype),
                    "precision_mode": precision_mode,
                    "chunk_size": self.chunk_size,
                    "elapsed_s": time.time() - t0,
                    "receptor_cache_hit": bool(features.get("_receptor_cache_hit", False)),
                    "fallback_used": attempt_idx > 0,
                }
                if attempt_idx > 0:
                    self._stats["fp32_fallbacks"] += 1
                    logger.warning(
                        "Recovered non-finite tFold features via %s fallback",
                        precision_mode,
                    )
                return features
            except Exception as e:
                attempt_errors.append(f"{precision_mode}: {e}")
                if attempt_idx + 1 < len(attempts):
                    logger.warning(
                        "tFold %s path failed for feature extraction; retrying with next precision mode: %s",
                        precision_mode,
                        e,
                    )
                else:
                    logger.error(
                        "Feature extraction failed after %d attempt(s): %s",
                        len(attempts),
                        "; ".join(attempt_errors),
                    )
        return None

    def _forward_feature_dict(
        self,
        chains: List[Dict],
        precision_mode: str,
        amp_dtype: Optional[torch.dtype],
    ) -> Dict[str, Any]:
        """
        Run the frozen tFold path needed by the V3.4 classifier.

        This intentionally stops after docking evoformer because the downstream
        classifier only consumes:
          - post-evoformer single/pair features
          - monomer coordinates from ligand/receptor models

        Final complex coordinates from docking `af2_smod` are not needed.
        """
        from tfold.utils.tensor import to_device

        self._stats["calls"] += 1
        inputs = self.model._build_inputs(chains)
        inputs = to_device(inputs, self.device)

        complex_id = inputs["base"]["complex_id"]
        ligand_id = inputs["base"]["ligand_id"]
        receptor_id = inputs["base"]["receptor_id"]

        ligand_seqs = [inputs[x]["base"]["seq"] for x in ligand_id.split(":")]
        receptor_seqs = [inputs[x]["base"]["seq"] for x in receptor_id.split(":")]

        with torch.inference_mode():
            with self._autocast_ctx(amp_dtype):
                plm_out_lig = self.model.plm_featurizer(ligand_seqs)
                inputs[ligand_id]["feat"]["sfea"] = plm_out_lig["sfea"]
                inputs[ligand_id]["feat"]["pfea"] = plm_out_lig["pfea"]

                li_outputs = self.model.model.ligand_model(
                    ligand_seqs,
                    s_init=inputs[ligand_id]["feat"]["sfea"],
                    z_init=inputs[ligand_id]["feat"]["pfea"],
                    asym_id=inputs[ligand_id].get("asym_id", None).squeeze(0),
                    chunk_size=self.chunk_size,
                )

                receptor_key = self._make_receptor_cache_key(chains)
                receptor_cached = self._get_receptor_cache(receptor_key)
                receptor_cache_hit = receptor_cached is not None
                if receptor_cached is None:
                    self._stats["receptor_cache_misses"] += 1
                    plm_out_rec = self.model.plm_featurizer(receptor_seqs)
                    inputs[receptor_id]["feat"]["sfea"] = plm_out_rec["sfea"]
                    inputs[receptor_id]["feat"]["pfea"] = plm_out_rec["pfea"]
                    re_outputs = self.model.model.receptor_model(
                        receptor_seqs,
                        s_init=inputs[receptor_id]["feat"]["sfea"],
                        z_init=inputs[receptor_id]["feat"]["pfea"],
                        asym_id=inputs[receptor_id].get("asym_id", None).squeeze(0),
                        chunk_size=self.chunk_size,
                    )
                    receptor_cached = {
                        "sfea": re_outputs["sfea"].detach(),
                        "pfea": re_outputs["pfea"].detach(),
                        "cord": re_outputs["3d"]["cord"][-1].detach(),
                        "seq": inputs[receptor_id]["base"]["seq"],
                    }
                    if self._all_finite(receptor_cached):
                        self._put_receptor_cache(receptor_key, receptor_cached)
                else:
                    self._stats["receptor_cache_hits"] += 1

                docking = self.model.model.docking_model
                ligand_feat = {
                    "sfea": li_outputs["sfea"],
                    "pfea": li_outputs["pfea"],
                    "cord": li_outputs["cord"].detach(),
                    "seq": inputs[ligand_id]["base"]["seq"],
                }
                receptor_feat = receptor_cached

                lengths = [
                    len(inputs[x]["base"]["seq"]) for x in [ligand_id, receptor_id]
                ]
                sfea_tns, pfea_tns = docking.net["mono2mult"](ligand_feat, receptor_feat)
                asym_id = inputs[complex_id]["asym_id"][0]
                pfea_tns = pfea_tns + docking.crpe_encoder(lengths, asym_id)
                sfea_tns, pfea_tns = docking.net["evoformer"](
                    sfea_tns, pfea_tns, chunk_size=self.chunk_size,
                )

        finite_payload = {
            "li_sfea": li_outputs["sfea"],
            "li_pfea": li_outputs["pfea"],
            "li_cord": li_outputs["cord"],
            "re_sfea": receptor_cached["sfea"],
            "re_pfea": receptor_cached["pfea"],
            "re_cord": receptor_cached["cord"],
            "dock_sfea": sfea_tns,
            "dock_pfea": pfea_tns,
        }
        if not self._all_finite(finite_payload):
            raise ValueError(f"non-finite tensors detected in {precision_mode} path")

        interface_idx = self._get_interface_indices(chains)
        pfea = pfea_tns.detach().cpu()[0]
        raw_sfea = sfea_tns.detach().cpu()[0]
        mono_cord_lig = li_outputs["cord"].detach().cpu()
        mono_cord_rec = receptor_feat["cord"].detach().cpu()
        lig_len = len(inputs[ligand_id]["base"]["seq"])
        rec_len = len(inputs[receptor_id]["base"]["seq"])

        pfea_cdr3b_pep, pfea_cdr3a_pep = self._extract_interface_pfea(
            pfea, interface_idx
        )
        ca_coords = torch.cat([mono_cord_lig[:, 1, :], mono_cord_rec[:, 1, :]], dim=0)

        return {
            "raw_sfea": raw_sfea.half(),
            "mono_cord_lig": mono_cord_lig.half(),
            "mono_cord_rec": mono_cord_rec.half(),
            "lig_len": lig_len,
            "rec_len": rec_len,
            "L_total": lig_len + rec_len,
            "ca_coords": ca_coords.half(),
            "pfea_cdr3b_pep": pfea_cdr3b_pep.half() if pfea_cdr3b_pep is not None else None,
            "pfea_cdr3a_pep": pfea_cdr3a_pep.half() if pfea_cdr3a_pep is not None else None,
            "chains": chains,
            "cdr3b_range": interface_idx["cdr3b"],
            "cdr3a_range": interface_idx["cdr3a"],
            "pep_range": interface_idx["pep"],
            "cdr3b_global_range": interface_idx["cdr3b"],
            "cdr3a_global_range": interface_idx["cdr3a"],
            "pep_global_range": interface_idx["pep"],
            "_receptor_cache_hit": receptor_cache_hit,
        }

    def _resolve_amp_dtype(self, amp_dtype: Optional[str]) -> Optional[torch.dtype]:
        if not self.use_amp:
            return None
        if amp_dtype is not None:
            normalized = amp_dtype.lower()
            if normalized in {"bf16", "bfloat16"}:
                return torch.bfloat16
            if normalized in {"fp16", "float16", "half"}:
                return torch.float16
            raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float16

    def _autocast_ctx(self, amp_dtype: Optional[torch.dtype]):
        if amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=amp_dtype)

    def _precision_attempts(self) -> List[Tuple[str, Optional[torch.dtype]]]:
        attempts: List[Tuple[str, Optional[torch.dtype]]] = []
        if self.use_amp and self.amp_dtype is not None:
            attempts.append((f"amp_{self.amp_dtype_name}", self.amp_dtype))
        if self.fallback_to_fp32:
            attempts.append(("fp32", None))
        if not attempts:
            attempts.append(("fp32", None))
        return attempts

    @staticmethod
    def _dtype_name(dtype: Optional[torch.dtype]) -> str:
        if dtype is None:
            return "fp32"
        if dtype == torch.bfloat16:
            return "bf16"
        if dtype == torch.float16:
            return "fp16"
        return str(dtype)

    def _all_finite(self, obj: Any) -> bool:
        if torch.is_tensor(obj):
            return bool(torch.isfinite(obj).all().item())
        if isinstance(obj, dict):
            return all(self._all_finite(value) for value in obj.values())
        if isinstance(obj, (list, tuple)):
            return all(self._all_finite(value) for value in obj)
        return True

    def _make_receptor_cache_key(self, chains: List[Dict]) -> str:
        receptor_parts = []
        for chain in chains:
            if chain["id"] in {"M", "N", "P"}:
                receptor_parts.append(f"{chain['id']}:{chain['sequence']}")
        return "|".join(receptor_parts)

    def _get_receptor_cache(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._receptor_cache.get(key)
        if entry is None:
            return None
        self._receptor_cache.move_to_end(key)
        return entry

    def _put_receptor_cache(self, key: str, value: Dict[str, Any]) -> None:
        self._receptor_cache[key] = value
        self._receptor_cache.move_to_end(key)
        while len(self._receptor_cache) > self.receptor_cache_size:
            self._receptor_cache.popitem(last=False)

    def _extract_interface_pfea(
        self,
        pfea: torch.Tensor,
        interface_idx: Dict[str, Optional[Tuple[int, int]]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        pfea_cdr3b_pep = None
        pfea_cdr3a_pep = None
        pep_range = interface_idx["pep"]
        if pep_range is None:
            return pfea_cdr3b_pep, pfea_cdr3a_pep

        p_s, p_e = pep_range
        if interface_idx["cdr3b"] is not None:
            b_s, b_e = interface_idx["cdr3b"]
            pfea_cdr3b_pep = pfea[b_s:b_e, p_s:p_e, :]
        if interface_idx["cdr3a"] is not None:
            a_s, a_e = interface_idx["cdr3a"]
            pfea_cdr3a_pep = pfea[a_s:a_e, p_s:p_e, :]
        return pfea_cdr3b_pep, pfea_cdr3a_pep

    @property
    def stats(self) -> Dict[str, int]:
        return {
            **self._stats,
            "receptor_cache_size": len(self._receptor_cache),
            "amp_dtype": self.amp_dtype_name,
        }

    def _get_interface_indices(self, chains: List[Dict]) -> Dict:
        """检测 CDR3 和 peptide 区域的全局索引。"""
        import re

        offsets = {}
        offset = 0
        for c in chains:
            cid = c["id"]
            clen = len(c["sequence"])
            offsets[cid] = (offset, offset + clen, c["sequence"])
            offset += clen

        result = {"cdr3b": None, "cdr3a": None, "pep": None}

        # CDR3β from chain B
        if "B" in offsets:
            b_s, b_e = self._find_cdr3(offsets["B"][2])
            if b_s is not None:
                result["cdr3b"] = (offsets["B"][0] + b_s, offsets["B"][0] + b_e)

        # CDR3α from chain A
        if "A" in offsets:
            a_s, a_e = self._find_cdr3(offsets["A"][2])
            if a_s is not None:
                result["cdr3a"] = (offsets["A"][0] + a_s, offsets["A"][0] + a_e)

        # Peptide from chain P
        if "P" in offsets:
            result["pep"] = (offsets["P"][0], offsets["P"][1])

        return result

    @staticmethod
    def _find_cdr3(seq: str):
        """查找 CDR3 区域（IMGT motifs）。"""
        import re

        # Find FG.G motif (CDR3 end)
        end_matches = []
        for m in re.finditer(r"[FW]G.G", seq):
            if m.start() > len(seq) * 0.25:
                end_matches.append(m.start())
        if not end_matches:
            return None, None

        cdr3_end = end_matches[0]

        # Find conserved Cys (CDR3 start)
        cdr3_start = None
        for i in range(cdr3_end - 3, max(cdr3_end - 30, -1), -1):
            if i >= 0 and seq[i] == "C":
                cdr3_start = i + 1
                break

        if cdr3_start is None:
            return None, None

        return cdr3_start, cdr3_end


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """测试 AMP 包装器。"""
    import time

    logging.basicConfig(level=logging.INFO)

    # Test sample
    chains = [
        {
            "id": "B",
            "sequence": "NAGVTQTPKFQVLKTGQSMTLQCSQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYSIRGSRGEQFFGPGTRLTVL"
        },
        {
            "id": "A",
            "sequence": "AQEVTQIPAALSVPEGENLVLNCSFTDSAIYNLQWFRQDPGKGLTSLLLIQSSQREQTSGRLNASLDKSSGRSTLYIAASQPGDSATYLCAVTNQAGTALIFGKGTTLSVSS"
        },
        {
            "id": "P",
            "sequence": "GILGFVFTL"
        },
        {
            "id": "M",
            "sequence": "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWE"
        },
        {
            "id": "N",
            "sequence": "GPHSLRYFVTAVSRPGLGEPRYMEVGYVDDTEFVRFDSDAENPRYEPRARWMEQEGPEYWERETQKAKGQEQWFRVSLRNLLGYYNQSAGGSHTLQQMSGCDLGPDGRLLRGHDQYAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGECVEWLRRYLKNGNATLLRTDSPKAHVTHHSRPEDKVTLRCWALGFYPADITLTWQLNGEELIQDMELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPVTLRWE"
        },
    ]

    print("=" * 70)
    print("tFold AMP Wrapper Test")
    print("=" * 70)

    # Test with AMP
    print("\n[1/2] With AMP")
    wrapper_amp = TFoldAMPWrapper(device="cuda", use_amp=True)

    start = time.time()
    features_amp = wrapper_amp.extract_features(chains)
    time_amp = time.time() - start

    print(f"Time: {time_amp:.2f}s")
    if features_amp:
        print(f"Features extracted:")
        print(f"  raw_sfea: {features_amp['raw_sfea'].shape}")
        print(f"  ca_coords: {features_amp['ca_coords'].shape}")
        print(f"  pfea_cdr3b_pep: {features_amp['pfea_cdr3b_pep'].shape if features_amp['pfea_cdr3b_pep'] is not None else None}")

    # Test without AMP
    print("\n[2/2] Without AMP (FP32)")
    wrapper_fp32 = TFoldAMPWrapper(device="cuda", use_amp=False)

    start = time.time()
    features_fp32 = wrapper_fp32.extract_features(chains)
    time_fp32 = time.time() - start

    print(f"Time: {time_fp32:.2f}s")
    print(f"\nSpeedup: {time_fp32/time_amp:.2f}×")


if __name__ == "__main__":
    main()
