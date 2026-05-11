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
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("tfold_amp_wrapper")


class TFoldAMPWrapper:
    """tFold 推理包装器，支持 AMP 加速。"""

    def __init__(
        self,
        ppi_path: Optional[str] = None,
        trunk_path: Optional[str] = None,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        """
        Args:
            ppi_path: ESM-PPI 模型路径（None = 自动下载）
            trunk_path: tFold trunk 路径（None = 自动下载）
            device: 'cuda' or 'cpu'
            use_amp: 是否启用自动混合精度（推荐 True）
        """
        self.device = device
        self.use_amp = use_amp and (device == "cuda")

        if self.use_amp:
            logger.info("AMP enabled (automatic mixed precision)")
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

        logger.info("✓ tFold model loaded")

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
            with torch.amp.autocast('cuda'):
                inputs, outputs = self.model.infer(chains)
        else:
            inputs, outputs = self.model.infer(chains)

        return inputs, outputs

    def extract_features(self, chains: List[Dict]) -> Optional[Dict]:
        """
        提取结构特征（用于 V3.4 分类器）。

        Args:
            chains: List of dicts with 'id' and 'sequence' keys
                    Expected: B (TCRβ), A (TCRα), P (peptide), M (MHC), N (β2m)

        Returns:
            Dict with:
              - raw_sfea: [L_total, 192] per-residue structure features
              - ca_coords: [L_total, 3] Cα coordinates
              - pfea_cdr3b_pep: [Lb, Lp, 128] pairwise features
              - pfea_cdr3a_pep: [La, Lp, 128]
              - chains: original chain info
              - cdr3b_range, cdr3a_range, pep_range: (start, end) indices

            Returns None if inference fails.
        """
        try:
            inputs, outputs = self.infer(chains)
        except Exception as e:
            logger.error(f"tFold inference failed: {e}")
            return None

        # Extract features from outputs
        try:
            features = self._parse_outputs(inputs, outputs, chains)
            return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def _parse_outputs(self, inputs, outputs, chains) -> Dict:
        """解析 tFold 输出为特征字典。"""
        # Get complex ID
        complex_id = inputs["base"]["complex_id"]

        # Extract raw sfea and coordinates
        raw_sfea = outputs[complex_id]["sfea"][0].cpu()  # [L, 192]
        cord = outputs[complex_id]["cord"].cpu()  # [L, 14, 3]
        ca_coords = cord[:, 1, :]  # Cα is atom index 1

        # Extract pfea (full pairwise features)
        pfea = outputs[complex_id]["pfea"][0].cpu()  # [L, L, 128]

        # Detect CDR3 and peptide regions
        interface_idx = self._get_interface_indices(chains)

        # Extract interface pfea blocks
        pfea_cdr3b_pep = None
        pfea_cdr3a_pep = None

        if interface_idx["cdr3b"] is not None and interface_idx["pep"] is not None:
            b_s, b_e = interface_idx["cdr3b"]
            p_s, p_e = interface_idx["pep"]
            pfea_cdr3b_pep = pfea[b_s:b_e, p_s:p_e, :]  # [Lb, Lp, 128]

        if interface_idx["cdr3a"] is not None and interface_idx["pep"] is not None:
            a_s, a_e = interface_idx["cdr3a"]
            p_s, p_e = interface_idx["pep"]
            pfea_cdr3a_pep = pfea[a_s:a_e, p_s:p_e, :]  # [La, Lp, 128]

        return {
            "raw_sfea": raw_sfea.half(),  # Convert to FP16 to save memory
            "ca_coords": ca_coords.half(),
            "pfea_cdr3b_pep": pfea_cdr3b_pep.half() if pfea_cdr3b_pep is not None else None,
            "pfea_cdr3a_pep": pfea_cdr3a_pep.half() if pfea_cdr3a_pep is not None else None,
            "chains": chains,
            "cdr3b_range": interface_idx["cdr3b"],
            "cdr3a_range": interface_idx["cdr3a"],
            "pep_range": interface_idx["pep"],
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
            if m.start() > len(seq) * 0.5:
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
