#!/usr/bin/env python
"""tFold feature extraction server — runs in the `tfold` conda env.

Long-lived process that loads the tFold model ONCE, then accepts feature
extraction requests via Unix domain socket. This avoids the ~5min model
loading penalty per subprocess call.

Usage:
    # Start server (in tfold env):
    /path/to/tfold/python scripts/tfold_feature_server.py \
        --socket /tmp/tfold_server.sock \
        --gpu 0

    # The server will:
    # 1. Load tFold model (takes ~5 min)
    # 2. Print "READY" when ready to accept connections
    # 3. Accept connections, read JSON requests, return JSON responses

Protocol:
    Client sends: 4-byte big-endian length + JSON bytes
    Server sends: 4-byte big-endian length + JSON bytes

    Request JSON:
    {
      "cmd": "extract",
      "samples": [{"cdr3b": "...", "peptide": "...", "hla": "..."}]
    }

    Response JSON:
    {
      "status": "ok",
      "features": [<base64-encoded .pt bytes>, ...],
      "errors": [null, "msg", ...]
    }

    Or: {"cmd": "ping"} → {"status": "pong"}
    Or: {"cmd": "shutdown"} → server exits
"""

import argparse
import base64
import io
import json
import os
import signal
import socket
import struct
import sys
import time
import traceback
from typing import Dict, List, Optional

import torch

# Add tFold root to path
TFOLD_ROOT = "/share/liuyutian/tfold"
sys.path.insert(0, TFOLD_ROOT)

import importlib

# Import tFold components
fe = importlib.import_module("TCR_PMHC_pred.4_16.feature_extraction")
dp = importlib.import_module("TCR_PMHC_pred.4_16.data_pipeline")
hla_mod = importlib.import_module("TCR_PMHC_pred.4_16.hla_mapping")


# ---------------------------------------------------------------------------
# Template V-region scaffolds (same as tfold_feature_worker.py)
# ---------------------------------------------------------------------------

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

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def splice_cdr3_into_scaffold(scaffold: str, cdr3: str, cdr3_pos: int,
                               fgxg_suffix: str = "FGXG") -> str:
    """Splice a CDR3 sequence into a V-region scaffold."""
    framework = scaffold[:cdr3_pos]
    return framework + cdr3 + fgxg_suffix


def clean_seq(s: str) -> str:
    """Remove non-standard amino acids."""
    return "".join(c for c in s if c in STANDARD_AA)


def build_chains_from_cdr3b(
    cdr3b: str, peptide: str, hla: str, hla_lookup: Dict,
    hla_seq_lookup: Optional[Dict] = None,
) -> Optional[List[Dict]]:
    """Build tFold-compatible chain list from CDR3beta + target info."""
    vregion_beta = splice_cdr3_into_scaffold(
        _DEFAULT_VREGION_BETA_SCAFFOLD, cdr3b, _CDR3_POS_BETA, "FGSG"
    )
    vregion_alpha = splice_cdr3_into_scaffold(
        _DEFAULT_VREGION_ALPHA_SCAFFOLD, _DEFAULT_CDR3_ALPHA, 90, "FGKG"
    )

    mhc_info = hla_lookup.get(hla)
    if mhc_info is None:
        try:
            if hla_seq_lookup is not None:
                chains_mhc = hla_mod.resolve_mhc_chains(hla, hla_seq_lookup)
            else:
                chains_mhc = None
            if chains_mhc is None:
                return None
            mhc_full = chains_mhc.get("M", "")
            b2m = chains_mhc.get("N", "")
        except Exception:
            return None
    else:
        mhc_full = mhc_info.get("M", "")
        b2m = mhc_info.get("N", "")

    if not mhc_full:
        return None

    chains = [
        {"id": "B", "sequence": clean_seq(vregion_beta)},
        {"id": "A", "sequence": clean_seq(vregion_alpha)},
        {"id": "M", "sequence": clean_seq(mhc_full)},
    ]
    if b2m:
        chains.append({"id": "N", "sequence": clean_seq(b2m)})
    chains.append({"id": "P", "sequence": clean_seq(peptide)})

    return chains


def extract_v34_features(predictor, chains: List[Dict], device: str) -> Optional[Dict]:
    """Run full tFold pipeline and extract V3.4 classifier features."""
    try:
        raw = fe.extract_features_from_chains(
            predictor, chains, device=device, chunk_size=16
        )
        if raw is None:
            print(f"  DEBUG: raw features is None", flush=True)
            return None
        print(f"  DEBUG: raw type={type(raw)}, keys={list(raw.keys()) if isinstance(raw, dict) else 'N/A'}", flush=True)
        structured = fe.extract_structured_features(raw)
        if structured is None:
            print(f"  DEBUG: structured features is None (raw was not None)", flush=True)
            return None
        print(f"  DEBUG: structured keys={list(structured.keys())}", flush=True)

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
    except Exception:
        traceback.print_exc()
        return None


def features_to_base64(features: Dict) -> str:
    """Serialize feature dict to base64 string."""
    buf = io.BytesIO()
    torch.save(features, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def recv_msg(conn: socket.socket) -> Optional[bytes]:
    """Receive a length-prefixed message."""
    # Read 4-byte length header
    header = b""
    while len(header) < 4:
        chunk = conn.recv(4 - len(header))
        if not chunk:
            return None
        header += chunk
    msg_len = struct.unpack(">I", header)[0]
    if msg_len > 100 * 1024 * 1024:  # 100MB safety limit
        return None
    # Read message body
    data = b""
    while len(data) < msg_len:
        chunk = conn.recv(min(msg_len - len(data), 65536))
        if not chunk:
            return None
        data += chunk
    return data


def send_msg(conn: socket.socket, data: bytes) -> None:
    """Send a length-prefixed message."""
    header = struct.pack(">I", len(data))
    conn.sendall(header + data)


class TFoldFeatureServer:
    """Persistent tFold feature extraction server."""

    def __init__(self, socket_path: str, gpu_id: int = 0):
        self.socket_path = socket_path
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.predictor = None
        self.hla_lookup = {}       # Resolved HLA -> {"M": seq, "N": seq} cache
        self._hla_seq_lookup = {}  # Raw allele -> protein sequence lookup (built from FASTA DB)
        self._running = False

    def _load_model(self):
        """Load the tFold predictor (slow, ~5min) and HLA lookup."""
        print(f"Loading tFold predictor on {self.device}...", flush=True)
        t0 = time.time()
        self.predictor = fe.load_tfold_predictor(
            ppi_path=os.path.join(TFOLD_ROOT, "checkpoints", "esm_ppi_650m_tcr.pth"),
            trunk_path=os.path.join(TFOLD_ROOT, "checkpoints", "tfold_tcr_pmhc_trunk.pth"),
            device=self.device,
        )
        elapsed = time.time() - t0
        print(f"tFold predictor loaded in {elapsed:.1f}s", flush=True)

        # Build HLA allele -> protein sequence lookup from IPD-IMGT/HLA database.
        # This is required by resolve_mhc_chains() as its second argument.
        print("Building HLA lookup from IPD-IMGT/HLA database...", flush=True)
        self._hla_seq_lookup = hla_mod.build_hla_lookup()
        print(f"HLA lookup built: {len(self._hla_seq_lookup)} entries", flush=True)

    def _resolve_hla(self, hla: str) -> Optional[Dict]:
        """Resolve an HLA allele, caching the result."""
        if hla in self.hla_lookup:
            return self.hla_lookup[hla]
        try:
            mhc_chains = hla_mod.resolve_mhc_chains(hla, self._hla_seq_lookup)
            if mhc_chains:
                self.hla_lookup[hla] = mhc_chains
                return mhc_chains
        except Exception:
            pass
        return None

    def _handle_extract(self, samples: List[Dict]) -> Dict:
        """Handle an extract request."""
        features_b64 = []
        errors = []

        # Pre-resolve HLAs
        for s in samples:
            self._resolve_hla(s.get("hla", "HLA-A*02:01"))

        for i, sample in enumerate(samples):
            cdr3b = sample.get("cdr3b", "")
            peptide = sample.get("peptide", "")
            hla = sample.get("hla", "HLA-A*02:01")

            try:
                chains = build_chains_from_cdr3b(cdr3b, peptide, hla, self.hla_lookup, self._hla_seq_lookup)
                if chains is None:
                    features_b64.append(None)
                    errors.append("Failed to build chains (HLA resolution failed)")
                    continue

                # Debug: print chain sequences
                for ch in chains:
                    if ch["id"] in ["B", "A"]:
                        print(f"  DEBUG: Chain {ch['id']}: {ch['sequence'][:50]}...", flush=True)

                feats = extract_v34_features(self.predictor, chains, self.device)
                if feats is None:
                    features_b64.append(None)
                    errors.append("Feature extraction failed")
                else:
                    features_b64.append(features_to_base64(feats))
                    errors.append(None)

            except Exception as e:
                features_b64.append(None)
                errors.append(str(e))
                traceback.print_exc()

            if (i + 1) % 10 == 0:
                print(f"  Extracted {i+1}/{len(samples)}", flush=True)

        n_ok = sum(1 for e in errors if e is None)
        print(f"  Batch done: {n_ok}/{len(samples)} successful", flush=True)

        return {
            "status": "ok",
            "features": features_b64,
            "errors": errors,
        }

    def _handle_connection(self, conn: socket.socket):
        """Handle a single client connection (may have multiple requests)."""
        try:
            while self._running:
                data = recv_msg(conn)
                if data is None:
                    break

                request = json.loads(data.decode("utf-8"))
                cmd = request.get("cmd", "")

                if cmd == "ping":
                    response = {"status": "pong"}
                elif cmd == "shutdown":
                    response = {"status": "shutting_down"}
                    send_msg(conn, json.dumps(response).encode("utf-8"))
                    self._running = False
                    break
                elif cmd == "extract":
                    samples = request.get("samples", [])
                    print(f"Extract request: {len(samples)} samples", flush=True)
                    t0 = time.time()
                    response = self._handle_extract(samples)
                    elapsed = time.time() - t0
                    print(f"Extraction took {elapsed:.1f}s ({elapsed/max(len(samples),1):.1f}s/sample)",
                          flush=True)
                elif cmd == "stats":
                    response = {
                        "status": "ok",
                        "hla_cache_size": len(self.hla_lookup),
                        "device": self.device,
                    }
                else:
                    response = {"status": "error", "error": f"Unknown command: {cmd}"}

                send_msg(conn, json.dumps(response).encode("utf-8"))
        except Exception as e:
            print(f"Connection error: {e}", flush=True)
            traceback.print_exc()
        finally:
            conn.close()

    def serve(self):
        """Start the server."""
        # Load model first
        self._load_model()

        # Clean up old socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Create Unix domain socket
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        server.listen(1)  # Single client at a time
        server.settimeout(1.0)  # 1s timeout for accept() to allow clean shutdown

        self._running = True

        # Write PID file
        pid_path = self.socket_path + ".pid"
        with open(pid_path, "w") as f:
            f.write(str(os.getpid()))

        # Signal handler for clean shutdown
        def _shutdown(signum, frame):
            print(f"\nReceived signal {signum}, shutting down...", flush=True)
            self._running = False

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        print(f"READY", flush=True)
        print(f"Server listening on {self.socket_path} (pid={os.getpid()})", flush=True)

        while self._running:
            try:
                conn, _ = server.accept()
                print(f"Client connected", flush=True)
                self._handle_connection(conn)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"Accept error: {e}", flush=True)

        # Cleanup
        server.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        if os.path.exists(pid_path):
            os.unlink(pid_path)
        print("Server shut down cleanly", flush=True)


def main():
    parser = argparse.ArgumentParser(description="tFold feature extraction server")
    parser.add_argument("--socket", default="/tmp/tfold_server.sock",
                       help="Unix domain socket path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    server = TFoldFeatureServer(
        socket_path=args.socket,
        gpu_id=args.gpu,
    )
    server.serve()


if __name__ == "__main__":
    main()
