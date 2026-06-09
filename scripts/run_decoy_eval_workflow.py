#!/usr/bin/env python
"""Run the standardized tFold decoy-reward evaluation workflow.

The workflow starts a dedicated tFold feature server, runs
eval_checkpoint_decoy_reward_tfold.py with a named decoy mode, then shuts the
server down. It does not use or stop any training server.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TFOLD_PYTHON = "/home/liuyutian/server/miniconda3/envs/tfold/bin/python"
DECOY_LIB_PYTHON = "/home/liuyutian/server/miniconda3/bin/python"
DECOY_LIBRARY_ROOT = Path("/share/liuyutian/pMHC_decoy_library")


def recv_msg(sock: socket.socket) -> bytes:
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("server closed before header")
        header += chunk
    length = struct.unpack(">I", header)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(length - len(data), 65536))
        if not chunk:
            raise ConnectionError("server closed during payload")
        data += chunk
    return data


def send_server_cmd(socket_path: str, cmd: str) -> dict:
    payload = json.dumps({"cmd": cmd}).encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect(socket_path)
    sock.sendall(struct.pack(">I", len(payload)) + payload)
    response = json.loads(recv_msg(sock).decode("utf-8"))
    sock.close()
    return response


def wait_ready(log_path: Path, socket_path: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if log_path.exists() and "\nREADY\n" in f"\n{log_path.read_text(errors='ignore')}\n":
            response = send_server_cmd(socket_path, "ping")
            if response.get("status") == "pong":
                return
        time.sleep(5)
    tail = log_path.read_text(errors="ignore")[-4000:] if log_path.exists() else ""
    raise TimeoutError(f"tFold server did not become READY within {timeout_s}s\n{tail}")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def load_targets(path: str) -> list[str]:
    target_path = Path(path)
    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path
    return [
        line.strip()
        for line in target_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def required_decoy_count(mode: str) -> int:
    if mode in {"1", "a1"}:
        return 1
    if mode in {"4", "abbd4"}:
        return 4
    if mode in {"16", "a4b8d4"}:
        return 16
    raise ValueError(f"Unsupported decoy mode for workflow: {mode}")


def existing_hard_decoy_count(target: str, decoy_library_path: Path) -> int:
    from tcrppo_v2.data.decoy_sampler import DecoySampler

    sampler = DecoySampler(decoy_library_path=str(decoy_library_path), targets=[target], seed=42)
    tiers = sampler.decoys.get(target, {})
    seen = set()
    for tier in ("A", "B", "D"):
        for peptide in tiers.get(tier, []):
            if peptide and peptide != target:
                seen.add(peptide)
    return len(seen)


def ensure_decoys(args: argparse.Namespace) -> None:
    """Construct missing hard decoys in pMHC-decoy-library before evaluation."""
    if args.no_ensure_decoys:
        print("[decoys] ensure step skipped", flush=True)
        return

    targets = load_targets(args.targets_file)
    required = required_decoy_count(args.decoy_mode)
    decoy_library = Path(args.decoy_library_path)
    missing = [
        target
        for target in targets
        if existing_hard_decoy_count(target, decoy_library) < required
    ]
    if not missing:
        print(f"[decoys] all targets have >= {required} A/B/D decoys; no construction needed", flush=True)
        return

    strategies = list(dict.fromkeys(args.construct_strategies))
    print(
        f"[decoys] constructing decoys for {len(missing)} targets with < {required} A/B/D decoys: "
        + ", ".join(missing),
        flush=True,
    )
    for target in missing:
        cmd = [
            DECOY_LIB_PYTHON,
            "run_decoy.py",
            target,
            *strategies,
            "--hla",
            args.construct_hla,
        ]
        if args.construct_skip_structural and "b" in strategies:
            cmd.append("--skip-structural")
        if "d" in strategies:
            cmd.extend(["--designs", str(args.construct_d_designs), "--top-k", str(args.construct_d_top_k)])
        print(f"[decoys] {shell_join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=DECOY_LIBRARY_ROOT, check=True)
        after = existing_hard_decoy_count(target, decoy_library)
        print(f"[decoys] {target}: now has {after} A/B/D decoys", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="Name used for output/log/socket paths.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--config", default="configs/test51c.yaml")
    parser.add_argument("--targets-file", default="data/tfold_excellent_peptides.txt")
    parser.add_argument("--output-dir", default="", help="Default: results/<run-name>")
    parser.add_argument("--decoy-mode", choices=["1", "4", "16", "a1", "abbd4", "a4b8d4"], default="4")
    parser.add_argument("--n-tcrs", type=int, default=1)
    parser.add_argument("--decoy-top-k", nargs="+", type=int, default=[1, 3, 4])
    parser.add_argument("--gpu", type=int, default=6, help="Physical GPU for the dedicated eval server.")
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--extract-batch-size", type=int, default=64)
    parser.add_argument("--tfold-cache-path", default="data/tfold_feature_cache.db")
    parser.add_argument("--server-timeout-s", type=int, default=900)
    parser.add_argument("--keep-server", action="store_true", help="Leave the dedicated eval server running.")
    parser.add_argument("--no-ensure-decoys", action="store_true", help="Do not construct missing A/B/D decoys before eval.")
    parser.add_argument("--decoy-library-path", default=str(DECOY_LIBRARY_ROOT))
    parser.add_argument(
        "--construct-strategies",
        nargs="+",
        choices=["a", "b", "d"],
        default=["a", "b"],
        help="pMHC-decoy-library strategies to run when hard decoys are insufficient.",
    )
    parser.add_argument("--construct-hla", default="HLA-A*02:01")
    parser.add_argument(
        "--construct-skip-structural",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use physchem-only Decoy B construction by default for speed.",
    )
    parser.add_argument("--construct-d-designs", type=int, default=1000)
    parser.add_argument("--construct-d-top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name
    output_dir = Path(args.output_dir or PROJECT_ROOT / "results" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)

    ensure_decoys(args)

    socket_path = f"/tmp/tfold_server_eval_{run_name}.sock"
    server_log = PROJECT_ROOT / "logs" / f"tfold_server_eval_{run_name}.log"
    completion_log = PROJECT_ROOT / "logs" / f"tfold_completion_eval_{run_name}.log"
    for path in (Path(socket_path), Path(socket_path + ".pid")):
        if path.exists():
            path.unlink()

    server_cmd = [
        TFOLD_PYTHON,
        "scripts/tfold_feature_server.py",
        "--socket",
        socket_path,
        "--gpu",
        "0",
        "--use-amp-wrapper",
        "--chunk-size",
        "64",
        "--completion-log",
        str(completion_log),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with server_log.open("w") as log_f:
        server = subprocess.Popen(
            server_cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    try:
        print(f"[server] pid={server.pid} socket={socket_path}", flush=True)
        wait_ready(server_log, socket_path, args.server_timeout_s)
        print("[server] READY", flush=True)

        eval_cmd = [
            sys.executable,
            "scripts/eval_checkpoint_decoy_reward_tfold.py",
            "--checkpoint-dir",
            args.checkpoint_dir,
            "--checkpoints",
            *args.checkpoints,
            "--config",
            args.config,
            "--targets-file",
            args.targets_file,
            "--output-dir",
            str(output_dir),
            "--n-tcrs",
            str(args.n_tcrs),
            "--decoy-mode",
            args.decoy_mode,
            "--decoy-top-k",
            *[str(k) for k in args.decoy_top_k],
            "--device",
            "cuda",
            "--tfold-server-socket",
            socket_path,
            "--tfold-cache-path",
            args.tfold_cache_path,
            "--score-batch-size",
            str(args.score_batch_size),
            "--extract-batch-size",
            str(args.extract_batch_size),
        ]
        eval_env = os.environ.copy()
        eval_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        run_log = output_dir / "run.log"
        print(f"[eval] {shell_join(eval_cmd)}", flush=True)
        with run_log.open("w") as log_f:
            subprocess.run(eval_cmd, cwd=PROJECT_ROOT, env=eval_env, stdout=log_f, stderr=subprocess.STDOUT, check=True)
        print(f"[eval] complete: {output_dir / 'REPORT.md'}", flush=True)
    finally:
        if args.keep_server:
            print("[server] kept running by request", flush=True)
        else:
            try:
                print("[server] shutting down", flush=True)
                send_server_cmd(socket_path, "shutdown")
            except Exception as exc:
                print(f"[server] graceful shutdown failed: {exc}; terminating pid={server.pid}", flush=True)
                os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)
            print("[server] stopped", flush=True)


if __name__ == "__main__":
    main()
