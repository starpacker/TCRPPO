#!/usr/bin/env python3
"""Run a trace11-style max_steps curriculum as staged PPO jobs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import sys
import time
from typing import Optional


ROOT = Path("/share/liuyutian/tcrppo_v2")
TRAIN_PYTHON = Path("/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python")
TFOLD_PYTHON = Path("/home/liuyutian/server/miniconda3/envs/tfold/bin/python")
EP_RE = re.compile(
    r"Episode\s+(\d+)\s+\|\s+Step\s+([0-9,]+)\s+\|\s+R=\s*([-0-9.]+)"
)
LATEST_RE = re.compile(r"\*\* Latest checkpoint refreshed:\s+([0-9,]+)\s+steps")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"{now()} {msg}", flush=True)


def parse_stage_spec(spec: str) -> list[tuple[int, Optional[int]]]:
    stages: list[tuple[int, Optional[int]]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        max_steps_s, episodes_s = item.split(":", 1)
        episodes = int(episodes_s)
        stages.append((int(max_steps_s), None if episodes <= 0 else episodes))
    if not stages:
        raise ValueError("empty stage spec")
    return stages


def round_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def ping_server(sock_path: Path, timeout_s: float = 5.0) -> bool:
    if not sock_path.exists():
        return False
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout_s)
        sock.connect(str(sock_path))
        payload = json.dumps({"cmd": "ping"}).encode("utf-8")
        sock.sendall(struct.pack(">I", len(payload)) + payload)
        header = sock.recv(4)
        if len(header) != 4:
            return False
        msg_len = struct.unpack(">I", header)[0]
        buf = b""
        while len(buf) < msg_len:
            chunk = sock.recv(msg_len - len(buf))
            if not chunk:
                return False
            buf += chunk
        return json.loads(buf.decode("utf-8")).get("status") == "pong"
    except Exception:
        return False
    finally:
        try:
            sock.close()
        except Exception:
            pass


def start_server(args: argparse.Namespace, env: dict[str, str]) -> subprocess.Popen:
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    args.completion_log.parent.mkdir(parents=True, exist_ok=True)
    args.socket_path.unlink(missing_ok=True)
    server_cmd = [
        str(TFOLD_PYTHON),
        str(ROOT / "scripts/tfold_feature_server.py"),
        "--socket",
        str(args.socket_path),
        "--gpu",
        "0",
        "--use-amp-wrapper",
        "--chunk-size",
        "64",
        "--completion-log",
        str(args.completion_log),
    ]
    log(f"starting tFold server on physical GPU {args.gpu}: {' '.join(server_cmd)}")
    server_f = args.server_log.open("w")
    proc = subprocess.Popen(
        server_cmd,
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=server_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + args.server_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"tFold server exited early with code {proc.returncode}")
        if "READY" in args.server_log.read_text(errors="replace") and ping_server(args.socket_path):
            log(f"tFold server ready pid={proc.pid} socket={args.socket_path}")
            return proc
        time.sleep(2)
    raise TimeoutError(f"tFold server did not become ready within {args.server_timeout_s}s")


def checkpoint_step(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu")
        return int(ckpt.get("global_step", 0))
    except Exception:
        return None


def read_rewards(log_path: Path) -> tuple[list[float], int, int, Optional[int]]:
    rewards: list[float] = []
    last_ep = 0
    last_step = 0
    latest_step = None
    if not log_path.exists():
        return rewards, last_ep, last_step, latest_step
    for line in log_path.read_text(errors="replace").splitlines():
        ep_m = EP_RE.search(line)
        if ep_m:
            last_ep = int(ep_m.group(1))
            last_step = int(ep_m.group(2).replace(",", ""))
            rewards.append(float(ep_m.group(3)))
            continue
        latest_m = LATEST_RE.search(line)
        if latest_m:
            latest_step = int(latest_m.group(1).replace(",", ""))
    return rewards, last_ep, last_step, latest_step


def has_plateaued(
    rewards: list[float],
    window: int,
    patience: int,
    min_delta: float,
    min_episodes: int,
) -> bool:
    if len(rewards) < min_episodes or len(rewards) < window * (patience + 1):
        return False
    means = [
        sum(rewards[i - window : i]) / window
        for i in range(window, len(rewards) + 1, window)
    ]
    if len(means) < patience + 1:
        return False
    best_before = max(means[: -patience])
    recent_best = max(means[-patience:])
    return recent_best <= best_before + min_delta


def terminate_process(proc: subprocess.Popen, label: str, timeout_s: int = 60) -> None:
    if proc.poll() is not None:
        return
    log(f"terminating {label} pid={proc.pid}")
    os.killpg(proc.pid, signal.SIGTERM)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)
    log(f"killing {label} pid={proc.pid}")
    os.killpg(proc.pid, signal.SIGKILL)


def run_stage(
    args: argparse.Namespace,
    env: dict[str, str],
    stage_idx: int,
    max_steps: int,
    target_episodes: Optional[int],
    resume_from: Optional[Path],
    current_step: int,
) -> tuple[Path, int]:
    rollout = args.n_envs * args.n_steps
    if target_episodes is None:
        total_timesteps = args.final_total_timesteps
    else:
        stage_steps = round_up(target_episodes * max_steps, rollout)
        total_timesteps = current_step + stage_steps

    run_name = f"{args.run_prefix}_{args.trace_tag}_ms{max_steps}"
    train_log = ROOT / "logs" / f"{run_name}_train.log"
    out_dir = ROOT / "output" / run_name / "checkpoints"
    latest_ckpt = out_dir / "latest.pt"
    final_ckpt = out_dir / "final.pt"
    train_log.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(TRAIN_PYTHON),
        "-u",
        "tcrppo_v2/ppo_trainer.py",
        "--config",
        str(args.config.relative_to(ROOT)),
        "--run_name",
        run_name,
        "--seed",
        "42",
        "--reward_mode",
        "v2_no_decoy_delta",
        "--affinity_scorer",
        "tfold",
        "--tfold_server_socket",
        str(args.socket_path),
        "--encoder",
        "esm2",
        "--total_timesteps",
        str(total_timesteps),
        "--n_envs",
        str(args.n_envs),
        "--learning_rate",
        "3e-4",
        "--entropy_coef",
        "0.02",
        "--hidden_dim",
        "512",
        "--max_steps",
        str(max_steps),
        "--n_contrast_decoys",
        "0",
        "--w_affinity",
        "1.0",
        "--w_decoy",
        "0.0",
        "--w_naturalness",
        "0.05",
        "--w_diversity",
        "0.02",
        "--curriculum_l0",
        "0.5",
        "--curriculum_l1",
        "0.0",
        "--curriculum_l2",
        "0.5",
        "--train_targets",
        "data/tfold_excellent_peptides.txt",
        "--tfold_cache_path",
        str(args.tfold_cache_path),
        "--decoy_library_path",
        "/share/liuyutian/pMHC_decoy_library",
        "--ban_stop",
        "--terminal_reward_only",
    ]
    if resume_from is not None:
        cmd.extend(["--resume_from", str(resume_from)])
    if args.reset_optimizer_each_stage and resume_from is not None:
        cmd.append("--resume_reset_optimizer")

    log(
        f"stage {stage_idx} start max_steps={max_steps} target_episodes="
        f"{target_episodes or 'final'} total_timesteps={total_timesteps} "
        f"resume={resume_from or 'none'} run_name={run_name}"
    )
    with train_log.open("w") as train_f:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=train_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        plateau_requested = False
        while proc.poll() is None:
            time.sleep(args.monitor_interval_s)
            rewards, last_ep, last_step, latest_step = read_rewards(train_log)
            if rewards:
                recent = rewards[-min(100, len(rewards)) :]
                mean_recent = sum(recent) / len(recent)
                log(
                    f"stage {stage_idx} max_steps={max_steps} pid={proc.pid} "
                    f"eps={last_ep} step={last_step} recent100_R={mean_recent:.3f} "
                    f"latest_ckpt_step={latest_step or 'none'}"
                )
            if target_episodes is None:
                continue
            if args.enable_plateau and not plateau_requested and has_plateaued(
                rewards,
                args.plateau_window,
                args.plateau_patience,
                args.plateau_min_delta,
                args.plateau_min_episodes,
            ):
                plateau_requested = True
                log(f"stage {stage_idx} plateau detected; waiting for a fresh latest checkpoint")
            if plateau_requested:
                latest_step = checkpoint_step(latest_ckpt)
                if latest_step is not None and latest_step > current_step:
                    terminate_process(proc, f"stage {stage_idx} trainer")
                    break
        rc = proc.wait()
    if rc != 0 and not plateau_requested:
        raise RuntimeError(f"stage {stage_idx} trainer exited with code {rc}; see {train_log}")

    ckpt = final_ckpt if final_ckpt.exists() else latest_ckpt
    step = checkpoint_step(ckpt)
    if step is None:
        raise RuntimeError(f"stage {stage_idx} did not produce a usable checkpoint in {out_dir}")
    log(f"stage {stage_idx} done checkpoint={ckpt} step={step}")
    return ckpt, step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--trace-tag", default="trace25_maxstep_curriculum")
    parser.add_argument("--run-prefix", default="test56_maxstep_curriculum")
    parser.add_argument("--stage-spec", default="1:2000,2:2000,4:2000,8:0")
    parser.add_argument("--final-total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--monitor-interval-s", type=int, default=60)
    parser.add_argument("--server-timeout-s", type=int, default=900)
    parser.add_argument("--enable-plateau", action="store_true")
    parser.add_argument("--plateau-window", type=int, default=200)
    parser.add_argument("--plateau-patience", type=int, default=4)
    parser.add_argument("--plateau-min-delta", type=float, default=0.02)
    parser.add_argument("--plateau-min-episodes", type=int, default=1200)
    parser.add_argument("--reset-optimizer-each-stage", action="store_true")
    parser.add_argument("--initial-resume", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/test56_maxstep_curriculum.yaml",
    )
    args = parser.parse_args()

    args.config = args.config.resolve()
    args.socket_path = Path(f"/tmp/tfold_server_{args.trace_tag}.sock")
    args.server_log = ROOT / "logs" / f"{args.run_prefix}_tfold_amp_server_{args.trace_tag}.log"
    args.completion_log = ROOT / "logs" / f"{args.run_prefix}_tfold_completion_{args.trace_tag}.log"
    args.tfold_cache_path = Path("data/tfold_feature_cache.db")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    stages = parse_stage_spec(args.stage_spec)

    log(f"starting curriculum controller stages={stages} gpu={args.gpu}")
    server_proc = start_server(args, env)
    resume_from = args.initial_resume
    current_step = checkpoint_step(resume_from) if resume_from else 0
    current_step = current_step or 0
    try:
        for idx, (max_steps, target_episodes) in enumerate(stages, start=1):
            resume_from, current_step = run_stage(
                args,
                env,
                idx,
                max_steps,
                target_episodes,
                resume_from,
                current_step,
            )
        log(f"curriculum complete final_checkpoint={resume_from} final_step={current_step}")
    finally:
        terminate_process(server_proc, "tFold server")
        args.socket_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
