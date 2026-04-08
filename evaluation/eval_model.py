#!/usr/bin/env python3
"""Run a trained TCRPPO model on test data and collect results.

This script loads a trained PPO checkpoint, runs inference on test TCR-peptide
pairs using vectorized environments, and writes results in the standard format:
    <peptide> <init_tcr> <final_tcr> <ergo_score> <edit_dist> <gmm_likelihood>

Usage:
    python eval_model.py \
        --model_path <path_to_ppo_tcr> \
        --ergo_model <ae_mcpas|ae_vdjdb|path_to_pt> \
        --peptide_file <peptide_file> \
        [--tcr_file <test_tcr_file>] \
        [--num_envs 4] [--num_tcrs 1000] [--rollout 1] \
        [--out results.txt]
"""
import argparse
import os
import sys
import time
import numpy as np

from eval_utils import (
    CODE_DIR, REPO_ROOT, ERGO_MODELS, PEPTIDE_FILES, TEST_TCR_FILE,
    find_model_checkpoint, load_peptides, load_tcrs, ensure_dir,
)

# We need code/ on path before importing project modules
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


def resolve_ergo_model(ergo_arg):
    """Resolve ergo model argument to an actual file path."""
    if ergo_arg in ERGO_MODELS:
        return ERGO_MODELS[ergo_arg]
    if os.path.isfile(ergo_arg):
        return ergo_arg
    if os.path.isfile(ergo_arg + ".pt"):
        return ergo_arg + ".pt"
    raise FileNotFoundError(f"Cannot find ERGO model: {ergo_arg}")


def resolve_peptide_file(pep_arg):
    """Resolve peptide file argument."""
    if pep_arg in PEPTIDE_FILES:
        return PEPTIDE_FILES[pep_arg]
    if os.path.isfile(pep_arg):
        return pep_arg
    raise FileNotFoundError(f"Cannot find peptide file: {pep_arg}")


def build_args_namespace(args):
    """Build the argparse.Namespace expected by TCREnv from our eval args."""
    ns = argparse.Namespace(
        reward_type="game",
        terminal=False,
        discount_penalty=0.8,
        mod_pos_penalty=1,
        no_mod_penalty=-0.5,
        mod_neg_penalty=-1,
        allow_imm_rew=None,
        allow_final_rew=True,
        beta=args.beta,
        rate=10,
        anneal_nomod_step=10000,
        anneal_nomod_rate=0.05,
        max_len=27,
        use_step=False,
        use_gmm=True,
        score_stop_criteria=args.score_stop_criteria,
        gmm_stop_criteria=args.gmm_stop_criteria,
        num_envs=args.num_envs,
        n_steps=20,
        max_step=args.max_step,
        peptide_path=args.peptide_file,
        sample_rate=0.8,
        device="cpu",
        hour=args.hour,
        max_size=args.max_size,
    )
    return ns



import torch
import gym
import multiprocessing as mp
from typing import Any, Callable, List, Optional, Tuple, Type, Union

from ppo import PPO
from tcr_env import TCREnv
from data_utils import num2seq, edit_sequence
import config as cfg
from reward import Reward
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper, VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn,
)
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs

# --- Worker and VecEnv (adapted from test_RL_tcrs.py) ---
def _worker(remote, parent_remote, env_fn_wrapper):
    from stable_baselines3.common.env_util import is_wrapped
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data[0], data[1])
                if done:
                    info["terminal_observation"] = observation
                remote.send((observation, reward, done, info))
            elif cmd == "reset":
                if data[0]:
                    if type(data[1]) is str:
                        observation = env.reset(peptide=data[1])
                    else:
                        observation = env.reset(peptide=data[1][0], init_tcr=data[1][1])
                else:
                    observation = data[2]
                remote.send(observation)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` not implemented")
        except EOFError:
            break

class EvalSubprocVecEnv(VecEnv):
    def __init__(self, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        self._last_obs = None
        self.reward_model = None
        n_envs = len(env_fns)
        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "spawn"
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            p_args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=p_args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.remotes[0].send(("get_attr", "max_tcr_len"))
        self.max_tcr_len = self.remotes[0].recv()
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def set_reward_model(self, reward_model):
        self.reward_model = reward_model

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def step(self, actions):
        tcrs = num2seq(self._last_obs[:, :self.max_tcr_len])
        peptides = num2seq(self._last_obs[:, self.max_tcr_len:])
        new_tcrs = edit_sequence(tcrs, actions)
        rewards = self.reward_model.reward(new_tcrs, peptides)
        self.step_async((actions, rewards))
        return self.step_wait()

    def step_async(self, data):
        actions, rewards = data
        for remote, action, reward in zip(self.remotes, actions, rewards):
            remote.send(("step", (action, reward)))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rew, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rew), np.stack(dones), infos

    def reset(self, dones, peptides, obs):
        self.reset_async(dones, peptides, obs)
        self._last_obs = self.reset_wait()
        return self._last_obs

    def reset_async(self, dones, peptides, obs):
        for remote, done, peptide, ob in zip(self.remotes, dones, peptides, obs):
            remote.send(("reset", (done, peptide, ob)))
        self.waiting = True

    def reset_wait(self):
        obs = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return _flatten_obs(obs, self.observation_space)

    def get_attr(self, attr_name, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]


def run_inference(args):
    """Run model inference and return results dict."""


    # --- Setup ---
    max_len = 27
    ergo_model_path = resolve_ergo_model(args.ergo_model)
    peptide_file = resolve_peptide_file(args.peptide_file)
    peptides = load_peptides(peptide_file)

    tcr_file = args.tcr_file if args.tcr_file else TEST_TCR_FILE
    all_tcrs = load_tcrs(tcr_file, max_len=max_len)
    if args.num_tcrs and args.num_tcrs < len(all_tcrs):
        np.random.seed(42)
        indices = np.random.choice(len(all_tcrs), args.num_tcrs, replace=False)
        tcrs = [all_tcrs[i] for i in sorted(indices)]
    else:
        tcrs = all_tcrs

    print(f"Peptides: {len(peptides)}")
    print(f"Test TCRs: {len(tcrs)}")
    print(f"ERGO model: {ergo_model_path}")
    print(f"Model path: {args.model_path}")

    env_ns = build_args_namespace(args)
    env_ns.peptide_path = peptide_file

    action_space = gym.spaces.multi_discrete.MultiDiscrete([max_len, 20])
    observation_space = gym.spaces.MultiDiscrete([20] * (25 + max_len))

    reward_model = Reward(args.beta, args.gmm_stop_criteria, ergo_model_file=ergo_model_path)
    m_env_kwargs = {
        "action_space": action_space,
        "observation_space": observation_space,
        "args": env_ns,
        "max_tcr_len": max_len,
    }

    m_env = make_vec_env(TCREnv, n_envs=args.num_envs, env_kwargs=m_env_kwargs, vec_env_cls=EvalSubprocVecEnv)
    m_env.set_reward_model(reward_model)

    model = PPO.load(args.model_path, env=m_env)

    # --- Run rollouts ---
    results = {peptide: [] for peptide in peptides}
    rollout_pairs = [(pep, tcr) for pep in peptides for tcr in tcrs for _ in range(args.rollout)]
    total = len(rollout_pairs)

    batch_peptides = rollout_pairs[:args.num_envs]
    batch_idxs = np.arange(args.num_envs)

    obs = m_env.reset([True] * len(batch_peptides), batch_peptides, [None] * len(batch_peptides))

    rollout = 0
    st_time = time.time()
    last_report = 0

    while rollout < total:
        with torch.no_grad():
            obs_tensor = obs_as_tensor(obs, cfg.device)
            actions, values, log_probs = model.policy.forward(obs_tensor)
        actions = actions.cpu().numpy()
        new_obs, rewards, dones, infos = m_env.step(actions)

        for idx, done in enumerate(dones):
            if done:
                peptide = rollout_pairs[batch_idxs[idx]][0]
                if max(batch_idxs) < total - 1:
                    batch_idxs[idx] = max(batch_idxs) + 1
                    batch_peptides[idx] = rollout_pairs[batch_idxs[idx]]
                rollout += 1
                if rollout <= total:
                    results[peptide].append({
                        "init_tcr": infos[idx].get("init_tcr", ""),
                        "final_tcr": infos[idx].get("new_tcr", ""),
                        "ergo_score": float(infos[idx].get("score", 0)),
                        "edit_dist": float(infos[idx].get("score1", 0)),
                        "gmm_likelihood": float(infos[idx].get("score2", 0)),
                    })

        # Progress reporting
        if rollout - last_report >= max(total // 20, 1):
            elapsed = time.time() - st_time
            rate = rollout / elapsed if elapsed > 0 else 0
            eta = (total - rollout) / rate if rate > 0 else 0
            print(f"  Progress: {rollout}/{total} ({rollout/total*100:.1f}%) "
                  f"| {rate:.1f} eps/s | ETA: {eta:.0f}s")
            last_report = rollout

        obs = m_env.reset(dones, batch_peptides, new_obs)

        ck_time = time.time()
        if ck_time - st_time >= args.hour * 3600:
            print(f"  Time limit reached ({args.hour}h). Stopping.")
            break

    m_env.close()
    elapsed = time.time() - st_time
    total_collected = sum(len(v) for v in results.values())
    print(f"\nInference complete: {total_collected} results in {elapsed:.1f}s")

    return results, peptides


def write_results(results, peptides, out_file):
    """Write results in standard TCRPPO output format."""
    ensure_dir(os.path.dirname(os.path.abspath(out_file)))
    with open(out_file, "w") as f:
        for peptide in peptides:
            for r in results.get(peptide, []):
                f.write(f"{peptide} {r['init_tcr']} {r['final_tcr']} "
                        f"{r['ergo_score']:.4f} {r['edit_dist']:.4f} "
                        f"{r['gmm_likelihood']:.4f}\n")
    print(f"Results written to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Run TCRPPO model evaluation on test data")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained PPO model (without .zip). Auto-detected if omitted.")
    parser.add_argument("--ergo_model", type=str, default="ae_mcpas",
                        help="ERGO model: 'ae_mcpas', 'ae_vdjdb', or path to .pt file")
    parser.add_argument("--peptide_file", type=str, default="ae_mcpas",
                        help="Peptide file: 'ae_mcpas', 'ae_vdjdb', or path to file")
    parser.add_argument("--tcr_file", type=str, default=None,
                        help="Test TCR file (default: data/tcrdb/test_uniq_tcr_seqs.txt)")
    parser.add_argument("--num_tcrs", type=int, default=1000,
                        help="Number of test TCRs to sample (default: 1000, 0=all)")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--rollout", type=int, default=1,
                        help="Rollout per (peptide, TCR) pair")
    parser.add_argument("--max_step", type=int, default=8,
                        help="Max steps per episode")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--score_stop_criteria", type=float, default=0.9)
    parser.add_argument("--gmm_stop_criteria", type=float, default=1.2577)
    parser.add_argument("--hour", type=int, default=5, help="Max run time in hours")
    parser.add_argument("--max_size", type=int, default=50000)
    parser.add_argument("--out", type=str, default=None,
                        help="Output result file path")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = find_model_checkpoint(ergo_key=args.ergo_model)
        if args.model_path is None:
            print("ERROR: No model checkpoint found. Specify --model_path.")
            sys.exit(1)
        print(f"Auto-detected model: {args.model_path}")

    if args.num_tcrs == 0:
        args.num_tcrs = None

    if args.out is None:
        ergo_tag = args.ergo_model if args.ergo_model in ERGO_MODELS else "custom"
        args.out = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results",
            f"eval_{ergo_tag}_n{args.num_tcrs or 'all'}.txt"
        )

    results, peptides = run_inference(args)
    write_results(results, peptides, args.out)
    print(f"\nTo analyze: python eval_metrics.py --result {args.out}")


if __name__ == "__main__":
    main()
