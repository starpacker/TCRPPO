"""Custom PPO trainer with autoregressive action space support.

Supports:
- VecEnv with n_envs parallel environments
- Autoregressive 3-head rollout collection
- GAE advantage estimation
- PPO clipped objective with entropy bonus
- Online SpecificityCallback
- Checkpointing at milestones
- TensorBoard logging
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.utils.constants import (
    ERGO_MODEL_DIR, ERGO_AE_FILE, NUM_OPS, NUM_AMINO_ACIDS,
    MAX_TCR_LEN, OP_SUB, OP_INS, OP_STOP, PROJECT_ROOT,
)
from tcrppo_v2.policy import ActorCritic


class RolloutBuffer:
    """Buffer for storing rollout data from VecEnv."""

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, max_tcr_len: int = 20, device: str = "cpu"):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.max_tcr_len = max_tcr_len
        self.device = device
        self.ptr = 0

        # Pre-allocate arrays
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.ops = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.positions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.tokens = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.valid = np.ones((n_steps, n_envs), dtype=bool)

        # Action masks
        self.op_masks = np.zeros((n_steps, n_envs, NUM_OPS), dtype=bool)
        self.pos_masks = np.zeros((n_steps, n_envs, max_tcr_len), dtype=bool)
        self.token_masks = np.ones((n_steps, n_envs, max_tcr_len, NUM_AMINO_ACIDS), dtype=bool)

        # Computed after rollout
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        op: np.ndarray,
        pos: np.ndarray,
        tok: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        op_mask: np.ndarray,
        pos_mask: np.ndarray,
        token_mask: np.ndarray,
    ) -> None:
        """Add one step of data."""
        if pos_mask.shape[-1] != self.max_tcr_len:
            raise ValueError(
                f"pos_mask length mismatch: expected {self.max_tcr_len}, got {pos_mask.shape[-1]}"
            )
        if token_mask.shape[-2:] != (self.max_tcr_len, NUM_AMINO_ACIDS):
            raise ValueError(
                "token_mask shape mismatch: expected "
                f"(*, {self.max_tcr_len}, {NUM_AMINO_ACIDS}), got {token_mask.shape}"
            )
        self.obs[self.ptr] = obs
        self.ops[self.ptr] = op
        self.positions[self.ptr] = pos
        self.tokens[self.ptr] = tok
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.op_masks[self.ptr] = op_mask
        self.pos_masks[self.ptr] = pos_mask
        self.token_masks[self.ptr] = token_mask
        self.valid[self.ptr] = True
        self.ptr += 1

    def clip_episode_to_best_step(
        self,
        env_idx: int,
        start_row: int,
        end_row: int,
        keep_row: int,
        reward: float,
    ) -> None:
        """Keep an episode prefix through keep_row and drop later transitions."""
        start_row = max(0, int(start_row))
        end_row = min(self.n_steps - 1, int(end_row))
        keep_row = min(max(start_row, int(keep_row)), end_row)
        env_idx = int(env_idx)

        self.rewards[start_row:end_row + 1, env_idx] = 0.0
        self.dones[start_row:end_row + 1, env_idx] = 0.0
        self.valid[start_row:end_row + 1, env_idx] = True

        self.rewards[keep_row, env_idx] = float(np.nan_to_num(
            reward, nan=0.0, posinf=0.0, neginf=0.0
        ))
        self.dones[keep_row, env_idx] = 1.0
        if keep_row < end_row:
            self.valid[keep_row + 1:end_row + 1, env_idx] = False

    def compute_gae(self, last_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and returns."""
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            invalid = ~self.valid[t]
            if invalid.any():
                self.advantages[t, invalid] = 0.0
                self.returns[t, invalid] = 0.0
                last_gae[invalid] = 0.0

            if t == self.n_steps - 1:
                next_values = last_value
            else:
                next_values = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            last_gae[invalid] = 0.0
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values
        self.returns[~self.valid] = 0.0

    def get_batches(self, batch_size: int) -> List[dict]:
        """Flatten and yield minibatches."""
        total = self.n_steps * self.n_envs
        indices = np.flatnonzero(self.valid.reshape(total))
        np.random.shuffle(indices)

        # Flatten
        flat_obs = self.obs.reshape(total, -1)
        flat_ops = self.ops.reshape(total)
        flat_pos = self.positions.reshape(total)
        flat_tok = self.tokens.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_op_masks = self.op_masks.reshape(total, NUM_OPS)
        flat_pos_masks = self.pos_masks.reshape(total, self.max_tcr_len)
        flat_token_masks = self.token_masks.reshape(total, self.max_tcr_len, NUM_AMINO_ACIDS)

        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": torch.FloatTensor(flat_obs[idx]).to(self.device),
                "ops": torch.LongTensor(flat_ops[idx]).to(self.device),
                "positions": torch.LongTensor(flat_pos[idx]).to(self.device),
                "tokens": torch.LongTensor(flat_tok[idx]).to(self.device),
                "old_log_probs": torch.FloatTensor(flat_log_probs[idx]).to(self.device),
                "advantages": torch.FloatTensor(flat_advantages[idx]).to(self.device),
                "returns": torch.FloatTensor(flat_returns[idx]).to(self.device),
                "op_masks": torch.BoolTensor(flat_op_masks[idx]).to(self.device),
                "pos_masks": torch.BoolTensor(flat_pos_masks[idx]).to(self.device),
                "token_masks": torch.BoolTensor(flat_token_masks[idx]).to(self.device),
            }

    def reset(self) -> None:
        """Reset buffer pointer."""
        self.ptr = 0
        self.valid.fill(False)


class EliteBuffer:
    """Tracks best TCRs found during training for tFold re-scoring.

    Stores completed episodes whose ERGO score exceeds a threshold.
    The top-K entries are periodically re-scored with tFold to provide
    correction gradients that prevent ERGO exploitation.
    """

    def __init__(self, max_size: int = 500, score_threshold: float = 0.7):
        self.max_size = max_size
        self.score_threshold = score_threshold
        # Each entry: (ergo_score, tcr, peptide, last_obs, last_op, last_pos, last_tok, last_logprob)
        self.buffer: List[Tuple] = []
        self.seen: set = set()

    def add_episode(
        self,
        tcr: str,
        peptide: str,
        ergo_score: float,
        last_obs: np.ndarray,
        last_op: int,
        last_pos: int,
        last_tok: int,
        last_logprob: float,
    ) -> None:
        """Add a completed episode if it exceeds the score threshold."""
        if ergo_score < self.score_threshold:
            return
        key = (tcr, peptide)
        if key in self.seen:
            return
        self.seen.add(key)
        self.buffer.append((ergo_score, tcr, peptide, last_obs, last_op, last_pos, last_tok, last_logprob))
        if len(self.buffer) > self.max_size:
            self.buffer.sort(key=lambda x: x[0], reverse=True)
            removed = self.buffer[self.max_size:]
            self.buffer = self.buffer[:self.max_size]
            for item in removed:
                self.seen.discard((item[1], item[2]))

    def get_top_k(self, k: int = 32) -> List[Tuple]:
        """Get top-K entries by ERGO score."""
        self.buffer.sort(key=lambda x: x[0], reverse=True)
        return self.buffer[:k]

    def __len__(self) -> int:
        return len(self.buffer)


class PPOTrainer:
    """Custom PPO implementation for autoregressive TCR editing."""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cuda")
        self.total_timesteps = config.get("total_timesteps", 10_000_000)
        self.n_envs = config.get("n_envs", 20)
        self.n_steps = config.get("n_steps", 128)
        self.batch_size = config.get("batch_size", 256)
        self.n_epochs = config.get("n_epochs", 4)
        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.90)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_range = config.get("clip_range", 0.2)
        self.target_kl = config.get("target_kl", None)
        self.entropy_coef = config.get("entropy_coef", 0.05)
        self.entropy_coef_final = config.get("entropy_coef_final", None)  # None = no decay
        self.entropy_decay_start = config.get("entropy_decay_start", 1000000)  # start decay at this step
        self.vf_coef = config.get("vf_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.seed = config.get("seed", 42)
        self.reward_mode = config.get("reward_mode", "v2_full")
        self.run_name = config.get("run_name", "v2_run")

        # Paths
        self.output_dir = config.get("output_dir", "output")
        self.results_dir = config.get("results_dir", "results")
        self.milestones = config.get("milestones", [500000, 1000000, 2000000, 5000000, 10000000])
        self.checkpoint_interval = config.get("checkpoint_interval", 100000)
        self.latest_checkpoint_interval = config.get("latest_checkpoint_interval", 2000)
        self.tfold_use_cache = config.get("tfold_use_cache", True)
        self.active_clipping = config.get("active_clipping", False)

        # Eval
        self.eval_interval = config.get("eval_interval", 100000)
        self.eval_n_tcrs = config.get("eval_n_tcrs", 5)
        self.eval_n_decoys = config.get("eval_n_decoys", 50)
        self.eval_abort_threshold = config.get("eval_abort_threshold", 0.40)
        self.eval_warmup = config.get("eval_warmup", 500000)

        # Set seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # tFold correction config
        self.tfold_correction = config.get("tfold_correction", False)
        self.tfold_rescore_interval = config.get("tfold_rescore_interval", 50)
        self.tfold_top_k = config.get("tfold_top_k", 32)
        self.tfold_correction_alpha = config.get("tfold_correction_alpha", 2.0)
        self.elite_buffer_size = config.get("elite_buffer_size", 500)
        self.elite_score_threshold = config.get("elite_score_threshold", 0.7)

        # Best TCR tracking
        self.best_tcr_log_path = config.get("best_tcr_log_path", None)
        self.best_tcr_min_affinity = config.get("best_tcr_min_affinity", -0.5)
        self.best_tcr_max_decoy_violation = config.get("best_tcr_max_decoy_violation", 1.0)

        # Will be initialized in setup()
        self.policy = None
        self.optimizer = None
        self.vec_env = None
        self.buffer = None
        self.logger = None
        self.tfold_scorer = None
        self.elite_buffer = None
        self.tcr_pool = None
        self.online_tcr_pool_events_path = None
        self.online_tcr_pool_snapshot_path = None

    def _build_pmhc_obs_transform(self, esm_cache, pmhc_loader, targets: List[str]) -> Optional[Dict[str, object]]:
        """Build an observation-only pMHC embedding transform.

        The transform is applied only when constructing policy observations.
        Reward/scorer inputs keep using the raw sequence or raw ESM path.
        """
        mode = self.config.get("pmhc_embedding_transform", self.config.get("peptide_centering", "none"))
        if isinstance(mode, bool):
            mode = "center" if mode else "none"
        mode = str(mode).strip().lower()
        if mode in ("", "none", "raw", "off", "false", "no"):
            return None

        valid_modes = {"center", "mean_center", "center_only", "center_layernorm", "mean_center_layernorm"}
        if mode not in valid_modes:
            raise ValueError(
                "Unsupported pmhc_embedding_transform={!r}; expected one of {}".format(
                    mode, sorted(valid_modes | {"none"})
                )
            )

        out_dir = os.path.join(self.output_dir, self.run_name)
        os.makedirs(out_dir, exist_ok=True)
        center_path = self.config.get(
            "pmhc_center_path",
            os.path.join(out_dir, "pmhc_embedding_center.pt"),
        )
        self.config["pmhc_center_path"] = center_path

        if os.path.exists(center_path):
            payload = torch.load(center_path, map_location=self.device)
            center = payload["center"].to(self.device).float()
            source_targets = payload.get("targets", targets)
            print(
                f"  pMHC obs transform: loaded center from {center_path} "
                f"(mode={mode}, n={len(source_targets)})",
                flush=True,
            )
        else:
            if not targets:
                raise ValueError("Cannot compute pMHC embedding center with an empty target list")

            pmhc_strings = [pmhc_loader.get_pmhc_string(pep) for pep in targets]
            embs = []
            with torch.no_grad():
                for pmhc_string in pmhc_strings:
                    embs.append(esm_cache.encode_pmhc(pmhc_string).float())
            X = torch.stack(embs, dim=0)
            center = X.mean(dim=0)

            payload = {
                "center": center.detach().cpu(),
                "mode": mode,
                "targets": list(targets),
                "pmhc_strings": pmhc_strings,
                "source": "train_targets",
            }
            torch.save(payload, center_path)
            norms = X.norm(dim=1)
            centered = X - center
            centered_norm = centered.norm(dim=1)
            print(
                "  pMHC obs transform: computed center "
                f"(mode={mode}, n={len(targets)}, path={center_path})",
                flush=True,
            )
            if len(targets) > 1:
                Xn = X / (norms[:, None] + 1e-8)
                raw_cos = Xn @ Xn.T
                centered_n = centered / (centered_norm[:, None] + 1e-8)
                centered_cos = centered_n @ centered_n.T
                mask = ~torch.eye(len(targets), dtype=torch.bool, device=X.device)
                raw_off = raw_cos[mask]
                centered_off = centered_cos[mask]
                print(
                    "    raw cos mean/min/max="
                    f"{raw_off.mean().item():.4f}/{raw_off.min().item():.4f}/{raw_off.max().item():.4f}; "
                    "centered cos mean/min/max="
                    f"{centered_off.mean().item():.4f}/{centered_off.min().item():.4f}/{centered_off.max().item():.4f}",
                    flush=True,
                )
            print(
                "    center_norm={:.4f}, raw_norm_mean={:.4f}, centered_norm_mean={:.4f}".format(
                    center.norm().item(), norms.mean().item(), centered_norm.mean().item()
                ),
                flush=True,
            )

        layer_norm = mode in ("center_layernorm", "mean_center_layernorm")
        return {"center": center.detach(), "layer_norm": layer_norm, "mode": mode}

    def setup(self) -> None:
        """Initialize all components."""
        print(f"Setting up PPO trainer: {self.run_name}")
        print(f"  reward_mode={self.reward_mode}, n_envs={self.n_envs}")
        print(f"  weights: aff={self.config.get('w_affinity', 1.0)}, "
              f"decoy={self.config.get('w_decoy', 0.8)}, "
              f"nat={self.config.get('w_naturalness', 0.5)}, "
              f"div={self.config.get('w_diversity', 0.2)}")
        if self.config.get("min_steps", 0) > 0:
            print(f"  min_steps={self.config['min_steps']}, "
                  f"penalty={self.config.get('min_steps_penalty', 0.0)}")
        if self.active_clipping:
            print("  active_clipping=True (train on best-affinity episode prefix)")

        # Build scorers
        from tcrppo_v2.utils.esm_cache import ESMCache
        from tcrppo_v2.data.pmhc_loader import PMHCLoader, EVAL_TARGETS
        from tcrppo_v2.data.tcr_pool import TCRPool
        from tcrppo_v2.reward_manager import RewardManager
        from tcrppo_v2.env import VecTCREditEnv

        # pMHC loader — optionally filter to specific targets
        train_targets_cfg = self.config.get("train_targets", None)
        if train_targets_cfg:
            if os.path.isfile(train_targets_cfg):
                with open(train_targets_cfg) as f:
                    target_list = [line.strip() for line in f if line.strip()]
            else:
                target_list = [t.strip() for t in train_targets_cfg.split(",")]
            pmhc_loader = PMHCLoader(targets=target_list)
            print(f"  Filtered targets: {len(target_list)} peptides from {train_targets_cfg}")
        else:
            pmhc_loader = PMHCLoader(mode="train")
        targets = pmhc_loader.get_target_list()

        # Affinity scorer — selected by config["affinity_model"]
        affinity_model = self.config.get("affinity_model", "ergo")
        if affinity_model == "nettcr":
            from tcrppo_v2.scorers.affinity_nettcr_pytorch import AffinityNetTCRPyTorchScorer
            affinity_scorer = AffinityNetTCRPyTorchScorer(device=self.device)
            print("  NetTCR-PyTorch loaded")
        elif affinity_model == "tfold_amp":
            from tcrppo_v2.scorers.affinity_tfold_amp import AffinityTFoldAMPScorer
            affinity_scorer = AffinityTFoldAMPScorer(
                device=self.device,
                gpu_id=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
                cache_path=self.config.get("tfold_cache_path", "data/tfold_feature_cache.db"),
                use_amp=True,
                fallback_to_subprocess=False,  # Disable fallback - AMP should handle everything
                cache_read_only=self.config.get("tfold_cache_read_only", False),
            )
            print("  tFold AMP loaded (3.97× faster, fallback disabled)")
        elif affinity_model == "tfold":
            from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer
            _socket_path = self.config.get("tfold_server_socket", "/tmp/tfold_server.sock")
            print(f"  [DEBUG] tfold_server_socket from config: {_socket_path}", flush=True)
            affinity_scorer = AffinityTFoldScorer(
                device=self.device,
                gpu_id=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
                server_socket_path=_socket_path,
                cache_path=self.config.get("tfold_cache_path", "data/tfold_feature_cache.db"),
                use_cache=self.tfold_use_cache,
                cache_read_only=self.config.get("tfold_cache_read_only", False),
            )
            print(
                f"  tFold V3.4 loaded (cache={affinity_scorer.cache_stats['cache_size']} "
                f"enabled={bool(affinity_scorer.cache_stats.get('cache_enabled', 1))})",
                flush=True,
            )
        elif affinity_model == "ensemble":
            from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
            from tcrppo_v2.scorers.affinity_nettcr import AffinityNetTCRScorer
            from tcrppo_v2.scorers.affinity_ensemble import EnsembleAffinityScorer
            model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
            ergo_scorer = AffinityERGOScorer(
                model_file=model_file,
                ae_file=ERGO_AE_FILE,
                device=self.device,
                mc_samples=self.config.get("affinity_mc_samples", 10),
            )
            print("  ERGO loaded")
            nettcr_scorer = AffinityNetTCRScorer(device=self.device)
            print("  NetTCR loaded")
            affinity_scorer = EnsembleAffinityScorer(
                scorers=[ergo_scorer, nettcr_scorer],
                weights=[0.5, 0.5],
            )
        elif affinity_model == "tcbind":
            from tcrppo_v2.scorers.affinity_tcbind import AffinityTCBindScorer
            tcbind_weights = self.config.get(
                "tcbind_weights", "runs/binding_classifier_v2/best_model.pt"
            )
            affinity_scorer = AffinityTCBindScorer(
                weights_path=tcbind_weights,
                device=self.device,
            )
            print(f"  TCBind loaded ({tcbind_weights})")
        elif affinity_model == "ensemble_ergo_tcbind":
            from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
            from tcrppo_v2.scorers.affinity_tcbind import AffinityTCBindScorer
            from tcrppo_v2.scorers.affinity_ensemble import EnsembleAffinityScorer
            model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
            ergo_scorer = AffinityERGOScorer(
                model_file=model_file,
                ae_file=ERGO_AE_FILE,
                device=self.device,
                mc_samples=self.config.get("affinity_mc_samples", 10),
            )
            print("  ERGO loaded")
            tcbind_weights = self.config.get(
                "tcbind_weights", "runs/binding_classifier_v2/best_model.pt"
            )
            tcbind_scorer = AffinityTCBindScorer(
                weights_path=tcbind_weights,
                device=self.device,
            )
            print(f"  TCBind loaded ({tcbind_weights})")
            affinity_scorer = EnsembleAffinityScorer(
                scorers=[ergo_scorer, tcbind_scorer],
                weights=[0.5, 0.5],
            )
        elif affinity_model == "ensemble_ergo_tfold":
            from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
            from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer
            from tcrppo_v2.scorers.affinity_ensemble import EnsembleAffinityScorer
            model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
            ergo_scorer = AffinityERGOScorer(
                model_file=model_file,
                ae_file=ERGO_AE_FILE,
                device=self.device,
                mc_samples=self.config.get("affinity_mc_samples", 10),
            )
            print("  ERGO loaded")
            tfold_scorer = AffinityTFoldScorer(
                device=self.device,
                gpu_id=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
            )
            print(f"  tFold V3.4 loaded (cache={tfold_scorer.cache_stats['cache_size']} entries)")
            affinity_scorer = EnsembleAffinityScorer(
                scorers=[ergo_scorer, tfold_scorer],
                weights=[0.5, 0.5],
            )
        elif affinity_model == "tfold_cascade":
            from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
            from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer, TFoldCascadeScorer
            model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")

            # Initialize tFold scorer (with cache)
            tfold_scorer = AffinityTFoldScorer(
                device=self.device,
                gpu_id=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
            )
            print(f"  tFold V3.4 loaded (cache={tfold_scorer.cache_stats['cache_size']} entries)")

            # Wrap with cascade logic
            affinity_scorer = TFoldCascadeScorer(
                ergo_model_file=model_file,
                tfold_scorer=tfold_scorer,
                uncertainty_threshold=self.config.get("cascade_threshold", 0.15),
                mc_samples=self.config.get("affinity_mc_samples", 10),
                ergo_device=self.device,
            )
            print(f"  ERGO+tFold cascade initialized (threshold={self.config.get('cascade_threshold', 0.15)})")
        elif affinity_model == "hybrid":
            from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
            from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer
            from tcrppo_v2.scorers.hybrid_scorer import HybridScorer

            # Initialize ERGO (primary, fast)
            model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
            ergo_scorer = AffinityERGOScorer(
                model_file=model_file,
                ae_file=ERGO_AE_FILE,
                device=self.device,
                mc_samples=self.config.get("affinity_mc_samples", 10),
            )
            print("  ERGO loaded (primary)")

            # Initialize tFold (secondary, accurate)
            tfold_scorer = AffinityTFoldScorer(
                device=self.device,
                gpu_id=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
            )
            print(f"  tFold V3.4 loaded (secondary, cache={tfold_scorer.cache_stats['cache_size']} entries)")

            # Wrap with hybrid scorer
            tfold_ratio = self.config.get("hybrid_tfold_ratio", 0.1)
            affinity_scorer = HybridScorer(
                primary_scorer=ergo_scorer,
                secondary_scorer=tfold_scorer,
                secondary_ratio=tfold_ratio,
                seed=self.config.get("seed", 42),
            )
            print(f"  Hybrid scorer (ERGO {1-tfold_ratio:.0%} + tFold {tfold_ratio:.0%})")
        elif affinity_model == "cascade":
            from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
            from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer
            from tcrppo_v2.scorers.cascade_scorer import CascadeScorer

            # Initialize ERGO (primary, fast)
            model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
            ergo_scorer = AffinityERGOScorer(
                model_file=model_file,
                ae_file=ERGO_AE_FILE,
                device=self.device,
                mc_samples=self.config.get("affinity_mc_samples", 10),
            )
            print("  ERGO loaded (primary)")

            # Initialize tFold (secondary, accurate)
            tfold_scorer = AffinityTFoldScorer(
                device=self.device,
                gpu_id=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
            )
            print(f"  tFold V3.4 loaded (secondary, cache={tfold_scorer.cache_stats['cache_size']} entries)")

            # Wrap with cascade scorer
            cascade_threshold = self.config.get("cascade_threshold", 0.3)
            tfold_weight = self.config.get("cascade_tfold_weight", 0.7)
            ergo_weight = self.config.get("cascade_ergo_weight", 0.3)

            affinity_scorer = CascadeScorer(
                primary_scorer=ergo_scorer,
                secondary_scorer=tfold_scorer,
                threshold=cascade_threshold,
                tfold_weight=tfold_weight,
                ergo_weight=ergo_weight,
            )
            print(f"  Cascade scorer (ERGO → tFold if score > {cascade_threshold})")
        elif affinity_model == "none":
            # Pretrain mode: no affinity scoring
            affinity_scorer = None
            print("  Affinity scorer: NONE (pretrain mode)")
        else:  # default: ergo
            from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
            model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
            affinity_scorer = AffinityERGOScorer(
                model_file=model_file,
                ae_file=ERGO_AE_FILE,
                device=self.device,
                mc_samples=self.config.get("affinity_mc_samples", 10),
            )
            print("  ERGO loaded")

        # State encoder (ESM-2 or lightweight BiLSTM)
        print("  Loading state encoder...", flush=True)
        encoder_type = self.config.get("encoder", "esm2")
        if encoder_type == "lightweight":
            from tcrppo_v2.utils.lightweight_encoder import LightweightEncoder
            encoder_dim = self.config.get("encoder_dim", 256)
            esm_cache = LightweightEncoder(
                device=self.device,
                encoder_output_dim=encoder_dim,
                tcr_cache_size=self.config.get("esm_tcr_cache_size", 4096),
                disk_cache_path=self.config.get("esm_cache_path"),
            )
            print(f"  Lightweight encoder (dim={esm_cache.output_dim})")
        else:
            esm_cache = ESMCache(
                device=self.device,
                tcr_cache_size=self.config.get("esm_tcr_cache_size", 4096),
                disk_cache_path=self.config.get("esm_cache_path"),
            )
            print(f"  ESM-2 loaded (dim={esm_cache.embed_dim}, disk_cache=ready)", flush=True)

        print(f"  pMHC loader: {len(targets)} targets", flush=True)

        # TCR pool with L1 seeds enabled for enhanced curriculum
        print("  Loading TCR pool...", flush=True)
        l1_dir = os.path.join(PROJECT_ROOT, "data", "l1_seeds")
        if not os.path.isdir(l1_dir):
            l1_dir = None  # Fall back to L0+L2 only
        tcr_pool = TCRPool(
            tcrdb_path=self.config.get("tcrdb_path", "/share/liuyutian/TCRPPO/data/tcrdb"),
            l1_seeds_dir=l1_dir,
            l0_mutation_range=tuple(self.config.get("l0_mutation_range", (3, 5))),
            l1_top_k=self.config.get("l1_top_k", 500),
            curriculum_schedule=self.config.get("curriculum_schedule"),
            seed=self.seed,
            online_pool_enabled=self.config.get("online_tcr_pool_enabled", False),
            online_pool_start_step=self.config.get("online_tcr_pool_start_step", 0),
            online_pool_warmup_steps=self.config.get("online_tcr_pool_warmup_steps", 100000),
            online_pool_max_ratio=self.config.get("online_tcr_pool_max_ratio", 0.5),
            online_pool_max_per_target=self.config.get("online_tcr_pool_max_per_target", 256),
            online_pool_mutate_prob=self.config.get("online_tcr_pool_mutate_prob", 0.0),
            online_pool_mutation_range=tuple(self.config.get("online_tcr_pool_mutation_range", (1, 2))),
            online_pool_min_hamming=self.config.get("online_tcr_pool_min_hamming", 2),
            online_pool_sample_bands=self.config.get("online_tcr_pool_sample_bands"),
            online_pool_snapshot_path=self.config.get("online_tcr_pool_snapshot_path"),
            online_pool_elite_ratio=self.config.get("online_tcr_pool_elite_ratio", 0.0),
            online_pool_elite_min_affinity=self.config.get("online_tcr_pool_elite_min_affinity", -0.5),
            online_pool_dynamic_mode=self.config.get("online_tcr_pool_dynamic_mode", "band"),
            online_pool_dynamic_below=self.config.get("online_tcr_pool_dynamic_below", 0.7),
            online_pool_dynamic_above=self.config.get("online_tcr_pool_dynamic_above", 0.0),
        )
        self.tcr_pool = tcr_pool
        
        # Apply adaptive bands patch if enabled
        if self.config.get("online_tcr_pool_adaptive_bands", False):
            print("  Loading trace96 adaptive bands patch...", flush=True)
            import tcrppo_v2.data.tcr_pool_trace96_adaptive
            print("  Adaptive bands patch applied", flush=True)
        
        if self.config.get("online_tcr_pool_snapshot_path"):
            stats = tcr_pool.get_online_pool_stats() if hasattr(tcr_pool, "get_online_pool_stats") else {}
            print(
                f"  Loaded online pool snapshot: targets={len(stats)}, total={sum(stats.values())}",
                flush=True,
            )
        print("  TCRPool initialized", flush=True)
        # Load L0 seeds from decoy D + tc-hard known binders
        decoy_lib_path = self.config.get("decoy_library_path", "/share/liuyutian/pMHC_decoy_library")
        print(f"  Loading L0 seeds from {decoy_lib_path}...", flush=True)
        tcr_pool.load_l0_from_decoy_d(decoy_lib_path, targets)
        print("  L0 seeds from decoy_d loaded", flush=True)
        # Also load tc-hard CDR3b binders as L0 seeds
        l0_tchard_dir = os.path.join(PROJECT_ROOT, "data", "l0_seeds_tchard")
        if os.path.isdir(l0_tchard_dir):
            print(f"  Loading L0 seeds from {l0_tchard_dir}...", flush=True)
            tcr_pool.load_l0_from_dir(l0_tchard_dir)
            print("  L0 seeds from tchard loaded", flush=True)
        # Load additional L0 seeds from config-specified directory
        l0_custom_dir = self.config.get("l0_seeds_dir")
        if l0_custom_dir:
            l0_custom_path = os.path.join(PROJECT_ROOT, l0_custom_dir) if not os.path.isabs(l0_custom_dir) else l0_custom_dir
            if os.path.isdir(l0_custom_path):
                print(f"  Loading custom L0 seeds from {l0_custom_path}...", flush=True)
                tcr_pool.load_l0_from_dir(l0_custom_path)
                print("  Custom L0 seeds loaded", flush=True)
        l0_targets = tcr_pool.get_l0_targets()
        l1_targets = tcr_pool.get_l1_targets()
        print(f"  TCR pool: {tcr_pool.num_tcrdb_seqs} seqs, "
              f"L0 targets={len(l0_targets)}/{len(targets)}, "
              f"L1 targets={len(l1_targets)}/{len(targets)}", flush=True)
        if self.config.get("online_tcr_pool_enabled", False):
            out_dir = os.path.join(self.output_dir, self.run_name)
            os.makedirs(out_dir, exist_ok=True)
            self.online_tcr_pool_events_path = os.path.join(out_dir, "online_tcr_pool_events.jsonl")
            self.online_tcr_pool_snapshot_path = os.path.join(out_dir, "online_tcr_pool_snapshot.json")
            print(
                "  Online TCR pool enabled: "
                f"start={self.config.get('online_tcr_pool_start_step', 0)}, "
                f"warmup={self.config.get('online_tcr_pool_warmup_steps', 100000)}, "
                f"max_ratio={self.config.get('online_tcr_pool_max_ratio', 0.5)}, "
                f"min_affinity={self.config.get('online_tcr_pool_min_affinity', -1.0)}, "
                f"min_hamming={self.config.get('online_tcr_pool_min_hamming', 2)}, "
                f"max_per_target={self.config.get('online_tcr_pool_max_per_target', 256)}, "
                f"sample_bands={self.config.get('online_tcr_pool_sample_bands', None)}",
                flush=True,
            )

        # Determine if reward schedule requires pre-loading all scorers
        has_schedule = bool(self.config.get("reward_schedule"))

        # Decoy scorer (for reward, with LogSumExp penalty)
        decoy_scorer = None
        self.decoy_scorer = None
        # Load decoy scorer for any mode that uses decoy penalty or contrastive sampling
        decoy_reward_modes = (
            "v2_full",
            "v2_decoy_only",
            "raw_decoy",
            "raw_multi_penalty",
            "threshold_penalty",
            "contrastive_ergo",
            "v2_delta_minus_decoy",
            "v2_target_guarded_decoy",
            "v2_absolute_specificity",
            "v2_simple_target_gated_decoy",
            "v2_hybrid_abs_delta_gated_decoy",
            "v2_soft_target_decoy_gap",
        )
        if has_schedule or self.reward_mode in decoy_reward_modes:
            print("  Loading decoy scorer...", flush=True)
            from tcrppo_v2.scorers.decoy import DecoyScorer
            decoy_scorer = DecoyScorer(
                decoy_library_path=decoy_lib_path,
                targets=targets,
                tier_weights=self.config.get("decoy_tier_weights"),
                K=self.config.get("decoy_K", 32),
                tau=self.config.get("decoy_tau", 10.0),
                affinity_scorer=affinity_scorer,
                rng=np.random.default_rng(self.seed),
            )
            # Start with only tier A unlocked
            decoy_scorer.set_unlocked_tiers(["A"])
            if hasattr(decoy_scorer, "set_decoy_difficulty"):
                decoy_scorer.set_decoy_difficulty(self.config.get("decoy_difficulty", "easy"))
            self.decoy_scorer = decoy_scorer
            print(f"  Decoy scorer loaded (K={decoy_scorer.K})", flush=True)

        # Naturalness scorer (requires ESM-2 model — skip for lightweight encoder)
        naturalness_scorer = None
        naturalness_modes = (
            "v2_full", "v2_no_decoy", "v2_no_decoy_delta",
            "v2_no_decoy_delta_calibrated", "v2_no_decoy_sigmoid_delta",
            "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty",
            "contrastive_ergo", "tfold_delta_amplified", "v2_delta_minus_decoy",
            "v2_target_guarded_decoy", "v2_absolute_specificity",
            "v2_simple_target_gated_decoy",
            "v2_soft_target_decoy_gap",
        )
        if has_schedule or self.reward_mode in naturalness_modes:
            scorer_type = self.config.get("naturalness_scorer_type", "esm2")
            
            if scorer_type == "ae_gmm":
                # Use AE+GMM naturalness scorer from TCRPPO
                from tcrppo_v2.scorers.naturalness_ae_gmm import NaturalnessAEGMMScorer
                ae_model = self.config.get("naturalness_ae_model", "/share/liuyutian/TCRPPO/code/reward/ae_model")
                gmm_model = self.config.get("naturalness_gmm_model", "/share/liuyutian/TCRPPO/code/reward/gmm.pkl")
                blosum_file = self.config.get("naturalness_blosum", "/share/liuyutian/TCRPPO/code/blosum.txt")
                ae_threshold = self.config.get("naturalness_ae_threshold", 0.8)
                naturalness_scorer = NaturalnessAEGMMScorer(
                    ae_model_path=ae_model,
                    gmm_model_path=gmm_model,
                    blosum_path=blosum_file,
                    threshold=ae_threshold,
                    device=self.device,
                )
                print(f"  Naturalness scorer loaded: AE+GMM (threshold={ae_threshold})")
                
            elif scorer_type == "esm2":
                # Use ESM-2 perplexity naturalness scorer
                if encoder_type == "lightweight":
                    print("  Naturalness scorer SKIPPED (lightweight encoder has no ESM-2 model)")
                else:
                    from tcrppo_v2.scorers.naturalness import NaturalnessScorer
                    stats_file = self.config.get("cdr3_ppl_stats", "data/cdr3_ppl_stats.json")
                    naturalness_scorer = NaturalnessScorer(
                        esm_model=esm_cache.model,
                        esm_alphabet=esm_cache.alphabet,
                        esm_batch_converter=esm_cache.batch_converter,
                        device=self.device,
                        stats_file=stats_file,
                        threshold_zscore=self.config.get("naturalness_threshold_zscore", 2.0),
                    )
                    print("  Naturalness scorer loaded: ESM-2 perplexity")
            else:
                raise ValueError(f"Unknown naturalness_scorer_type: {scorer_type}")

        # Diversity scorer
        diversity_scorer = None
        diversity_modes = (
            "v2_full", "v2_no_decoy", "v2_no_decoy_delta",
            "v2_no_decoy_delta_calibrated", "v2_no_decoy_sigmoid_delta",
            "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty",
            "v2_delta_minus_decoy", "v2_target_guarded_decoy", "v2_absolute_specificity",
            "v2_simple_target_gated_decoy",
            "v2_soft_target_decoy_gap",
        )
        if has_schedule or self.reward_mode in diversity_modes:
            from tcrppo_v2.scorers.diversity import DiversityScorer
            diversity_scorer = DiversityScorer(
                buffer_size=self.config.get("diversity_buffer_size", 512),
                similarity_threshold=self.config.get(
                    "diversity_similarity_threshold",
                    self.config.get("diversity_threshold", 0.85),
                ),
            )
            print("  Diversity scorer loaded")

        # Reward manager
        delta_reward_modes = (
            "v1_ergo_delta",
            "v2_no_decoy_delta",
            "v2_no_decoy_delta_calibrated",
            "v2_no_decoy_sigmoid_delta",
            "tfold_stepwise",
            "tfold_delta_calibrated",
            "tfold_delta_amplified",
            "v2_delta_minus_decoy",
            "v2_target_guarded_decoy",
            "v2_absolute_specificity",
        )
        use_delta_reward = (
            True if self.reward_mode in delta_reward_modes
            else self.config.get("use_delta_reward", False)
        )
        self.reward_manager = RewardManager(
            affinity_scorer=affinity_scorer,
            decoy_scorer=decoy_scorer,
            naturalness_scorer=naturalness_scorer,
            diversity_scorer=diversity_scorer,
            reward_mode=self.reward_mode,
            use_delta_reward=use_delta_reward,
            w_affinity=self.config.get("w_affinity", 1.0),
            w_decoy=self.config.get("w_decoy", 0.8),
            w_naturalness=self.config.get("w_naturalness", 0.5),
            w_diversity=self.config.get("w_diversity", 0.2),
            n_contrast_decoys=self.config.get("n_contrast_decoys", 4),
            convex_alpha=self.config.get("convex_alpha", 3.0),
            contrastive_agg=self.config.get("contrastive_agg", "mean"),
            ood_threshold=self.config.get("ood_threshold", 0.15),
            ood_penalty_weight=self.config.get("ood_penalty_weight", 1.0),
            ood_penalty_mode=self.config.get("ood_penalty_mode", "soft"),
            w_absolute_affinity=self.config.get("w_absolute_affinity", 0.25),
            affinity_ref_logit=self.config.get("affinity_ref_logit", -4.5),
            delta_amp_thresholds=self.config.get("delta_amp_thresholds"),
            delta_amp_slopes=self.config.get("delta_amp_slopes"),
            delta_negative_scale=self.config.get("delta_negative_scale", 1.0),
            delta_deadband=self.config.get("delta_deadband", 0.0),
            delta_deadband_penalty=self.config.get("delta_deadband_penalty", 0.0),
            delta_amp_clip=self.config.get("delta_amp_clip"),
            naturalness_gate=self.config.get("naturalness_gate_affinity", False),
            naturalness_gate_threshold=self.config.get("naturalness_gate_threshold", 0.0),
            affinity_guard_logit=self.config.get("affinity_guard_logit", -3.0),
            affinity_guard_tolerance=self.config.get("affinity_guard_tolerance", 0.35),
            affinity_guard_weight=self.config.get("affinity_guard_weight", 4.0),
            specificity_margin=self.config.get("specificity_margin", 1.0),
            decoy_drop_weight=self.config.get("decoy_drop_weight", 0.25),
            target_gate_temperature=self.config.get("target_gate_temperature", 0.25),
            decoy_affinity_ceiling=self.config.get("decoy_affinity_ceiling", -4.5),
            target_surplus_cap=self.config.get("target_surplus_cap", 2.0),
            target_decoy_gate_logit=self.config.get("target_decoy_gate_logit", -2.0),
            target_pass_bonus=self.config.get("target_pass_bonus", 1.0),
            decoy_affinity_center=self.config.get("decoy_affinity_center", -3.0),
            decoy_fixed_tiers=self.config.get("decoy_fixed_tiers", ["A", "B"]),
            decoy_k_per_tier=self.config.get("decoy_k_per_tier", 1),
            hybrid_delta_weight=self.config.get("hybrid_delta_weight", 0.25),
            curriculum_gates=self.config.get("curriculum_gates"),
            curriculum_bonuses=self.config.get("curriculum_bonuses"),
            gap_margin=self.config.get("gap_margin", 1.0),
            decoy_activation_threshold=self.config.get("decoy_activation_threshold", 0.5),
            w_gap=self.config.get("w_gap", 1.0),
            soft_gate_affinity=self.config.get("soft_gate_affinity", 0.5),
            soft_gate_temperature=self.config.get("soft_gate_temperature", 0.12),
            soft_decoy_min_gate=self.config.get("soft_decoy_min_gate", 0.02),
            decoy_topk=self.config.get("decoy_topk", 2),
            w_decoy_mean=self.config.get("w_decoy_mean", 0.15),
            decoy_margin_clip=self.config.get("decoy_margin_clip", 3.0),
            preserve_high_init_threshold=self.config.get("preserve_high_init_threshold"),
            preserve_high_init_tolerance=self.config.get("preserve_high_init_tolerance", 0.10),
            preserve_high_init_weight=self.config.get("preserve_high_init_weight", 0.0),
            improve_low_init_threshold=self.config.get("improve_low_init_threshold"),
            improve_low_init_min_delta=self.config.get("improve_low_init_min_delta", 0.05),
            improve_low_init_weight=self.config.get("improve_low_init_weight", 0.0),
            improve_low_init_max_penalty=self.config.get("improve_low_init_max_penalty"),
            pretrain_naturalness_only=self.config.get("pretrain_naturalness_only", False),
        )

        pmhc_obs_transform = self._build_pmhc_obs_transform(esm_cache, pmhc_loader, targets)
        self.max_tcr_len = int(self.config.get("max_tcr_len", 20))

        # VecEnv
        self.vec_env = VecTCREditEnv(
            n_envs=self.n_envs,
            esm_cache=esm_cache,
            pmhc_loader=pmhc_loader,
            tcr_pool=tcr_pool,
            reward_manager=self.reward_manager,
            max_steps=self.config.get("max_steps", self.config.get("max_steps_per_episode", 8)),
            max_tcr_len=self.max_tcr_len,
            min_tcr_len=self.config.get("min_tcr_len", 8),
            reward_mode=self.reward_mode,
            min_steps=self.config.get("min_steps", 0),
            min_steps_penalty=self.config.get("min_steps_penalty", 0.0),
            ban_stop=self.config.get("ban_stop", False),
            sub_only=self.config.get("sub_only", False),
            terminal_reward_only=self.config.get("terminal_reward_only", False),
            active_clipping=self.active_clipping,
            use_biochem_features=self.config.get("use_biochem_features", False),
            include_state_scalars=self.config.get("include_state_scalars", False),
            allow_stop_at_step0=self.config.get("allow_stop_at_step0", False),
            stop_at_step0_min_init_affinity=self.config.get("stop_at_step0_min_init_affinity"),
            pmhc_obs_transform=pmhc_obs_transform,
        )
        print(
            f"  VecEnv: {self.n_envs} envs, obs_dim={self.vec_env.obs_dim}, "
            f"max_steps={self.config.get('max_steps', self.config.get('max_steps_per_episode', 8))}, "
            f"ban_stop={self.config.get('ban_stop', False)}, "
            f"sub_only={self.config.get('sub_only', False)}, "
            f"terminal_reward_only={self.config.get('terminal_reward_only', False)}, "
            f"active_clipping={self.active_clipping}, "
            f"use_biochem_features={self.config.get('use_biochem_features', False)}, "
            f"include_state_scalars={self.config.get('include_state_scalars', False)}, "
            f"pmhc_transform={self.config.get('pmhc_embedding_transform', self.config.get('peptide_centering', 'none'))}"
        )

        # Warmup: pre-cache all pMHC embeddings so encode_pmhc is 0ms during training
        import time as _time
        _t0 = _time.time()
        for _pep in targets:
            _pmhc_str = pmhc_loader.get_pmhc_string(_pep)
            esm_cache.encode_pmhc(_pmhc_str)
        _elapsed = _time.time() - _t0
        print(f"  pMHC warmup: {len(targets)} targets cached in {_elapsed:.1f}s")

        # tFold correction: load tFold scorer + elite buffer
        if self.tfold_correction:
            from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer
            cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
            gpu_id = int(cuda_env.split(",")[0]) if cuda_env else 0
            tfold_cache_path = self.config.get("tfold_cache_path")
            self.tfold_scorer = AffinityTFoldScorer(
                device=self.device,
                gpu_id=gpu_id,
                cache_path=tfold_cache_path,
            )
            self.elite_buffer = EliteBuffer(
                max_size=self.elite_buffer_size,
                score_threshold=self.elite_score_threshold,
            )
            stats = self.tfold_scorer.cache_stats
            print(f"  tFold correction enabled (cache={stats['cache_size']}, "
                  f"interval={self.tfold_rescore_interval}, top_k={self.tfold_top_k}, "
                  f"alpha={self.tfold_correction_alpha})")
            print(f"  Elite buffer: max_size={self.elite_buffer_size}, "
                  f"threshold={self.elite_score_threshold}")

        # Policy
        policy_class = self.config.get("policy_class", "ActorCritic")
        if policy_class == "ActorCriticCrossAttn":
            from tcrppo_v2.policy_cross_attn import ActorCriticCrossAttn
            self.policy = ActorCriticCrossAttn(
                obs_dim=self.vec_env.obs_dim,
                hidden_dim=self.config.get("hidden_dim", 512),
                max_tcr_len=self.max_tcr_len,
                use_cross_attn=self.config.get("use_cross_attn", True),
                n_attn_heads=self.config.get("n_attn_heads", 4),
            ).to(self.device)
            print(f"  Policy class: ActorCriticCrossAttn (cross_attn={self.config.get('use_cross_attn', True)}, heads={self.config.get('n_attn_heads', 4)})")
        else:
            self.policy = ActorCritic(
                obs_dim=self.vec_env.obs_dim,
                hidden_dim=self.config.get("hidden_dim", 512),
                max_tcr_len=self.max_tcr_len,
            ).to(self.device)
            print(f"  Policy class: ActorCritic (default)")
        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"  Policy: {n_params:,} parameters")

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

        # Buffer
        self.buffer = RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            obs_dim=self.vec_env.obs_dim,
            max_tcr_len=self.max_tcr_len,
            device=self.device,
        )

        # TensorBoard - DISABLED due to TensorFlow/PyTorch GPU conflict
        # TensorBoard internally uses TensorFlow which deadlocks with PyTorch on GPU
        self.logger = None
        # try:
        #     from torch.utils.tensorboard import SummaryWriter
        #     log_dir = os.path.join(self.output_dir, self.run_name, "tb_logs")
        #     os.makedirs(log_dir, exist_ok=True)
        #     self.logger = SummaryWriter(log_dir)
        # except ImportError:
        #     self.logger = None

        # Checkpoint dir
        self.ckpt_dir = os.path.join(self.output_dir, self.run_name, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _save_experiment_json(self) -> None:
        """Save experiment config to output/<run_name>/experiment.json at launch."""
        import subprocess
        git_hash = "unknown"
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            pass

        from datetime import datetime
        archive = {
            "name": self.run_name,
            "status": "training",
            "launched_at": datetime.now().isoformat(),
            "git_commit": git_hash,
            "config": {
                "seed": self.seed,
                "reward_mode": self.reward_mode,
                "total_timesteps": self.total_timesteps,
                "n_envs": self.n_envs,
                "learning_rate": self.lr,
                "hidden_dim": self.config.get("hidden_dim", 512),
                "max_steps": self.config.get("max_steps", 8),
                "affinity_scorer": self.config.get("affinity_model", "ergo"),
                "weights": {
                    "affinity": self.config.get("w_affinity", 1.0),
                    "decoy": self.config.get("w_decoy", 0.8),
                    "naturalness": self.config.get("w_naturalness", 0.5),
                    "diversity": self.config.get("w_diversity", 0.2),
                },
                "entropy_coef": self.entropy_coef,
                "tfold_use_cache": self.tfold_use_cache,
                "tfold_cache_read_only": self.config.get("tfold_cache_read_only", False),
                "use_znorm": False,  # z-norm removed in bugfix
                "tfold_correction": self.tfold_correction,
                "active_clipping": self.active_clipping,
                "n_contrast_decoys": self.config.get("n_contrast_decoys", 4),
                "decoy_K": self.config.get("decoy_K", 32),
                "pmhc_embedding_transform": self.config.get(
                    "pmhc_embedding_transform",
                    self.config.get("peptide_centering", "none"),
                ),
                "pmhc_center_path": self.config.get("pmhc_center_path"),
            },
            "notes": "",
        }
        if self.tfold_correction:
            archive["config"]["tfold_rescore_interval"] = self.tfold_rescore_interval
            archive["config"]["tfold_top_k"] = self.tfold_top_k
            archive["config"]["tfold_correction_alpha"] = self.tfold_correction_alpha
            archive["config"]["elite_buffer_size"] = self.elite_buffer_size
            archive["config"]["elite_score_threshold"] = self.elite_score_threshold
        if self.config.get("min_steps", 0) > 0:
            archive["config"]["min_steps"] = self.config["min_steps"]
            archive["config"]["min_steps_penalty"] = self.config.get("min_steps_penalty", 0.0)
        if self.config.get("ban_stop", False):
            archive["config"]["ban_stop"] = True
        if self.config.get("sub_only", False):
            archive["config"]["sub_only"] = True
        if self.config.get("terminal_reward_only", False):
            archive["config"]["terminal_reward_only"] = True
        if self.config.get("use_biochem_features", False):
            archive["config"]["use_biochem_features"] = True
        reward_shape_keys = (
            "delta_amp_thresholds",
            "delta_amp_slopes",
            "delta_negative_scale",
            "delta_deadband",
            "delta_deadband_penalty",
            "delta_amp_clip",
            "naturalness_gate",
            "affinity_guard_logit",
            "affinity_guard_tolerance",
            "affinity_guard_weight",
            "specificity_margin",
            "decoy_drop_weight",
            "target_gate_temperature",
            "decoy_affinity_ceiling",
            "target_surplus_cap",
            "soft_gate_affinity",
            "soft_gate_temperature",
            "soft_decoy_min_gate",
            "decoy_topk",
            "w_decoy_mean",
            "decoy_margin_clip",
        )
        for key in reward_shape_keys:
            if key in self.config:
                archive["config"][key] = self.config[key]
        if self.config.get("reward_schedule"):
            archive["config"]["reward_schedule"] = self.config["reward_schedule"]
        if self.config.get("train_targets"):
            archive["config"]["train_targets"] = self.config["train_targets"]
        if self.config.get("decoy_unlock_schedule"):
            archive["config"]["decoy_unlock_schedule"] = self.config["decoy_unlock_schedule"]

        out_dir = os.path.join(self.output_dir, self.run_name)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "experiment.json")
        with open(out_file, "w") as f:
            json.dump(archive, f, indent=2)
        print(f"  Saved experiment.json to {out_file}")

    def _score_active_clip_affinities(
        self,
        tcrs: List[str],
        peptides: List[str],
    ) -> List[float]:
        """Score candidate episode states for active clipping."""
        scorer = self.reward_manager.affinity_scorer
        if scorer is None or not tcrs:
            return []
        if hasattr(scorer, "score_batch_fast"):
            scores = scorer.score_batch_fast(tcrs, peptides)
        elif hasattr(scorer, "score_batch"):
            scores, _ = scorer.score_batch(tcrs, peptides)
        else:
            scores = [scorer.score(tcr, pep)[0] for tcr, pep in zip(tcrs, peptides)]
        return [
            self.reward_manager._finite_scalar(score, 0.0, "active_clip_affinity")
            for score in scores
        ]

    def _apply_active_clipping(
        self,
        env_idx: int,
        records: List[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        """Clip a completed episode to the post-edit state with best affinity."""
        if not records:
            return None

        tcrs = [str(record["tcr"]) for record in records]
        peptides = [str(record["peptide"]) for record in records]
        affinities = self._score_active_clip_affinities(tcrs, peptides)
        if not affinities:
            return None

        best_idx = int(np.argmax(np.asarray(affinities, dtype=np.float32)))
        best_record = records[best_idx]
        env = self.vec_env.envs[env_idx]

        reward, components = self.reward_manager.compute_reward(
            tcr=tcrs[best_idx],
            peptide=str(best_record["peptide"]),
            initial_affinity=env.initial_affinity,
            initial_tcr=env.initial_tcr,
            target=env.target,
        )
        components = dict(components)
        components["active_clipping"] = 1.0
        components["active_clip_best_step"] = float(best_idx + 1)
        components["active_clip_original_len"] = float(len(records))
        components["active_clip_best_affinity"] = float(affinities[best_idx])
        components["active_clip_final_affinity"] = float(affinities[-1])
        components["active_clip_affinity_gain_vs_final"] = float(
            affinities[best_idx] - affinities[-1]
        )

        self.buffer.clip_episode_to_best_step(
            env_idx=env_idx,
            start_row=int(records[0]["buffer_row"]),
            end_row=int(records[-1]["buffer_row"]),
            keep_row=int(best_record["buffer_row"]),
            reward=reward,
        )

        return {
            "reward": float(reward),
            "components": components,
            "kept_len": best_idx + 1,
            "original_len": len(records),
        }

    def _maybe_add_online_tcr_seed(
        self,
        *,
        target: str,
        tcr: str,
        affinity: float,
        delta_affinity: float,
        decoy_violation: float,
        global_step: int,
    ) -> bool:
        """Store a high-affinity final TCR in the per-target online seed pool."""
        if not self.config.get("online_tcr_pool_enabled", False):
            return False
        if self.tcr_pool is None:
            return False

        min_affinity = float(self.config.get("online_tcr_pool_min_affinity", -1.0))
        max_affinity = float(self.config.get("online_tcr_pool_max_affinity", 1e9))
        max_decoy_violation = float(self.config.get("online_tcr_pool_max_decoy_violation", 1e9))
        
        # CRITICAL FIX: If using dynamic bands, enforce minimum threshold of -4.0
        # to prevent pool pollution with low-quality TCRs (affinity < -4).
        # This ensures only TCRs within the band range [-4, 0.6] can enter the pool.
        if self.config.get("online_tcr_pool_use_dynamic_bands", False):
            min_affinity = max(min_affinity, -4.0)
        
        if affinity < min_affinity or affinity >= max_affinity or decoy_violation > max_decoy_violation:
            return False

        added = self.tcr_pool.add_online_tcr(
            target=target,
            tcr=tcr,
            affinity=affinity,
            delta_affinity=delta_affinity,
            decoy_violation=decoy_violation,
        )
        if added and self.online_tcr_pool_events_path:
            event = {
                "step": int(global_step),
                "target": target,
                "tcr": tcr,
                "affinity": float(affinity),
                "delta_affinity": float(delta_affinity),
                "decoy_violation": float(decoy_violation),
            }
            with open(self.online_tcr_pool_events_path, "a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        return added

    def _write_online_tcr_pool_snapshot(self, global_step: int) -> None:
        """Persist the current in-memory online TCR pool for local inspection."""
        if not self.online_tcr_pool_snapshot_path or self.tcr_pool is None:
            return
        if not hasattr(self.tcr_pool, "write_online_pool_snapshot"):
            return
        try:
            self.tcr_pool.write_online_pool_snapshot(
                self.online_tcr_pool_snapshot_path,
                step=global_step,
            )
        except Exception as exc:
            print(f"Warning: failed to write online TCR pool snapshot: {exc}", flush=True)

    def train(self) -> None:
        """Main training loop."""
        print(f"\nStarting training for {self.total_timesteps:,} timesteps...")
        self.setup()
        self._save_experiment_json()

        # Handle two-phase training (resume from checkpoint)
        resume_step = 0
        if getattr(self, '_resume_from', None):
            print(f"Resuming from checkpoint: {self._resume_from}")
            resume_step = self.load_checkpoint(self._resume_from)
            print(f"  Resumed at step {resume_step:,}")

            if getattr(self, '_resume_change_reward_mode', None):
                print(f"  Changing reward mode to: {self._resume_change_reward_mode}")
                self.reward_manager.reward_mode = self._resume_change_reward_mode

            if getattr(self, '_resume_reset_optimizer', False):
                print("  Resetting optimizer")
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

        global_step = resume_step
        self.vec_env.set_global_step(global_step)
        obs = self.vec_env.reset()
        n_updates = 0
        episode_rewards = []
        episode_lengths = []
        episode_components = []
        ep_reward_buf = np.zeros(self.n_envs)
        ep_length_buf = np.zeros(self.n_envs, dtype=int)
        next_milestone_idx = 0
        # Skip past milestones already reached (important for resume)
        while next_milestone_idx < len(self.milestones) and global_step >= self.milestones[next_milestone_idx]:
            next_milestone_idx += 1
        next_latest_ckpt_step = None
        if self.latest_checkpoint_interval and self.latest_checkpoint_interval > 0:
            next_latest_ckpt_step = (
                (global_step // self.latest_checkpoint_interval) + 1
            ) * self.latest_checkpoint_interval

        # Per-env trackers for elite buffer (last obs/action before STOP)
        ep_last_obs = [None] * self.n_envs
        ep_last_action = [None] * self.n_envs  # (op, pos, tok, logprob)
        ep_active_clip_records = [[] for _ in range(self.n_envs)]

        while global_step < self.total_timesteps:
            # Update decoy tier unlock schedule
            self._update_decoy_schedule(global_step)
            self._update_reward_schedule(global_step)
            self._update_gate_schedule(global_step)

            # Collect rollout
            self.buffer.reset()
            if self.active_clipping:
                ep_active_clip_records = [[] for _ in range(self.n_envs)]
            self.policy.eval()

            for step in range(self.n_steps):
                self.vec_env.set_global_step(global_step)

                reset_indices, reset_obs = self.vec_env.reset_done()
                if reset_indices:
                    for j, i in enumerate(reset_indices):
                        obs[i] = reset_obs[j]

                # Get action masks
                masks = self.vec_env.get_action_masks()
                op_masks = np.stack([m["op_mask"] for m in masks])
                pos_masks = np.stack([m["pos_mask"] for m in masks])
                token_masks = np.stack([m["token_mask"] for m in masks])

                # Get actions from policy
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                mask_dict = {
                    "op_mask": torch.BoolTensor(op_masks).to(self.device),
                    "pos_mask": torch.BoolTensor(pos_masks).to(self.device),
                    "token_mask": torch.BoolTensor(token_masks).to(self.device),
                }

                with torch.no_grad():
                    ops, positions, tokens, values = self.policy(obs_tensor, mask_dict)

                ops_np = ops.cpu().numpy()
                pos_np = positions.cpu().numpy()
                tok_np = tokens.cpu().numpy()
                val_np = values.cpu().numpy()

                # Compute log-probs for the sampled actions
                with torch.no_grad():
                    log_probs, _, _, _ = self.policy(
                        obs_tensor, mask_dict,
                        actions=(ops, positions, tokens),
                    )
                    lp_np = log_probs.cpu().numpy()

                # Track which envs are already done BEFORE stepping
                # (these will auto-reset and produce phantom transitions)
                was_done = np.array([env.done for env in self.vec_env.envs], dtype=bool)

                # Step environments
                actions = [(int(ops_np[i]), int(pos_np[i]), int(tok_np[i])) for i in range(self.n_envs)]
                next_obs, rewards, dones, infos = self.vec_env.step(actions)

                # Fix phantom transitions: envs that were done before stepping
                # got auto-reset with reward=0, done=False. Mark them done=True
                # so GAE correctly cuts the bootstrap at episode boundaries.
                effective_dones = dones | was_done

                # Store in buffer (with corrected dones)
                self.buffer.add(obs, ops_np, pos_np, tok_np, lp_np,
                               rewards, effective_dones.astype(np.float32),
                               val_np, op_masks, pos_masks, token_masks)
                buffer_row = self.buffer.ptr - 1

                # Track episode stats (skip phantom transitions)
                for i in range(self.n_envs):
                    if not was_done[i]:
                        ep_reward_buf[i] += rewards[i]
                        ep_length_buf[i] += 1
                        # Track last obs/action for elite buffer
                        ep_last_obs[i] = obs[i].copy()
                        ep_last_action[i] = (int(ops_np[i]), int(pos_np[i]), int(tok_np[i]), float(lp_np[i]))
                        if self.active_clipping:
                            ep_active_clip_records[i].append({
                                "buffer_row": buffer_row,
                                "tcr": infos[i].get("new_tcr", self.vec_env.envs[i].current_tcr),
                                "peptide": self.vec_env.envs[i].peptide,
                            })
                    if dones[i] and not was_done[i]:
                        terminal_components = infos[i].get("reward_components", {}) or {}
                        active_clip_result = None
                        if (
                            self.active_clipping
                            and self.config.get("terminal_reward_only", False)
                            and ep_active_clip_records[i]
                        ):
                            active_clip_result = self._apply_active_clipping(
                                i, ep_active_clip_records[i]
                            )
                            if active_clip_result is not None:
                                ep_reward_buf[i] = float(active_clip_result["reward"])
                                terminal_components = active_clip_result["components"]
                                infos[i]["reward_components"] = terminal_components

                        effective_ep_len = ep_length_buf[i]
                        if active_clip_result is not None:
                            effective_ep_len = int(active_clip_result["kept_len"])
                        episode_rewards.append(ep_reward_buf[i])
                        episode_lengths.append(effective_ep_len)
                        episode_components.append(terminal_components)
                        aff_raw = float(terminal_components.get("affinity_raw", 0.0))
                        init_aff = float(terminal_components.get("initial_affinity", 0.0))
                        delta_aff = float(terminal_components.get("affinity_step_delta", aff_raw - init_aff))
                        nat_raw = float(terminal_components.get("naturalness_raw", 0.0))
                        div_raw = float(terminal_components.get("diversity_raw", 0.0))
                        dec_delta_raw = terminal_components.get("decoy_delta")
                        decoy_str = ""
                        if dec_delta_raw is not None:
                            decoy_str = f" DecDelta={float(dec_delta_raw):.4f}"
                        if "decoy_affinity_violation" in terminal_components:
                            decoy_str += (
                                f" DecViol={float(terminal_components.get('decoy_affinity_violation', 0.0)):.4f}"
                                f" DecA={float(terminal_components.get('decoy_affinity_for_penalty', 0.0)):.4f}"
                            )
                        if "target_affinity_shortfall" in terminal_components:
                            decoy_str += (
                                f" TargetShort={float(terminal_components.get('target_affinity_shortfall', 0.0)):.4f}"
                                f" TargetSat={float(terminal_components.get('target_affinity_satisfied', 0.0)):.4f}"
                            )
                        if "improve_low_init_penalty" in terminal_components:
                            decoy_str += (
                                f" ImprovePen={float(terminal_components.get('improve_low_init_penalty', 0.0)):.4f}"
                            )
                        if "preserve_high_init_penalty" in terminal_components:
                            decoy_str += (
                                f" PreservePen={float(terminal_components.get('preserve_high_init_penalty', 0.0)):.4f}"
                            )
                        clip_str = ""
                        if active_clip_result is not None:
                            best_aff = float(terminal_components.get("active_clip_best_affinity", aff_raw))
                            final_aff = float(terminal_components.get("active_clip_final_affinity", aff_raw))
                            clip_str = (
                                f" Clip={effective_ep_len}/{ep_length_buf[i]}"
                                f" BestA={best_aff:.4f} FinalA={final_aff:.4f}"
                            )
                        env_i = self.vec_env.envs[i]
                        final_tcr_for_pool = env_i.current_tcr
                        peptide_for_pool = env_i.peptide
                        online_added = self._maybe_add_online_tcr_seed(
                            target=peptide_for_pool,
                            tcr=final_tcr_for_pool,
                            affinity=aff_raw,
                            delta_affinity=delta_aff,
                            decoy_violation=float(terminal_components.get("decoy_affinity_violation", 0.0)),
                            global_step=global_step,
                        )
                        # Record affinity for dynamic band selection (trace61)
                        if self.tcr_pool and hasattr(self.tcr_pool, 'record_episode_affinity'):
                            self.tcr_pool.record_episode_affinity(peptide_for_pool, aff_raw)
                        online_str = " OnlinePool=add" if online_added else ""
                        init_source = getattr(env_i, "initial_tcr_source", "")
                        init_source_str = f" InitSrc={init_source}" if init_source else ""
                        # Print episode completion immediately
                        print(
                            f"Episode {len(episode_rewards)} | Step {global_step} | "
                            f"R={ep_reward_buf[i]:.3f} | Len={effective_ep_len} | "
                            f"A={aff_raw:.4f} InitA={init_aff:.4f} DeltaA={delta_aff:.4f} "
                            f"Nat={nat_raw:.4f} Div={div_raw:.4f}",
                            f"{decoy_str}{clip_str}{init_source_str}{online_str}",
                            flush=True,
                        )

                        # Log best TCR if criteria met
                        if self.best_tcr_log_path is not None:
                            decoy_viol = float(terminal_components.get("decoy_affinity_violation", 0.0))
                            if (aff_raw >= self.best_tcr_min_affinity and
                                decoy_viol <= self.best_tcr_max_decoy_violation):
                                with open(self.best_tcr_log_path, "a") as f:
                                    f.write(
                                        f"{global_step}\t{final_tcr_for_pool}\t"
                                        f"{aff_raw:.4f}\t{decoy_viol:.4f}\t{delta_aff:.4f}\t"
                                        f"{peptide_for_pool}\n"
                                    )
                                print(f"  ★ Best TCR logged: A={aff_raw:.4f}, DecViol={decoy_viol:.4f}", flush=True)
                        # Submit to elite buffer if tFold correction enabled
                        if self.elite_buffer is not None and ep_last_obs[i] is not None:
                            final_tcr = env_i.current_tcr
                            peptide = env_i.peptide
                            # Use the affinity component from the terminal reward
                            ergo_score = ep_reward_buf[i] / max(ep_length_buf[i], 1)
                            # Better: get the raw affinity score for the final TCR
                            scorer = self.reward_manager.affinity_scorer
                            if scorer is not None and hasattr(scorer, 'score_batch_fast'):
                                ergo_score = scorer.score_batch_fast([final_tcr], [peptide])[0]
                            op, pos, tok, lp = ep_last_action[i]
                            self.elite_buffer.add_episode(
                                tcr=final_tcr,
                                peptide=peptide,
                                ergo_score=ergo_score,
                                last_obs=ep_last_obs[i],
                                last_op=op,
                                last_pos=pos,
                                last_tok=tok,
                                last_logprob=lp,
                            )
                        ep_reward_buf[i] = 0.0
                        ep_length_buf[i] = 0
                        ep_last_obs[i] = None
                        ep_last_action[i] = None
                        ep_active_clip_records[i] = []
                    elif was_done[i]:
                        # Auto-reset happened, start fresh counter
                        ep_reward_buf[i] = 0.0
                        ep_length_buf[i] = 0
                        ep_last_obs[i] = None
                        ep_last_action[i] = None
                        ep_active_clip_records[i] = []

                obs = next_obs
                global_step += self.n_envs

            # Compute GAE
            with torch.no_grad():
                last_values = self.policy.get_value(
                    torch.FloatTensor(obs).to(self.device)
                ).cpu().numpy()
            self.buffer.compute_gae(last_values, self.gamma, self.gae_lambda)

            # PPO update
            self.policy.train()
            total_pg_loss = 0
            total_vf_loss = 0
            total_entropy = 0
            total_approx_kl = 0
            total_clip_frac = 0
            total_ratio_mean = 0
            n_batches = 0
            early_stop_kl = False

            for epoch in range(self.n_epochs):
                for batch in self.buffer.get_batches(self.batch_size):
                    # Normalize advantages
                    adv = batch["advantages"]
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Evaluate current policy on stored data
                    log_probs, entropy, values, _ = self.policy(
                        batch["obs"],
                        action_masks={
                            "op_mask": batch["op_masks"],
                            "pos_mask": batch["pos_masks"],
                            "token_mask": batch["token_masks"],
                        },
                        actions=(batch["ops"], batch["positions"], batch["tokens"]),
                    )

                    # PPO clipped objective
                    ratio = torch.exp(log_probs - batch["old_log_probs"])
                    with torch.no_grad():
                        approx_kl = (batch["old_log_probs"] - log_probs).mean()
                        clip_frac = (
                            (ratio - 1.0).abs() > self.clip_range
                        ).float().mean()

                    if (
                        self.target_kl is not None
                        and approx_kl.item() > 1.5 * float(self.target_kl)
                    ):
                        early_stop_kl = True
                        break

                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    vf_loss = F.mse_loss(values, batch["returns"])

                    # Entropy bonus (with optional decay)
                    entropy_loss = -entropy.mean()
                    current_ent_coef = self._get_entropy_coef(global_step)

                    # Total loss
                    loss = pg_loss + self.vf_coef * vf_loss + current_ent_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    total_pg_loss += pg_loss.item()
                    total_vf_loss += vf_loss.item()
                    total_entropy += (-entropy_loss).item()
                    total_approx_kl += approx_kl.item()
                    total_clip_frac += clip_frac.item()
                    total_ratio_mean += ratio.mean().item()
                    n_batches += 1
                if early_stop_kl:
                    break

            n_updates += 1

            # tFold correction: re-score elite TCRs and run correction gradient
            if (self.tfold_correction and self.elite_buffer is not None
                    and n_updates % self.tfold_rescore_interval == 0
                    and len(self.elite_buffer) >= self.tfold_top_k):
                self._run_tfold_correction(global_step)

            # Logging
            if n_batches > 0:
                avg_pg = total_pg_loss / n_batches
                avg_vf = total_vf_loss / n_batches
                avg_ent = total_entropy / n_batches
                avg_kl = total_approx_kl / n_batches
                avg_clip_frac = total_clip_frac / n_batches
                avg_ratio_mean = total_ratio_mean / n_batches

            if n_updates % 1 == 0 and episode_rewards:
                recent = episode_rewards[-100:]
                mean_r = np.mean(recent)
                mean_l = np.mean(episode_lengths[-100:])
                recent_components = episode_components[-100:]
                comp_stats_str = ""
                if recent_components:
                    mean_aff = np.mean([float(c.get("affinity_raw", 0.0)) for c in recent_components])
                    mean_init_aff = np.mean([float(c.get("initial_affinity", 0.0)) for c in recent_components])
                    mean_delta_aff = np.mean([float(c.get("affinity_step_delta", 0.0)) for c in recent_components])
                    mean_nat = np.mean([float(c.get("naturalness_raw", 0.0)) for c in recent_components])
                    mean_div = np.mean([float(c.get("diversity_raw", 0.0)) for c in recent_components])
                    comp_stats_str = (
                        f" | A: {mean_aff:.4f}"
                        f" | InitA: {mean_init_aff:.4f}"
                        f" | DeltaA: {mean_delta_aff:.4f}"
                        f" | Nat: {mean_nat:.4f}"
                        f" | Div: {mean_div:.4f}"
                    )
                    if any("decoy_delta" in c for c in recent_components):
                        mean_decoy_delta = np.mean([
                            float(c.get("decoy_delta", 0.0)) for c in recent_components
                        ])
                        mean_decoy_n = np.mean([
                            float(c.get("decoy_n", 0.0)) for c in recent_components
                        ])
                        comp_stats_str += f" | DecDelta: {mean_decoy_delta:.4f} | DecN: {mean_decoy_n:.1f}"
                    if any("decoy_affinity_violation" in c for c in recent_components):
                        mean_decoy_violation = np.mean([
                            float(c.get("decoy_affinity_violation", 0.0)) for c in recent_components
                        ])
                        mean_decoy_affinity = np.mean([
                            float(c.get("decoy_affinity_for_penalty", 0.0)) for c in recent_components
                        ])
                        mean_target_shortfall = np.mean([
                            float(c.get("target_affinity_shortfall", 0.0)) for c in recent_components
                        ])
                        comp_stats_str += (
                            f" | DecViol: {mean_decoy_violation:.4f}"
                            f" | DecA: {mean_decoy_affinity:.4f}"
                            f" | TargetShort: {mean_target_shortfall:.4f}"
                        )

                # Get OOD stats if in OOD penalty mode
                ood_stats_str = ""
                if self.reward_mode == "v1_ergo_ood_penalty":
                    ood_stats = self.reward_manager.get_ood_stats()
                    ood_rate = ood_stats["ood_trigger_rate"]
                    ood_stats_str = f" | OOD: {ood_rate:>5.1%}"

                # Get tFold cache stats if using tFold scorer
                cache_stats_str = ""
                if hasattr(self.reward_manager.affinity_scorer, 'cache_stats'):
                    cs = self.reward_manager.affinity_scorer.cache_stats
                    total_cache = cs['cache_hits'] + cs['cache_misses']
                    if total_cache > 0:
                        hit_rate = cs['cache_hits'] / total_cache
                        cache_stats_str = f" | Cache: {hit_rate:.0%}({cs['cache_size']})"

                # Get hybrid scorer stats if using hybrid scorer
                hybrid_stats_str = ""
                if hasattr(self.reward_manager.affinity_scorer, 'get_stats'):
                    hs = self.reward_manager.affinity_scorer.get_stats()
                    if hs['total_calls'] > 0:
                        # Check if it's hybrid or cascade
                        if 'secondary_ratio_actual' in hs:
                            # Hybrid scorer
                            hybrid_stats_str = f" | Hybrid: {hs['secondary_ratio_actual']:.1%} tFold"
                        elif 'cascade_ratio' in hs:
                            # Cascade scorer
                            hybrid_stats_str = f" | Cascade: {hs['cascade_ratio']:.1%} tFold"

                online_pool_str = ""
                if self.config.get("online_tcr_pool_enabled", False) and self.tcr_pool is not None:
                    stats = self.tcr_pool.get_online_pool_stats()
                    total_online = sum(stats.values())
                    active_targets = sum(1 for size in stats.values() if size > 0)
                    ratio = self.tcr_pool.get_online_pool_ratio(global_step)
                    self._write_online_tcr_pool_snapshot(global_step)
                    online_pool_str = (
                        f" | OnlinePool: {total_online}"
                        f"/{active_targets}t"
                        f" p={ratio:.2f}"
                    )

                print(
                    f"Step {global_step:>10,} | "
                    f"Eps: {len(episode_rewards):>6} | "
                    f"R: {mean_r:>8.3f} | "
                    f"Len: {mean_l:>5.1f} | "
                    f"PG: {avg_pg:>8.4f} | "
                    f"VF: {avg_vf:>8.4f} | "
                    f"Ent: {avg_ent:>6.3f} | "
                    f"KL: {avg_kl:>7.5f} | "
                    f"Clip: {avg_clip_frac:>5.2f} | "
                    f"Ratio: {avg_ratio_mean:>5.2f}"
                    f"{comp_stats_str}"
                    f"{ood_stats_str}"
                    f"{cache_stats_str}"
                    f"{hybrid_stats_str}"
                    f"{online_pool_str}"
                )

                if self.logger:
                    self.logger.add_scalar("train/mean_reward", mean_r, global_step)
                    self.logger.add_scalar("train/mean_episode_length", mean_l, global_step)
                    self.logger.add_scalar("train/pg_loss", avg_pg, global_step)
                    self.logger.add_scalar("train/vf_loss", avg_vf, global_step)
                    self.logger.add_scalar("train/entropy", avg_ent, global_step)
                    self.logger.add_scalar("train/approx_kl", avg_kl, global_step)
                    self.logger.add_scalar("train/clip_fraction", avg_clip_frac, global_step)
                    self.logger.add_scalar("train/ratio_mean", avg_ratio_mean, global_step)

                    # Log OOD stats
                    if self.reward_mode == "v1_ergo_ood_penalty":
                        ood_stats = self.reward_manager.get_ood_stats()
                        self.logger.add_scalar("train/ood_trigger_rate", ood_stats["ood_trigger_rate"], global_step)
                        self.logger.add_scalar("train/ood_triggered", ood_stats["ood_triggered"], global_step)
                        # Reset counters after logging
                        self.reward_manager.reset_ood_stats()

                    # Log tFold cache stats
                    if hasattr(self.reward_manager.affinity_scorer, 'cache_stats'):
                        cache_stats = self.reward_manager.affinity_scorer.cache_stats
                        if cache_stats['cache_hits'] + cache_stats['cache_misses'] > 0:
                            hit_rate = cache_stats['cache_hits'] / (cache_stats['cache_hits'] + cache_stats['cache_misses'])
                            self.logger.add_scalar("train/tfold_cache_hit_rate", hit_rate, global_step)
                            self.logger.add_scalar("train/tfold_cache_size", cache_stats['cache_size'], global_step)

            # Checkpointing
            while next_milestone_idx < len(self.milestones) and global_step >= self.milestones[next_milestone_idx]:
                ms = self.milestones[next_milestone_idx]
                self.save_checkpoint(f"milestone_{ms}", global_step)
                print(f"  ** Milestone checkpoint saved: {ms:,} steps")
                next_milestone_idx += 1

            if next_latest_ckpt_step is not None and global_step >= next_latest_ckpt_step:
                self.save_checkpoint("latest", global_step)
                print(f"  ** Latest checkpoint refreshed: {global_step:,} steps")
                next_latest_ckpt_step = (
                    (global_step // self.latest_checkpoint_interval) + 1
                ) * self.latest_checkpoint_interval

        # Final checkpoint
        self.save_checkpoint("final", global_step)
        print(f"\nTraining complete: {global_step:,} steps, {len(episode_rewards)} episodes")

        if self.logger:
            self.logger.close()

    def _update_decoy_schedule(self, global_step: int) -> None:
        """Update decoy tier unlock based on training progress."""
        if self.decoy_scorer is None:
            return
        configured_schedule = self.config.get("decoy_unlock_schedule")
        tiers = None
        if configured_schedule:
            if isinstance(configured_schedule, dict):
                for threshold, scheduled_tiers in sorted(
                    configured_schedule.items(), key=lambda item: int(item[0])
                ):
                    if global_step >= int(threshold):
                        tiers = list(scheduled_tiers)
            elif isinstance(configured_schedule, list):
                for entry in configured_schedule:
                    threshold = int(entry.get("step", entry.get("until", 0)) or 0)
                    if global_step >= threshold:
                        tiers = list(entry.get("tiers", []))
        if tiers is None:
            if global_step < 2_000_000:
                tiers = ["A"]
            elif global_step < 5_000_000:
                tiers = ["A", "B"]
            elif global_step < 8_000_000:
                tiers = ["A", "B", "D"]
            else:
                tiers = ["A", "B", "D", "C"]
        self.decoy_scorer.set_unlocked_tiers(tiers)

        difficulty_schedule = self.config.get("decoy_difficulty_schedule")
        difficulty = self.config.get("decoy_difficulty", None)
        if difficulty_schedule:
            if isinstance(difficulty_schedule, dict):
                for threshold, scheduled_difficulty in sorted(
                    difficulty_schedule.items(), key=lambda item: int(item[0])
                ):
                    if global_step >= int(threshold):
                        difficulty = scheduled_difficulty
            elif isinstance(difficulty_schedule, list):
                for entry in difficulty_schedule:
                    threshold = int(entry.get("step", entry.get("until", 0)) or 0)
                    if global_step >= threshold:
                        difficulty = entry.get("difficulty", difficulty)
        if difficulty and hasattr(self.decoy_scorer, "set_decoy_difficulty"):
            self.decoy_scorer.set_decoy_difficulty(difficulty)

    def _update_reward_schedule(self, global_step: int) -> None:
        """Apply curriculum reward schedule transitions based on step count."""
        schedule = self.config.get("reward_schedule")
        if not schedule:
            return
        # Find the latest phase that should be active
        current_phase = None
        for phase in schedule:
            if global_step >= phase["step"]:
                current_phase = phase
        if current_phase is None:
            return
        phase_step = int(current_phase["step"])
        if getattr(self, "_last_reward_schedule_step", None) == phase_step:
            return
        new_mode = current_phase.get("mode")
        changed = False
        if new_mode and new_mode != self.reward_manager.reward_mode:
            print(f"\n[Schedule] Step {global_step:,}: reward mode "
                  f"{self.reward_manager.reward_mode} -> {new_mode}")
            self.reward_manager.reward_mode = new_mode
            changed = True
        if "w_nat" in current_phase:
            self.reward_manager.weights["naturalness"] = current_phase["w_nat"]
            changed = True
        if "w_affinity" in current_phase:
            self.reward_manager.weights["affinity"] = current_phase["w_affinity"]
            changed = True
        if "w_decoy" in current_phase:
            self.reward_manager.weights["decoy"] = current_phase["w_decoy"]
            changed = True
        if "w_decoy_mean" in current_phase:
            self.reward_manager.weights["decoy_mean"] = current_phase["w_decoy_mean"]
            changed = True
        if "w_diversity" in current_phase:
            self.reward_manager.weights["diversity"] = current_phase["w_diversity"]
            changed = True
        if "n_decoys" in current_phase:
            self.reward_manager.n_contrast_decoys = current_phase["n_decoys"]
            changed = True
        if "contrastive_agg" in current_phase:
            self.reward_manager.contrastive_agg = current_phase["contrastive_agg"]
            changed = True
        if "decoy_affinity_ceiling" in current_phase:
            self.reward_manager.decoy_affinity_ceiling = current_phase["decoy_affinity_ceiling"]
            changed = True
        if "affinity_guard_logit" in current_phase:
            self.reward_manager.affinity_guard_logit = current_phase["affinity_guard_logit"]
            changed = True
        if "target_surplus_cap" in current_phase:
            self.reward_manager.target_surplus_cap = current_phase["target_surplus_cap"]
            changed = True
        if "affinity_guard_weight" in current_phase:
            self.reward_manager.affinity_guard_weight = current_phase["affinity_guard_weight"]
            changed = True
        if "soft_gate_affinity" in current_phase:
            self.reward_manager.soft_gate_affinity = current_phase["soft_gate_affinity"]
            changed = True
        if "soft_gate_temperature" in current_phase:
            self.reward_manager.soft_gate_temperature = max(1e-6, current_phase["soft_gate_temperature"])
            changed = True
        if "lr" in current_phase:
            new_lr = current_phase["lr"]
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr
            print(f"  Learning rate -> {new_lr}")
            changed = True
        if changed:
            self._last_reward_schedule_step = phase_step
            print(f"\n[Schedule] Step {global_step:,}: "
                  f"mode={self.reward_manager.reward_mode}, "
                  f"aff={self.reward_manager.weights['affinity']}, "
                  f"nat={self.reward_manager.weights['naturalness']}, "
                  f"decoy={self.reward_manager.weights['decoy']}, "
                  f"decoy_mean={self.reward_manager.weights.get('decoy_mean', 0.0)}, "
                  f"n_decoys={self.reward_manager.n_contrast_decoys}, "
                  f"agg={self.reward_manager.contrastive_agg}, "
                  f"decoy_ceiling={getattr(self.reward_manager, 'decoy_affinity_ceiling', 'n/a')}")

    def _update_gate_schedule(self, global_step: int) -> None:
        """Apply curriculum gate schedule transitions based on step count."""
        gate_schedule = self.config.get("gate_schedule")
        if not gate_schedule:
            return
        
        # gate_schedule is a dict: {step: gate_value}
        # Find the latest gate that should be active
        active_steps = sorted([int(s) for s in gate_schedule.keys() if int(s) <= global_step])
        if not active_steps:
            return
        
        latest_step = active_steps[-1]
        new_gate = float(gate_schedule[latest_step])
        
        # Check if we already applied this gate
        if getattr(self, "_last_gate_schedule_step", None) == latest_step:
            return
        
        # Update the gate in reward_manager
        old_gate = getattr(self.reward_manager, 'target_decoy_gate_logit', 
                          getattr(self.reward_manager, 'target_affinity_gate', None))
        
        # Update both possible gate attributes (for compatibility)
        if hasattr(self.reward_manager, 'target_decoy_gate_logit'):
            self.reward_manager.target_decoy_gate_logit = new_gate
        if hasattr(self.reward_manager, 'target_affinity_gate'):
            self.reward_manager.target_affinity_gate = new_gate
        
        self._last_gate_schedule_step = latest_step
        print(f"\n[Gate Schedule] Step {global_step:,}: gate {old_gate} -> {new_gate}")

    def _get_entropy_coef(self, global_step: int) -> float:
        """Get entropy coefficient with optional linear decay."""
        if self.entropy_coef_final is None:
            return self.entropy_coef  # No decay
        if global_step < self.entropy_decay_start:
            return self.entropy_coef  # Before decay start
        # Linear decay from entropy_coef to entropy_coef_final
        # over remaining steps after decay_start
        remaining_frac = (global_step - self.entropy_decay_start) / max(
            1, self.total_timesteps - self.entropy_decay_start
        )
        remaining_frac = min(1.0, remaining_frac)
        return self.entropy_coef + (self.entropy_coef_final - self.entropy_coef) * remaining_frac

    def _run_tfold_correction(self, global_step: int) -> None:
        """Re-score top-K elite TCRs with tFold and run correction gradient steps."""
        import time as _time
        t0 = _time.time()

        top_k = self.elite_buffer.get_top_k(self.tfold_top_k)
        if not top_k:
            return

        # Extract data from elite entries
        ergo_scores = [e[0] for e in top_k]
        tcrs = [e[1] for e in top_k]
        peptides = [e[2] for e in top_k]
        obs_list = [e[3] for e in top_k]
        ops_list = [e[4] for e in top_k]
        pos_list = [e[5] for e in top_k]
        tok_list = [e[6] for e in top_k]
        old_lps = [e[7] for e in top_k]

        # Re-score with tFold
        tfold_scores = self.tfold_scorer.score_batch_fast(tcrs, peptides)

        # Compute correction advantages: alpha * (tfold - ergo)
        corrections = []
        for i in range(len(top_k)):
            diff = tfold_scores[i] - ergo_scores[i]
            corrections.append(self.tfold_correction_alpha * diff)

        # Build correction batch tensors
        batch_obs = torch.FloatTensor(np.stack(obs_list)).to(self.device)
        batch_ops = torch.LongTensor(ops_list).to(self.device)
        batch_pos = torch.LongTensor(pos_list).to(self.device)
        batch_tok = torch.LongTensor(tok_list).to(self.device)
        batch_old_lps = torch.FloatTensor(old_lps).to(self.device)
        batch_advantages = torch.FloatTensor(corrections).to(self.device)

        # Normalize correction advantages
        if len(batch_advantages) > 1:
            adv_std = batch_advantages.std()
            if adv_std > 1e-8:
                batch_advantages = batch_advantages / (adv_std + 1e-8)

        # Generate action masks for these obs (all ops allowed, all positions allowed)
        batch_op_masks = torch.ones(len(top_k), NUM_OPS, dtype=torch.bool, device=self.device)
        correction_max_tcr_len = int(getattr(self.policy, "max_tcr_len", self.max_tcr_len))
        batch_pos_masks = torch.ones(
            len(top_k), correction_max_tcr_len, dtype=torch.bool, device=self.device
        )

        # Run 2 correction gradient steps
        self.policy.train()
        correction_losses = []
        for _ in range(2):
            log_probs, entropy, values, _ = self.policy(
                batch_obs,
                action_masks={
                    "op_mask": batch_op_masks,
                    "pos_mask": batch_pos_masks,
                },
                actions=(batch_ops, batch_pos, batch_tok),
            )

            ratio = torch.exp(log_probs - batch_old_lps)
            pg_loss1 = -batch_advantages * ratio
            pg_loss2 = -batch_advantages * torch.clamp(
                ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            entropy_loss = -entropy.mean()
            loss = pg_loss + self._get_entropy_coef(global_step) * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            correction_losses.append(pg_loss.item())

        elapsed = _time.time() - t0

        # Log stats
        mean_ergo = np.mean(ergo_scores)
        mean_tfold = np.mean(tfold_scores)
        mean_correction = np.mean(corrections)
        n_positive = sum(1 for c in corrections if c > 0)
        n_negative = sum(1 for c in corrections if c < 0)

        print(
            f"  [tFold correction] K={len(top_k)} | "
            f"ERGO={mean_ergo:.3f} | tFold={mean_tfold:.3f} | "
            f"corr={mean_correction:+.3f} | "
            f"+{n_positive}/-{n_negative} | "
            f"loss={np.mean(correction_losses):.4f} | "
            f"elite_buf={len(self.elite_buffer)} | "
            f"{elapsed:.1f}s"
        )

        if self.logger:
            self.logger.add_scalar("tfold/mean_ergo_score", mean_ergo, global_step)
            self.logger.add_scalar("tfold/mean_tfold_score", mean_tfold, global_step)
            self.logger.add_scalar("tfold/mean_correction", mean_correction, global_step)
            self.logger.add_scalar("tfold/correction_loss", np.mean(correction_losses), global_step)
            self.logger.add_scalar("tfold/elite_buffer_size", len(self.elite_buffer), global_step)
            self.logger.add_scalar("tfold/n_positive", n_positive, global_step)
            self.logger.add_scalar("tfold/n_negative", n_negative, global_step)

    def save_checkpoint(self, name: str, global_step: int = 0) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.ckpt_dir, f"{name}.pt")
        tmp_path = f"{path}.tmp"
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "global_step": global_step,
        }, tmp_path)
        os.replace(tmp_path, path)

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns the step count stored in checkpoint, or from filename."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Prefer step count stored in checkpoint
        if "global_step" in ckpt:
            return int(ckpt["global_step"])

        # Fallback: extract step count from filename (e.g. milestone_1000000.pt)
        import re
        basename = os.path.basename(path)
        match = re.search(r'(\d+)', basename)
        if match and 'milestone' in basename:
            return int(match.group(1))
        return 0


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="TCRPPO v2 Training")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--run_name", default=None, help="Run name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--reward_mode", default=None, help="Reward mode override")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total timesteps")
    parser.add_argument("--n_envs", type=int, default=None, help="Num envs")
    parser.add_argument("--n_epochs", type=int, default=None, help="PPO epochs per rollout")
    parser.add_argument("--batch_size", type=int, default=None, help="PPO minibatch size")
    parser.add_argument("--clip_range", type=float, default=None, help="PPO clip range")
    parser.add_argument("--vf_coef", type=float, default=None, help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Gradient clipping norm")
    parser.add_argument("--target_kl", type=float, default=None, help="Stop PPO epochs early above this approximate KL")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--w_affinity", type=float, default=None, help="Affinity weight")
    parser.add_argument("--w_decoy", type=float, default=None, help="Decoy weight")
    parser.add_argument("--w_naturalness", type=float, default=None, help="Naturalness weight")
    parser.add_argument("--w_diversity", type=float, default=None, help="Diversity weight")
    parser.add_argument("--w_gap", type=float, default=None, help="Specificity gap weight (curriculum climbing)")
    parser.add_argument("--w_absolute_affinity", type=float, default=None, help="Absolute logit term weight for calibrated delta reward")
    parser.add_argument("--affinity_ref_logit", type=float, default=None, help="Reference logit for calibrated delta reward")
    parser.add_argument("--affinity_guard_logit", type=float, default=None, help="Absolute target logit floor for v2_target_guarded_decoy")
    parser.add_argument("--affinity_guard_tolerance", type=float, default=None, help="Allowed target logit drop from episode initial affinity")
    parser.add_argument("--affinity_guard_weight", type=float, default=None, help="Penalty weight for target affinity shortfall")
    parser.add_argument("--specificity_margin", type=float, default=None, help="Required target-vs-decoy logit margin")
    parser.add_argument("--decoy_drop_weight", type=float, default=None, help="Bonus weight for reducing decoy affinity when target is guarded")
    parser.add_argument("--target_gate_temperature", type=float, default=None, help="Smooth gate temperature for decoy-drop bonus")
    parser.add_argument("--decoy_affinity_ceiling", type=float, default=None, help="Absolute decoy non-binding logit ceiling")
    parser.add_argument("--target_surplus_cap", type=float, default=None, help="Cap for target satisfied reward above the affinity floor")
    parser.add_argument("--min_steps", type=int, default=None, help="Min steps before STOP")
    parser.add_argument("--min_steps_penalty", type=float, default=None, help="Penalty for early STOP")
    # Curriculum climbing parameters
    parser.add_argument("--curriculum_gates", type=float, nargs='+', default=None, help="Curriculum gates for climbing reward")
    parser.add_argument("--curriculum_bonuses", type=float, nargs='+', default=None, help="Curriculum bonuses for climbing reward")
    parser.add_argument("--gap_margin", type=float, default=None, help="Gap margin for curriculum climbing")
    parser.add_argument("--decoy_activation_threshold", type=float, default=None, help="Decoy activation threshold for curriculum climbing")
    parser.add_argument("--soft_gate_affinity", type=float, default=None, help="Affinity center for v2_soft_target_decoy_gap")
    parser.add_argument("--soft_gate_temperature", type=float, default=None, help="Soft gate temperature for v2_soft_target_decoy_gap")
    parser.add_argument("--soft_decoy_min_gate", type=float, default=None, help="Skip decoy scoring below this soft gate value")
    parser.add_argument("--decoy_topk", type=int, default=None, help="Top-k strongest decoys for v2_soft_target_decoy_gap")
    parser.add_argument("--w_decoy_mean", type=float, default=None, help="Mean-decoy margin weight for v2_soft_target_decoy_gap")
    parser.add_argument("--decoy_margin_clip", type=float, default=None, help="Absolute clip for target-decoy margins")
    # Two-phase training support
    parser.add_argument("--resume_from", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--resume_change_reward_mode", default=None, help="Change reward mode on resume")
    parser.add_argument("--resume_reset_optimizer", action="store_true", help="Reset optimizer on resume")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Policy hidden dim override")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate override")
    parser.add_argument("--entropy_coef", type=float, default=None, help="Entropy coefficient override")
    parser.add_argument("--affinity_scorer", default=None, help="Affinity scorer: ergo, nettcr, tcbind, tfold, tfold_cascade, hybrid, ensemble, ensemble_ergo_tcbind, ensemble_ergo_tfold")
    parser.add_argument("--cascade_threshold", type=float, default=None, help="ERGO uncertainty threshold for tFold cascade (default 0.15)")
    parser.add_argument("--cascade_tfold_weight", type=float, default=None, help="Cascade scorer: tFold weight in combination (default 0.7)")
    parser.add_argument("--cascade_ergo_weight", type=float, default=None, help="Cascade scorer: ERGO weight in combination (default 0.3)")
    parser.add_argument("--hybrid_tfold_ratio", type=float, default=None, help="Hybrid scorer: ratio of tFold calls (default 0.1 = 10 percent)")
    parser.add_argument("--encoder", default=None, choices=["esm2", "lightweight"], help="State encoder: esm2 (default) or lightweight (CPU-friendly BiLSTM)")
    parser.add_argument("--encoder_dim", type=int, default=None, help="Lightweight encoder output dim (default 256)")
    parser.add_argument("--esm_cache_path", type=str, default=None, help="Path to ESM cache DB (default: data/esm_cache.db)")
    parser.add_argument("--pmhc_embedding_transform", default=None,
                        choices=["none", "center", "mean_center", "center_only", "center_layernorm", "mean_center_layernorm"],
                        help="Observation-only pMHC embedding transform")
    parser.add_argument("--pmhc_center_path", type=str, default=None,
                        help="Path to fixed pMHC embedding center .pt file")
    parser.add_argument("--tfold_cache_path", type=str, default=None, help="Path to tFold feature cache DB")
    parser.add_argument("--tfold_server_socket", type=str, default=None, help="Path to tFold feature server Unix socket")
    parser.add_argument("--latest_checkpoint_interval", type=int, default=None, help="Refresh latest.pt every N steps")
    parser.add_argument("--disable_tfold_cache", action="store_true", help="Disable tFold feature cache reads and writes")
    # tFold correction
    parser.add_argument("--tfold_correction", action="store_true", help="Enable tFold elite re-scoring correction")
    parser.add_argument("--tfold_rescore_interval", type=int, default=None, help="Re-score elite TCRs every N rollouts (default 50)")
    parser.add_argument("--tfold_top_k", type=int, default=None, help="Top-K elite TCRs to re-score (default 32)")
    parser.add_argument("--tfold_correction_alpha", type=float, default=None, help="Correction advantage scale (default 2.0)")
    parser.add_argument("--elite_buffer_size", type=int, default=None, help="Max elite buffer size (default 500)")
    parser.add_argument("--elite_score_threshold", type=float, default=None, help="Min ERGO score for elite (default 0.7)")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per episode (default 8)")
    parser.add_argument("--ban_stop", action="store_true", help="Ban STOP action — agent must use all max_steps")
    parser.add_argument("--sub_only", action="store_true", help="Allow only substitution edits (plus STOP when permitted)")
    parser.add_argument("--terminal_reward_only", action="store_true", help="Only compute reward at episode end (for slow scorers like tFold)")
    parser.add_argument("--active_clipping", action="store_true", help="After a full terminal-only episode, train only on the best-affinity prefix")

    # Curriculum overrides
    parser.add_argument("--curriculum_l0", type=float, default=None, help="L0 curriculum ratio (known binder variants)")
    parser.add_argument("--curriculum_l1", type=float, default=None, help="L1 curriculum ratio (ERGO top-K)")
    parser.add_argument("--curriculum_l2", type=float, default=None, help="L2 curriculum ratio (random TCRdb)")

    # Contrastive reward
    parser.add_argument("--n_contrast_decoys", type=int, default=None, help="Number of decoys for contrastive_ergo reward")
    parser.add_argument("--contrastive_agg", default=None, choices=["mean", "max"], help="Aggregation for contrastive decoy scores: mean or max")
    parser.add_argument("--convex_alpha", type=float, default=None, help="Exponent for v1_ergo_convex reward mode")
    parser.add_argument("--entropy_coef_final", type=float, default=None, help="Final entropy coefficient (enables linear decay)")
    parser.add_argument("--entropy_decay_start", type=int, default=None, help="Step to begin entropy decay (default: 1M)")
    parser.add_argument("--decoy_library_path", type=str, default=None, help="Path to pMHC decoy library (default: /share/liuyutian/pMHC_decoy_library)")

    # OOD penalty (for v1_ergo_ood_penalty mode)
    parser.add_argument("--ood_threshold", type=float, default=None, help="Uncertainty threshold for OOD detection (ERGO MC Dropout std)")
    parser.add_argument("--ood_penalty_weight", type=float, default=None, help="Weight for OOD penalty")
    parser.add_argument("--ood_penalty_mode", default=None, choices=["soft", "hard"], help="OOD penalty mode: soft (penalize excess) or hard (penalize full uncertainty)")

    # Target peptide filtering
    parser.add_argument("--train_targets", default=None, help="Comma-separated peptides or path to txt file (one peptide per line). Training uses only these; eval still uses all 12 McPAS.")

    # Curriculum reward schedule
    parser.add_argument("--reward_schedule", default=None,
        help='JSON string defining reward phase schedule. '
             'Format: [{"step":0,"mode":"v1_ergo_only"}, '
             '{"step":500000,"mode":"raw_multi_penalty","w_nat":0.1,"w_decoy":0.02}, '
             '{"step":1500000,"mode":"contrastive_ergo","n_decoys":16}]')

    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.run_name:
        config["run_name"] = args.run_name
    if args.seed is not None:
        config["seed"] = args.seed
    if args.reward_mode:
        config["reward_mode"] = args.reward_mode
    if args.total_timesteps:
        config["total_timesteps"] = args.total_timesteps
    if args.n_envs:
        config["n_envs"] = args.n_envs
    if args.n_epochs is not None:
        config["n_epochs"] = args.n_epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.clip_range is not None:
        config["clip_range"] = args.clip_range
    if args.vf_coef is not None:
        config["vf_coef"] = args.vf_coef
    if args.max_grad_norm is not None:
        config["max_grad_norm"] = args.max_grad_norm
    if args.target_kl is not None:
        config["target_kl"] = args.target_kl
    if args.device is not None:
        config["device"] = args.device
    if args.w_affinity is not None:
        config["w_affinity"] = args.w_affinity
    if args.w_decoy is not None:
        config["w_decoy"] = args.w_decoy
    if args.w_naturalness is not None:
        config["w_naturalness"] = args.w_naturalness
    if args.w_diversity is not None:
        config["w_diversity"] = args.w_diversity
    if args.w_absolute_affinity is not None:
        config["w_absolute_affinity"] = args.w_absolute_affinity
    if args.affinity_ref_logit is not None:
        config["affinity_ref_logit"] = args.affinity_ref_logit
    if args.affinity_guard_logit is not None:
        config["affinity_guard_logit"] = args.affinity_guard_logit
    if args.affinity_guard_tolerance is not None:
        config["affinity_guard_tolerance"] = args.affinity_guard_tolerance
    if args.affinity_guard_weight is not None:
        config["affinity_guard_weight"] = args.affinity_guard_weight
    if args.specificity_margin is not None:
        config["specificity_margin"] = args.specificity_margin
    if args.decoy_drop_weight is not None:
        config["decoy_drop_weight"] = args.decoy_drop_weight
    if args.target_gate_temperature is not None:
        config["target_gate_temperature"] = args.target_gate_temperature
    if args.decoy_affinity_ceiling is not None:
        config["decoy_affinity_ceiling"] = args.decoy_affinity_ceiling
    if args.target_surplus_cap is not None:
        config["target_surplus_cap"] = args.target_surplus_cap
    if args.min_steps is not None:
        config["min_steps"] = args.min_steps
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.min_steps_penalty is not None:
        config["min_steps_penalty"] = args.min_steps_penalty
    # Curriculum climbing parameters
    if args.curriculum_gates is not None:
        config["curriculum_gates"] = args.curriculum_gates
    if args.curriculum_bonuses is not None:
        config["curriculum_bonuses"] = args.curriculum_bonuses
    if args.gap_margin is not None:
        config["gap_margin"] = args.gap_margin
    if args.decoy_activation_threshold is not None:
        config["decoy_activation_threshold"] = args.decoy_activation_threshold
    if args.soft_gate_affinity is not None:
        config["soft_gate_affinity"] = args.soft_gate_affinity
    if args.soft_gate_temperature is not None:
        config["soft_gate_temperature"] = args.soft_gate_temperature
    if args.soft_decoy_min_gate is not None:
        config["soft_decoy_min_gate"] = args.soft_decoy_min_gate
    if args.decoy_topk is not None:
        config["decoy_topk"] = args.decoy_topk
    if args.w_decoy_mean is not None:
        config["w_decoy_mean"] = args.w_decoy_mean
    if args.decoy_margin_clip is not None:
        config["decoy_margin_clip"] = args.decoy_margin_clip
    if args.hidden_dim is not None:
        config["hidden_dim"] = args.hidden_dim
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.entropy_coef is not None:
        config["entropy_coef"] = args.entropy_coef
    if args.affinity_scorer is not None:
        config["affinity_model"] = args.affinity_scorer
    if args.encoder is not None:
        config["encoder"] = args.encoder
    if args.encoder_dim is not None:
        config["encoder_dim"] = args.encoder_dim
    if args.esm_cache_path is not None:
        config["esm_cache_path"] = args.esm_cache_path
    if args.pmhc_embedding_transform is not None:
        config["pmhc_embedding_transform"] = args.pmhc_embedding_transform
    if args.pmhc_center_path is not None:
        config["pmhc_center_path"] = args.pmhc_center_path
    if args.tfold_cache_path is not None:
        config["tfold_cache_path"] = args.tfold_cache_path
    if args.tfold_server_socket is not None:
        config["tfold_server_socket"] = args.tfold_server_socket
    if args.latest_checkpoint_interval is not None:
        config["latest_checkpoint_interval"] = args.latest_checkpoint_interval
    if args.disable_tfold_cache:
        config["tfold_use_cache"] = False
    if args.tfold_correction:
        config["tfold_correction"] = True
    if args.tfold_rescore_interval is not None:
        config["tfold_rescore_interval"] = args.tfold_rescore_interval
    if args.tfold_top_k is not None:
        config["tfold_top_k"] = args.tfold_top_k
    if args.tfold_correction_alpha is not None:
        config["tfold_correction_alpha"] = args.tfold_correction_alpha
    if args.elite_buffer_size is not None:
        config["elite_buffer_size"] = args.elite_buffer_size
    if args.elite_score_threshold is not None:
        config["elite_score_threshold"] = args.elite_score_threshold
    if args.cascade_threshold is not None:
        config["cascade_threshold"] = args.cascade_threshold
    if args.cascade_tfold_weight is not None:
        config["cascade_tfold_weight"] = args.cascade_tfold_weight
    if args.cascade_ergo_weight is not None:
        config["cascade_ergo_weight"] = args.cascade_ergo_weight
    if args.hybrid_tfold_ratio is not None:
        config["hybrid_tfold_ratio"] = args.hybrid_tfold_ratio
    if args.ban_stop:
        config["ban_stop"] = True
    if args.sub_only:
        config["sub_only"] = True
    if args.terminal_reward_only:
        config["terminal_reward_only"] = True
    if args.active_clipping:
        config["active_clipping"] = True
    if args.decoy_library_path is not None:
        config["decoy_library_path"] = args.decoy_library_path

    # Curriculum overrides
    if args.curriculum_l0 is not None or args.curriculum_l1 is not None or args.curriculum_l2 is not None:
        l0 = args.curriculum_l0 if args.curriculum_l0 is not None else 0.0
        l1 = args.curriculum_l1 if args.curriculum_l1 is not None else 0.0
        l2 = args.curriculum_l2 if args.curriculum_l2 is not None else 1.0
        # Override curriculum schedule with single static entry
        config["curriculum_schedule"] = [{"until": None, "L0": l0, "L1": l1, "L2": l2}]

    # Contrastive reward config
    if args.n_contrast_decoys is not None:
        config["n_contrast_decoys"] = args.n_contrast_decoys
    if args.contrastive_agg:
        config["contrastive_agg"] = args.contrastive_agg
    if args.convex_alpha is not None:
        config["convex_alpha"] = args.convex_alpha
    if args.entropy_coef_final is not None:
        config["entropy_coef_final"] = args.entropy_coef_final
    if args.entropy_decay_start is not None:
        config["entropy_decay_start"] = args.entropy_decay_start
    if args.train_targets is not None:
        config["train_targets"] = args.train_targets
    if args.reward_schedule is not None:
        config["reward_schedule"] = json.loads(args.reward_schedule)

    # OOD penalty config
    if args.ood_threshold is not None:
        config["ood_threshold"] = args.ood_threshold
    if args.ood_penalty_weight is not None:
        config["ood_penalty_weight"] = args.ood_penalty_weight
    if args.ood_penalty_mode:
        config["ood_penalty_mode"] = args.ood_penalty_mode

    config.setdefault("run_name", "v2_run")

    trainer = PPOTrainer(config)

    # Two-phase training: store resume args for processing after setup()
    trainer._resume_from = args.resume_from
    trainer._resume_change_reward_mode = args.resume_change_reward_mode
    trainer._resume_reset_optimizer = args.resume_reset_optimizer

    try:
        trainer.train()
    except Exception as e:
        print(f"\nFATAL ERROR during training:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
