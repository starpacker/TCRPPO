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

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, device: str = "cpu"):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
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

        # Action masks
        self.op_masks = np.zeros((n_steps, n_envs, NUM_OPS), dtype=bool)
        self.pos_masks = np.zeros((n_steps, n_envs, MAX_TCR_LEN), dtype=bool)

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
    ) -> None:
        """Add one step of data."""
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
        self.ptr += 1

    def compute_gae(self, last_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and returns."""
        last_gae = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_value
            else:
                next_values = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int) -> List[dict]:
        """Flatten and yield minibatches."""
        total = self.n_steps * self.n_envs
        indices = np.arange(total)
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
        flat_pos_masks = self.pos_masks.reshape(total, MAX_TCR_LEN)

        for start in range(0, total, batch_size):
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
            }

    def reset(self) -> None:
        """Reset buffer pointer."""
        self.ptr = 0


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

        # Will be initialized in setup()
        self.policy = None
        self.optimizer = None
        self.vec_env = None
        self.buffer = None
        self.logger = None
        self.tfold_scorer = None
        self.elite_buffer = None

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

        # Build scorers
        from tcrppo_v2.utils.esm_cache import ESMCache
        from tcrppo_v2.data.pmhc_loader import PMHCLoader, EVAL_TARGETS
        from tcrppo_v2.data.tcr_pool import TCRPool
        from tcrppo_v2.reward_manager import RewardManager
        from tcrppo_v2.env import VecTCREditEnv

        # pMHC loader — train mode loads all tc-hard targets (~163)
        pmhc_loader = PMHCLoader(mode="train")
        targets = pmhc_loader.get_target_list()

        # Affinity scorer — selected by config["affinity_model"]
        affinity_model = self.config.get("affinity_model", "ergo")
        if affinity_model == "nettcr":
            from tcrppo_v2.scorers.affinity_nettcr import AffinityNetTCRScorer
            affinity_scorer = AffinityNetTCRScorer(device=self.device)
            print("  NetTCR loaded")
        elif affinity_model == "tfold":
            from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer
            affinity_scorer = AffinityTFoldScorer(
                device=self.device,
                gpu_id=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
                cache_only=self.config.get("tfold_cache_only", False),
                cache_miss_score=self.config.get("tfold_cache_miss_score", 0.5),
            )
            print(f"  tFold V3.4 loaded (cache={affinity_scorer.cache_stats['cache_size']} entries, "
                  f"cache_only={affinity_scorer.cache_only})")
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
                cache_only=self.config.get("tfold_cache_only", False),
                cache_miss_score=self.config.get("tfold_cache_miss_score", 0.5),
            )
            print(f"  tFold V3.4 loaded (cache={tfold_scorer.cache_stats['cache_size']} entries, "
                  f"cache_only={tfold_scorer.cache_only})")
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
                cache_only=self.config.get("tfold_cache_only", False),
                cache_miss_score=self.config.get("tfold_cache_miss_score", 0.5),
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
            print(f"  ESM-2 loaded (dim={esm_cache.embed_dim}, disk_cache={esm_cache.disk_cache_size} seqs)")

        print(f"  pMHC loader: {len(targets)} targets")

        # TCR pool with L1 seeds enabled for enhanced curriculum
        l1_dir = os.path.join(PROJECT_ROOT, "data", "l1_seeds")
        if not os.path.isdir(l1_dir):
            l1_dir = None  # Fall back to L0+L2 only
        tcr_pool = TCRPool(
            l1_seeds_dir=l1_dir,
            curriculum_schedule=self.config.get("curriculum_schedule"),
            seed=self.seed,
        )
        # Load L0 seeds from decoy D + tc-hard known binders
        decoy_lib_path = self.config.get("decoy_library_path", "/share/liuyutian/pMHC_decoy_library")
        tcr_pool.load_l0_from_decoy_d(decoy_lib_path, targets)
        # Also load tc-hard CDR3b binders as L0 seeds
        l0_tchard_dir = os.path.join(PROJECT_ROOT, "data", "l0_seeds_tchard")
        if os.path.isdir(l0_tchard_dir):
            tcr_pool.load_l0_from_dir(l0_tchard_dir)
        l0_targets = tcr_pool.get_l0_targets()
        l1_targets = tcr_pool.get_l1_targets()
        print(f"  TCR pool: {tcr_pool.num_tcrdb_seqs} seqs, "
              f"L0 targets={len(l0_targets)}/{len(targets)}, "
              f"L1 targets={len(l1_targets)}/{len(targets)}")

        # Decoy scorer (for reward, with LogSumExp penalty)
        decoy_scorer = None
        self.decoy_scorer = None
        # Load decoy scorer for any mode that uses decoy penalty or contrastive sampling
        if self.reward_mode in ("v2_full", "v2_decoy_only", "raw_decoy", "raw_multi_penalty", "threshold_penalty", "contrastive_ergo"):
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
            self.decoy_scorer = decoy_scorer
            print(f"  Decoy scorer loaded (K={decoy_scorer.K})")

        # Naturalness scorer (requires ESM-2 model — skip for lightweight encoder)
        naturalness_scorer = None
        if self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty"):
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
                )
                print("  Naturalness scorer loaded")

        # Diversity scorer
        diversity_scorer = None
        if self.reward_mode in ("v2_full", "v2_no_decoy", "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty"):
            from tcrppo_v2.scorers.diversity import DiversityScorer
            diversity_scorer = DiversityScorer(
                buffer_size=self.config.get("diversity_buffer_size", 512),
                similarity_threshold=self.config.get("diversity_threshold", 0.85),
            )
            print("  Diversity scorer loaded")

        # Reward manager
        self.reward_manager = RewardManager(
            affinity_scorer=affinity_scorer,
            decoy_scorer=decoy_scorer,
            naturalness_scorer=naturalness_scorer,
            diversity_scorer=diversity_scorer,
            reward_mode=self.reward_mode,
            w_affinity=self.config.get("w_affinity", 1.0),
            w_decoy=self.config.get("w_decoy", 0.8),
            w_naturalness=self.config.get("w_naturalness", 0.5),
            w_diversity=self.config.get("w_diversity", 0.2),
            n_contrast_decoys=self.config.get("n_contrast_decoys", 4),
            convex_alpha=self.config.get("convex_alpha", 3.0),
            contrastive_agg=self.config.get("contrastive_agg", "mean"),
        )

        # VecEnv
        self.vec_env = VecTCREditEnv(
            n_envs=self.n_envs,
            esm_cache=esm_cache,
            pmhc_loader=pmhc_loader,
            tcr_pool=tcr_pool,
            reward_manager=self.reward_manager,
            reward_mode=self.reward_mode,
            min_steps=self.config.get("min_steps", 0),
            min_steps_penalty=self.config.get("min_steps_penalty", 0.0),
            ban_stop=self.config.get("ban_stop", False),
        )
        print(f"  VecEnv: {self.n_envs} envs, obs_dim={self.vec_env.obs_dim}, ban_stop={self.config.get('ban_stop', False)}")

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
                cache_only=False,  # full scoring with server
                cache_miss_score=0.5,
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
        self.policy = ActorCritic(
            obs_dim=self.vec_env.obs_dim,
            hidden_dim=self.config.get("hidden_dim", 512),
        ).to(self.device)
        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"  Policy: {n_params:,} parameters")

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

        # Buffer
        self.buffer = RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            obs_dim=self.vec_env.obs_dim,
            device=self.device,
        )

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(self.output_dir, self.run_name, "tb_logs")
            os.makedirs(log_dir, exist_ok=True)
            self.logger = SummaryWriter(log_dir)
        except ImportError:
            self.logger = None

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
                "use_znorm": False,  # z-norm removed in bugfix
                "tfold_correction": self.tfold_correction,
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

        out_dir = os.path.join(self.output_dir, self.run_name)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "experiment.json")
        with open(out_file, "w") as f:
            json.dump(archive, f, indent=2)
        print(f"  Saved experiment.json to {out_file}")

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
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        obs = self.vec_env.reset()
        global_step = resume_step
        n_updates = 0
        episode_rewards = []
        episode_lengths = []
        ep_reward_buf = np.zeros(self.n_envs)
        ep_length_buf = np.zeros(self.n_envs, dtype=int)
        next_milestone_idx = 0
        # Skip past milestones already reached (important for resume)
        while next_milestone_idx < len(self.milestones) and global_step >= self.milestones[next_milestone_idx]:
            next_milestone_idx += 1

        # Per-env trackers for elite buffer (last obs/action before STOP)
        ep_last_obs = [None] * self.n_envs
        ep_last_action = [None] * self.n_envs  # (op, pos, tok, logprob)

        while global_step < self.total_timesteps:
            # Update decoy tier unlock schedule
            self._update_decoy_schedule(global_step)

            # Collect rollout
            self.buffer.reset()
            self.policy.eval()

            for step in range(self.n_steps):
                self.vec_env.set_global_step(global_step)

                # Get action masks
                masks = self.vec_env.get_action_masks()
                op_masks = np.stack([m["op_mask"] for m in masks])
                pos_masks = np.stack([m["pos_mask"] for m in masks])

                # Get actions from policy
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                mask_dict = {
                    "op_mask": torch.BoolTensor(op_masks).to(self.device),
                    "pos_mask": torch.BoolTensor(pos_masks).to(self.device),
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
                               val_np, op_masks, pos_masks)

                # Track episode stats (skip phantom transitions)
                for i in range(self.n_envs):
                    if not was_done[i]:
                        ep_reward_buf[i] += rewards[i]
                        ep_length_buf[i] += 1
                        # Track last obs/action for elite buffer
                        ep_last_obs[i] = obs[i].copy()
                        ep_last_action[i] = (int(ops_np[i]), int(pos_np[i]), int(tok_np[i]), float(lp_np[i]))
                    if dones[i] and not was_done[i]:
                        episode_rewards.append(ep_reward_buf[i])
                        episode_lengths.append(ep_length_buf[i])
                        # Submit to elite buffer if tFold correction enabled
                        if self.elite_buffer is not None and ep_last_obs[i] is not None:
                            env_i = self.vec_env.envs[i]
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
                    elif was_done[i]:
                        # Auto-reset happened, start fresh counter
                        ep_reward_buf[i] = 0.0
                        ep_length_buf[i] = 0
                        ep_last_obs[i] = None
                        ep_last_action[i] = None

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
            n_batches = 0

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
                        },
                        actions=(batch["ops"], batch["positions"], batch["tokens"]),
                    )

                    # PPO clipped objective
                    ratio = torch.exp(log_probs - batch["old_log_probs"])
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
                    n_batches += 1

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

            if n_updates % 10 == 0 and episode_rewards:
                recent = episode_rewards[-100:]
                mean_r = np.mean(recent)
                mean_l = np.mean(episode_lengths[-100:])
                print(
                    f"Step {global_step:>10,} | "
                    f"Eps: {len(episode_rewards):>6} | "
                    f"R: {mean_r:>8.3f} | "
                    f"Len: {mean_l:>5.1f} | "
                    f"PG: {avg_pg:>8.4f} | "
                    f"VF: {avg_vf:>8.4f} | "
                    f"Ent: {avg_ent:>6.3f}"
                )

                if self.logger:
                    self.logger.add_scalar("train/mean_reward", mean_r, global_step)
                    self.logger.add_scalar("train/mean_episode_length", mean_l, global_step)
                    self.logger.add_scalar("train/pg_loss", avg_pg, global_step)
                    self.logger.add_scalar("train/vf_loss", avg_vf, global_step)
                    self.logger.add_scalar("train/entropy", avg_ent, global_step)

            # Checkpointing
            if next_milestone_idx < len(self.milestones) and global_step >= self.milestones[next_milestone_idx]:
                ms = self.milestones[next_milestone_idx]
                self.save_checkpoint(f"milestone_{ms}", global_step)
                print(f"  ** Milestone checkpoint saved: {ms:,} steps")
                next_milestone_idx += 1

            if global_step % self.checkpoint_interval < self.n_envs * self.n_steps:
                self.save_checkpoint("latest", global_step)

        # Final checkpoint
        self.save_checkpoint("final", global_step)
        print(f"\nTraining complete: {global_step:,} steps, {len(episode_rewards)} episodes")

        if self.logger:
            self.logger.close()

    def _update_decoy_schedule(self, global_step: int) -> None:
        """Update decoy tier unlock based on training progress."""
        if self.decoy_scorer is None:
            return
        if global_step < 2_000_000:
            tiers = ["A"]
        elif global_step < 5_000_000:
            tiers = ["A", "B"]
        elif global_step < 8_000_000:
            tiers = ["A", "B", "D"]
        else:
            tiers = ["A", "B", "D", "C"]
        self.decoy_scorer.set_unlocked_tiers(tiers)

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
        batch_pos_masks = torch.ones(len(top_k), MAX_TCR_LEN, dtype=torch.bool, device=self.device)

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
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "global_step": global_step,
        }, path)

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
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--w_affinity", type=float, default=None, help="Affinity weight")
    parser.add_argument("--w_decoy", type=float, default=None, help="Decoy weight")
    parser.add_argument("--w_naturalness", type=float, default=None, help="Naturalness weight")
    parser.add_argument("--w_diversity", type=float, default=None, help="Diversity weight")
    parser.add_argument("--min_steps", type=int, default=None, help="Min steps before STOP")
    parser.add_argument("--min_steps_penalty", type=float, default=None, help="Penalty for early STOP")
    # Two-phase training support
    parser.add_argument("--resume_from", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--resume_change_reward_mode", default=None, help="Change reward mode on resume")
    parser.add_argument("--resume_reset_optimizer", action="store_true", help="Reset optimizer on resume")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Policy hidden dim override")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate override")
    parser.add_argument("--affinity_scorer", default=None, help="Affinity scorer: ergo, nettcr, tcbind, tfold, tfold_cascade, ensemble, ensemble_ergo_tcbind, ensemble_ergo_tfold")
    parser.add_argument("--tfold_cache_only", action="store_true", help="tFold: skip server extraction for cache misses")
    parser.add_argument("--tfold_cache_miss_score", type=float, default=None, help="tFold: score for cache misses (default 0.5)")
    parser.add_argument("--cascade_threshold", type=float, default=None, help="ERGO uncertainty threshold for tFold cascade (default 0.15)")
    parser.add_argument("--encoder", default=None, choices=["esm2", "lightweight"], help="State encoder: esm2 (default) or lightweight (CPU-friendly BiLSTM)")
    parser.add_argument("--encoder_dim", type=int, default=None, help="Lightweight encoder output dim (default 256)")
    parser.add_argument("--esm_cache_path", type=str, default=None, help="Path to ESM cache DB (default: data/esm_cache.db)")
    parser.add_argument("--tfold_cache_path", type=str, default=None, help="Path to tFold feature cache DB")
    # tFold correction
    parser.add_argument("--tfold_correction", action="store_true", help="Enable tFold elite re-scoring correction")
    parser.add_argument("--tfold_rescore_interval", type=int, default=None, help="Re-score elite TCRs every N rollouts (default 50)")
    parser.add_argument("--tfold_top_k", type=int, default=None, help="Top-K elite TCRs to re-score (default 32)")
    parser.add_argument("--tfold_correction_alpha", type=float, default=None, help="Correction advantage scale (default 2.0)")
    parser.add_argument("--elite_buffer_size", type=int, default=None, help="Max elite buffer size (default 500)")
    parser.add_argument("--elite_score_threshold", type=float, default=None, help="Min ERGO score for elite (default 0.7)")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per episode (default 8)")
    parser.add_argument("--ban_stop", action="store_true", help="Ban STOP action — agent must use all max_steps")

    # Curriculum overrides
    parser.add_argument("--curriculum_l0", type=float, default=None, help="L0 curriculum ratio (known binder variants)")
    parser.add_argument("--curriculum_l1", type=float, default=None, help="L1 curriculum ratio (ERGO top-K)")
    parser.add_argument("--curriculum_l2", type=float, default=None, help="L2 curriculum ratio (random TCRdb)")

    # Contrastive reward
    parser.add_argument("--n_contrast_decoys", type=int, default=4, help="Number of decoys for contrastive_ergo reward")
    parser.add_argument("--contrastive_agg", default="mean", choices=["mean", "max"], help="Aggregation for contrastive decoy scores: mean or max")
    parser.add_argument("--convex_alpha", type=float, default=3.0, help="Exponent for v1_ergo_convex reward mode")
    parser.add_argument("--entropy_coef_final", type=float, default=None, help="Final entropy coefficient (enables linear decay)")
    parser.add_argument("--entropy_decay_start", type=int, default=None, help="Step to begin entropy decay (default: 1M)")
    parser.add_argument("--decoy_library_path", type=str, default=None, help="Path to pMHC decoy library (default: /share/liuyutian/pMHC_decoy_library)")
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
    config["device"] = args.device
    if args.w_affinity is not None:
        config["w_affinity"] = args.w_affinity
    if args.w_decoy is not None:
        config["w_decoy"] = args.w_decoy
    if args.w_naturalness is not None:
        config["w_naturalness"] = args.w_naturalness
    if args.w_diversity is not None:
        config["w_diversity"] = args.w_diversity
    if args.min_steps is not None:
        config["min_steps"] = args.min_steps
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.min_steps_penalty is not None:
        config["min_steps_penalty"] = args.min_steps_penalty
    if args.hidden_dim is not None:
        config["hidden_dim"] = args.hidden_dim
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.affinity_scorer is not None:
        config["affinity_model"] = args.affinity_scorer
    if args.tfold_cache_only:
        config["tfold_cache_only"] = True
    if args.tfold_cache_miss_score is not None:
        config["tfold_cache_miss_score"] = args.tfold_cache_miss_score
    if args.encoder is not None:
        config["encoder"] = args.encoder
    if args.encoder_dim is not None:
        config["encoder_dim"] = args.encoder_dim
    if args.esm_cache_path is not None:
        config["esm_cache_path"] = args.esm_cache_path
    if args.tfold_cache_path is not None:
        config["tfold_cache_path"] = args.tfold_cache_path
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
    if args.ban_stop:
        config["ban_stop"] = True
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

    config.setdefault("run_name", "v2_run")

    trainer = PPOTrainer(config)

    # Two-phase training: store resume args for processing after setup()
    trainer._resume_from = args.resume_from
    trainer._resume_change_reward_mode = args.resume_change_reward_mode
    trainer._resume_reset_optimizer = args.resume_reset_optimizer

    trainer.train()


if __name__ == "__main__":
    main()
