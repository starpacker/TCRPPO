"""Integration tests for the TCR editing environment."""

import sys
import os
import pytest
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.utils.constants import (
    OP_SUB, OP_INS, OP_DEL, OP_STOP, NUM_OPS, NUM_AMINO_ACIDS,
    MIN_TCR_LEN, MAX_TCR_LEN, MAX_STEPS_PER_EPISODE,
)


@pytest.fixture(scope="module")
def env_components():
    """Set up all shared components for env tests."""
    from tcrppo_v2.utils.esm_cache import ESMCache
    from tcrppo_v2.data.pmhc_loader import PMHCLoader
    from tcrppo_v2.data.tcr_pool import TCRPool
    from tcrppo_v2.reward_manager import RewardManager

    # Use mock affinity scorer for speed
    class MockAffinityScorer:
        def score(self, tcr, peptide="", **kwargs):
            # Simple length-based score for testing
            return min(len(tcr) / 20.0, 1.0), 1.0

        def score_batch(self, tcrs, peptides, **kwargs):
            scores = [min(len(t) / 20.0, 1.0) for t in tcrs]
            return scores, [1.0] * len(tcrs)

    esm_cache = ESMCache(device="cuda", tcr_cache_size=128)
    pmhc_loader = PMHCLoader(targets=["GILGFVFTL", "NLVPMVATV"])
    tcr_pool = TCRPool(seed=42)
    reward_manager = RewardManager(
        affinity_scorer=MockAffinityScorer(),
        reward_mode="v1_ergo_only",  # Simple mode for testing
    )

    return {
        "esm_cache": esm_cache,
        "pmhc_loader": pmhc_loader,
        "tcr_pool": tcr_pool,
        "reward_manager": reward_manager,
    }


@pytest.fixture(scope="module")
def env(env_components):
    """Create a single TCREditEnv."""
    from tcrppo_v2.env import TCREditEnv
    return TCREditEnv(**env_components, reward_mode="v1_ergo_only")


class TestTCREditEnv:
    def test_reset(self, env):
        obs = env.reset(peptide="GILGFVFTL")
        assert obs.shape == (env.obs_dim,)
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()
        print(f"\nObs dim: {env.obs_dim}, TCR: {env.current_tcr}")

    def test_step_sub(self, env):
        env.reset(peptide="GILGFVFTL")
        old_tcr = env.current_tcr
        obs, reward, done, info = env.step((OP_SUB, 0, 5))
        assert obs.shape == (env.obs_dim,)
        assert not np.isnan(reward)
        assert env.current_tcr[0] != old_tcr[0] or env.current_tcr == old_tcr
        assert info["action_name"] == "SUB"

    def test_step_ins(self, env):
        env.reset(peptide="GILGFVFTL")
        old_len = len(env.current_tcr)
        if old_len < MAX_TCR_LEN:
            obs, reward, done, info = env.step((OP_INS, 0, 3))
            assert len(env.current_tcr) == old_len + 1
            assert info["action_name"] == "INS"

    def test_step_del(self, env):
        env.reset(peptide="GILGFVFTL")
        old_len = len(env.current_tcr)
        if old_len > MIN_TCR_LEN:
            obs, reward, done, info = env.step((OP_DEL, 0, 0))
            assert len(env.current_tcr) == old_len - 1
            assert info["action_name"] == "DEL"

    def test_step_stop(self, env):
        env.reset(peptide="GILGFVFTL")
        # Must take at least one step before STOP
        env.step((OP_SUB, 0, 5))
        obs, reward, done, info = env.step((OP_STOP, 0, 0))
        assert done
        assert info["action_name"] == "STOP"

    def test_action_mask(self, env):
        env.reset(peptide="GILGFVFTL")
        masks = env.get_action_mask()
        assert masks["op_mask"].shape == (NUM_OPS,)
        assert masks["pos_mask"].shape == (MAX_TCR_LEN,)
        # Step 0: STOP should be masked
        assert not masks["op_mask"][OP_STOP]
        # Valid positions should be True
        assert masks["pos_mask"][:len(env.current_tcr)].all()
        assert not masks["pos_mask"][len(env.current_tcr):].any()

    def test_max_steps_termination(self, env):
        env.reset(peptide="GILGFVFTL")
        for i in range(MAX_STEPS_PER_EPISODE):
            if env.done:
                break
            obs, reward, done, info = env.step((OP_SUB, 0, i % NUM_AMINO_ACIDS))
        assert env.done

    def test_length_constraints(self, env):
        """Sequence length should stay within bounds."""
        rng = np.random.default_rng(42)
        env.reset(peptide="GILGFVFTL")

        for _ in range(MAX_STEPS_PER_EPISODE):
            if env.done:
                break
            masks = env.get_action_mask()
            # Randomly pick a valid op
            valid_ops = np.where(masks["op_mask"])[0]
            op = rng.choice(valid_ops)
            pos = rng.integers(0, len(env.current_tcr))
            tok = rng.integers(0, NUM_AMINO_ACIDS)
            env.step((op, pos, tok))

            assert MIN_TCR_LEN <= len(env.current_tcr) <= MAX_TCR_LEN, (
                f"TCR length {len(env.current_tcr)} out of bounds"
            )


class TestRandomEpisodes:
    """Run 100 random episodes to verify consistency."""

    def test_100_episodes(self, env_components):
        from tcrppo_v2.env import TCREditEnv

        env = TCREditEnv(**env_components, reward_mode="v1_ergo_only")
        rng = np.random.default_rng(123)

        n_episodes = 100
        total_steps = 0
        all_rewards = []
        all_lengths = []

        for ep in range(n_episodes):
            obs = env.reset()
            assert obs.shape == (env.obs_dim,), f"Episode {ep}: bad obs shape"
            ep_reward = 0.0

            for step in range(MAX_STEPS_PER_EPISODE):
                if env.done:
                    break
                masks = env.get_action_mask()
                valid_ops = np.where(masks["op_mask"])[0]
                op = rng.choice(valid_ops)
                pos = rng.integers(0, max(1, len(env.current_tcr)))
                tok = rng.integers(0, NUM_AMINO_ACIDS)

                obs, reward, done, info = env.step((op, pos, tok))

                assert obs.shape == (env.obs_dim,), f"Ep {ep} step {step}: bad obs"
                assert not np.isnan(obs).any(), f"Ep {ep} step {step}: NaN in obs"
                assert not np.isinf(obs).any(), f"Ep {ep} step {step}: Inf in obs"
                assert np.isfinite(reward), f"Ep {ep} step {step}: bad reward"
                assert MIN_TCR_LEN <= len(env.current_tcr) <= MAX_TCR_LEN

                ep_reward += reward
                total_steps += 1

            all_rewards.append(ep_reward)
            all_lengths.append(len(env.current_tcr))

        print(f"\n100 episodes completed:")
        print(f"  Total steps: {total_steps}")
        print(f"  Mean ep reward: {np.mean(all_rewards):.4f}")
        print(f"  Mean final TCR len: {np.mean(all_lengths):.1f}")
        print(f"  TCR len range: [{min(all_lengths)}, {max(all_lengths)}]")

        assert total_steps > 0
        assert all(np.isfinite(r) for r in all_rewards)


class TestVecEnv:
    def test_vec_env_creation(self, env_components):
        from tcrppo_v2.env import VecTCREditEnv

        vec_env = VecTCREditEnv(n_envs=3, **env_components, reward_mode="v1_ergo_only")
        obs = vec_env.reset()
        assert obs.shape == (3, vec_env.obs_dim)

    def test_vec_env_step(self, env_components):
        from tcrppo_v2.env import VecTCREditEnv

        vec_env = VecTCREditEnv(n_envs=3, **env_components, reward_mode="v1_ergo_only")
        vec_env.reset()

        actions = [(OP_SUB, 0, 5), (OP_SUB, 1, 3), (OP_SUB, 2, 7)]
        obs, rewards, dones, infos = vec_env.step(actions)
        assert obs.shape == (3, vec_env.obs_dim)
        assert rewards.shape == (3,)
        assert dones.shape == (3,)
        assert len(infos) == 3

    def test_vec_env_masks(self, env_components):
        from tcrppo_v2.env import VecTCREditEnv

        vec_env = VecTCREditEnv(n_envs=2, **env_components, reward_mode="v1_ergo_only")
        vec_env.reset()
        masks = vec_env.get_action_masks()
        assert len(masks) == 2
        assert masks[0]["op_mask"].shape == (NUM_OPS,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
