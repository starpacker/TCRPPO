"""Unit tests for the TCR-SAC migration components."""

import os
import sys
import sqlite3

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.sac_trainer import ActionQNetwork, SACReplayBuffer, TCRSACTrainer
from tcrppo_v2.utils.constants import MAX_TCR_LEN, NUM_AMINO_ACIDS, NUM_OPS


def test_sac_replay_buffer_shapes():
    obs_dim = 10
    n_envs = 3
    buf = SACReplayBuffer(capacity=16, obs_dim=obs_dim, device="cpu")
    obs = np.random.randn(n_envs, obs_dim).astype(np.float32)
    next_obs = np.random.randn(n_envs, obs_dim).astype(np.float32)
    actions = (
        np.zeros(n_envs, dtype=np.int64),
        np.ones(n_envs, dtype=np.int64),
        np.full(n_envs, 2, dtype=np.int64),
    )
    rewards = np.ones(n_envs, dtype=np.float32)
    dones = np.zeros(n_envs, dtype=np.float32)
    op_masks = np.ones((n_envs, NUM_OPS), dtype=bool)
    pos_masks = np.ones((n_envs, MAX_TCR_LEN), dtype=bool)

    buf.add_batch(obs, actions, rewards, next_obs, dones, op_masks, pos_masks, op_masks, pos_masks)
    batch = buf.sample(batch_size=5)

    assert batch["obs"].shape == (5, obs_dim)
    assert batch["next_obs"].shape == (5, obs_dim)
    assert batch["ops"].shape == (5,)
    assert batch["op_masks"].shape == (5, NUM_OPS)
    assert batch["pos_masks"].shape == (5, MAX_TCR_LEN)


def test_action_q_network_forward_and_gradients():
    obs_dim = 12
    q = ActionQNetwork(obs_dim=obs_dim, hidden_dim=32)
    obs = torch.randn(6, obs_dim)
    ops = torch.randint(0, NUM_OPS, (6,))
    pos = torch.randint(0, MAX_TCR_LEN, (6,))
    tok = torch.randint(0, NUM_AMINO_ACIDS, (6,))

    values = q(obs, ops, pos, tok)
    assert values.shape == (6,)
    assert torch.isfinite(values).all()

    loss = values.mean()
    loss.backward()
    grads = [p.grad for p in q.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_random_action_sampler_respects_masks():
    trainer = TCRSACTrainer({"device": "cpu"})
    op_masks = np.zeros((4, NUM_OPS), dtype=bool)
    op_masks[:, 0] = True
    pos_masks = np.zeros((4, MAX_TCR_LEN), dtype=bool)
    pos_masks[:, 2:5] = True

    ops, pos, tok = trainer._sample_random_actions(op_masks, pos_masks)
    assert np.all(ops == 0)
    assert np.all((pos >= 2) & (pos < 5))
    assert np.all((tok >= 0) & (tok < NUM_AMINO_ACIDS))


def test_tfold_feature_cache_read_only_no_writes(tmp_path):
    from tcrppo_v2.scorers.affinity_tfold import TFoldFeatureCache

    db_path = tmp_path / "tfold_cache.db"
    writable = TFoldFeatureCache(str(db_path))
    writable.put("k1", {"x": torch.tensor([1.0])})
    writable.close()

    readonly = TFoldFeatureCache(str(db_path), read_only=True)
    item = readonly.get("k1")
    assert item is not None
    assert torch.equal(item["x"], torch.tensor([1.0]))

    readonly.put("k2", {"x": torch.tensor([2.0])})
    readonly.put_batch([("k3", {"x": torch.tensor([3.0])})])
    readonly.close()

    with sqlite3.connect(db_path) as conn:
        n_rows = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    assert n_rows == 1
