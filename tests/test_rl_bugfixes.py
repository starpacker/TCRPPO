"""Regression tests for RL training bugs found from trace reward audits."""

import os
import sys

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _make_tcrdb(tmp_path):
    tcrdb = tmp_path / "tcrdb"
    tcrdb.mkdir()
    (tcrdb / "train_uniq_tcr_seqs.txt").write_text(
        "CASSIRSSYEQYF\n"
        "ASSIRSSYEQYF\n"
        "CASSINVALID*\n"
        "CASSLGQAYEQYF\n"
    )
    return tcrdb


def test_decoy_d_generic_sequence_column_is_not_loaded_as_l0(tmp_path):
    from tcrppo_v2.data.tcr_pool import TCRPool

    tcrdb = _make_tcrdb(tmp_path)
    decoy_root = tmp_path / "decoys"
    peptide_dir = decoy_root / "data" / "decoy_d" / "PEPTIDE"
    peptide_dir.mkdir(parents=True)
    (peptide_dir / "decoy_d_results.csv").write_text(
        "sequence,score\n"
        "MLPPVDAPV,1.0\n"
        "CASSIRSSYEQYF,0.9\n"
    )

    pool = TCRPool(tcrdb_path=str(tcrdb), seed=7)
    pool.load_l0_from_decoy_d(str(decoy_root), ["PEPTIDE"])

    assert "PEPTIDE" not in pool.l0_seeds


def test_decoy_d_explicit_tcr_column_filters_to_cdr3_like_l0(tmp_path):
    from tcrppo_v2.data.tcr_pool import TCRPool

    tcrdb = _make_tcrdb(tmp_path)
    decoy_root = tmp_path / "decoys"
    peptide_dir = decoy_root / "data" / "decoy_d" / "PEPTIDE"
    peptide_dir.mkdir(parents=True)
    (peptide_dir / "decoy_d_results.csv").write_text(
        "cdr3b,score\n"
        "CASSIRSSYEQYF,1.0\n"
        "MLPPVDAPV,0.8\n"
        "ASSIRSSYEQYF,0.7\n"
    )

    pool = TCRPool(tcrdb_path=str(tcrdb), seed=7)
    pool.load_l0_from_decoy_d(str(decoy_root), ["PEPTIDE"])

    assert pool.l0_seeds["PEPTIDE"] == ["CASSIRSSYEQYF"]


def test_l0_sampling_preserves_leading_cys(tmp_path):
    from tcrppo_v2.data.tcr_pool import TCRPool

    tcrdb = _make_tcrdb(tmp_path)
    pool = TCRPool(tcrdb_path=str(tcrdb), l0_mutation_range=(5, 5), seed=13)
    pool.l0_seeds["PEPTIDE"] = ["CASSIRSSYEQYF"]

    for _ in range(20):
        sampled = pool._sample_l0("PEPTIDE")
        assert sampled.startswith("C")
        assert len(sampled) == len("CASSIRSSYEQYF")


class FakeESMCache:
    output_dim = 2

    def encode_pmhc(self, pmhc):
        return torch.tensor([1.0, 2.0])

    def encode_tcr(self, tcr):
        return torch.tensor([float(len(tcr)), float(tcr.startswith("C"))])

    def encode_tcr_batch(self, tcrs):
        return torch.stack([self.encode_tcr(tcr) for tcr in tcrs])


class FakePMHCLoader:
    def sample_target(self):
        return "PEPTIDE"

    def get_pmhc_string(self, peptide):
        return peptide


class FakeTCRPool:
    def sample_tcr(self, target, step=0, reward_mode="v2_full"):
        return "CASSIRSS", "L0"


class FakeRewardManager:
    affinity_scorer = None

    def compute_reward(self, **kwargs):
        return 0.0, {"total": 0.0}


def test_env_masks_and_actions_preserve_leading_cys():
    from tcrppo_v2.env import TCREditEnv
    from tcrppo_v2.utils.constants import OP_DEL, OP_INS, OP_SUB

    env = TCREditEnv(
        esm_cache=FakeESMCache(),
        pmhc_loader=FakePMHCLoader(),
        tcr_pool=FakeTCRPool(),
        reward_manager=FakeRewardManager(),
        max_steps=8,
    )
    env.reset(init_tcr="CASSIRSS", peptide="PEPTIDE")

    masks = env.get_action_mask()
    assert not masks["pos_mask"][0]
    assert masks["pos_mask"][1:len(env.current_tcr)].all()

    original = env.current_tcr
    env.step((OP_SUB, 0, 1))
    assert env.current_tcr == original

    env.step((OP_DEL, 0, 0))
    assert env.current_tcr == original

    env.step((OP_INS, 0, 1))
    assert env.current_tcr.startswith("C")
    assert len(env.current_tcr) == len(original) + 1


def test_reward_manager_applies_affinity_weight_and_calibrated_delta():
    from tcrppo_v2.reward_manager import RewardManager

    class MockAffinityScorer:
        def score(self, tcr, peptide="", **kwargs):
            return -6.0, 1.0

        def score_batch_fast(self, tcrs, peptides):
            return [-6.0 for _ in tcrs]

    rm_abs = RewardManager(
        affinity_scorer=MockAffinityScorer(),
        reward_mode="v2_no_decoy",
        w_affinity=0.2,
        w_naturalness=0.0,
        w_diversity=0.0,
    )
    total, comp = rm_abs.compute_reward("CASSIRSS", "PEPTIDE", initial_affinity=-8.0)
    assert np.isclose(total, -1.2)
    assert np.isclose(comp["affinity_raw"], -6.0)

    rm_delta = RewardManager(
        affinity_scorer=MockAffinityScorer(),
        reward_mode="v2_no_decoy_delta_calibrated",
        w_affinity=1.0,
        w_naturalness=0.0,
        w_diversity=0.0,
        w_absolute_affinity=0.25,
        affinity_ref_logit=-4.5,
    )
    total, comp = rm_delta.compute_reward(
        "CASSIRSS", "PEPTIDE", initial_affinity=-8.0
    )
    assert np.isclose(comp["affinity_delta"], 2.0)
    assert np.isclose(comp["affinity_abs_centered"], -1.5)
    assert np.isclose(total, 1.625)


def test_reward_manager_delta_minus_decoy_uses_episode_initial_tcr():
    from tcrppo_v2.reward_manager import RewardManager

    class MockAffinityScorer:
        def score_batch_fast(self, tcrs, peptides):
            table = {
                ("INITIAL", "TARGET"): -7.0,
                ("FINAL", "TARGET"): -5.0,
                ("INITIAL", "DECOY1"): -6.0,
                ("FINAL", "DECOY1"): -4.0,
                ("INITIAL", "DECOY2"): -4.0,
                ("FINAL", "DECOY2"): -5.0,
            }
            return [table[(tcr, pep)] for tcr, pep in zip(tcrs, peptides)]

        def score(self, tcr, peptide="", **kwargs):
            return self.score_batch_fast([tcr], [peptide])[0], 1.0

    class MockDecoyScorer:
        K = 2

        def sample_decoys(self, target, k=None):
            return ["DECOY1", "DECOY2"][:k]

    rm = RewardManager(
        affinity_scorer=MockAffinityScorer(),
        decoy_scorer=MockDecoyScorer(),
        reward_mode="v2_delta_minus_decoy",
        n_contrast_decoys=2,
        w_affinity=1.0,
        w_decoy=1.0,
        w_naturalness=0.0,
        w_diversity=0.0,
    )

    total, comp = rm.compute_reward(
        "FINAL",
        "TARGET",
        initial_affinity=-7.0,
        initial_tcr="INITIAL",
    )

    assert np.isclose(comp["affinity_delta"], 2.0)
    assert np.isclose(comp["decoy_delta_mean"], 0.5)
    assert np.isclose(comp["decoy_delta"], 0.5)
    assert np.isclose(total, 1.5)

    rewards, comps = rm.compute_reward_batch(
        ["FINAL"],
        ["TARGET"],
        [-7.0],
        initial_tcrs=["INITIAL"],
        targets=["TARGET"],
    )
    assert np.isclose(comps[0]["decoy_delta"], 0.5)
    assert np.isclose(rewards[0], 1.5)


def test_reward_manager_target_guarded_decoy_protects_target_floor():
    from tcrppo_v2.reward_manager import RewardManager

    class MockAffinityScorer:
        def __init__(self, final_target):
            self.final_target = final_target

        def score_batch_fast(self, tcrs, peptides):
            table = {
                ("INITIAL", "TARGET"): -2.4,
                ("FINAL", "TARGET"): self.final_target,
                ("INITIAL", "DECOY1"): -2.0,
                ("FINAL", "DECOY1"): -4.0,
                ("INITIAL", "DECOY2"): -2.2,
                ("FINAL", "DECOY2"): -4.2,
            }
            return [table[(tcr, pep)] for tcr, pep in zip(tcrs, peptides)]

        def score(self, tcr, peptide="", **kwargs):
            return self.score_batch_fast([tcr], [peptide])[0], 1.0

    class MockDecoyScorer:
        K = 2

        def sample_decoys(self, target, k=None):
            return ["DECOY1", "DECOY2"][:k]

    base_kwargs = dict(
        decoy_scorer=MockDecoyScorer(),
        reward_mode="v2_target_guarded_decoy",
        n_contrast_decoys=2,
        contrastive_agg="max",
        w_affinity=1.0,
        w_decoy=0.45,
        w_naturalness=0.0,
        w_diversity=0.0,
        w_absolute_affinity=0.60,
        affinity_guard_logit=-3.0,
        affinity_guard_tolerance=0.35,
        affinity_guard_weight=6.0,
        specificity_margin=1.0,
        decoy_drop_weight=0.15,
    )

    guarded = RewardManager(affinity_scorer=MockAffinityScorer(-2.6), **base_kwargs)
    guarded_total, guarded_comp = guarded.compute_reward(
        "FINAL",
        "TARGET",
        initial_affinity=-2.4,
        initial_tcr="INITIAL",
    )
    assert np.isclose(guarded_comp["affinity_guard_floor"], -2.75)
    assert np.isclose(guarded_comp["affinity_guard_shortfall"], 0.0)
    assert np.isclose(guarded_comp["decoy_margin_violation"], 0.0)
    assert guarded_comp["decoy_drop"] > 0.0

    collapsed = RewardManager(affinity_scorer=MockAffinityScorer(-4.0), **base_kwargs)
    collapsed_total, collapsed_comp = collapsed.compute_reward(
        "FINAL",
        "TARGET",
        initial_affinity=-2.4,
        initial_tcr="INITIAL",
    )
    assert collapsed_comp["affinity_guard_shortfall"] > 1.0
    assert collapsed_comp["target_guard_penalty"] > 10.0
    assert collapsed_total < guarded_total - 10.0


def test_reward_manager_absolute_specificity_uses_absolute_thresholds():
    from tcrppo_v2.reward_manager import RewardManager

    class MockAffinityScorer:
        def __init__(self, final_target, final_decoy):
            self.final_target = final_target
            self.final_decoy = final_decoy

        def score_batch_fast(self, tcrs, peptides):
            table = {
                ("INITIAL", "TARGET"): -2.4,
                ("FINAL", "TARGET"): self.final_target,
                ("INITIAL", "DECOY1"): -2.0,
                ("FINAL", "DECOY1"): self.final_decoy,
            }
            return [table[(tcr, pep)] for tcr, pep in zip(tcrs, peptides)]

        def score(self, tcr, peptide="", **kwargs):
            return self.score_batch_fast([tcr], [peptide])[0], 1.0

    class MockDecoyScorer:
        K = 1

        def sample_decoys(self, target, k=None):
            return ["DECOY1"]

    kwargs = dict(
        decoy_scorer=MockDecoyScorer(),
        reward_mode="v2_absolute_specificity",
        n_contrast_decoys=1,
        contrastive_agg="max",
        w_affinity=1.0,
        w_decoy=0.8,
        w_naturalness=0.0,
        w_diversity=0.0,
        affinity_guard_logit=-3.0,
        decoy_affinity_ceiling=-4.5,
        affinity_guard_weight=6.0,
        target_surplus_cap=2.0,
    )

    good = RewardManager(affinity_scorer=MockAffinityScorer(-2.4, -5.0), **kwargs)
    good_total, good_comp = good.compute_reward(
        "FINAL",
        "TARGET",
        initial_affinity=-2.4,
        initial_tcr="INITIAL",
    )
    assert np.isclose(good_comp["target_affinity_satisfied"], 0.6)
    assert np.isclose(good_comp["decoy_affinity_violation"], 0.0)
    assert np.isclose(good_total, 0.6)

    bad_decoy = RewardManager(affinity_scorer=MockAffinityScorer(-2.4, -3.5), **kwargs)
    bad_decoy_total, bad_decoy_comp = bad_decoy.compute_reward(
        "FINAL",
        "TARGET",
        initial_affinity=-2.4,
        initial_tcr="INITIAL",
    )
    assert np.isclose(bad_decoy_comp["decoy_affinity_violation"], 1.0)
    assert bad_decoy_total < good_total

    bad_target = RewardManager(affinity_scorer=MockAffinityScorer(-4.0, -5.0), **kwargs)
    bad_target_total, bad_target_comp = bad_target.compute_reward(
        "FINAL",
        "TARGET",
        initial_affinity=-2.4,
        initial_tcr="INITIAL",
    )
    assert np.isclose(bad_target_comp["target_affinity_shortfall"], 1.0)
    assert bad_target_total < -10.0


def test_reward_manager_simple_target_gated_decoy_ignores_decoys_until_target_passes():
    from tcrppo_v2.reward_manager import RewardManager

    class MockAffinityScorer:
        def __init__(self, target_score):
            self.target_score = target_score

        def score_batch_fast(self, tcrs, peptides):
            table = {
                ("FINAL", "TARGET"): self.target_score,
                ("FINAL", "DECOY_A"): -1.0,
                ("FINAL", "DECOY_B"): -3.0,
            }
            return [table[(tcr, pep)] for tcr, pep in zip(tcrs, peptides)]

        def score(self, tcr, peptide="", **kwargs):
            return self.score_batch_fast([tcr], [peptide])[0], 1.0

    class MockDecoyScorer:
        K = 2

        def sample_decoys_by_difficulty(self, target, tiers=None, k_per_tier=1):
            return ["DECOY_A", "DECOY_B"]

    kwargs = dict(
        decoy_scorer=MockDecoyScorer(),
        reward_mode="v2_simple_target_gated_decoy",
        w_affinity=1.0,
        w_decoy=0.3,
        w_naturalness=0.0,
        w_diversity=0.0,
        target_decoy_gate_logit=-2.0,
        target_pass_bonus=1.0,
        decoy_affinity_center=-3.0,
        decoy_fixed_tiers=["A", "B"],
    )

    below = RewardManager(affinity_scorer=MockAffinityScorer(-2.1), **kwargs)
    below_total, below_comp = below.compute_reward("FINAL", "TARGET")
    assert np.isclose(below_total, -2.1)
    assert below_comp["target_decoy_gate_passed"] == 0.0
    assert np.isclose(below_comp["decoy_term"], 0.0)

    passed = RewardManager(affinity_scorer=MockAffinityScorer(-1.5), **kwargs)
    passed_total, passed_comp = passed.compute_reward("FINAL", "TARGET")
    assert passed_comp["target_decoy_gate_passed"] == 1.0
    assert np.isclose(passed_comp["decoy_final_mean"], -2.0)
    assert np.isclose(passed_comp["decoy_centered"], 1.0)
    assert np.isclose(passed_total, -1.5 + 1.0 - 0.3)


def test_decoy_scorer_samples_ab_by_difficulty_without_c(tmp_path):
    from tcrppo_v2.scorers.decoy import DecoyScorer
    import json

    root = tmp_path / "decoys"
    a_dir = root / "data" / "decoy_a" / "TARGET"
    b_dir = root / "data" / "decoy_b" / "TARGET"
    c_dir = root / "data" / "decoy_c"
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)
    c_dir.mkdir(parents=True)
    (a_dir / "decoy_a_results.json").write_text(json.dumps([
        {"sequence": "A1", "hamming_distance": 1},
        {"sequence": "A2", "hamming_distance": 2},
        {"sequence": "A4", "hamming_distance": 4},
    ]))
    (b_dir / "decoy_b_results.json").write_text(json.dumps([
        {"sequence": "B2", "hamming_distance": 2},
        {"sequence": "B5", "hamming_distance": 5},
        {"sequence": "B8", "hamming_distance": 8},
        {"sequence": "B10", "hamming_distance": 10},
    ]))
    (c_dir / "decoy_library.json").write_text(json.dumps([
        {"peptide_info": {"decoy_sequence": "CDECOY"}}
    ]))

    scorer = DecoyScorer(
        decoy_library_path=str(root),
        targets=["TARGET"],
        K=2,
        rng=np.random.default_rng(1),
    )

    scorer.set_decoy_difficulty("easy")
    easy = scorer.sample_decoys_by_difficulty("TARGET", tiers=["A", "B", "C"])
    assert easy[0] == "A4"
    assert easy[1] in {"B8", "B10"}
    assert "CDECOY" not in easy

    scorer.set_decoy_difficulty("medium")
    medium = scorer.sample_decoys_by_difficulty("TARGET", tiers=["A", "B"])
    assert medium[0] in {"A2", "A4"}
    assert medium[1] in {"B5", "B8"}

    scorer.set_decoy_difficulty("hard")
    hard = scorer.sample_decoys_by_difficulty("TARGET", tiers=["A", "B"])
    assert hard[0] in {"A1", "A2", "A4"}
    assert hard[1] in {"B2", "B5"}


def test_tfold_stepwise_reward_uses_previous_action_affinity():
    from tcrppo_v2.env import TCREditEnv
    from tcrppo_v2.reward_manager import RewardManager
    from tcrppo_v2.utils.constants import OP_INS, OP_SUB

    class LengthAffinityScorer:
        def score_batch_fast(self, tcrs, peptides):
            return [float(len(tcr)) for tcr in tcrs]

        def score(self, tcr, peptide="", **kwargs):
            return float(len(tcr)), 1.0

    env = TCREditEnv(
        esm_cache=FakeESMCache(),
        pmhc_loader=FakePMHCLoader(),
        tcr_pool=FakeTCRPool(),
        reward_manager=RewardManager(
            affinity_scorer=LengthAffinityScorer(),
            reward_mode="tfold_stepwise",
        ),
        reward_mode="tfold_stepwise",
        terminal_reward_only=False,
        max_steps=3,
    )
    env.reset(init_tcr="CASSIRSS", peptide="PEPTIDE")

    _, reward, _, info = env.step((OP_INS, 1, 1))
    assert np.isclose(reward, 1.0)
    assert np.isclose(info["reward_components"]["affinity_step_delta"], 1.0)

    _, reward, _, info = env.step((OP_SUB, 1, 2))
    assert np.isclose(reward, 0.0)
    assert np.isclose(info["reward_components"]["affinity_step_delta"], 0.0)


def test_active_clipping_keeps_best_affinity_prefix_in_rollout_buffer():
    from tcrppo_v2.ppo_trainer import PPOTrainer, RolloutBuffer
    from tcrppo_v2.reward_manager import RewardManager

    class MockAffinityScorer:
        table = {"LOW": 1.0, "HIGH": 5.0, "MID": 3.0}

        def score_batch_fast(self, tcrs, peptides):
            return [self.table[tcr] for tcr in tcrs]

        def score(self, tcr, peptide="", **kwargs):
            return self.table[tcr], 1.0

    class FakeEnv:
        initial_affinity = 0.0
        initial_tcr = "START"
        target = "PEPTIDE"

    class FakeVecEnv:
        envs = [FakeEnv()]

    trainer = PPOTrainer({
        "device": "cpu",
        "n_envs": 1,
        "n_steps": 3,
        "active_clipping": True,
    })
    trainer.reward_manager = RewardManager(
        affinity_scorer=MockAffinityScorer(),
        reward_mode="v2_no_decoy",
        w_affinity=1.0,
        w_naturalness=0.0,
        w_diversity=0.0,
    )
    trainer.buffer = RolloutBuffer(n_steps=3, n_envs=1, obs_dim=2, device="cpu")
    trainer.vec_env = FakeVecEnv()

    for _ in range(3):
        trainer.buffer.add(
            obs=np.zeros((1, 2), dtype=np.float32),
            op=np.zeros(1, dtype=np.int64),
            pos=np.zeros(1, dtype=np.int64),
            tok=np.zeros(1, dtype=np.int64),
            log_prob=np.zeros(1, dtype=np.float32),
            reward=np.zeros(1, dtype=np.float32),
            done=np.zeros(1, dtype=np.float32),
            value=np.zeros(1, dtype=np.float32),
            op_mask=np.ones((1, 4), dtype=bool),
            pos_mask=np.ones((1, 20), dtype=bool),
        )

    result = trainer._apply_active_clipping(0, [
        {"buffer_row": 0, "tcr": "LOW", "peptide": "PEPTIDE"},
        {"buffer_row": 1, "tcr": "HIGH", "peptide": "PEPTIDE"},
        {"buffer_row": 2, "tcr": "MID", "peptide": "PEPTIDE"},
    ])

    assert result["kept_len"] == 2
    assert np.isclose(result["reward"], 5.0)
    assert trainer.buffer.valid[:, 0].tolist() == [True, True, False]
    assert trainer.buffer.dones[:, 0].tolist() == [0.0, 1.0, 0.0]
    assert trainer.buffer.rewards[:, 0].tolist() == [0.0, 5.0, 0.0]
