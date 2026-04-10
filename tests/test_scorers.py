"""Unit tests for all scorer modules and reward manager."""

import sys
import os
import json
import pytest
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.utils.constants import ERGO_MODEL_DIR, ERGO_AE_FILE, ERGO_DIR
from tcrppo_v2.utils.encoding import (
    levenshtein_similarity, is_valid_tcr, mutate_sequence, random_aa_sequence
)


# --- Encoding Utils ---

class TestEncodingUtils:
    def test_levenshtein_identical(self):
        assert levenshtein_similarity("CASSIR", "CASSIR") == 1.0

    def test_levenshtein_different(self):
        sim = levenshtein_similarity("CASSIR", "XYZABC")
        assert 0.0 <= sim < 1.0

    def test_levenshtein_empty(self):
        assert levenshtein_similarity("", "") == 1.0

    def test_valid_tcr(self):
        assert is_valid_tcr("CASSIRSSYEQYF")
        assert not is_valid_tcr("CASSIR123")
        assert not is_valid_tcr("CASSIR ssyeqyf")

    def test_mutate_sequence(self):
        rng = np.random.default_rng(42)
        seq = "CASSIRSSYEQYF"
        mutated = mutate_sequence(seq, 3, rng)
        assert len(mutated) == len(seq)
        diffs = sum(a != b for a, b in zip(seq, mutated))
        assert diffs == 3

    def test_random_aa_sequence(self):
        rng = np.random.default_rng(42)
        seq = random_aa_sequence(15, rng)
        assert len(seq) == 15
        assert is_valid_tcr(seq)


# --- Affinity ERGO Scorer ---

class TestAffinityERGO:
    @pytest.fixture(scope="class")
    def scorer(self):
        from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
        model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
        return AffinityERGOScorer(
            model_file=model_file,
            ae_file=ERGO_AE_FILE,
            device="cuda",
            mc_samples=10,
        )

    def test_known_binder_scores_higher(self, scorer):
        """Known GILGFVFTL binder should score higher than random sequences."""
        known_binder = "CASSIRSSYEQYF"  # Known strong binder to GILGFVFTL
        random_tcrs = ["AAAAAAAAAAAA", "WWWWWWWWWWWW", "GGGGGGGGGGG"]
        peptide = "GILGFVFTL"

        binder_score, binder_conf = scorer.score(known_binder, peptide)

        random_scores = []
        for tcr in random_tcrs:
            s, c = scorer.score(tcr, peptide)
            random_scores.append(s)

        mean_random = np.mean(random_scores)
        print(f"\nBinder score: {binder_score:.4f} (conf={binder_conf:.4f})")
        print(f"Random mean: {mean_random:.4f}")
        assert binder_score > mean_random, (
            f"Known binder ({binder_score:.4f}) should score higher "
            f"than random mean ({mean_random:.4f})"
        )

    def test_batch_scoring(self, scorer):
        """Batch scoring should produce same results as individual scoring."""
        tcrs = ["CASSIRSSYEQYF", "CASSLAPGATNEKLFF"]
        peps = ["GILGFVFTL", "GILGFVFTL"]
        scores, confs = scorer.score_batch(tcrs, peps)
        assert len(scores) == 2
        assert len(confs) == 2
        assert all(0.0 <= c <= 1.0 for c in confs)

    def test_mc_dropout_produces_std(self, scorer):
        """MC Dropout should produce non-zero std for most inputs."""
        tcrs = ["CASSIRSSYEQYF", "CASSLGQAYEQYF"]
        peps = ["GILGFVFTL", "GILGFVFTL"]
        means, stds = scorer.mc_dropout_score(tcrs, peps)
        assert len(means) == 2
        assert len(stds) == 2
        # At least one should have nonzero std
        print(f"\nMC means: {means}, stds: {stds}")

    def test_fast_scoring(self, scorer):
        """Fast scoring (no MC Dropout) should work."""
        preds = scorer.score_batch_fast(["CASSIRSSYEQYF"], ["GILGFVFTL"])
        assert len(preds) == 1
        assert 0.0 <= preds[0] <= 1.0


# --- Decoy Scorer ---

class TestDecoyScorer:
    @pytest.fixture(scope="class")
    def affinity_scorer(self):
        from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
        model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
        return AffinityERGOScorer(
            model_file=model_file,
            ae_file=ERGO_AE_FILE,
            device="cuda",
            mc_samples=5,
        )

    @pytest.fixture(scope="class")
    def scorer(self, affinity_scorer):
        from tcrppo_v2.scorers.decoy import DecoyScorer
        return DecoyScorer(
            decoy_library_path="/share/liuyutian/pMHC_decoy_library",
            targets=["GILGFVFTL"],
            affinity_scorer=affinity_scorer,
            K=32,  # Larger K for stable statistics
            tau=10.0,
            rng=np.random.default_rng(42),
        )

    def test_decoy_loading(self, scorer):
        """Should load decoys for GILGFVFTL from all available tiers."""
        decoys = scorer.decoys.get("GILGFVFTL", {})
        print(f"\nDecoy counts: A={len(decoys.get('A', []))}, "
              f"B={len(decoys.get('B', []))}, "
              f"C={len(decoys.get('C', []))}, "
              f"D={len(decoys.get('D', []))}")
        # GILGFVFTL should have tier A decoys
        assert len(decoys.get("A", [])) > 0

    def test_sampling(self, scorer):
        """Should sample K decoys with tier weighting."""
        sampled = scorer.sample_decoys("GILGFVFTL", k=8)
        assert len(sampled) == 8
        assert all(isinstance(s, str) for s in sampled)

    def test_universal_binder_higher_penalty(self, scorer):
        """A TCR that binds everything should get higher decoy penalty."""
        # Use known strong binder as proxy for "universal binder"
        universal_like = "CASSIRSSYEQYF"  # Known strong binder
        # Use a very weak/random TCR
        weak_tcr = "AAAAAAAAAAAA"

        penalty_universal, _ = scorer.score(universal_like, "GILGFVFTL", target="GILGFVFTL")
        penalty_weak, _ = scorer.score(weak_tcr, "GILGFVFTL", target="GILGFVFTL")

        print(f"\nUniversal-like penalty: {penalty_universal:.4f}")
        print(f"Weak TCR penalty: {penalty_weak:.4f}")
        assert penalty_universal > penalty_weak, (
            f"Strong binder ({penalty_universal:.4f}) should have higher "
            f"decoy penalty than weak ({penalty_weak:.4f})"
        )


# --- Diversity Scorer ---

class TestDiversityScorer:
    def test_identical_sequences_penalty(self):
        from tcrppo_v2.scorers.diversity import DiversityScorer
        scorer = DiversityScorer(buffer_size=10, similarity_threshold=0.85)

        # First sequence: no penalty
        s1, _ = scorer.score("CASSIRSSYEQYF")
        assert s1 == 0.0

        # Identical sequence: should get penalty
        s2, _ = scorer.score("CASSIRSSYEQYF")
        assert s2 < 0.0, f"Identical seq should get penalty, got {s2}"
        print(f"\nIdentical penalty: {s2:.4f}")

    def test_diverse_sequences_no_penalty(self):
        from tcrppo_v2.scorers.diversity import DiversityScorer
        scorer = DiversityScorer(buffer_size=10, similarity_threshold=0.85)

        scorer.score("CASSIRSSYEQYF")
        s, _ = scorer.score("AAAAAAAAAAAA")  # Very different
        assert s == 0.0, f"Very different seq should get no penalty, got {s}"

    def test_buffer_reset(self):
        from tcrppo_v2.scorers.diversity import DiversityScorer
        scorer = DiversityScorer(buffer_size=10, similarity_threshold=0.85)

        scorer.score("CASSIRSSYEQYF")
        scorer.reset()
        s, _ = scorer.score("CASSIRSSYEQYF")
        assert s == 0.0  # Buffer was cleared


# --- Reward Manager ---

class TestRewardManager:
    def test_running_normalizer(self):
        from tcrppo_v2.reward_manager import RunningNormalizer
        norm = RunningNormalizer(window=100, warmup=10)

        # During warmup (first 10 values), returns raw values
        for i in range(9):
            v = norm.normalize(float(i))
        # 9th value (0-indexed) is still warmup (buffer has 9 items)
        assert abs(v - 8.0) < 1e-6

        # 10th value triggers warmup completion
        v = norm.normalize(10.0)
        assert norm.is_warmed_up

        # After warmup, should be z-scored
        for i in range(100):
            v = norm.normalize(50.0)

        # After many identical values, normalized should be ~0
        result = norm.normalize(50.0)
        print(f"\nNormalized constant=50.0 -> {result:.4f}")
        assert abs(result) < 0.5

    def test_reward_mode_v1(self):
        """v1_ergo_only mode should only use raw affinity."""
        from tcrppo_v2.reward_manager import RewardManager

        class MockScorer:
            def score(self, tcr, peptide="", **kwargs):
                return 0.75, 1.0

        rm = RewardManager(
            affinity_scorer=MockScorer(),
            reward_mode="v1_ergo_only",
        )
        total, comp = rm.compute_reward("CASSIR", "GILGFVFTL")
        assert abs(total - 0.75) < 1e-6, f"v1 mode should return raw affinity, got {total}"

    def test_reward_mode_v2_full(self):
        """v2_full mode should combine all components."""
        from tcrppo_v2.reward_manager import RewardManager

        class MockAffinityScorer:
            def score(self, tcr, peptide="", **kwargs):
                return 0.8, 1.0

        class MockDecoyScorer:
            def score(self, tcr, peptide="", **kwargs):
                return 0.5, 1.0

        class MockNatScorer:
            def score(self, tcr, peptide="", **kwargs):
                return -0.1, 1.0

        class MockDivScorer:
            def score(self, tcr, peptide="", **kwargs):
                return 0.0, 1.0

        rm = RewardManager(
            affinity_scorer=MockAffinityScorer(),
            decoy_scorer=MockDecoyScorer(),
            naturalness_scorer=MockNatScorer(),
            diversity_scorer=MockDivScorer(),
            reward_mode="v2_full",
            norm_warmup=1,  # Skip warmup for test
        )

        total, comp = rm.compute_reward("CASSIR", "GILGFVFTL", initial_affinity=0.5)
        assert "total" in comp
        assert "affinity_raw" in comp
        assert "decoy_raw" in comp
        print(f"\nv2_full total: {total:.4f}, components: { {k: round(v, 4) for k, v in comp.items()} }")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
