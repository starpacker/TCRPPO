"""Unit tests for data pipeline modules."""

import sys
import os
import json
import pytest
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# --- pMHC Loader ---

class TestPMHCLoader:
    def test_default_targets(self):
        from tcrppo_v2.data.pmhc_loader import PMHCLoader
        loader = PMHCLoader()
        targets = loader.get_target_list()
        assert len(targets) == 12
        assert "GILGFVFTL" in targets

    def test_train_mode_targets(self):
        from tcrppo_v2.data.pmhc_loader import PMHCLoader
        loader = PMHCLoader(mode="train")
        targets = loader.get_target_list()
        # tc-hard provides ~163 targets, should be well over 100
        assert len(targets) > 100, f"Expected >100 train targets, got {len(targets)}"
        # All 12 eval targets must be included
        for t in ["GILGFVFTL", "NLVPMVATV", "GLCTLVAML", "KLGGALQAK"]:
            assert t in targets

    def test_hla_assignment(self):
        from tcrppo_v2.data.pmhc_loader import PMHCLoader
        loader = PMHCLoader()
        info = loader.get_target_info("GILGFVFTL")
        assert info["hla"] == "HLA-A*02:01"
        assert len(info["pseudoseq"]) > 0

    def test_pmhc_string(self):
        from tcrppo_v2.data.pmhc_loader import PMHCLoader
        loader = PMHCLoader()
        pmhc = loader.get_pmhc_string("GILGFVFTL")
        assert pmhc.startswith("GILGFVFTL")
        assert len(pmhc) > len("GILGFVFTL")  # Has pseudoseq appended

    def test_sample_target(self):
        from tcrppo_v2.data.pmhc_loader import PMHCLoader
        loader = PMHCLoader()
        rng = np.random.default_rng(42)
        target = loader.sample_target(rng)
        assert target in loader.get_target_list()

    def test_weighted_sampling(self):
        from tcrppo_v2.data.pmhc_loader import PMHCLoader
        loader = PMHCLoader()
        rng = np.random.default_rng(42)
        # Give GILGFVFTL very high weight
        weights = {t: 0.01 for t in loader.get_target_list()}
        weights["GILGFVFTL"] = 100.0
        counts = {}
        for _ in range(100):
            t = loader.sample_target_weighted(weights, rng)
            counts[t] = counts.get(t, 0) + 1
        assert counts.get("GILGFVFTL", 0) > 80


# --- TCR Pool ---

class TestTCRPool:
    @pytest.fixture(scope="class")
    def pool(self):
        from tcrppo_v2.data.tcr_pool import TCRPool
        return TCRPool(seed=42)

    def test_tcrdb_loading(self, pool):
        assert pool.num_tcrdb_seqs > 100000
        print(f"\nTCRdb loaded: {pool.num_tcrdb_seqs} sequences")

    def test_random_sampling(self, pool):
        tcr = pool.get_random_tcr()
        assert 8 <= len(tcr) <= 27
        assert all(c in "ARNDCQEGHILKMFPSTWYV" for c in tcr)

    def test_curriculum_weights(self, pool):
        w0 = pool.get_curriculum_weights(0)
        assert w0 == (0.7, 0.0, 0.3)

        w1 = pool.get_curriculum_weights(2_000_000)
        assert w1 == (0.4, 0.0, 0.6)

        w2 = pool.get_curriculum_weights(7_000_000)
        assert w2 == (0.1, 0.0, 0.9)

    def test_l2_sampling(self, pool):
        """Without L0/L1 seeds, should fall back to L2."""
        tcr, level = pool.sample_tcr("GILGFVFTL", step=0)
        assert level == "L2"  # No L0/L1 loaded
        assert 8 <= len(tcr) <= 27

    def test_no_curriculum_mode(self, pool):
        """v2_no_curriculum always uses L2."""
        tcr, level = pool.sample_tcr(
            "GILGFVFTL", step=0, reward_mode="v2_no_curriculum"
        )
        assert level == "L2"

    def test_l0_from_tchard(self):
        """L0 seeds load from tc-hard directory."""
        from tcrppo_v2.data.tcr_pool import TCRPool
        import os
        pool = TCRPool(seed=42)
        l0_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "l0_seeds_tchard"
        )
        if os.path.isdir(l0_dir):
            pool.load_l0_from_dir(l0_dir)
            l0_targets = pool.get_l0_targets()
            assert len(l0_targets) > 50, f"Expected >50 L0 targets from tc-hard, got {len(l0_targets)}"
            # GILGFVFTL should have lots of L0 seeds
            assert "GILGFVFTL" in l0_targets
            assert len(pool.l0_seeds["GILGFVFTL"]) > 100


# --- Decoy Sampler ---

class TestDecoySampler:
    @pytest.fixture(scope="class")
    def sampler(self):
        from tcrppo_v2.data.decoy_sampler import DecoySampler
        return DecoySampler(targets=["GILGFVFTL", "NLVPMVATV"])

    def test_tier_c_loading(self, sampler):
        assert sampler.get_tier_c_count() > 0
        print(f"\nTier C global: {sampler.get_tier_c_count()} decoys")

    def test_target_decoy_loading(self, sampler):
        tiers = sampler.get_available_tiers("GILGFVFTL")
        print(f"\nGILGFVFTL tiers: {tiers}")
        # GILGFVFTL should have at least tier C
        assert "C" in tiers

    def test_sampling(self, sampler):
        decoys = sampler.sample_decoys("GILGFVFTL", k=16)
        assert len(decoys) == 16
        assert all(isinstance(s, str) for s in decoys)

    def test_unlock_schedule(self, sampler):
        sampler.update_unlocked_tiers(0)
        assert sampler.unlocked_tiers == {"A"}

        sampler.update_unlocked_tiers(3_000_000)
        assert sampler.unlocked_tiers == {"A", "B"}

        sampler.update_unlocked_tiers(10_000_000)
        assert sampler.unlocked_tiers == {"A", "B", "D", "C"}

        # Reset for other tests
        sampler.update_unlocked_tiers(0)

    def test_all_tiers_sampling(self, sampler):
        """When all tiers unlocked, should sample from all available."""
        sampler.update_unlocked_tiers(10_000_000)
        decoys = sampler.sample_decoys("GILGFVFTL", k=64)
        assert len(decoys) == 64
        # Reset
        sampler.update_unlocked_tiers(0)


# --- ESM Cache ---

class TestESMCache:
    @pytest.fixture(scope="class")
    def cache(self):
        from tcrppo_v2.utils.esm_cache import ESMCache
        return ESMCache(device="cuda", tcr_cache_size=32)

    def test_encode_sequence(self, cache):
        emb = cache.encode_sequence("CASSIRSSYEQYF")
        assert emb.shape == (cache.embed_dim,)
        assert not emb.isnan().any()
        print(f"\nESM embed dim: {cache.embed_dim}")

    def test_tcr_caching(self, cache):
        import time
        seq = "CASSIRSSYEQYF"

        # First call - compute
        t0 = time.time()
        emb1 = cache.encode_tcr(seq)
        t1 = time.time() - t0

        # Second call - from cache
        t0 = time.time()
        emb2 = cache.encode_tcr(seq)
        t2 = time.time() - t0

        assert torch.allclose(emb1, emb2)
        assert cache.tcr_cache_size_current >= 1
        print(f"\nFirst call: {t1:.4f}s, Cached call: {t2:.6f}s")

    def test_pmhc_caching(self, cache):
        pmhc = "GILGFVFTLYFAMYQENAAHTLRWEPYSEGAEYLERTCEW"
        emb1 = cache.encode_pmhc(pmhc)
        emb2 = cache.encode_pmhc(pmhc)
        assert torch.allclose(emb1, emb2)
        assert cache.pmhc_cache_size_current >= 1

    def test_batch_encoding(self, cache):
        seqs = ["CASSIRSSYEQYF", "CASSLGQAYEQYF", "AAAAAAAAAAAA"]
        embeddings = cache.encode_tcr_batch(seqs)
        assert embeddings.shape == (3, cache.embed_dim)
        assert not embeddings.isnan().any()

    def test_different_sequences_different_embeddings(self, cache):
        emb1 = cache.encode_tcr("CASSIRSSYEQYF")
        emb2 = cache.encode_tcr("AAAAAAAAAAAA")
        # Different sequences should have different embeddings
        assert not torch.allclose(emb1, emb2)

    def test_cache_eviction(self, cache):
        """Fill cache beyond capacity, verify eviction works."""
        cache.clear_tcr_cache()
        # Cache size is 32, insert 35 sequences
        for i in range(35):
            seq = "C" + "A" * (i % 20 + 5) + "F"
            cache.encode_tcr(seq)
        assert cache.tcr_cache_size_current <= 32


# Need torch import for assertions
import torch


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
