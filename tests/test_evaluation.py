"""Unit tests for 3-tier evaluation modules."""

import sys
import os
import json
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# --- Sequence Analysis (Tier 3) ---

class TestSequenceAnalysis:
    def test_levenshtein_identical(self):
        from tcrppo_v2.evaluation.sequence_analysis import levenshtein_distance
        assert levenshtein_distance("CASSIRSS", "CASSIRSS") == 0

    def test_levenshtein_one_edit(self):
        from tcrppo_v2.evaluation.sequence_analysis import levenshtein_distance
        assert levenshtein_distance("CASSIRSS", "CASSIRSA") == 1

    def test_levenshtein_empty(self):
        from tcrppo_v2.evaluation.sequence_analysis import levenshtein_distance
        assert levenshtein_distance("", "ABC") == 3
        assert levenshtein_distance("ABC", "") == 3

    def test_kmer_frequencies(self):
        from tcrppo_v2.evaluation.sequence_analysis import compute_kmer_frequencies
        seqs = ["CASSIRSS"]
        freq = compute_kmer_frequencies(seqs, k=3)
        # "CASSIRSS" has 6 3-mers: CAS, ASS, SSI, SIR, IRS, RSS
        assert len(freq) == 6
        assert abs(sum(freq.values()) - 1.0) < 1e-6

    def test_aa_composition(self):
        from tcrppo_v2.evaluation.sequence_analysis import compute_aa_composition
        seqs = ["AAACCC"]
        comp = compute_aa_composition(seqs)
        assert abs(comp["A"] - 0.5) < 1e-6
        assert abs(comp["C"] - 0.5) < 1e-6

    def test_length_distribution(self):
        from tcrppo_v2.evaluation.sequence_analysis import compute_length_distribution
        seqs = ["AAA", "AAA", "AAAA"]
        dist = compute_length_distribution(seqs)
        assert abs(dist[3] - 2/3) < 1e-6
        assert abs(dist[4] - 1/3) < 1e-6

    def test_sequence_analyzer_diversity(self):
        from tcrppo_v2.evaluation.sequence_analysis import SequenceAnalyzer
        analyzer = SequenceAnalyzer(tcrdb_path="/nonexistent", l0_seeds_dir="/nonexistent")
        tcrs = ["CASSIRSSYEQYF", "CASSIRSTYEQYF", "CASSIRSAYEQYF",
                "CASSIRSSYEQYF", "CSVGATNEKLFF"]
        result = analyzer._compute_diversity(tcrs)
        assert result["n_total"] == 5
        assert result["n_unique"] == 4
        assert result["uniqueness_ratio"] == 4/5
        assert result["mean_pairwise_levenshtein"] > 0

    def test_analyze_generated_tcrs(self):
        from tcrppo_v2.evaluation.sequence_analysis import analyze_generated_tcrs
        tcrs = {
            "GILGFVFTL": ["CASSIRSSYEQYF", "CSVGATNEKLFF", "CASSLGQAYEQYF"],
        }
        results = analyze_generated_tcrs(tcrs, tcrdb_path="/nonexistent", l0_seeds_dir="/nonexistent")
        assert "GILGFVFTL" in results
        r = results["GILGFVFTL"]
        assert "diversity" in r
        assert "length_distribution" in r
        assert "aa_composition" in r
        assert "kmer_enrichment" in r
        assert "distance_to_binders" in r


# --- NetTCR Scorer (Tier 2) ---

class TestNetTCRScorer:
    def test_blosum_encoding(self):
        from tcrppo_v2.evaluation.nettcr_scorer import encode_sequences, BLOSUM50_20AA
        seqs = ["CASSIRSS", "GILGFVFTL"]
        encoded = encode_sequences(seqs, max_len=30)
        assert encoded.shape == (2, 30, 20)
        # First position of first sequence should be Cysteine encoding
        np.testing.assert_array_equal(encoded[0, 0], BLOSUM50_20AA["C"])
        # Position beyond sequence should be zeros
        np.testing.assert_array_equal(encoded[0, 8], np.zeros(20))

    def test_build_model(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from tcrppo_v2.evaluation.nettcr_scorer import build_nettcr_beta_model
        model = build_nettcr_beta_model()
        # Model should have 2 inputs (cdr3b, peptide) and 1 output
        assert len(model.inputs) == 2
        assert model.output_shape == (None, 1)

    def test_scorer_loads_weights(self):
        weights_path = os.path.join(PROJECT_ROOT, "data", "nettcr_model.weights.h5")
        if not os.path.exists(weights_path):
            pytest.skip("NetTCR weights not trained yet")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        from tcrppo_v2.evaluation.nettcr_scorer import NetTCRScorer
        scorer = NetTCRScorer(model_path=weights_path)
        # Score a known binder
        score = scorer.score("CASSIRSSYEQYF", "GILGFVFTL")
        assert 0.0 <= score <= 1.0

    def test_scorer_batch(self):
        weights_path = os.path.join(PROJECT_ROOT, "data", "nettcr_model.weights.h5")
        if not os.path.exists(weights_path):
            pytest.skip("NetTCR weights not trained yet")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        from tcrppo_v2.evaluation.nettcr_scorer import NetTCRScorer
        scorer = NetTCRScorer(model_path=weights_path)
        scores = scorer.score_batch(
            ["CASSIRSSYEQYF", "AAAAAAAAAAAA"],
            ["GILGFVFTL", "GILGFVFTL"],
        )
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)
        # Known binder should score higher than random
        assert scores[0] > scores[1], f"Expected binder > random: {scores[0]:.4f} vs {scores[1]:.4f}"


# --- 3-Tier Evaluation Orchestrator ---

class TestEvaluate3Tier:
    def test_load_generated_tcrs(self, tmp_path):
        from tcrppo_v2.evaluation.evaluate_3tier import load_generated_tcrs
        # Create test JSON files
        data = {"generated_tcrs": ["CASSIRSSYEQYF", "CSVGATNEKLFF"]}
        (tmp_path / "GILGFVFTL.json").write_text(json.dumps(data))
        data2 = {"tcrs": ["CASSLGQAYEQYF"]}
        (tmp_path / "NLVPMVATV.json").write_text(json.dumps(data2))

        tcrs = load_generated_tcrs(str(tmp_path))
        assert "GILGFVFTL" in tcrs
        assert len(tcrs["GILGFVFTL"]) == 2
        assert "NLVPMVATV" in tcrs
        assert len(tcrs["NLVPMVATV"]) == 1

    def test_load_nonexistent_dir(self):
        from tcrppo_v2.evaluation.evaluate_3tier import load_generated_tcrs
        tcrs = load_generated_tcrs("/nonexistent/dir")
        assert tcrs == {}

    def test_tier3_runs(self):
        from tcrppo_v2.evaluation.evaluate_3tier import run_tier3_sequence
        tcrs = {
            "GILGFVFTL": ["CASSIRSSYEQYF", "CSVGATNEKLFF", "CASSLGQAYEQYF"],
        }
        results = run_tier3_sequence(tcrs)
        assert "GILGFVFTL" in results
        assert "diversity" in results["GILGFVFTL"]
        assert results["GILGFVFTL"]["diversity"]["n_total"] == 3
