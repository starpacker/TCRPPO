#!/usr/bin/env python
"""Baseline experiments: Random search vs Genetic Algorithm vs RL (trace73).

Three baselines for TCR design against tfold_excellent_peptides:
  1. Pure random CDR3β generation (random amino acid strings)
  2. TCRdb random sampling (real CDR3β from the database)
  3. Genetic Algorithm (GA) — evolve a population of TCRdb CDR3β using
     mutation, crossover, and tournament selection guided by tFold affinity

Usage:
    # Use the trace73 tfold server (already running on GPU 3)
    python scripts/baseline_random_and_ga.py \
        --mode all \
        --peptides data/tfold_excellent_peptides.txt \
        --n_random 5000 \
        --ga_pop_size 100 \
        --ga_generations 50 \
        --tfold_socket /tmp/tfold_server_trace73_curriculum_exploration.sock \
        --output_dir results/baseline_random_ga

    # Or use the AMP scorer directly (needs a free GPU)
    python scripts/baseline_random_and_ga.py \
        --mode all \
        --use_amp --gpu 1 \
        --cache_path data/tfold_feature_cache_baseline.db
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Suppress verbose tFoldScore per-sample logs
logging.getLogger("tcrppo_v2.scorers.affinity_tfold").setLevel(logging.WARNING)
logging.getLogger("tcrppo_v2.scorers.affinity_tfold_amp").setLevel(logging.WARNING)

# Monkey-patch print to always flush
import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
TCRDB_PATH = "/share/liuyutian/TCRPPO/data/tcrdb/train_uniq_tcr_seqs.txt"


# ============================================================================
# Scorer wrapper
# ============================================================================

class TFoldScorerWrapper:
    """Wraps either AMP scorer or socket-based scorer."""

    def __init__(self, use_amp: bool = False, gpu: int = 1,
                 socket_path: str = "/tmp/tfold_server_baseline.sock",
                 cache_path: str = "data/tfold_feature_cache_baseline.db"):
        self.n_scored = 0
        self.n_failed = 0
        self.total_time = 0.0
        self.default_score = -20.0

        if use_amp:
            from tcrppo_v2.scorers.affinity_tfold_amp import AffinityTFoldAMPScorer
            self.scorer = AffinityTFoldAMPScorer(
                device=f"cuda:{gpu}" if gpu >= 0 else "cpu",
                gpu_id=gpu,
                cache_path=cache_path,
                fallback_to_subprocess=False,  # No subprocess fallback
            )
            print(f"[Scorer] AMP scorer on GPU {gpu}, cache: {cache_path}")
        else:
            from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer
            self.scorer = AffinityTFoldScorer(
                device="cuda",
                server_socket_path=socket_path,
                cache_path=cache_path,
            )
            print(f"[Scorer] Socket scorer via {socket_path}, cache: {cache_path}")

    def score_batch(self, tcrs: List[str], peptides: List[str]) -> List[float]:
        """Score a batch of (tcr, peptide) pairs. Returns list of affinity logits.
        Handles failures gracefully by scoring one-by-one on batch failure."""
        t0 = time.time()
        try:
            scores = self.scorer.score_batch_fast(tcrs, peptides)
        except Exception as e:
            print(f"  [Scorer] Batch failed ({e}), scoring one-by-one...")
            scores = []
            for tcr, pep in zip(tcrs, peptides):
                try:
                    s = self.scorer.score_batch_fast([tcr], [pep])[0]
                except Exception:
                    s = self.default_score
                    self.n_failed += 1
                scores.append(s)
        dt = time.time() - t0
        self.n_scored += len(tcrs)
        self.total_time += dt
        if self.n_scored % 100 == 0 or len(tcrs) >= 32:
            print(f"  [Progress] {self.n_scored} scored, "
                  f"{self.n_failed} failed, "
                  f"{self.total_time:.1f}s elapsed, "
                  f"{self.total_time/max(self.n_scored,1)*1000:.1f}ms/sample")
        return scores

    def score_single(self, tcr: str, peptide: str) -> float:
        return self.score_batch([tcr], [peptide])[0]


# ============================================================================
# CDR3β generators
# ============================================================================

def load_tcrdb(path: str = TCRDB_PATH, max_seqs: int = 100000) -> List[str]:
    """Load CDR3β sequences from TCRdb."""
    seqs = []
    with open(path) as f:
        for line in f:
            seq = line.strip()
            if 8 <= len(seq) <= 20 and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
                seqs.append(seq)
                if len(seqs) >= max_seqs:
                    break
    print(f"[TCRdb] Loaded {len(seqs)} CDR3β sequences")
    return seqs


def generate_random_cdr3(n: int, min_len: int = 10, max_len: int = 18,
                          rng: np.random.Generator = None) -> List[str]:
    """Generate random CDR3β sequences (start with C, random AAs)."""
    if rng is None:
        rng = np.random.default_rng(42)
    seqs = []
    for _ in range(n):
        length = rng.integers(min_len, max_len + 1)
        # CDR3β typically starts with C
        body = "".join(rng.choice(AMINO_ACIDS) for _ in range(length - 1))
        seqs.append("C" + body)
    return seqs


def sample_tcrdb(tcrdb_seqs: List[str], n: int,
                  rng: np.random.Generator = None) -> List[str]:
    """Randomly sample n CDR3β from TCRdb."""
    if rng is None:
        rng = np.random.default_rng(42)
    indices = rng.choice(len(tcrdb_seqs), size=min(n, len(tcrdb_seqs)), replace=False)
    return [tcrdb_seqs[i] for i in indices]


# ============================================================================
# Genetic Algorithm
# ============================================================================

def mutate_cdr3(seq: str, n_mutations: int = 1,
                rng: np.random.Generator = None) -> str:
    """Point-mutate a CDR3β sequence (substitute random positions)."""
    if rng is None:
        rng = np.random.default_rng()
    seq = list(seq)
    # Don't mutate position 0 (conserved C)
    mutable = list(range(1, len(seq)))
    if not mutable:
        return "".join(seq)
    positions = rng.choice(mutable, size=min(n_mutations, len(mutable)), replace=False)
    for pos in positions:
        seq[pos] = rng.choice(AMINO_ACIDS)
    return "".join(seq)


def crossover_cdr3(parent1: str, parent2: str,
                    rng: np.random.Generator = None) -> str:
    """Single-point crossover between two CDR3β sequences."""
    if rng is None:
        rng = np.random.default_rng()
    # Use shorter length as the limit
    min_len = min(len(parent1), len(parent2))
    if min_len <= 2:
        return parent1
    # Crossover point (keep C at position 0)
    xp = rng.integers(1, min_len)
    child = parent1[:xp] + parent2[xp:]
    # Trim if too long
    if len(child) > 20:
        child = child[:20]
    return child


def run_genetic_algorithm(
    scorer: TFoldScorerWrapper,
    peptide: str,
    tcrdb_seqs: List[str],
    pop_size: int = 100,
    n_generations: int = 50,
    mutation_rate: float = 0.8,
    crossover_rate: float = 0.3,
    n_mutations: int = 2,
    tournament_size: int = 5,
    elite_size: int = 5,
    rng: np.random.Generator = None,
) -> Dict:
    """Run GA to optimize TCR affinity for a single peptide.

    Returns dict with history and best TCR found.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Initialize population from TCRdb
    pop_indices = rng.choice(len(tcrdb_seqs), size=pop_size, replace=False)
    population = [tcrdb_seqs[i] for i in pop_indices]

    # Score initial population
    peptides_batch = [peptide] * len(population)
    fitness = scorer.score_batch(population, peptides_batch)

    history = {
        "best_fitness": [],
        "mean_fitness": [],
        "median_fitness": [],
        "n_positive": [],  # affinity > 0.0
        "best_tcr": [],
        "total_evals": 0,
    }

    best_overall_fitness = max(fitness)
    best_overall_tcr = population[int(np.argmax(fitness))]
    total_evals = len(population)

    for gen in range(n_generations):
        gen_best = max(fitness)
        gen_mean = float(np.mean(fitness))
        gen_median = float(np.median(fitness))
        n_positive = sum(1 for f in fitness if f > 0.0)

        history["best_fitness"].append(float(gen_best))
        history["mean_fitness"].append(gen_mean)
        history["median_fitness"].append(gen_median)
        history["n_positive"].append(n_positive)
        history["best_tcr"].append(population[int(np.argmax(fitness))])

        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
            best_overall_tcr = population[int(np.argmax(fitness))]

        if gen % 10 == 0 or n_positive > 0:
            print(f"  Gen {gen:3d} | Best={gen_best:.4f} Mean={gen_mean:.4f} "
                  f"Median={gen_median:.4f} #Pos={n_positive} "
                  f"Overall_Best={best_overall_fitness:.4f}")

        # Selection + reproduction
        new_population = []

        # Elitism: keep top-k
        sorted_idx = np.argsort(fitness)[::-1]
        for i in range(elite_size):
            new_population.append(population[sorted_idx[i]])

        while len(new_population) < pop_size:
            # Tournament selection
            t_idx = rng.choice(len(population), size=tournament_size, replace=False)
            t_fitness = [fitness[i] for i in t_idx]
            winner_idx = t_idx[int(np.argmax(t_fitness))]
            parent = population[winner_idx]

            # Crossover
            if rng.random() < crossover_rate:
                t_idx2 = rng.choice(len(population), size=tournament_size, replace=False)
                t_fitness2 = [fitness[i] for i in t_idx2]
                parent2_idx = t_idx2[int(np.argmax(t_fitness2))]
                child = crossover_cdr3(parent, population[parent2_idx], rng)
            else:
                child = parent

            # Mutation
            if rng.random() < mutation_rate:
                n_mut = rng.integers(1, n_mutations + 1)
                child = mutate_cdr3(child, n_mutations=n_mut, rng=rng)

            new_population.append(child)

        population = new_population

        # Score new population
        peptides_batch = [peptide] * len(population)
        fitness = scorer.score_batch(population, peptides_batch)
        total_evals += len(population)

    # Final stats
    final_best = max(fitness)
    if final_best > best_overall_fitness:
        best_overall_fitness = final_best
        best_overall_tcr = population[int(np.argmax(fitness))]

    history["best_fitness"].append(float(max(fitness)))
    history["mean_fitness"].append(float(np.mean(fitness)))
    history["median_fitness"].append(float(np.median(fitness)))
    history["n_positive"].append(sum(1 for f in fitness if f > 0.0))
    history["best_tcr"].append(population[int(np.argmax(fitness))])
    history["total_evals"] = total_evals

    return {
        "best_fitness": float(best_overall_fitness),
        "best_tcr": best_overall_tcr,
        "history": history,
        "final_population_top10": [
            {"tcr": population[i], "fitness": float(fitness[i])}
            for i in np.argsort(fitness)[::-1][:10]
        ],
    }


# ============================================================================
# Random search with early-stop tracking
# ============================================================================

def run_random_search(
    scorer: TFoldScorerWrapper,
    peptide: str,
    tcr_sequences: List[str],
    batch_size: int = 64,
    label: str = "random",
    best_file: Optional[str] = None,
) -> Dict:
    """Score all TCR sequences against a peptide, tracking when first positive is found.

    Returns summary dict.
    """
    all_scores = []
    first_positive_idx = None
    best_score = -999.0
    best_tcr = ""
    last_best_update = 0

    for start in range(0, len(tcr_sequences), batch_size):
        batch_tcrs = tcr_sequences[start:start + batch_size]
        batch_peps = [peptide] * len(batch_tcrs)
        scores = scorer.score_batch(batch_tcrs, batch_peps)

        for j, s in enumerate(scores):
            idx = start + j
            all_scores.append(s)
            if s > best_score:
                best_score = s
                best_tcr = batch_tcrs[j]
                last_best_update = idx
                # Write to best file immediately
                if best_file:
                    with open(best_file, "w") as f:
                        f.write(f"Peptide: {peptide}\n")
                        f.write(f"Label: {label}\n")
                        f.write(f"Progress: {idx+1}/{len(tcr_sequences)} ({(idx+1)/len(tcr_sequences)*100:.2f}%)\n")
                        f.write(f"Best Score: {best_score:.6f}\n")
                        f.write(f"Best TCR: {best_tcr}\n")
                        f.write(f"Found at index: {last_best_update}\n")
                        f.write(f"Positive samples: {sum(1 for x in all_scores if x > 0.0)}/{len(all_scores)}\n")
                        f.write(f"Mean score: {np.mean(all_scores):.6f}\n")
                        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if s > 0.0 and first_positive_idx is None:
                first_positive_idx = idx
                print(f"  [{label}] First positive at idx={idx}: "
                      f"score={s:.4f} tcr={batch_tcrs[j]}")

        # Print periodic summary every 500 samples
        if (start + len(batch_tcrs)) % 500 == 0 or (start + len(batch_tcrs)) >= len(tcr_sequences):
            n_done = len(all_scores)
            n_pos = sum(1 for x in all_scores if x > 0.0)
            print(f"  [{label}] Progress: {n_done}/{len(tcr_sequences)} | "
                  f"Best={best_score:.4f} (at idx={last_best_update}) | "
                  f"#Positive={n_pos} ({n_pos/n_done*100:.2f}%)")

    scores_arr = np.array(all_scores)
    return {
        "label": label,
        "peptide": peptide,
        "n_total": len(all_scores),
        "best_score": float(best_score),
        "best_tcr": best_tcr,
        "mean_score": float(scores_arr.mean()),
        "median_score": float(np.median(scores_arr)),
        "std_score": float(scores_arr.std()),
        "n_positive": int((scores_arr > 0.0).sum()),
        "pct_positive": float((scores_arr > 0.0).mean() * 100),
        "first_positive_idx": first_positive_idx,
        "percentiles": {
            "p5": float(np.percentile(scores_arr, 5)),
            "p25": float(np.percentile(scores_arr, 25)),
            "p50": float(np.percentile(scores_arr, 50)),
            "p75": float(np.percentile(scores_arr, 75)),
            "p95": float(np.percentile(scores_arr, 95)),
            "p99": float(np.percentile(scores_arr, 99)),
            "max": float(scores_arr.max()),
        },
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline: Random + GA vs RL")
    parser.add_argument("--mode", default="all", choices=["random", "tcrdb", "ga", "all"])
    parser.add_argument("--peptides", default="data/tfold_excellent_peptides.txt")
    parser.add_argument("--n_random", type=int, default=5000,
                        help="Number of random CDR3β to generate")
    parser.add_argument("--n_tcrdb", type=int, default=5000,
                        help="Number of TCRdb sequences to sample")
    parser.add_argument("--ga_pop_size", type=int, default=100)
    parser.add_argument("--ga_generations", type=int, default=50)
    parser.add_argument("--ga_n_mutations", type=int, default=2)
    parser.add_argument("--ga_elite_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--tfold_socket", default="/tmp/tfold_server_trace73_curriculum_exploration.sock")
    parser.add_argument("--cache_path", default="data/tfold_feature_cache_baseline.db")
    parser.add_argument("--output_dir", default="results/baseline_random_ga")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_peptides", type=int, default=None,
                        help="Only test first N peptides (for quick test)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load peptides
    with open(args.peptides) as f:
        peptides = [line.strip() for line in f if line.strip()]
    if args.n_peptides:
        peptides = peptides[:args.n_peptides]
    print(f"\n{'='*70}")
    print(f"Baseline experiment: {len(peptides)} peptides")
    print(f"Peptides: {peptides}")
    print(f"{'='*70}\n")

    # Initialize scorer
    scorer = TFoldScorerWrapper(
        use_amp=args.use_amp,
        gpu=args.gpu,
        socket_path=args.tfold_socket,
        cache_path=args.cache_path,
    )

    # Load TCRdb
    tcrdb_seqs = load_tcrdb(max_seqs=max(args.n_tcrdb, args.ga_pop_size * 2))

    results = {"random": {}, "tcrdb": {}, "ga": {}}
    run_modes = [args.mode] if args.mode != "all" else ["random", "tcrdb", "ga"]

    for peptide in peptides:
        print(f"\n{'='*60}")
        print(f"Peptide: {peptide}")
        print(f"{'='*60}")

        # --- Baseline 1: Pure random CDR3β ---
        if "random" in run_modes:
            print(f"\n[1] Pure random CDR3β ({args.n_random} sequences)...")
            random_tcrs = generate_random_cdr3(args.n_random, rng=rng)
            best_file = os.path.join(args.output_dir, f"best_{peptide}_random.txt")
            result = run_random_search(scorer, peptide, random_tcrs,
                                       batch_size=args.batch_size, label="pure_random",
                                       best_file=best_file)
            results["random"][peptide] = result
            print(f"  Best={result['best_score']:.4f} Mean={result['mean_score']:.4f} "
                  f"#Positive={result['n_positive']}/{result['n_total']} "
                  f"({result['pct_positive']:.1f}%)")
            if result['first_positive_idx'] is not None:
                print(f"  ✅ First positive found at attempt #{result['first_positive_idx']+1}")
            else:
                print(f"  ❌ No positive found in {result['n_total']} attempts")

        # --- Baseline 2: TCRdb random sampling ---
        if "tcrdb" in run_modes:
            print(f"\n[2] TCRdb random sampling ({args.n_tcrdb} sequences)...")
            tcrdb_sample = sample_tcrdb(tcrdb_seqs, args.n_tcrdb, rng=rng)
            best_file = os.path.join(args.output_dir, f"best_{peptide}_tcrdb.txt")
            result = run_random_search(scorer, peptide, tcrdb_sample,
                                       batch_size=args.batch_size, label="tcrdb_random",
                                       best_file=best_file)
            results["tcrdb"][peptide] = result
            print(f"  Best={result['best_score']:.4f} Mean={result['mean_score']:.4f} "
                  f"#Positive={result['n_positive']}/{result['n_total']} "
                  f"({result['pct_positive']:.1f}%)")
            if result['first_positive_idx'] is not None:
                print(f"  ✅ First positive found at attempt #{result['first_positive_idx']+1}")
            else:
                print(f"  ❌ No positive found in {result['n_total']} attempts")

        # --- Baseline 3: Genetic Algorithm ---
        if "ga" in run_modes:
            print(f"\n[3] Genetic Algorithm (pop={args.ga_pop_size}, "
                  f"gen={args.ga_generations})...")
            ga_result = run_genetic_algorithm(
                scorer=scorer,
                peptide=peptide,
                tcrdb_seqs=tcrdb_seqs,
                pop_size=args.ga_pop_size,
                n_generations=args.ga_generations,
                n_mutations=args.ga_n_mutations,
                elite_size=args.ga_elite_size,
                rng=np.random.default_rng(args.seed),
            )
            results["ga"][peptide] = ga_result
            print(f"  Best={ga_result['best_fitness']:.4f} "
                  f"(after {ga_result['history']['total_evals']} evals)")
            n_pos = ga_result["history"]["n_positive"][-1]
            print(f"  Final #Positive in pop: {n_pos}/{args.ga_pop_size}")
            if ga_result['best_fitness'] > 0.0:
                print(f"  ✅ GA found positive: {ga_result['best_tcr']} "
                      f"→ {ga_result['best_fitness']:.4f}")
            else:
                print(f"  ❌ GA best is still negative: {ga_result['best_fitness']:.4f}")

    # ========================================================================
    # Summary table
    # ========================================================================
    print(f"\n\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")

    header = f"{'Peptide':<20} | {'Random Best':>12} {'#Pos':>5} | " \
             f"{'TCRdb Best':>12} {'#Pos':>5} | " \
             f"{'GA Best':>12} {'GA #Pos':>7} {'GA Evals':>9}"
    print(header)
    print("-" * len(header))

    for peptide in peptides:
        r_rand = results["random"].get(peptide, {})
        r_tcrdb = results["tcrdb"].get(peptide, {})
        r_ga = results["ga"].get(peptide, {})

        rand_best = f"{r_rand.get('best_score', float('nan')):.4f}" if r_rand else "N/A"
        rand_npos = str(r_rand.get('n_positive', '-')) if r_rand else "-"
        tcrdb_best = f"{r_tcrdb.get('best_score', float('nan')):.4f}" if r_tcrdb else "N/A"
        tcrdb_npos = str(r_tcrdb.get('n_positive', '-')) if r_tcrdb else "-"
        ga_best = f"{r_ga.get('best_fitness', float('nan')):.4f}" if r_ga else "N/A"
        ga_npos = str(r_ga.get('history', {}).get('n_positive', ['-'])[-1]) if r_ga else "-"
        ga_evals = str(r_ga.get('history', {}).get('total_evals', '-')) if r_ga else "-"

        print(f"{peptide:<20} | {rand_best:>12} {rand_npos:>5} | "
              f"{tcrdb_best:>12} {tcrdb_npos:>5} | "
              f"{ga_best:>12} {ga_npos:>7} {ga_evals:>9}")

    # ========================================================================
    # Comparison with trace73
    # ========================================================================
    print(f"\n\nComparison context:")
    print(f"  trace73 RL: current mean affinity ≈ -0.87 (best episodes ≈ -0.21)")
    print(f"  trace73 uses 8 editing steps from TCRdb seed → final TCR")
    print(f"  Positive affinity (> 0.0) = predicted binding")

    # Total scoring stats
    print(f"\n\nScoring stats:")
    print(f"  Total scored: {scorer.n_scored}")
    print(f"  Total time: {scorer.total_time:.1f}s")
    if scorer.n_scored > 0:
        print(f"  Avg time per sample: {scorer.total_time / scorer.n_scored * 1000:.1f}ms")

    # Save results
    output_path = os.path.join(args.output_dir, "baseline_results.json")
    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=to_serializable)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
