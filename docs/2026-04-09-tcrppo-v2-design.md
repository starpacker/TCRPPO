# TCRPPO v2 Design Specification

**Date:** 2026-04-09
**Status:** Draft
**Scope:** Complete redesign of TCR generation pipeline with contrastive reward, indel actions, ESM state encoder, and curriculum learning.

---

## 1. Motivation

The current TCRPPO (v1) achieves mean AUROC = 0.45 across 12 targets on decoy specificity evaluation -- worse than random. Root causes:

1. **Reward only optimizes on-target ERGO score** -- no specificity constraint. PPO learns a "universal binder" shortcut.
2. **Delayed reward** -- only terminal reward after 8 steps. Credit assignment is broken.
3. **Weak state encoder** -- LSTM + BLOSUM/OneHot lacks deep biochemical understanding.
4. **No indel** -- fixed-length substitution only, CDR3 length diversity is cut off.
5. **Random initialization** -- 99.9% of starting TCRs have zero affinity. Wasted exploration.

## 2. Architecture Overview

### 2.1 Directory Layout

```
tcrppo_v2/                     # Self-contained, zero imports from code/
  env.py                       # Gym environment: pMHC state + indel + STOP
  policy.py                    # Actor-Critic with ESM-2 frozen state encoder
  ppo_trainer.py               # Training entrypoint + online specificity callback
  test_tcrs.py                 # Inference / generation script
  scorers/
    base.py                    # Abstract Scorer interface: score() -> (float, float)
    affinity_ergo.py           # ERGO binding predictor (retained from v1) + MC Dropout confidence
    decoy.py                   # LogSumExp contrastive penalty over decoy library
    naturalness.py             # ESM perplexity with CDR3 z-score normalization
    diversity.py               # Recent-buffer similarity penalty
  reward_manager.py            # Combine scorers + running mean/std normalization
  data/
    pmhc_loader.py             # pMHC loading + HLA pseudosequence encoding (34 AAs)
    tcr_pool.py                # TCRdb loading + curriculum sampler (L0/L1/L2)
    decoy_sampler.py           # Tiered decoy sampling (A>B>D>C) with unlocking
  utils/
    constants.py               # AA alphabet, max_len, PAD token, etc.
    encoding.py                # BLOSUM, seq2num, num2seq (copied from code/)
    esm_cache.py               # Frozen ESM inference with caching
  configs/
    default.yaml               # All hyperparameters in one place
```

**Key principle:** `tcrppo_v2/` does not import from `code/`. Small utilities (BLOSUM dict, AA alphabet, seq/num conversion) are copied as constants. This prevents contamination from v1's load-bearing quirks.

### 2.2 Data Flow (One Episode)

```
reset():
  1. curriculum_sampler -> pick (target_pMHC, init_TCR) based on L0/L1/L2 schedule
  2. pmhc_loader -> encode HLA pseudosequence (34 AAs) + peptide via ESM-2 (cached per target)
  3. esm_cache -> encode init_TCR via ESM-2
  4. state = [TCR_emb | pMHC_emb | remaining_steps | cumulative_reward]

step(action):
  1. decode action -> (op_type, position, token)
  2. apply edit to TCR string (SUB/INS/DEL/STOP)
  3. re-encode TCR via ESM-2
  4. reward_manager.score(tcr, peptide, decoys) -> delta reward
  5. state = [new_TCR_emb | pMHC_emb | remaining_steps | cumulative_reward]
  6. if STOP or max_steps -> terminal
```

## 3. Action Space: Two-Head Autoregressive with Indel + STOP

### 3.1 Action Heads

```
Head 1: op_type in {SUB=0, INS=1, DEL=2, STOP=3}    # 4-way categorical
Head 2: position in [0, L-1]                           # L-way categorical
Head 3: token in [0, 19]                               # 20-way categorical
                                                        # (only sampled if op_type in {SUB, INS})
```

Autoregressive: Head 2 conditioned on Head 1. Head 3 conditioned on Head 1 + Head 2. If `op_type = DEL` or `STOP`, Head 3 is skipped (not sampled, not in log-prob).

### 3.2 Action Masking (Hard Constraints)

| Condition | Mask |
|---|---|
| `len(seq) >= max_len (27)` | INS logit -> `-inf` |
| `len(seq) <= min_len (8)` | DEL logit -> `-inf` |
| `position >= len(seq)` (pointing at PAD) | position logit -> `-inf` |
| `step == 0` | STOP logit -> `-inf` (must edit at least once) |

### 3.3 Sequence Editing

- **SUB**: Replace amino acid at `position` with `token`.
- **INS**: Insert `token` at `position`. Sequence shifts right, last PAD consumed.
- **DEL**: Remove amino acid at `position`. Sequence shifts left, PAD appended.
- **STOP**: Terminate episode. No sequence change.

Length constraints: min=8, max=27 (CDR3beta biological range).

## 4. State Representation

### 4.1 ESM-2 Frozen State Encoder (replaces LSTM SeqEmbed)

```python
class ESMStateEncoder(nn.Module):
    """Frozen ESM-2 replaces SeqEmbed's LSTM for state encoding."""

    def __init__(self, esm_model_name="esm2_t33_650M_UR50D", frozen=True):
        self.esm = load_esm(esm_model_name)
        if frozen:
            for param in self.esm.parameters():
                param.requires_grad = False

    def encode_tcr(self, tcr_seq: str) -> torch.Tensor:
        # Mean-pool ESM hidden states over non-PAD positions
        # Returns: [D_esm]

    def encode_pmhc(self, peptide: str, hla_pseudoseq: str) -> torch.Tensor:
        # Concatenate peptide + HLA pseudosequence, single ESM pass
        # Returns: [D_esm]
```

- **TCR**: re-encoded every step (sequence changes)
- **pMHC**: encoded once at episode start, cached for the episode
- **ESM weights frozen**: only MLP actor/critic heads are trained by PPO

### 4.2 State Vector

```
state = concat([
    esm_encode(TCR),           # [D_esm]  -- per step
    esm_encode(pMHC),          # [D_esm]  -- cached per episode
    remaining_steps / max,     # [1]      -- normalized to [0, 1]
    cumulative_delta,          # [1]      -- Score(s_t) - Score(s_0)
])
```

Optional future extension: TFold structural features concatenated after pMHC embedding. Interface placeholder exists (`structural_features: Optional[Tensor] = None`).

### 4.3 HLA Encoding

HLA pseudosequence (34 residues, NetMHC-style). Sufficient to distinguish alleles (e.g., HLA-A*02:01 vs A*24:02). Concatenated with peptide before ESM encoding.

## 5. Reward System

### 5.1 Reward Formula

```
R_t = w1 * delta_affinity - w2 * R_decoy - w3 * R_naturalness - w4 * R_diversity
```

Each component is z-score normalized by its own running mean/std. Weights are relative.

### 5.2 Component 1: Target Affinity (ERGO -- retained)

- **Model**: ERGO AE-LSTM classifier (same as v1: `ae_mcpas1.pt` etc.)
- **Input**: `(TCR_CDR3beta, peptide)` -- note: MHC-unaware in reward, but MHC enters the state via ESM
- **Output**: `(binding_score, confidence)`
- **Confidence**: MC Dropout with N=10 forward passes. Score = mean, confidence = 1.0 - std.
- **Credit assignment**: Step-wise delta reward: `delta_t = score(s_t) - score(s_0)` (improvement over initial TCR)
- **Terminal**: At STOP or max_steps, full evaluation score is used

### 5.3 Component 2: Decoy Contrastive Penalty (LogSumExp)

```
R_decoy = (1/tau) * log(sum(exp(tau * Affinity(TCR, p_neg))))
```

- **Scorer**: Same ERGO model as Component 1 (ensures consistent scale)
- **Sampling**: K=32 decoys per step, randomly sampled from current target's decoy pool
- **Temperature**: tau=10 (configurable). tau -> inf becomes hard max; moderate tau gives gradient from multiple high-risk decoys
- **Confidence**: Mean confidence across sampled decoys

#### Tiered Sampling

Importance order: A > B > D > C

| Tier | Content | Weight | Rationale |
|---|---|---|---|
| A | 1-2 AA point mutants | 3 | Closest to target, highest clinical risk |
| B | 2-3 AA mutants | 3 | Still dangerous |
| D | Other known binders (VDJdb/IEDB) | 2 | Real-world cross-reactivity |
| C | 1900 unrelated peptides | 1 | Global safety; best for final screening |

#### Tiered Unlocking Schedule

| Phase | Steps | Unlocked Tiers |
|---|---|---|
| 1 | 0 -- 2M | A only |
| 2 | 2M -- 5M | A + B |
| 3 | 5M -- 8M | A + B + D |
| 4 | 8M+ | A + B + D + C |

Within a tier, sampling can optionally weight by sequence similarity to target (higher similarity = higher sampling probability).

### 5.4 Component 3: Naturalness (ESM Perplexity)

- **Model**: ESM-2 (same instance as state encoder, or separate lighter model)
- **Threshold-based**: Penalize below threshold, zero reward above
- **CDR3 z-score normalization**: Offline compute ESM perplexity distribution over TCRdb CDR3beta sequences. Use z-score instead of raw log-likelihood. This prevents ESM from penalizing CDR3 sequences simply for being hypervariable regions.

```python
z = (perplexity - mean_cdr3_ppl) / std_cdr3_ppl
if z >= threshold:  # e.g., -2.0
    return 0.0      # natural enough
else:
    return z - threshold  # negative penalty
```

### 5.5 Component 4: Diversity Penalty

- **Buffer**: Size=512 recent generations
- **Metric**: Sequence similarity (e.g., normalized Levenshtein or BLOSUM62 alignment score)
- **Threshold**: 0.85 similarity
- **Penalty**: If `max_sim > threshold`, penalty = `-(max_sim - threshold)`
- **PPO entropy coef**: Raised from 0.01 to 0.05

### 5.6 Reward Normalization (RewardManager)

Each reward component maintains its own running mean/std (window=10000). After warmup (1000 steps), all components are z-score normalized before weighting.

```python
class RunningNormalizer:
    def __init__(self, window=10000, warmup=1000):
        self.buffer = deque(maxlen=window)
        self.warmup = warmup

    def normalize(self, value):
        self.buffer.append(value)
        if len(self.buffer) < self.warmup:
            return value  # raw during warmup
        return (value - np.mean(self.buffer)) / (np.std(self.buffer) + 1e-8)
```

### 5.7 Scorer Interface

```python
class BaseScorer(ABC):
    @abstractmethod
    def score(self, *args) -> Tuple[float, float]:
        """Returns (score, confidence) where confidence in [0, 1]."""
        pass
```

All scorers return `(score, confidence)`. Confidence can optionally gate reward contribution (low confidence -> discounted weight).

## 6. Curriculum Learning

### 6.1 TCR Initialization Pool (3 Levels)

| Level | Source | Details | Difficulty |
|---|---|---|---|
| L0 | VDJdb known binders, 3-5 AA random mutations | Hamming dist 3-5 from real binder. Agent must find and fix mutations. | Easy-Medium |
| L1 | TCRdb top-500 per target by ERGO score (offline) | Not known binders, but ERGO predicts weak affinity. One-time offline computation. | Medium |
| L2 | Random TCRdb sequences | No prior, pure exploration. Must always maintain nonzero fraction. | Hard |

### 6.2 Curriculum Schedule

| Steps | L0 | L1 | L2 |
|---|---|---|---|
| 0 -- 1M | 0.7 | 0.2 | 0.1 |
| 1M -- 3M | 0.4 | 0.4 | 0.2 |
| 3M -- 6M | 0.2 | 0.4 | 0.4 |
| 6M+ | 0.1 | 0.3 | 0.6 |

L0 never drops to 0 to prevent catastrophic forgetting of easy targets.

### 6.3 L1 Seed Generation (Offline, One-Time)

```python
for target in targets:
    ergo_scores = ergo.score_all(tcrdb_seqs, target)  # fast, ERGO is cheap
    l1_seeds[target] = tcrdb_seqs[argsort(ergo_scores)[-500:]]
    save(f"data/l1_seeds/{target}.txt", l1_seeds[target])
```

TCRdb has ~2.7M CDR3beta sequences. ERGO scoring is fast (thousands/second), so this is tractable.

### 6.4 Target Peptide Sampling

Difficulty-weighted: targets with lower recent AUROC (from online eval) are sampled more often.

```python
weights = [1.0 / (auroc + 0.1) for auroc in recent_aurocs]
target = random.choices(targets, weights=weights, k=1)[0]
```

## 7. Training Pipeline

### 7.1 PPO Configuration

| Parameter | Value | Notes |
|---|---|---|
| total_timesteps | 10,000,000 | Same as v1 |
| n_envs | 20 | Same as v1 |
| n_steps | 128 | Rollout buffer per env |
| batch_size | 256 | |
| n_epochs | 4 | PPO epochs per update |
| learning_rate | 3e-4 | |
| gamma | 0.90 | Same as v1 |
| gae_lambda | 0.95 | |
| clip_range | 0.2 | |
| entropy_coef | 0.05 | Increased from 0.01 |
| vf_coef | 0.5 | |
| max_grad_norm | 0.5 | |
| max_steps_per_episode | 8 | Same as v1 |

### 7.2 Online Specificity Validation Callback

```python
class SpecificityCallback:
    eval_interval = 100_000     # every 100K steps
    n_tcrs_per_target = 5       # small, fast
    n_decoys_per_target = 50    # sampled
    abort_threshold = 0.40      # abort if avg AUROC < this after warmup
    warmup_steps = 500_000      # don't abort during initial exploration
```

Logs AUROC, target/decoy score distributions, and diversity metrics to TensorBoard/wandb.

### 7.3 Ablation Support

Config switch `reward_mode`:
- `"v2_full"`: Full v2 reward (affinity delta + decoy + naturalness + diversity)
- `"v1_ergo_only"`: ERGO terminal reward only (for baseline comparison)
- `"v2_no_decoy"`: v2 without decoy penalty (ablation)
- `"v2_no_curriculum"`: v2 with random init only (ablation)

## 8. What Does NOT Change from v1

- **ERGO** as the binding predictor (reward signal source)
- **PPO** as the RL algorithm (no GFlowNets)
- **CDR3beta only** (no alpha chain)
- VecEnv parallelism (n_envs=20)
- Vendored SB3 stays frozen for v1; v2 uses its own PPO implementation (custom `ppo_trainer.py`) because the autoregressive action space and action masking require non-standard policy rollout logic that SB3 doesn't natively support

## 9. Deferred to Future Versions

- **TFold structural features**: Interface placeholder exists. Requires offline pipeline (5-7 day effort).
- **NetTCR-2.2 ensemble**: Second affinity head for reward hacking mitigation.
- **PopArt reward normalization**: More sophisticated than running mean/std.
- **GFlowNets**: If diversity bonus insufficient.
- **Alpha chain**: v2.1 scope.
- **Custom ESM-2 fine-tuned binding classifier**: If ERGO proves insufficient.

## 10. Full Hyperparameter Reference (default.yaml)

```yaml
# === PPO ===
total_timesteps: 10_000_000
n_envs: 20
n_steps: 128
batch_size: 256
n_epochs: 4
learning_rate: 3.0e-4
gamma: 0.90
gae_lambda: 0.95
clip_range: 0.2
entropy_coef: 0.05
vf_coef: 0.5
max_grad_norm: 0.5

# === Environment ===
max_tcr_len: 27
min_tcr_len: 8
max_pep_len: 25
max_steps_per_episode: 8
hla_pseudoseq_len: 34

# === ESM State Encoder ===
esm_model: "esm2_t33_650M_UR50D"   # try 650M first, downgrade if slow
esm_frozen: true
esm_device: "cuda"

# === Reward Weights (relative, after z-score normalization) ===
w_affinity: 1.0
w_decoy: 0.8
w_naturalness: 0.5
w_diversity: 0.2

# === Credit Assignment ===
use_delta_reward: true              # score(s_t) - score(s_0)

# === Reward Normalization ===
norm_window: 10000
norm_warmup: 1000

# === Affinity Scorer (ERGO) ===
affinity_model: "ergo"
ergo_model_file: "ae_mcpas1.pt"
affinity_mc_samples: 10

# === Decoy Scorer ===
decoy_K: 32
decoy_tau: 10.0
decoy_tier_weights:
  A: 3
  B: 3
  D: 2
  C: 1
decoy_unlock_schedule:
  0: [A]
  2000000: [A, B]
  5000000: [A, B, D]
  8000000: [A, B, D, C]

# === Naturalness Scorer (ESM Perplexity) ===
naturalness_threshold_zscore: -2.0
# Offline stats (computed from TCRdb CDR3beta):
# naturalness_mean_ppl: <to be computed>
# naturalness_std_ppl: <to be computed>

# === Diversity Scorer ===
diversity_buffer_size: 512
diversity_similarity_threshold: 0.85

# === Curriculum ===
l0_mutation_range: [3, 5]           # random mutations on VDJdb binders
l1_top_k: 500                      # ERGO-scored top-K from TCRdb per target
curriculum_schedule:
  - {until: 1000000,  L0: 0.7, L1: 0.2, L2: 0.1}
  - {until: 3000000,  L0: 0.4, L1: 0.4, L2: 0.2}
  - {until: 6000000,  L0: 0.2, L1: 0.4, L2: 0.4}
  - {until: null,      L0: 0.1, L1: 0.3, L2: 0.6}

# === Target Sampling ===
target_sampling: "difficulty_weighted"  # or "round_robin"

# === Online Eval ===
eval_interval: 100000
eval_n_tcrs: 5
eval_n_decoys: 50
eval_abort_threshold: 0.40
eval_warmup: 500000

# === Checkpointing ===
checkpoint_interval: 100000
milestones: [500000, 1000000, 2000000, 5000000, 10000000]

# === Ablation ===
reward_mode: "v2_full"              # v2_full | v1_ergo_only | v2_no_decoy | v2_no_curriculum

# === TFold (placeholder, not active in v2.0) ===
use_tfold: false
tfold_features_dir: null
```
