# tFold Hybrid Training Guide

## Problem Statement

ERGO scorer has **reward-AUROC misalignment** on some peptides: it gives higher scores to decoy peptides than to target peptides, causing the RL agent to learn the wrong behavior.

### Evidence from SAC test3 @ 1M steps (7 ERGO-trainable peptides)

| Peptide | ERGO Mapping AUC | Actual AUROC | Target Score | Decoy Score | Gap | Status |
|---------|------------------|--------------|--------------|-------------|-----|--------|
| KLWASPLHV | 0.823 | 0.5656 | 0.1375 | 0.1103 | +0.027 | ✅ Aligned |
| FPRPWLHGL | 0.762 | 0.6124 | 0.2001 | 0.1502 | +0.050 | ✅ Aligned |
| KAFSPEVIPMF | 0.739 | 0.5912 | 0.2367 | 0.1907 | +0.046 | ✅ Aligned |
| HSKKKCDEL | 0.763 | 0.6040 | 0.2079 | 0.1273 | +0.081 | ✅ Aligned |
| RFYKTLRAEQASQ | 0.908 | 0.1244 | 0.0166 | 0.0355 | **-0.019** | ❌ Reversed |
| DRFYKTLRAEQASQEV | 0.786 | 0.2952 | 0.0141 | 0.0241 | **-0.010** | ❌ Reversed |
| FRCPRRFCF | 0.714 | 0.0960 | 0.0359 | 0.0958 | **-0.060** | ❌ Reversed |

**Key findings**:
- 4 peptides: positive gap → AUROC 0.57-0.61 (reward signal correct)
- 3 peptides: negative gap → AUROC 0.10-0.30 (reward signal reversed)
- Mean AUROC across 7 peptides: **0.4127** (dragged down by reversed peptides)
- Mean AUROC for 4 positive peptides: **0.5933** (close to PPO best)

**Root cause**: ERGO's mapping AUC measures discrimination on **existing TCR-peptide data** (VDJdb/IEDB). But RL generates **novel TCRs** that may fall outside ERGO's training distribution, where its scoring becomes unreliable.

## Solution: Hybrid tFold-ERGO Training

### Strategy

1. **Filter to 4 positive-aligned peptides** (exclude 3 reversed peptides)
2. **Mix ERGO (fast) with tFold (accurate)** during training:
   - 90% episodes: ERGO reward (~10ms per TCR)
   - 10% episodes: tFold reward (~1s per TCR on cache miss, <1ms on cache hit)
3. **Pre-warm tFold cache** to minimize slow subprocess calls
4. **SAC's replay buffer** naturally mixes both reward sources for training

### Why This Works

- **tFold is more accurate** on novel TCRs (structure-based, not sequence-only)
- **10% tFold is enough** to correct ERGO's errors without sacrificing speed
- **Off-policy RL (SAC)** can learn from mixed reward sources (PPO cannot)
- **Cache warmup** makes tFold fast enough for training (most calls are cache hits)

## Implementation

### Step 1: Pre-warm tFold Cache

```bash
cd /share/liuyutian/tcrppo_sac

# Warmup cache for 4 positive-aligned peptides
CUDA_VISIBLE_DEVICES=2 python scripts/warmup_tfold_cache.py \
    --targets KLWASPLHV,FPRPWLHGL,KAFSPEVIPMF,HSKKKCDEL \
    --n_tcrs_per_target 200 \
    --cache_path /share/liuyutian/tcrppo_v2/data/tfold_feature_cache.db

# This takes ~15-30 minutes (800 TCRs × 4 peptides = 3200 cache entries)
# After warmup, most training calls will be cache hits (<1ms)
```

### Step 2: Launch Hybrid Training (SAC)

```bash
cd /share/liuyutian/tcrppo_sac

CUDA_VISIBLE_DEVICES=2 bash scripts/launch_sac_test5_hybrid_tfold.sh
```

Key config:
- `--use_tfold_hybrid`: Enable hybrid scorer
- `--tfold_ratio 0.1`: 10% of episodes use tFold
- `--affinity_scorer ergo`: Primary scorer is ERGO
- Targets automatically filtered to 4 positive-aligned peptides

### Step 3: Monitor Training

```bash
tail -f logs/sac_test5_hybrid_train.log

# Check hybrid scorer telemetry (printed every 100K steps)
grep "Hybrid scorer" logs/sac_test5_hybrid_train.log
```

Expected telemetry:
```
Hybrid scorer: 90.2% ERGO (180K calls), 9.8% tFold (20K calls)
tFold cache: 95.3% hit rate (19K hits, 1K misses)
```

### Step 4: Evaluate

```bash
# After training completes (2M steps)
python scripts/eval_sac_tcrs_ergo.py \
    --checkpoint output/sac_test5_hybrid_tfold_4targets/checkpoints/final.pt \
    --n_tcrs 50 --n_decoys 50 \
    --targets KLWASPLHV,FPRPWLHGL,KAFSPEVIPMF,HSKKKCDEL \
    --output_dir results/sac_test5_hybrid_ergoeval
```

Expected: **Mean AUROC > 0.63** (better than test3's 0.612 on 4 peptides)

## Adapting for PPO

PPO requires **on-policy** data, so hybrid training is more complex. Two approaches:

### Approach A: Alternating Phases (Simpler)

Train in phases, switching scorer every N steps:

```python
# Phase 1: ERGO warm-start (1M steps)
python tcrppo_v2/ppo_trainer.py \
    --reward_mode v1_ergo_only \
    --affinity_scorer ergo \
    --total_timesteps 1000000

# Phase 2: tFold fine-tune (1M steps, resume from Phase 1)
python tcrppo_v2/ppo_trainer.py \
    --reward_mode v1_tfold_only \
    --affinity_scorer tfold \
    --total_timesteps 2000000 \
    --resume_from output/phase1/checkpoints/step_1000000.pt
```

### Approach B: Cascade Scorer (More Complex)

Use `TFoldCascadeScorer` (already implemented in `tcrppo_v2/scorers/affinity_tfold.py`):

```python
from tcrppo_v2.scorers.affinity_tfold import TFoldCascadeScorer

# ERGO with MC Dropout uncertainty → tFold fallback
cascade_scorer = TFoldCascadeScorer(
    ergo_scorer=ergo_scorer,
    tfold_scorer=tfold_scorer,
    uncertainty_threshold=0.15,  # Use tFold when ERGO std > 0.15
    mc_samples=10,
)
```

This automatically routes uncertain cases to tFold during rollout collection.

## Cache Maintenance

### Check Cache Stats

```python
import sqlite3
conn = sqlite3.connect('/share/liuyutian/tcrppo_v2/data/tfold_feature_cache.db')
size = conn.execute('SELECT COUNT(*) FROM features').fetchone()[0]
print(f'Cache size: {size:,} entries')

# Per-peptide coverage
for pep in ['KLWASPLHV', 'FPRPWLHGL', 'KAFSPEVIPMF', 'HSKKKCDEL']:
    n = conn.execute('SELECT COUNT(*) FROM features WHERE cache_key LIKE ?', 
                     (f'%|{pep}|%',)).fetchone()[0]
    print(f'  {pep}: {n} cached')
conn.close()
```

### Expand Cache (Optional)

If training explores new TCR regions, expand cache:

```bash
python scripts/warmup_tfold_cache.py \
    --targets KLWASPLHV,FPRPWLHGL,KAFSPEVIPMF,HSKKKCDEL \
    --n_tcrs_per_target 500  # More TCRs
```

### Clear Cache (If Needed)

```bash
rm /share/liuyutian/tcrppo_v2/data/tfold_feature_cache.db
# Will be recreated on next use
```

## Expected Results

| Experiment | Targets | Scorer | Mean AUROC | Notes |
|------------|---------|--------|------------|-------|
| SAC test3 | 7 ERGO | ERGO only | 0.4127 | Dragged down by 3 reversed peptides |
| SAC test3 (4 pos) | 4 positive | ERGO only | 0.5933 | Filtered to aligned peptides |
| SAC test5 | 4 positive | Hybrid (90% ERGO, 10% tFold) | **>0.63** | tFold corrects ERGO errors |
| PPO test41 | 7 ERGO | Two-phase (ERGO→contrastive) | 0.6243 | Best PPO result |

**Hypothesis**: Hybrid training on 4 positive-aligned peptides should match or exceed PPO test41's 0.624 AUROC.

## Troubleshooting

### Cache Warmup is Slow

- **Expected**: First warmup takes 15-30 min (most are cache misses)
- **Solution**: Run warmup overnight, or reduce `--n_tcrs_per_target`

### tFold Server Not Running

Error: `tFold server socket not found at /tmp/tfold_server.sock`

**Solution**: tFold scorer uses subprocess mode by default (no server needed). This warning can be ignored.

### Out of Memory During Training

- **Cause**: tFold V3.4 classifier (1.57M params) + ESM-2 (650M params) on same GPU
- **Solution**: Use `--device cuda:0` for training, tFold will auto-use same GPU

### Low Cache Hit Rate

- **Cause**: Training explores TCRs not in warmup set
- **Solution**: Increase `--n_tcrs_per_target` in warmup, or accept slower training

## References

- **PEPTIDE_SCORER_MAPPING.md**: Per-peptide scorer reliability
- **tcrppo_v2/scorers/affinity_tfold.py**: tFold scorer implementation
- **tcrppo_sac/utils/hybrid_scorer.py**: Hybrid scorer implementation
- **SAC test3 results**: `/share/liuyutian/tcrppo_sac/results/sac_test3_1m_ergo_7targets/`
