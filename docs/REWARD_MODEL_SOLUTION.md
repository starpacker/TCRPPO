# Reward Model Bottleneck: Analysis and Solutions

**Date**: 2026-05-05  
**Problem**: Current reward models face accuracy-speed tradeoff  
**Status**: Solution implemented (test48), long-term plan defined

---

## 🎯 Problem Statement

### Core Bottleneck

| Reward Model | Accuracy (AUROC) | Speed | Usability |
|--------------|------------------|-------|-----------|
| **ERGO** | 0.62 (medium) | ~10ms ✅ | Fast but limited accuracy |
| **tFold** | 0.50-0.90 (high, peptide-dependent) | ~1s ❌ | Accurate but 100x slower |

**Contradiction**: We need **both** high accuracy (for good specificity) **and** fast speed (for practical training time).

### Evidence of the Problem

#### 1. ERGO Accuracy Limitation (test41 tFold evaluation)

**test41 (pure ERGO training)**:
- ERGO AUROC: 0.6243
- tFold AUROC: 0.4017 (gap: -0.2227)

**Root cause**: RL model overfits to ERGO's sequence biases, generates TCRs that lack structural plausibility (tFold's criterion).

**Per-peptide breakdown**:
- 5/12 peptides: tFold better than ERGO (e.g., NLVPMVATV: tFold 0.80 vs ERGO 0.49)
- 7/12 peptides: tFold shows reversed discrimination (e.g., LLWNGPMAV: tFold 0.04 vs ERGO 0.80)

**Conclusion**: ERGO is not the "ground truth" — it has blind spots that tFold can correct.

#### 2. tFold Speed Problem (test44)

**test44 (pure tFold training)**:
- Progress: 10K/2M steps (0.5%) in 30 hours
- Estimated completion: 22 days
- Cache hit rate: 3% (97% cache misses require ~1s feature extraction)

**Conclusion**: Pure tFold training is impractical for production use.

---

## 💡 Solution Matrix

I propose **5 solutions**, ranked by implementation difficulty and expected impact:

### Solution 1: Hybrid Scorer (90% ERGO + 10% tFold) ⭐⭐⭐⭐⭐

**Status**: ⚠️ **TESTING** (test48, launched but VERY SLOW)

**CRITICAL FINDING (2026-05-05 16:23)**: First rollout taking 30+ minutes due to contrastive reward amplification:
- Contrastive reward scores 1 target + 16 decoys = 17 peptides per sample
- 1024 samples/rollout × 17 peptides × 10% tFold = ~1,741 tFold calls
- At ~1s per cache miss, first rollout takes 30+ minutes
- **Conclusion**: Hybrid approach is IMPRACTICAL with contrastive reward

**Revised recommendation**: 
- Either reduce tFold ratio to 2-5% (test48b)
- Or use cascade/active learning (Solutions 3/5) to reduce tFold calls
- Or apply hybrid only to target peptide, not decoys (requires code change)

#### How It Works

```python
class HybridScorer:
    def score(self, tcr, peptide):
        if random() < 0.1:  # 10% probability
            return tfold.score(tcr, peptide)  # Slow but accurate
        else:
            return ergo.score(tcr, peptide)   # Fast but biased
```

#### Performance Estimates

| Metric | Value | Calculation |
|--------|-------|-------------|
| **Speed** | 109ms/sample | 0.9×10ms + 0.1×1000ms |
| **Training time** | ~30 hours | 1M steps, 8 envs, 128 steps/rollout |
| **vs test41** | 2-3x slower | Acceptable tradeoff |
| **vs test44** | 17x faster | Practical for production |

#### Expected Results

- **ERGO AUROC**: 0.63-0.65 (slightly better than test41's 0.6243)
- **tFold AUROC**: 0.45-0.50 (significantly better than test41's 0.4017)
- **Gap reduction**: From -0.22 to -0.15

#### Advantages

✅ **Immediate**: 2 hours implementation, ready to launch  
✅ **Practical speed**: 30h training (vs 22 days for pure tFold)  
✅ **Proven concept**: Similar to ensemble methods in ML  
✅ **Tunable**: Can adjust ratio (5%, 10%, 20%)

#### Risks

⚠️ 10% tFold signal may be too weak (mitigation: try 20%)  
⚠️ Still 2-3x slower than pure ERGO (acceptable for better accuracy)

#### Launch Command

```bash
bash scripts/launch_test48_hybrid.sh
# GPU 2, ~30 hours training time
```

---

### Solution 2: Knowledge Distillation (Train Fast Student Model) ⭐⭐⭐⭐⭐

**Status**: 🔄 **RECOMMENDED** for long-term (5-7 days implementation)

#### How It Works

**Step 1**: Generate training dataset (offline, one-time)
```python
# For each peptide, generate 10K TCR samples
# Score with tFold (slow, but only once)
# Save (TCR, peptide, tFold_score) tuples
# Time: 10K × 45 peptides × 1s = 450K seconds ≈ 5 days
# Parallelizable: 5 GPUs → 1 day
```

**Step 2**: Train lightweight neural network
```python
class DistilledTFoldScorer(nn.Module):
    def __init__(self):
        self.esm2 = load_frozen_esm2()  # Reuse existing
        self.mlp = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, tcr_emb, peptide_emb):
        combined = torch.cat([tcr_emb, peptide_emb], dim=-1)
        return self.mlp(combined)

# Train with MSE loss: minimize |predicted - tFold_score|^2
# Training time: 2-4 hours on 450K samples
```

**Step 3**: Use distilled model for RL training
```python
# Inference speed: ~10ms (same as ERGO)
# Accuracy: close to tFold (if distillation successful)
```

#### Performance Estimates

| Metric | Value | Notes |
|--------|-------|-------|
| **Speed** | ~10ms | Same as ERGO |
| **Accuracy** | 0.65-0.70 AUROC | Between ERGO and tFold |
| **Training time** | 12-16 hours | Same as test41 |
| **Setup cost** | 1 day (parallel) | One-time investment |

#### Advantages

✅ **Best of both worlds**: Fast inference + high accuracy  
✅ **One-time cost**: Dataset generation only needed once  
✅ **Reusable**: Can train multiple RL models with same dataset  
✅ **Interpretable**: Can analyze what the student learned

#### Risks

⚠️ Distillation may lose accuracy (student not as good as teacher)  
⚠️ Requires 5-7 days implementation + 1 day data generation  
⚠️ Need to verify distilled model generalizes to RL-generated TCRs

#### Implementation Plan

1. **Day 1**: Write data generation script (4 hours)
2. **Day 2**: Generate dataset in parallel (5 GPUs × 1 day)
3. **Day 3**: Train distilled model (2-4 hours)
4. **Day 4**: Integrate into ppo_trainer.py (2 hours)
5. **Day 5-7**: Launch test49_distilled_tfold, monitor results

---

### Solution 3: Cascade Scorer (ERGO Filter + tFold Verify) ⭐⭐⭐⭐

**Status**: 🔄 **READY** (4 hours implementation)

#### How It Works

```python
class CascadeScorer:
    def score(self, tcr, peptide):
        # Stage 1: ERGO quick filter
        ergo_score = ergo.score(tcr, peptide)
        
        # Stage 2: tFold verify only high-scoring candidates
        if ergo_score > 0.3:  # Threshold
            tfold_score = tfold.score(tcr, peptide)
            return 0.7 * tfold_score + 0.3 * ergo_score
        else:
            return ergo_score  # Low score, skip tFold
```

#### Performance Estimates

| Metric | Value | Notes |
|--------|-------|-------|
| **Speed (early)** | ~20ms | Most TCRs score low initially |
| **Speed (late)** | ~200ms | More high-scoring TCRs later |
| **Average** | ~100ms | Adaptive speed |
| **Training time** | 1-2 days | Faster than hybrid |

#### Advantages

✅ **Adaptive**: Fast early, accurate late  
✅ **Focused**: tFold only verifies promising TCRs  
✅ **Efficient**: Avoids wasting tFold on bad TCRs

#### Risks

⚠️ Threshold hard to tune (0.3? 0.5?)  
⚠️ May miss "ERGO low but tFold high" TCRs

---

### Solution 4: Ensemble Scorer (Multiple Fast Models) ⭐⭐⭐

**Status**: 🔄 **AVAILABLE** (1 day implementation)

#### How It Works

```python
class EnsembleScorer:
    def __init__(self):
        self.scorers = [
            ERGOScorer(),      # LSTM-based
            NetTCRScorer(),    # CNN-based
            LightweightScorer() # MLP-based
        ]
        self.weights = [0.5, 0.3, 0.2]
    
    def score(self, tcr, peptide):
        scores = [s.score(tcr, peptide) for s in self.scorers]
        return sum(w * s for w, s in zip(self.weights, scores))
```

#### Performance Estimates

| Metric | Value | Notes |
|--------|-------|-------|
| **Speed** | ~30ms | All models fast |
| **Accuracy** | 0.63-0.66 | Better than single ERGO |
| **Training time** | 12-16 hours | Same as test41 |

#### Advantages

✅ **Fast**: All models are sequence-based (~10-30ms)  
✅ **Robust**: Multiple models reduce single-model bias  
✅ **Existing**: ERGO and NetTCR already implemented

#### Risks

⚠️ All sequence models, may share same blind spots  
⚠️ Doesn't solve "lack of structure awareness" problem

---

### Solution 5: Active Learning (Uncertainty-Guided tFold) ⭐⭐⭐⭐

**Status**: 🔄 **READY** (4 hours implementation)

#### How It Works

```python
class ActiveLearningScorer:
    def score(self, tcr, peptide):
        # Stage 1: ERGO with MC Dropout uncertainty
        ergo_score, ergo_conf = ergo.score(tcr, peptide)
        uncertainty = 1.0 - ergo_conf
        
        # Stage 2: Use tFold only for high-uncertainty samples
        if uncertainty > 0.2 and tfold_budget_remaining():
            return tfold.score(tcr, peptide)  # ERGO unsure
        else:
            return ergo_score  # ERGO confident
```

#### Performance Estimates

| Metric | Value | Notes |
|--------|-------|-------|
| **Speed** | ~60ms | 95% ERGO + 5% tFold |
| **Accuracy** | 0.63-0.66 | Targets ERGO's blind spots |
| **Training time** | 1-2 days | Faster than hybrid |

#### Advantages

✅ **Efficient**: Only 5% tFold calls  
✅ **Targeted**: Focuses on ERGO's uncertain regions  
✅ **Budget-controlled**: Strict tFold call limit

#### Risks

⚠️ Depends on ERGO uncertainty (we know this is unreliable from test46)  
⚠️ May miss "ERGO confident but wrong" samples

---

## 📊 Solution Comparison

| Solution | Speed | Accuracy | Impl. Time | Training Time | Recommend |
|----------|-------|----------|------------|---------------|-----------|
| **1. Hybrid** | ⭐⭐⭐ (109ms) | ⭐⭐⭐⭐ (0.63-0.65) | 2h | 30h | ⭐⭐⭐⭐⭐ |
| **2. Distillation** | ⭐⭐⭐⭐⭐ (10ms) | ⭐⭐⭐⭐ (0.65-0.70) | 5-7d | 12-16h | ⭐⭐⭐⭐⭐ |
| **3. Cascade** | ⭐⭐⭐⭐ (20-200ms) | ⭐⭐⭐⭐ (0.64-0.68) | 4h | 1-2d | ⭐⭐⭐⭐ |
| **4. Ensemble** | ⭐⭐⭐⭐ (30ms) | ⭐⭐⭐ (0.63-0.66) | 1d | 12-16h | ⭐⭐⭐ |
| **5. Active Learning** | ⭐⭐⭐⭐ (60ms) | ⭐⭐⭐ (0.63-0.66) | 4h | 1-2d | ⭐⭐⭐⭐ |

---

## 🎯 Recommended Strategy: Revised Three-Phase Plan

### Phase 1: Immediate (Today, 4 hours)

**Implement and launch Cascade Scorer (Solution 3)**

✅ Fast to implement (4 hours)  
✅ Adaptive speed (20-200ms)  
✅ Practical training time (1-2 days)

**Goal**: Validate that ERGO pre-filter + tFold verify improves tFold AUROC

**Success criteria**:
- tFold AUROC > 0.45 (vs test41's 0.40)
- ERGO AUROC ≥ 0.62 (no regression)
- Training time < 48 hours

**Why cascade instead of hybrid**:
- test48 showed hybrid is too slow with contrastive reward (40+ min per rollout)
- Cascade is adaptive: fast early, accurate late
- Focuses tFold on promising TCRs only

---

### Phase 2: Parallel Preparation (This Week, 5-7 days)

**Prepare Knowledge Distillation (Solution 2)**

While test49 (cascade) runs, prepare the long-term solution:

1. **Day 1**: Write data generation script
2. **Day 2**: Generate 450K training samples (5 GPUs parallel)
3. **Day 3**: Train distilled model
4. **Day 4**: Integrate and test
5. **Day 5**: Launch test50_distilled_tfold

**Goal**: Create a fast (~10ms) model with tFold-like accuracy

---

### Phase 3: Optimization (Next Week)

**Based on test49 (cascade) results**:

- **If successful** (tFold AUROC > 0.45):
  - Tune cascade threshold (0.2? 0.4?)
  - Compare with distilled model (test50)
  
- **If failed** (tFold AUROC ≤ 0.42):
  - Try target-only hybrid (apply tFold to target, ERGO for decoys)
  - Investigate why cascade insufficient

---

## 🔬 Scientific Questions

### Q1: What is the minimum tFold ratio needed?

**Hypothesis**: 10% tFold signal is enough to correct ERGO biases

**Test**: Compare test48 (10%), test48b (20%), test48c (5%)

**Expected**: Diminishing returns above 15-20%

### Q2: Can distillation preserve tFold's accuracy?

**Hypothesis**: Lightweight MLP can learn tFold's scoring function

**Test**: Compare distilled model vs tFold on held-out TCRs

**Metric**: Pearson correlation > 0.85 = success

### Q3: Why does ERGO-tFold gap exist?

**Hypothesis**: ERGO learns sequence patterns, tFold learns structure

**Test**: Analyze TCRs where ERGO and tFold disagree

**Method**: 
- Score known VDJdb binders with both
- Analyze structural features (AlphaFold pLDDT, RMSD)
- Identify what tFold "sees" that ERGO doesn't

---

## 📋 Immediate Action Items

### Today (2026-05-05)

1. ✅ **Implement HybridScorer** (DONE)
2. ✅ **Create test48 documentation** (DONE)
3. ✅ **Create launch script** (DONE)
4. ⏳ **Launch test48** (READY)

```bash
bash scripts/launch_test48_hybrid.sh
```

5. ⏳ **Monitor for first 2 hours** (verify hybrid ratio ~10%)

### Tomorrow (2026-05-06)

1. Check test48 progress (~5% complete)
2. Start writing distillation data generation script
3. Check test47/test41_seed123/test41_seed7 progress

### This Week

1. Complete distillation dataset generation (parallel on 5 GPUs)
2. Train distilled model
3. Prepare test49 launch

---

## 📈 Expected Timeline

```
Today (Day 0):     Launch test48
Day 1:             test48 5% complete, start distillation prep
Day 2:             test48 15% complete, generate distillation data
Day 3:             test48 50% complete, train distilled model
Day 4:             test48 complete, evaluate results
Day 5:             Launch test49 (distilled) if test48 successful
Day 7:             test49 complete, compare all approaches
```

---

## 🎯 Success Metrics

### Short-term (test48, 1 week)

- **tFold AUROC**: > 0.45 (improvement over test41's 0.40)
- **ERGO AUROC**: ≥ 0.62 (no regression)
- **Training time**: < 40 hours (acceptable)

### Medium-term (test49, 2 weeks)

- **Speed**: ~10ms (same as ERGO)
- **Accuracy**: 0.65-0.70 AUROC (close to tFold)
- **Training time**: 12-16 hours (same as test41)

### Long-term (1 month)

- **Mean AUROC**: > 0.70 across all 12 peptides
- **Reproducible**: 3-seed std < 0.03
- **Production-ready**: < 24h training time

---

**Document created**: 2026-05-05  
**Status**: test48 ready to launch, distillation plan defined  
**Next review**: After test48 completes (~30 hours)
