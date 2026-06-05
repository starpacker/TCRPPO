# trace73 Success Analysis - 为什么它能产生高 affinity TCRs

**Date**: 2026-05-29
**Goal**: 理解 trace73 的成功策略，指导 trace78 改进

---

## 📊 Overall Performance

- **Total episodes**: 5,088
- **Mean affinity**: -1.172
- **Episodes > 0.0**: 73 (1.43%)
- **Episodes > -0.5**: 824 (16.19%)
- **Episodes > -1.0**: 2,244 (44.10%)

**Top 20 episodes mean**: 0.361 (best: 0.834)

---

## 🏆 Top TCRs Characteristics

### 1. CDR3 Length Pattern
**ALL top 30 TCRs have length 18**

This is NOT a coincidence. The model learned that 18-AA CDRs work best for these targets.

### 2. Sequence Patterns
Top TCRs share common motifs:
- **Start**: `CAL` or `CSL` or `CSV` (C-terminal anchor)
- **Middle**: Rich in S, L, G, A, Y, E, Q (flexible/polar residues)
- **End**: `YYCCC` or `YFCCC` or `QYYCCC` (conserved C-terminal)

Examples:
```
CALSETSSNYGATYYCCC  (A=0.834)
CALSPSYLSTDAQYYCCC  (A=0.776)
CALSFGASSTEAQYYCCC  (A=0.714)
```

### 3. Target Peptide Preferences
Top peptides that got high-affinity TCRs:
1. **YLQPRTFLL**: 6 TCRs (most successful!)
2. **YLPLNTFLL**: 4 TCRs
3. **GILGFVFTL**: 3 TCRs
4. **LLLDRLNQL**: 3 TCRs
5. **LLWRGSIYKL**: 3 TCRs

**Pattern**: Peptides with L, Y, F (hydrophobic) are easier targets.

---

## 🔑 Key Success Factors

### 1. Curriculum Learning
trace73 used curriculum with gradual difficulty increase:
- Started with L0 seeds (known binders + mutations)
- Progressed to L1 (top TCRdb sequences)
- Finally L2 (random exploration)

**Result**: Model learned from good examples before exploring.

### 2. Online Pool Strategy
From top episodes:
```
Episode 1674 | A=0.8342 | OnlinePool=add
Episode 4521 | A=0.7139 | OnlinePool=add
Episode 4591 | A=0.6711 | OnlinePool=add
```

**All top episodes were added to OnlinePool!**

This means:
- Good TCRs were kept and reused
- Model could build on successful sequences
- Exploration was guided by past success

### 3. Delta Reward Signal
```
Episode 1674: InitA=-4.8460 → A=0.8342 (DeltaA=+5.68)
Episode 4591: InitA=-5.9388 → A=0.6711 (DeltaA=+6.61)
Episode 4767: InitA=-6.3238 → A=0.4688 (DeltaA=+6.79)
```

**Key insight**: Model learned to make LARGE improvements from poor starting points.

### 4. Reward Components Balance
From top episodes:
```
Episode 1674: R=1.063 | A=0.834 | DecViol=2.57 | TargetSat=2.83
Episode 4521: R=1.005 | A=0.714 | DecViol=2.34 | TargetSat=2.21
```

- **TargetSat bonus**: 2.0-2.8 (huge reward for passing gate)
- **DecViol penalty**: 2.3-2.6 (but still net positive)
- **Naturalness/Diversity**: minimal impact (-0.1 to 0)

**Conclusion**: Target affinity + gate bonus dominated the reward.

---

## ❌ Why trace78 is Failing

### Problem 1: Online Pool Too Strict
```
trace78 config: min_affinity: -0.8
trace73 config: min_affinity: -10.0 (essentially no filter)
```

**Result**: trace78 pool is empty (3/20 targets), trace73 pool was full.

### Problem 2: No Curriculum
```
trace78: curriculum disabled (L2 only from start)
trace73: L0 → L1 → L2 progression
```

**Result**: trace78 starts from random TCRs, trace73 started from known binders.

### Problem 3: Wrong Baseline
```
trace78: resumed from step 682K (late in trace73's training)
trace73: started from step 0
```

**Result**: trace78 inherited trace73's policy but without its pool data.

---

## 💡 Recommended Strategy

### Option A: Fix trace78 (Quick)
1. **Relax pool filter**: `min_affinity: -1.5` (already done)
2. **Restart training**: Let pool rebuild from scratch
3. **Monitor**: Wait for pool to fill (should see 15+ targets with data)

### Option B: New trace79 (Better)
1. **Start fresh**: Don't resume from trace73
2. **Enable curriculum**: Use L0 → L1 → L2 progression
3. **Moderate pool filter**: `min_affinity: -1.2`
4. **Longer training**: 2M steps instead of 1M

### Option C: Extract trace73 Pool (Best)
1. **Parse trace73 log**: Extract all TCRs with `OnlinePool=add`
2. **Filter by affinity**: Keep only A > -0.5
3. **Use as L1 seeds**: Initialize trace79 with these proven TCRs
4. **Train with strict gate**: `target_affinity_gate: -0.3`

---

## 📈 Expected Outcomes

### If we fix trace78 (Option A):
- Pool should fill in ~100 episodes
- Mean affinity should improve to ~-1.0 in 500 episodes
- Might reach -0.5 in 2000+ episodes

### If we start trace79 with curriculum (Option B):
- Better learning trajectory
- Mean affinity ~-0.8 by 500K steps
- Might reach -0.5 by 1M steps

### If we use trace73 pool as seeds (Option C):
- **Fastest path to -0.5**
- Start with proven TCRs (A > -0.5)
- Focus on refinement, not discovery
- Could reach -0.5 in 200-500 episodes

---

## 🎯 Next Steps

1. **Immediate**: Stop trace78, restart with relaxed filter
2. **Short-term**: Extract trace73's best TCRs as L1 seeds
3. **Long-term**: Launch trace79 with curriculum + extracted seeds

**Estimated time to break -0.5**: 
- Option A: 6-12 hours
- Option B: 4-8 hours  
- Option C: 1-3 hours ⭐ **RECOMMENDED**
