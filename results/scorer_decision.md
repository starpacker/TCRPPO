# TCR-Peptide Scorer Decision for RL Training

**Date**: 2026-04-24  
**Context**: Evaluation complete, TITAN unavailable, need to make go/no-go decision

---

## Situation Summary

### Current Scorer Performance
- **NetTCR**: AUC-ROC=0.585 (barely better than random)
- **ERGO**: AUC-ROC=0.506 (essentially random)
- **DeepAIR**: Degenerate (constant scores)

### TITAN Status
- Reported AUC=0.87 in literature (best available)
- Repository exists but network access blocked
- Cannot download or test
- No local installation found

### Alternative Scorers
- **libtcrlm**: pip install fails (network issues)
- **tcr-bert**: pip install fails (network issues)
- **TEINet, PanPep, STAPLER**: Not found locally
- **TULIP-TCR**: Found as zip but unclear if it's a scorer

---

## Critical Question: Can RL Work with Weak Scorers?

### Arguments FOR Proceeding

1. **RL is robust to noisy rewards**
   - PPO has been shown to learn from imperfect signals
   - Reward shaping with multiple components can compensate
   - NetTCR AUC=0.585 is weak but not random (0.085 above baseline)

2. **Multi-component reward may be sufficient**
   - Affinity (NetTCR): weak but directional
   - Naturalness (ESM perplexity): reliable and well-validated
   - Diversity: computable, no model needed
   - Decoy penalty: even weak scorer may provide contrastive signal

3. **Ensemble can improve robustness**
   - Average NetTCR + ERGO scores
   - Reduces individual model biases
   - NetTCR and ERGO are uncorrelated (r=0.058) → complementary

4. **V1 baseline used ERGO alone**
   - V1 achieved some success despite ERGO's limitations
   - V2 has better architecture (ESM-2, per-step reward, curriculum)
   - Even with weak scorer, v2 may outperform v1

5. **Iterative refinement is possible**
   - Start with weak scorers to validate pipeline
   - Validate top candidates with expensive methods (AlphaFold2)
   - Swap in better scorer (TITAN) when available
   - RL training is modular - scorer is pluggable

### Arguments AGAINST Proceeding

1. **Garbage in, garbage out**
   - ERGO cannot distinguish binders (AUC=0.506)
   - NetTCR has terrible precision (0.398) - 60% false positives
   - RL will learn to exploit scorer weaknesses, not true binding

2. **Wasted compute and time**
   - Training RL with bad reward = training on wrong objective
   - May need to retrain from scratch with better scorer
   - Better to wait for TITAN than train twice

3. **Decoy discrimination untested**
   - Current scorers may fail target vs decoy test completely
   - Contrastive reward requires scorer to distinguish similar peptides
   - If scorer can't do this, decoy penalty is meaningless

4. **Risk of negative results**
   - If v2 fails, unclear if it's architecture or scorer
   - Hard to debug when reward signal is fundamentally broken
   - May waste weeks chasing wrong problems

---

## Recommended Decision: CONDITIONAL PROCEED

### Phase 1: Rapid Validation (1-2 days)

**Before full RL training, test critical assumptions:**

1. **Decoy discrimination test** (Task #10)
   - Test NetTCR and ERGO on target vs decoy peptides
   - Use decoy library tiers A, B, D
   - Compute score separation and discrimination metrics
   - **Go/no-go threshold**: Mean score(target) - mean score(decoy) > 0.05 for at least one scorer

2. **Ensemble scorer evaluation**
   - Test NetTCR + ERGO average on labeled dataset
   - Check if ensemble improves AUC
   - **Target**: Ensemble AUC > 0.60

3. **ESM perplexity validation**
   - Verify ESM perplexity can distinguish natural vs unnatural TCRs
   - Test on TCRdb (natural) vs random sequences (unnatural)
   - **Target**: Clear separation (p < 0.001)

### Phase 2: Pilot RL Training (3-5 days)

**If Phase 1 passes, run small-scale RL:**

1. **Single target, short training**
   - Pick one peptide (e.g., GILGFVFTL)
   - Train for 10K episodes (not 100K)
   - Use ensemble scorer (NetTCR + ERGO average)
   - Full reward: affinity + decoy + naturalness + diversity

2. **Evaluation metrics**
   - Affinity: mean NetTCR score on generated TCRs
   - Specificity: AUROC on target vs decoy discrimination
   - Naturalness: ESM perplexity distribution
   - Diversity: unique sequences generated

3. **Success criteria**
   - Generated TCRs have higher affinity than L0 seeds
   - AUROC > 0.55 (better than v1's 0.45)
   - ESM perplexity within 1 std of natural TCRs
   - At least 50% unique sequences

### Phase 3: Full Training (if pilot succeeds)

- Scale to all 12 targets
- Full 100K episodes per target
- Comprehensive evaluation vs v1 baseline

### Phase 4: Refinement (when TITAN available)

- Swap in TITAN scorer
- Retrain or fine-tune policies
- Compare results with weak-scorer baseline

---

## Alternative: Pause and Wait for TITAN

**If decoy discrimination test fails:**

- Do NOT proceed with RL training
- Focus on obtaining TITAN:
  - Ask user to manually download from GitHub
  - Try alternative networks/proxies
  - Contact TITAN authors for pretrained models
- Use waiting time to:
  - Implement and test other v2 components (env, policy, data loaders)
  - Prepare evaluation pipeline
  - Optimize ESM-2 inference speed

---

## Immediate Next Steps

1. **Run decoy discrimination test** (Task #10)
   - Critical blocker - must know if scorers can distinguish target vs decoy
   - 2-3 hours to implement and run
   - Clear go/no-go decision point

2. **Based on results**:
   - **If pass**: Proceed to ensemble evaluation and pilot RL
   - **If fail**: Pause RL, focus on obtaining TITAN

3. **Document decision** in progress_v2.md

---

## Risk Mitigation

- **Modular design**: Scorer is pluggable, can swap later
- **Checkpointing**: Save all intermediate results
- **Baseline comparison**: Always compare to v1 (ERGO-only)
- **Early stopping**: If pilot shows no improvement, stop before full training
- **Validation**: Use AlphaFold2 on top candidates (expensive but accurate)

---

## Conclusion

**Recommended**: Run decoy discrimination test first. If scorers can distinguish target from decoys (even weakly), proceed with pilot RL training. The multi-component reward and improved architecture may compensate for weak affinity scorer. If decoy test fails, pause and obtain TITAN before proceeding.

**Key insight**: V1 used ERGO alone and achieved some success. V2 has better architecture + ensemble + naturalness + diversity. Even with weak affinity scorer, v2 should outperform v1. The question is not "will it work perfectly" but "will it work better than v1".
