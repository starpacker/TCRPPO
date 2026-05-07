# test51: Pure tFold with Terminal-Only Reward

**Date**: 2026-05-06
**Status**: running
**GPU**: 4
**Priority**: P0
**Started**: 2026-05-06 16:45
**PID**: 3803344 (training), 3802876 (tFold server)

## Hypothesis

Terminal-only reward (computing reward only at episode end) dramatically reduces tFold inference calls, enabling practical training with structure-aware scoring in <100h.

**Key insight**: Step-wise reward with tFold requires scoring at every step:
- 4 steps × (1 target + 2 decoys) = 12 tFold calls per episode
- With terminal reward: 1 initial + 1 target + 2 decoys = 4 tFold calls per episode (3x speedup)

Adding naturalness bonus (+0.1 * naturalness) provides additional signal without extra tFold calls.

## Configuration

```bash
CUDA_VISIBLE_DEVICES=0 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test51_tfold_terminal \
    --seed 42 \
    --reward_mode contrastive_ergo \
    --affinity_scorer tfold \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 4 \
    --ban_stop \
    --terminal_reward_only \
    --n_contrast_decoys 2 \
    --contrastive_agg mean \
    --w_naturalness 0.1 \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.0 \
    --curriculum_l2 0.5 \
    --train_targets data/tfold_excellent_peptides.txt \
    --eval_interval 999999999 \
    --checkpoint_interval 200000
```

## Key Differences from Previous Experiments

**vs test44 (pure tFold, step-wise)**:
- Added `--terminal_reward_only` flag → 3x fewer tFold calls
- Reduced `--max_steps` from 8 to 4 → 2x fewer steps per episode
- Reduced `--n_contrast_decoys` from 16 to 2 → 8x fewer decoy evaluations
- Added `--w_naturalness 0.1` → naturalness bonus for sequence quality
- **Combined speedup**: ~48x faster per episode (3 × 2 × 8)

**vs test49 (cascade ERGO→tFold)**:
- Pure tFold instead of cascade → no ERGO pre-filter
- Terminal reward instead of step-wise → GAE handles credit assignment
- Smaller decoy set (2 vs 16) → faster but still informative

**vs test41 (NetTCR contrastive)**:
- tFold scorer instead of NetTCR → structure-aware vs sequence-only
- Terminal reward instead of step-wise → optimized for slow scorer
- Curriculum L0+L2 only (no L1) → L1 seeds are ERGO-specific

**NEW features**:
- First experiment with `terminal_reward_only` mode
- First to use naturalness bonus in contrastive_ergo mode
- First to train on filtered peptide set (tfold_excellent_peptides.txt, 20 peptides with tFold AUC > 0.7)

## Expected Outcome

**Training time**: <100h (target), estimated ~12-24h actual
- 2M steps / 8 envs / 4 steps = 62,500 episodes
- First rollout: 8 samples × 8s = ~64s (cold-start, all cache misses)
- Ongoing: 62,500 episodes × 4 tFold calls × 8s × (1 - cache_hit_rate)
  - With 50% cache hit rate: ~62,500 × 4 × 8s × 0.5 = ~139h (too slow)
  - With 80% cache hit rate: ~62,500 × 4 × 8s × 0.2 = ~56h (acceptable)
  - With 90% cache hit rate: ~62,500 × 4 × 8s × 0.1 = ~28h (good)
- **Key assumption**: Cache hit rate reaches 80-90% after warmup (500K steps)

**Metrics**:
- Mean AUROC > 0.65 (target, vs v1 baseline 0.45)
- Mean target affinity ≥ v1 level (no regression)
- Naturalness score > test41 (due to explicit naturalness bonus)

**If successful**: Proves terminal reward enables practical tFold training
**If failed**: May need to increase decoys or revert to cascade approach

## Dependencies

**Code changes needed**:
- ✅ Add `terminal_reward_only` parameter to `env.py` (completed)
- ✅ Add `terminal_reward_only` CLI arg to `ppo_trainer.py` (completed)
- ✅ Add naturalness support to `contrastive_ergo` mode in `reward_manager.py` (completed)

**Data requirements**:
- ✅ `data/tfold_excellent_peptides.txt` (20 peptides, already exists)
- ✅ tFold feature cache at `data/tfold_feature_cache.db` (4.4GB, already exists)
- ✅ TCRdb data for L0/L2 curriculum (already exists)

**Blocks**: None

## Reward Formula

```
reward = affinity(TCR, target) - mean(affinity(TCR, decoy_A), affinity(TCR, decoy_B)) + 0.1 * naturalness(TCR)
```

Where:
- `affinity`: tFold binding score (structure-aware)
- `decoy_A`: 1 peptide from tier A (1-2 AA point mutants)
- `decoy_B`: 1 peptide from tier B (2-3 AA mutants)
- `naturalness`: ESM perplexity z-score (lower perplexity = more natural)

## Checkpoint Strategy

- Every 200K steps: `milestone_200000.pt`, `milestone_400000.pt`, ..., `milestone_2000000.pt`
- Latest: `latest.pt` (every 100K steps, overwritten)
- Final: `final.pt` (at 2M steps)

Total: 10 milestone checkpoints + 1 final

## Evaluation Protocol

**Training evaluation**: Disabled (`--eval_interval 999999999`)

**Final evaluation** (after training completes):
```bash
python tcrppo_v2/test_tcrs.py \
    --checkpoint output/test51_tfold_terminal/checkpoints/final.pt \
    --n_tcrs 10 \
    --n_decoys 32 \
    --affinity_scorer tfold \
    --output_dir results/test51_tfold_terminal/
```

Metrics:
- Per-target AUROC (10 TCRs vs 32 decoys per target)
- Mean AUROC across 20 targets
- Mean target affinity
- Mean decoy affinity
- Naturalness score distribution

## Risk Assessment

**Low risk**:
- Terminal reward is well-established in RL (standard for episodic tasks)
- GAE handles credit assignment for terminal rewards
- tFold cache hit rate should be high after warmup

**Medium risk**:
- Only 2 decoys per episode → may not provide enough specificity signal
  - Mitigation: Decoys are tier A+B (hardest negatives), not random
- Curriculum L1 disabled → may slow early learning
  - Mitigation: L0 (known binder variants) provides good warm start

**High risk**: None

## Monitoring Plan

**During training** (via `tail -f logs/test51_tfold_terminal_train.log`):
- Episode reward trend (should increase)
- tFold cache hit rate (should reach 70-80% after 500K steps)
- Episode length (should stabilize around 4 steps due to ban_stop)
- Naturalness component (should be positive, indicating bonus not penalty)

**Red flags** (abort if observed):
- Reward not increasing after 500K steps
- Cache hit rate < 50% after 1M steps (indicates poor exploration)
- Naturalness component consistently negative (indicates unnatural sequences)

## Next Steps

**If successful** (AUROC > 0.65):
- test52: Increase decoys to 4 (2A + 2B) to test specificity ceiling
- test53: Add diversity penalty to reduce sequence repetition
- test54: Extend to 5M steps to test convergence

**If failed** (AUROC < 0.55):
- Diagnose: Check if reward-AUROC alignment holds for tFold
- Fallback: Revert to cascade approach (test49 style) with terminal reward

## Launch Issues and Fixes

**Issue 1: 10-minute timeout too short for cold-start structure predictions**
- Error: `tFold server request timed out (10 min)` after first launch
- Root cause: 8 cold-start structure predictions take ~1-2 min each = 8-16 min total
- Fix: Increased timeout from 600s to 1800s (30 min) in `affinity_tfold.py:440`
- Result: Still timed out at 30 minutes

**Issue 2: 30-minute timeout still insufficient**
- Error: `tFold server request timed out (30 min)` after second launch
- Investigation: Server completed 8/8 samples in 1989.3s (33.15 min), average 248.7s/sample
- Server log showed "Batch done: 8/8 successful" but client got "Broken pipe" error
- Initial conclusion: Structure prediction too slow (4.1 min/sample)

**Issue 3: Speed discrepancy discovered - ROOT CAUSE**
- Observation: GPU 0's main tFold server runs at 7.8s/sample (32x faster)
- Investigation: GPU 4 server log showed "Loading tFold predictor on cpu"
- Root cause: Server was running on CPU, not GPU
- Diagnosis: Launch script used `python` which resolved to base environment with PyTorch 2.8.0+cpu
- tcrppo_v2 environment had correct CUDA PyTorch (2.7.1+cu118) but wasn't being used

**Issue 4: Wrong Python environment**
- Discovery: `python -c "import torch; print(torch.__version__)"` showed 2.8.0+cpu
- But `pip list` in tcrppo_v2 showed torch 2.7.1+cu118 was installed
- Fix: Changed launch command to use full path: `/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python`

**Issue 5: Missing dependencies in tcrppo_v2 environment**
- Error: `ModuleNotFoundError: No module named 'Bio'`
- Fix: Installed biopython, ml-collections, dm-tree, einops

**Final Success (2026-05-06 16:45)**
- Server started successfully on cuda:0 (physical GPU 4)
- Log: "Loading tFold predictor on cuda:0..."
- GPU 4 memory: 3359 MiB (model loaded)
- Server status: "READY"
- **Actual tFold speed: ~8 seconds/sample on GPU** (not 4 minutes on CPU)
- First rollout expected: ~64 seconds for 8 samples (well within 30-minute timeout)

## Failure Analysis (Initial Launch Attempts)

**Initial outcome**: ABORTED after 33 minutes — first rollout never completed

**Initial diagnosis (INCORRECT)**: Cold-start structure predictions are too slow and variable:
- 7 samples completed in 30 minutes (~4.3 min/sample average)
- 8th sample took >6 minutes and still processing when timeout hit
- Total time for 8 samples: >33 minutes (vs 8-16 min estimate)
- 30-minute timeout was still too short

**Why this initial diagnosis was wrong**:
- The server was running on CPU (PyTorch 2.8.0+cpu), not GPU
- CPU tFold speed: ~248 seconds/sample (4.1 minutes)
- GPU tFold speed: ~8 seconds/sample (31x faster)
- The problem was environment configuration, not fundamental tFold limitations

**Corrected understanding**:
1. **First rollout bottleneck**: 8 envs × 1 initial TCR = 8 cold-start predictions = ~64 seconds on GPU (not 30-40 min)
2. **Ongoing cold-start penalty**: Every new TCR generated during training requires structure prediction (~8s on GPU)
3. **Cache warmup is practical**: At 8s/sample, warming up 1000 TCRs takes ~2.2 hours (not 66-100 hours)
4. **Prediction time is consistent**: GPU predictions are stable at ~8s/sample

**Comparison to test44 (pure tFold, step-wise reward)**:
- test44: 0.5% progress in 30 hours (failed due to step-wise reward amplification)
- test51: Terminal reward reduces tFold calls from 12/episode to 4/episode (3x speedup)
- With GPU execution, test51 is viable

**Conclusion**: Pure tFold training with terminal reward IS viable with correct GPU execution. The initial failure was due to using CPU-only PyTorch, not fundamental algorithmic limitations.

## Recommended Next Steps

**Current status (2026-05-06 16:45)**: test51 is now RUNNING with correct GPU execution

**Monitoring plan**:
1. Verify first rollout completes in ~1-2 minutes (8 samples × 8s = 64s)
2. Monitor training loop starts outputting step counts, rewards, loss values
3. Verify terminal_reward_only is working (rewards only computed at episode end)
4. Verify naturalness component is being calculated
5. Monitor tFold cache hit rate increases over time (target: 80-90% after 500K steps)

**If successful** (AUROC > 0.65 after 2M steps):
- test52: Increase decoys to 4 (2A + 2B) to test specificity ceiling
- test53: Add diversity penalty to reduce sequence repetition
- test54: Extend to 5M steps to test convergence

**If cache hit rate stays low** (<50% after 1M steps):
- Diagnose: Check if policy is exploring too broadly (generating too many unique TCRs)
- Mitigation: Reduce entropy coefficient to concentrate policy
- Alternative: Pre-warm cache with L0/L1 seed TCRs

**If training is too slow** (>100h projected):
- Fallback Option 1: Cascade scorer with terminal reward (ERGO pre-filter → tFold for high-scoring TCRs)
- Fallback Option 2: Hybrid ERGO-tFold training (90% ERGO episodes, 10% tFold episodes)
