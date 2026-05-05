# test44: Pure tFold PPO Training (No Cache-Only)

**Date**: 2026-05-03
**Status**: planned
**GPU**: 0
**Priority**: P0

## Hypothesis

tFold's structure-aware scoring provides better specificity than ERGO, achieving >0.65 AUROC (better than test41's 0.6243).

## Configuration

```bash
CUDA_VISIBLE_DEVICES=0 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test44_pure_tfold_nocache \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --affinity_scorer tfold \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 4 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/tfold_good_peptides.txt \
    --checkpoint_freq 50000
```

## Key Differences from Previous Experiments

- vs test41: Replaced ERGO with tFold as affinity scorer
- vs test41: Reduced n_envs from 8 to 4 (fewer parallel cache misses)
- vs test41: Reduced checkpoint_freq from 100K to 50K (early abort if too slow)
- vs test41: Using 29 tFold-trainable peptides (AUC ≥ 0.7) instead of 12 McPAS peptides
- NEW: cache_only=False (force real-time tFold scoring on all cache misses)
- NEW: Using tFold feature server (PID 1996296) instead of subprocess

## Target Peptides (29 tFold-trainable, AUC ≥ 0.7)

From PEPTIDE_SCORER_MAPPING.md:
- 20 excellent (AUC ≥ 0.8): GILGFVFTL, ELAGIGILTV, GLCTLVAML, RAKFKQLL, NLVPMVATV, CINGVCWTV, KLGGALQAK, LLWNGPMAV, YLQPRTFLL, AVFDRKSDAK, FLYALALLL, SLYNTVATL, IVTDFSVIK, SPRWYFYYL, RLRAEAQVK, ATDALMTGY, KRWIILGLNK, RLRPGGKKK, KAFSPEVIPMF, TPRVTGGGAM
- 9 good (0.7-0.8): IPSINVHHY, FLASKIGRLV, FRDYVDRFYKTLRAEQASQE, SLFNTVATLY, RLQSLQTYV, CLGGLLTMV, LLDFVRFMGV, EENLLDFVRF, HSKKKCDEL

## Speed Optimization Strategies

1. **tFold feature server**: Already running (PID 1996296), avoids ~5min model loading per subprocess
2. **Reduced n_envs**: 4 instead of 8 (fewer parallel cache misses)
3. **Smaller batch size**: max_subprocess_batch=32 (default)
4. **Pre-warmed cache**: 18,174 entries (9.8 GB) already cached
5. **Checkpoint frequency**: 50K steps (vs 100K) for early abort

## Expected Speed

- **Cache hit**: <1ms per sample (V3.4 classifier only)
- **Cache miss**: ~1s per sample (feature extraction via server)
- **Rollout size**: 4 envs × 128 steps = 512 samples
- **If 50% cache hit rate**: ~256 cache misses × 1s = 256s per rollout (~4 min)
- **If 90% cache hit rate**: ~51 cache misses × 1s = 51s per rollout (~1 min)
- **Total training time estimate**: 2M steps / (4 × 128) × 1-4 min = **33-533 hours (1.4-22 days)**

## Early Abort Criteria

- If cache hit rate < 50% at 100K steps → abort (training too slow)
- If average rollout time > 5 min at 200K steps → abort
- Monitor: `grep "Cache:" logs/test44_pure_tfold_train.log`

## Expected Outcome

- **Mean AUROC**: 0.65-0.70 (better than test41's 0.6243)
- **Cache hit rate**: Should increase over time as agent explores bounded CDR3 space
- **If successful**: Proves tFold's structure-aware scoring is superior to ERGO
- **If failed**: tFold may be too slow for practical RL training, or RL-generated TCRs are structurally implausible

## Risks

1. **Training too slow**: If cache hit rate stays low, training could take >1 month
   - Mitigation: Early abort at 100K steps if hit rate < 50%
2. **tFold server crashes**: Long-lived process may crash during training
   - Mitigation: Scorer has automatic restart logic
3. **RL-generated TCRs are structurally implausible**: tFold may score them poorly even if they bind
   - Evidence: TFOLD_EVAL_RESULTS_CORRECTED.md shows tFold has reversed discrimination on 7/12 peptides for RL-generated TCRs

## Dependencies

- tFold feature server running (PID 1996296)
- tFold cache at data/tfold_feature_cache.db (18,174 entries)
- 29 peptides in data/tfold_good_peptides.txt

## Monitoring

```bash
# Watch training progress
tail -f logs/test44_pure_tfold_train.log

# Check cache hit rate
grep "Cache:" logs/test44_pure_tfold_train.log | tail -20

# Check tFold server status
ps aux | grep tfold_feature_server

# Monitor GPU usage
watch -n 1 nvidia-smi
```
