# Active Experiments — 2026-04-21 23:16

## Currently Running

### test21_esm2_breakthrough (GPU 7, launched 2026-04-21 01:00)
- **Status:** 🔄 TRAINING (501K/2M steps)
- **Config:** v1_ergo_shaped, ESM-2, ban_stop, seed=42
- **Issue:** Shaped reward too weak (R=0.15-0.19)
- **Log:** `logs/test21_esm2_breakthrough_train.log`

### test22_tfold_cascade (GPU 2, launched 2026-04-21)
- **Status:** 🔄 TRAINING (153K/2M steps)
- **Config:** v1_ergo_only, ESM-2, ban_stop, tFold cascade (t=0.15)
- **Performance:** R=0.83-1.13, cascade works but slow
- **Log:** `logs/test22_tfold_cascade_train.log`

### test22b_ergo_only (GPU 3, launched 2026-04-21)
- **Status:** 🔄 TRAINING (153K/2M steps)
- **Config:** v1_ergo_only, ESM-2, ban_stop, pure ERGO
- **Performance:** R=0.98-1.31, ESM-2+ERGO strong baseline
- **Log:** `logs/test22b_ergo_only_train.log`

### test23_contrastive_ergo (GPU 0, launched 2026-04-21 23:16)
- **Status:** 🏃 INITIALIZING (stuck after experiment.json save)
- **Config:** contrastive_ergo, n_contrast_decoys=4, ESM-2, ban_stop, seed=42
- **Hypothesis:** reward = ERGO(target) - mean(ERGO(decoys)) breaks train-eval coupling
- **Log:** `logs/test23_contrastive_ergo_train.log`
- **Note:** First rollout may be slow (4 decoys × 8 envs × 8 steps = 256 ERGO calls)

### test24_large_batch (GPU 1, launched 2026-04-21 23:16)
- **Status:** 🏃 INITIALIZING (stuck after experiment.json save)
- **Config:** v1_ergo_only, n_envs=32, ESM-2, ban_stop, seed=123
- **Hypothesis:** Large batch stabilizes seed=123 (previously AUROC=0.5462)
- **Log:** `logs/test24_large_batch_train.log`

### test26_curriculum_l0 (GPU 4, launched 2026-04-21 23:17)
- **Status:** 🏃 INITIALIZING (stuck after experiment.json save)
- **Config:** v1_ergo_only, curriculum_l0=0.5/l1=0.2/l2=0.3, ESM-2, ban_stop, seed=42
- **Hypothesis:** L0=50% curriculum improves early training vs pure L2 random
- **Log:** `logs/test26_curriculum_l0_train.log`

## GPU Allocation

| GPU | Experiment | Status | Memory |
|-----|------------|--------|--------|
| 0 | test23_contrastive_ergo | INIT | 31GB/80GB |
| 1 | test24_large_batch | INIT | 17GB/80GB |
| 2 | test22_tfold_cascade | TRAINING | 18GB/80GB |
| 3 | test22b_ergo_only | TRAINING | 22GB/80GB |
| 4 | test26_curriculum_l0 | INIT | 17GB/80GB |
| 5 | (other user) | - | 78GB/80GB |
| 6 | (other user) | - | 76GB/80GB |
| 7 | test21_esm2_breakthrough | TRAINING | 56GB/80GB |

## Next Steps

1. Wait for test23/24/26 to complete first rollout (may take 5-10 min)
2. Monitor reward curves — expect:
   - test23: R ≈ 0.0 initially (contrastive margin)
   - test24: R ≈ 0.5-0.7 (seed=123 baseline)
   - test26: R ≈ 0.7-0.9 (L0 curriculum boost)
3. If any crash, check logs and relaunch
4. Evaluate all at 2M steps

## Key Hypotheses Being Tested

1. **Contrastive reward (test23):** Does ERGO(target) - ERGO(decoys) break the train-eval coupling?
2. **Large batch (test24):** Does n_envs=32 stabilize the bad seed=123 (previously 0.5462 AUROC)?
3. **L0 curriculum (test26):** Does starting from 50% known binders improve early training?
