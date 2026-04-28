# test43: ERGO + Curriculum Reward Schedule + Peptide Filtering (3 variants)

**Date**: 2026-04-28
**Status**: planned
**GPUs**: 0 (test43a), 4 (test43b), 6 (test43c)
**Priority**: P0

## Hypothesis

By (1) filtering training to 45 peptides where ERGO is discriminative (AUC >= 0.6), and
(2) applying curriculum reward schedules with naturalness and decoy components, we can
outperform test41's 0.6243 AUROC.

Three variants test different hypotheses on what drives improvement:
- **test43a**: Does 3-phase curriculum work from cold-start?
- **test43b**: Does naturalness improve test41's warm-start?
- **test43c (control)**: Is peptide filtering + more decoys sufficient without curriculum?

## Experiment Comparison Matrix

| Feature | test43a | test43b | test43c |
|---------|---------|---------|---------|
| Start | Cold | Warm (test41) | Warm (test41) |
| Peptide filter | 45 ERGO-good | 45 ERGO-good | 45 ERGO-good |
| Naturalness | Yes (phase 2) | Yes (phase 1) | **No** |
| Curriculum schedule | 3-phase | 2-phase | No schedule |
| Decoys (final) | 16 | 16 | **32** |
| Total steps | 3M | 2M | 2M |
| GPU | 0 | 4 | 6 |
| LR | 3e-4 (decays to 1e-4) | 1e-4 | 1e-4 |

## Configuration

### test43a: Cold-Start 3-Phase Curriculum (GPU 0)

```bash
CUDA_VISIBLE_DEVICES=0 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test43a_curriculum_cold \
    --seed 42 \
    --affinity_scorer ergo --encoder esm2 \
    --reward_mode v1_ergo_only \
    --total_timesteps 3000000 --n_envs 8 \
    --learning_rate 3e-4 --ban_stop \
    --train_targets data/ergo_good_peptides.txt \
    --reward_schedule '[{"step":0,"mode":"v1_ergo_only"},{"step":500000,"mode":"raw_multi_penalty","w_nat":0.1,"w_decoy":0.02,"w_diversity":0.0},{"step":1500000,"mode":"contrastive_ergo","n_decoys":16,"lr":1e-4}]' \
    --entropy_coef_final 0.01 --entropy_decay_start 500000 \
    --curriculum_l0 0.5 --curriculum_l1 0.2 --curriculum_l2 0.3
```

**Schedule**:
| Phase | Steps | Mode | Key Weights |
|-------|-------|------|-------------|
| 1 | 0-500K | v1_ergo_only | aff=1.0 |
| 2 | 500K-1.5M | raw_multi_penalty | nat=0.1, decoy=0.02 |
| 3 | 1.5M-3M | contrastive_ergo | 16 decoys, lr→1e-4 |

### test43b: Warm-Start + Naturalness Phase (GPU 4)

```bash
CUDA_VISIBLE_DEVICES=4 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test43b_curriculum_warm \
    --seed 42 \
    --affinity_scorer ergo --encoder esm2 \
    --reward_mode raw_multi_penalty \
    --total_timesteps 2000000 --n_envs 8 \
    --learning_rate 1e-4 --ban_stop \
    --train_targets data/ergo_good_peptides.txt \
    --resume_from output/test41_from_test33_1m_16decoys/checkpoints/final.pt \
    --resume_change_reward_mode raw_multi_penalty \
    --resume_reset_optimizer \
    --w_naturalness 0.1 --w_decoy 0.02 --w_diversity 0.0 \
    --reward_schedule '[{"step":0,"mode":"raw_multi_penalty","w_nat":0.1,"w_decoy":0.02,"w_diversity":0.0},{"step":500000,"mode":"contrastive_ergo","n_decoys":16}]' \
    --entropy_coef_final 0.01 --entropy_decay_start 0
```

**Schedule**:
| Phase | Steps | Mode | Key Weights |
|-------|-------|------|-------------|
| 1 | 0-500K | raw_multi_penalty | nat=0.1, decoy=0.02 |
| 2 | 500K-2M | contrastive_ergo | 16 decoys |

### test43c: 32 Decoys Control — No Curriculum (GPU 6)

```bash
CUDA_VISIBLE_DEVICES=6 python tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test43c_32decoys_filtered \
    --seed 42 \
    --affinity_scorer ergo --encoder esm2 \
    --reward_mode contrastive_ergo \
    --total_timesteps 2000000 --n_envs 8 \
    --learning_rate 1e-4 --ban_stop \
    --train_targets data/ergo_good_peptides.txt \
    --resume_from output/test41_from_test33_1m_16decoys/checkpoints/final.pt \
    --resume_change_reward_mode contrastive_ergo \
    --resume_reset_optimizer \
    --n_contrast_decoys 32 --contrastive_agg mean \
    --entropy_coef_final 0.01 --entropy_decay_start 0
```

No schedule — direct contrastive with 32 decoys (2x test41).

## Key Differences from Previous Experiments

- vs test41: All 3 add peptide filtering (45 vs 163 peptides) — NEVER tested before
- vs test41: test43a/b add naturalness component
- vs test41: test43c doubles decoy count (32 vs 16)
- vs test4 (raw_multi): Curriculum approach instead of static multi-penalty
- NEW: `--reward_schedule` auto-transitions reward modes within single run
- NEW: `--train_targets` filters training peptides to ERGO AUC >= 0.6

## What We Learn

- If **test43c > test43b**: naturalness hurts (consistent with Category 4 findings)
- If **test43b > test43c**: naturalness with curriculum helps after warm-start
- If **test43a > test41**: cold-start curriculum can match two-phase
- If **test43c > test41**: peptide filtering alone improves specificity
- All 3 test peptide filtering which has never been tested before

## Expected Outcome

- **test43a** (cold): AUROC >= 0.60 (comparable to test41 with naturalness benefit)
- **test43b** (warm): AUROC >= 0.65 (leveraging test41's warm-start + naturalness)
- **test43c** (control): AUROC >= 0.63 (peptide filtering + 32 decoys)

## Dependencies

- Code: `--train_targets`, `--reward_schedule` in ppo_trainer.py (DONE)
- Data: `data/ergo_good_peptides.txt` — 45 peptides (DONE)
- Checkpoint: `output/test41_from_test33_1m_16decoys/checkpoints/final.pt` (for test43b/c)
