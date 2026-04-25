# test42: NetTCR Two-Phase Training - Quick Summary

**Created**: 2026-04-25  
**Status**: Ready to launch  
**Estimated Time**: 20-24 hours total

## Motivation

test41 achieved **0.6243 AUROC** (current best) using ERGO scorer in a three-phase training strategy:
1. 2M pure ERGO warm-start
2. 1.5M contrastive with 8 decoys  
3. 1M contrastive with 16 decoys

**Question**: Can we replicate this success with NetTCR scorer?

## Why NetTCR?

- **Architectural independence**: CNN vs ERGO's LSTM - breaks train-eval coupling
- **Different binding patterns**: BLOSUM50 encoding may capture complementary features
- **Validation**: Proves two-phase contrastive training is scorer-agnostic

## Three-Phase Strategy

| Phase | Steps | Config | Goal |
|-------|-------|--------|------|
| 1 | 2M | Pure NetTCR affinity | R ≈ 2.0 (match test22b) |
| 2 | 1.5M | NetTCR contrastive (8 decoys) | AUROC ≈ 0.58 (match test33) |
| 3 | 1M | NetTCR contrastive (16 decoys) | AUROC > 0.62 (match test41) |

## Launch Commands

```bash
# Phase 1: Pure NetTCR warm-start
bash scripts/launch_test42_nettcr_phase1.sh

# Wait for completion (~8-10 hours), then:
# Phase 2: Contrastive with 8 decoys
bash scripts/launch_test42_nettcr_phase2.sh

# Wait for completion (~6-8 hours), then:
# Phase 3: Increase to 16 decoys
bash scripts/launch_test42_nettcr_phase3.sh

# After completion (~4-5 hours):
# Evaluate
python tcrppo_v2/test_tcrs.py \
    --checkpoint output/test42_nettcr_phase3/checkpoints/final.pt \
    --n_tcrs 50 --n_decoys 50 \
    --output_dir results/test42_nettcr_phase3/
```

## Success Criteria

- **Phase 1**: Final reward R > 2.0
- **Phase 2**: Mean AUROC > 0.55
- **Phase 3**: Mean AUROC > 0.62 (ideally > 0.6243 to beat test41)

## If Successful

- NetTCR is a viable alternative to ERGO
- Two-phase contrastive training is scorer-agnostic
- Opens door to ensemble approaches (ERGO + NetTCR)

## If Failed (AUROC < 0.55)

- NetTCR may lack ERGO's expressiveness
- Consider hybrid: ERGO warm-start → NetTCR contrastive
- Or: Ensemble reward during training

## Files Created

- `docs/experiments/test42_nettcr_twophase.md` - Full experiment design
- `scripts/launch_test42_nettcr_phase1.sh` - Phase 1 launch script
- `scripts/launch_test42_nettcr_phase2.sh` - Phase 2 launch script
- `scripts/launch_test42_nettcr_phase3.sh` - Phase 3 launch script
- `docs/all_experiments_tracker.md` - Updated with test42 entry

## Monitoring

```bash
# Phase 1
tail -f logs/test42_nettcr_phase1_train.log

# Phase 2
tail -f logs/test42_nettcr_phase2_train.log

# Phase 3
tail -f logs/test42_nettcr_phase3_train.log
```

## Key Differences from test41

| Aspect | test41 | test42 |
|--------|--------|--------|
| Scorer | ERGO (LSTM) | NetTCR (CNN) |
| Embeddings | Learned | BLOSUM50 |
| Train-eval coupling | Yes | No |
| Architecture | Recurrent | Convolutional |
