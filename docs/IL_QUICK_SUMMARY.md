# Imitation Learning Quick Summary

## What is it?

Behavior cloning pre-training to warm-start the RL policy for TCR design. The policy learns to imitate high-quality TCR editing demonstrations before RL fine-tuning.

## How does it work?

```
1. Collect high-affinity TCRs from RL logs + TC-Hard dataset
2. Generate editing demonstrations (trace replay + corrupt-reverse)
3. Train policy to predict expert actions (behavior cloning)
4. Use IL checkpoint to initialize RL training
```

## Quick Start

```bash
# Build IL dataset
python scripts/build_il_dataset.py \
  --targets data/tfold_excellent_peptides.txt \
  --trace-log logs/test62_*_train.log \
  --tc-hard /share/liuyutian/TCRdata/tc-hard/ds.csv \
  --out data/il/my_dataset.jsonl

# IL pre-training
python scripts/pretrain_il.py \
  --config configs/trace62_multi_gates.yaml \
  --dataset data/il/my_dataset.jsonl \
  --base-checkpoint output/test62_*/checkpoints/milestone_580000.pt \
  --out output/il_pretrain_my_run/checkpoints/latest.pt \
  --epochs 3

# RL fine-tuning
python -m tcrppo_v2.ppo_trainer \
  --config configs/trace62_multi_gates.yaml \
  --resume_from output/il_pretrain_my_run/checkpoints/latest.pt \
  --total_timesteps 2000000
```

## Current Results

| Metric | Base RL | IL Pre-trained |
|--------|---------|----------------|
| Mean Target Reward | -3.64 | -5.01 |
| Mean Target Prob | 5.19% | 1.24% |
| Training Loss | N/A | 5.79 |

**Note:** IL checkpoint underperforms base RL in standalone evaluation. This is expected—IL is designed as initialization for RL, not a standalone policy.

## Key Files

- **Dataset builder**: `scripts/build_il_dataset.py`
- **IL trainer**: `scripts/pretrain_il.py`
- **Datasets**: `data/il/*.jsonl`
- **Checkpoints**: `output/il_pretrain_*/checkpoints/`
- **Full guide**: `docs/IMITATION_LEARNING_GUIDE.md`

## Dataset Statistics

**Example: `highaff03_trace11_29_61_62_63_tchard_il.jsonl`**
- **Steps**: 11,555
- **Episodes**: 2,917
- **Endpoints**: 1,461 (93.5% from TC-Hard)
- **Methods**: 99.8% corrupt-reverse, 0.2% trace replay
- **Mean episode length**: 3.96 steps

## Training Configuration

```yaml
epochs: 3
batch_size: 128
learning_rate: 3e-5
optimizer: Adam (eps=1e-5)
gradient_clipping: 0.5
```

## Recommendations

1. **Use IL as warm-start**: Always follow with RL fine-tuning
2. **Filter endpoints**: Use affinity threshold ≥ 0.3
3. **Balance methods**: Mix trace replay and corrupt-reverse
4. **Validate quality**: Check dataset statistics before training
5. **Monitor convergence**: Loss should decrease to ~5-6 after 3 epochs

## Common Issues

**Issue**: IL checkpoint performs worse than base RL  
**Solution**: This is expected. Use IL checkpoint to initialize RL training, not as standalone policy.

**Issue**: Training loss doesn't converge  
**Solution**: Reduce learning rate to 1e-5, increase batch size to 256.

**Issue**: Out of memory  
**Solution**: Reduce batch size to 64 or 32.

## Next Steps

1. **Evaluate IL+RL**: Train RL from IL checkpoint and compare to RL-only baseline
2. **Improve dataset**: Increase trace replay ratio, filter low-affinity endpoints
3. **Ablation studies**: Test different affinity thresholds, corruption depths
4. **Online IL**: Collect demonstrations during RL training for continuous improvement

---

**See full documentation**: `docs/IMITATION_LEARNING_GUIDE.md`
