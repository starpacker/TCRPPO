# 🎯 Strategy to Break Mean Affinity -0.5 Today

## 📊 Current Status Analysis

### Best Performing Traces
1. **trace75**: Mean -1.059, Best -0.469 (only 13 episodes, too early)
2. **trace73**: Mean -1.182, Recent -1.128, Best 0.834 (RUNNING, 4288 episodes)
3. **trace70**: Mean -1.216, Best 0.702
4. **trace72**: Mean -1.237, Recent -1.122, Best 0.747 (13,352 episodes)

### Key Observations
- **trace73** recent episodes show affinities: -0.53, -0.67, -0.71, -0.84
- Some episodes already reaching **-0.4 to -0.5 range**
- Current gate schedule: -2.0 → -1.5 → -1.0 → -0.5
- Step 678,400 (should be at gate -1.5 or approaching -1.0)

### What's Working
✅ tFold scorer (more accurate than ERGO)
✅ Dynamic band selection from online pool
✅ Terminal reward only (cleaner signal)
✅ Target gating mechanism
✅ Curriculum learning

### What's Limiting
❌ Mean affinity still dragged down by poor initial TCRs
❌ Only 1.3% episodes > 0 (need to increase this)
❌ Gate schedule may be too conservative
❌ Online pool ratio only 0.5 (could be higher)

## 🚀 Action Plan to Break -0.5 Today

### Option 1: Aggressive Continuation from trace73 (RECOMMENDED)
**Start a new trace from trace73's latest checkpoint with aggressive settings**

```yaml
# trace78: Aggressive push from trace73
# Goal: Mean affinity > -0.5 within 50K steps

# Base from trace73 latest (step ~678K)
resume_from: output/trace73_curriculum_exploration/checkpoints/latest.pt

# AGGRESSIVE CHANGES:
1. Skip curriculum, go straight to gate -0.5
   target_affinity_gate: -0.5  # No more easy gates

2. Increase online pool ratio to 0.85 (from 0.5)
   online_tcr_pool_max_ratio: 0.85
   online_tcr_pool_min_affinity: -0.8  # Only keep good TCRs

3. Filter online pool to only positive-affinity TCRs
   online_tcr_pool_sample_bands: [[-0.8, 0.0], [0.0, 0.6], [0.6, 2.0]]
   # Focus on TCRs that already work

4. Increase target bonus
   target_affinity_bonus: 2.0  # Reward good TCRs more

5. Reduce exploration (we know what works)
   entropy_coef: 0.008  # Down from 0.020

6. Increase learning rate for faster adaptation
   learning_rate: 2.0e-4  # Up from 1.2e-4

7. Add elite preservation
   preserve_elite: true
   elite_threshold: 0.0
   elite_pool_size: 128
```

**Expected outcome:** Mean affinity -0.8 → -0.5 in ~30-50K steps

### Option 2: Fresh Start with Elite Initialization
**Start from scratch but initialize with best TCRs from trace73**

```yaml
# trace79: Elite-only training
# Goal: Train only on proven high-affinity TCRs

# Extract top 500 TCRs from trace73 (affinity > -0.5)
# Use as L1 seeds

curriculum_schedule:
  - {until: null, L0: 0.0, L1: 1.0, L2: 0.0}  # Only L1 (elite seeds)

online_tcr_pool_enabled: true
online_tcr_pool_start_step: 0
online_tcr_pool_max_ratio: 0.95  # Almost all from pool
online_tcr_pool_min_affinity: -0.5  # STRICT filter

target_affinity_gate: -0.3  # Even stricter
```

**Expected outcome:** Mean affinity > -0.5 from the start

### Option 3: Hybrid - Continue trace72 with Filtering
**trace72 has 13K episodes, good foundation**

```yaml
# trace80: Filtered continuation of trace72
resume_from: output/trace72_delta_from_trace70/checkpoints/latest.pt

# Add strict filtering
online_tcr_pool_min_affinity: -0.6
online_tcr_pool_max_ratio: 0.90

# Use delta reward (trace72's strength)
use_delta_reward: true

# Strict gate
target_affinity_gate: -0.4
```

## 🎯 My Recommendation: Option 1 (trace78)

**Why:**
1. trace73 is already close (recent mean -1.128)
2. Has good TCRs in pool (1.3% > 0, best 0.834)
3. Just needs aggressive push to filter out bad TCRs
4. Can achieve -0.5 in 1-2 hours of training

**Implementation:**

```bash
# 1. Stop trace73 (or let it continue, we'll branch)
# 2. Create config
cat > configs/trace78_aggressive_push.yaml << 'EOF'
# trace78: Aggressive push to break -0.5
# Base: trace73 latest checkpoint

total_timesteps: 100000  # Short run, focused goal
n_envs: 8
learning_rate: 2.0e-4  # Faster adaptation
entropy_coef: 0.008    # Less exploration
hidden_dim: 512
max_steps: 8
ban_stop: true
terminal_reward_only: true

reward_mode: "v2_simple_target_gated_decoy"
w_affinity: 1.0
w_decoy: 0.3
w_naturalness: 0.05
w_diversity: 0.02

# AGGRESSIVE SETTINGS
target_affinity_gate: -0.5  # Strict gate
target_affinity_bonus: 2.0  # High reward for passing

# Elite pool focus
online_tcr_pool_enabled: true
online_tcr_pool_start_step: 0
online_tcr_pool_max_ratio: 0.85  # Mostly from pool
online_tcr_pool_min_affinity: -0.8  # Only good TCRs
online_tcr_pool_max_per_target: 512  # Larger pool
online_tcr_pool_min_hamming: 2

# Dynamic band selection (focus on high-affinity bands)
online_tcr_pool_use_dynamic_bands: true
online_tcr_pool_recent_window: 100

# tFold scorer
affinity_model: "tfold"
tfold_server_socket: "/tmp/tfold_server_trace78_aggressive_push.sock"
tfold_cache_path: "data/tfold_feature_cache.db"  # Reuse existing cache

# L2 only (random TCRdb)
curriculum_schedule:
  - {until: null, L0: 0.0, L1: 0.0, L2: 1.0}

train_targets: "data/tfold_excellent_peptides.txt"
encoder: "esm2"
esm_device: "cuda:2"  # Use GPU 2

checkpoint_interval: 5000
milestones: [10000, 20000, 30000, 50000, 75000, 100000]
EOF

# 3. Launch tFold server
nohup python scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_trace78_aggressive_push.sock \
    --gpu 2 \
    --use-amp-wrapper \
    --chunk-size 64 \
    --completion-log logs/trace78_aggressive_push_tfold_completion.log \
    > logs/trace78_aggressive_push_tfold_server.log 2>&1 &

# 4. Launch training
CUDA_VISIBLE_DEVICES=2 nohup python -u tcrppo_v2/ppo_trainer.py \
    --config configs/trace78_aggressive_push.yaml \
    --run_name trace78_aggressive_push \
    --seed 42 \
    --resume_from output/trace73_curriculum_exploration/checkpoints/latest.pt \
    --resume_reset_optimizer \
    > logs/trace78_aggressive_push_train.log 2>&1 &
```

## 📈 Expected Timeline

- **0-10K steps**: Mean affinity -1.1 → -0.9 (pool builds up)
- **10K-30K steps**: Mean affinity -0.9 → -0.7 (filtering takes effect)
- **30K-50K steps**: Mean affinity -0.7 → -0.5 (target reached!)
- **50K-100K steps**: Stabilize around -0.4 to -0.5

**Estimated time:** 2-3 hours on GPU 2

## 🔍 Monitoring

```bash
# Watch progress
watch -n 10 'tail -30 logs/trace78_aggressive_push_train.log | grep "Step"'

# Check mean affinity
python analyze_affinity_distribution.py

# Detailed analysis
python visualize_single_trace.py 78
```

## 🎯 Success Criteria

✅ Mean affinity > -0.5
✅ Recent mean (last 500) > -0.4
✅ % episodes > 0: at least 5%
✅ % episodes > 0.6: at least 1%

## 🔄 Backup Plan

If trace78 doesn't reach -0.5 in 50K steps:

**Plan B: Extract elite TCRs and retrain**
```bash
# Extract all TCRs with affinity > -0.5 from trace73 online pool
# Use as L1 seeds for trace79
# Train with strict filtering from the start
```

---

**Ready to launch? Let me know and I'll help you set it up!**
