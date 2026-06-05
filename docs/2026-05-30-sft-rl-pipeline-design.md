# SFT → RL Pipeline Design for TCRPPO v2

**Date**: 2026-05-30  
**Goal**: Achieve mean affinity = 0.0 through supervised fine-tuning followed by RL fine-tuning  
**Status**: Design approved, ready for implementation

---

## Executive Summary

This design introduces a **two-stage training pipeline** to improve TCR design quality:

1. **Stage 1: Supervised Fine-Tuning (SFT)** — Train the policy to imitate high-quality editing trajectories extracted from 171M tFoldScore records across 157 training logs
2. **Stage 2: RL Fine-Tuning** — Refine the SFT policy using PPO + online pool to push mean affinity from ~-0.5 to 0.0

**Key Innovation**: Instead of learning from scratch (trace73: mean affinity -1.172), we bootstrap from expert demonstrations, dramatically reducing exploration cost.

---

## 1. Motivation

### Current Baseline (trace73)
- **Training**: 5,088 episodes, pure RL from curriculum initialization
- **Mean affinity**: -1.172
- **Episodes > 0.0**: 73 (1.43%)
- **Problem**: Massive wasted exploration on low-quality TCRs

### Proposed Approach
- **SFT**: Learn from 80K expert trajectories (affinity -4 to +0.8)
- **Expected SFT output**: mean affinity -0.5 to -0.3
- **RL fine-tuning**: Focus on refinement (-0.5 → 0.0), not discovery

**Analogy**: AlphaGo = SFT (human games) + RL (self-play)

---

## 2. Data Landscape

### Available Data Sources

| Source | Records | Contains TCR? | Contains Peptide? | Contains Affinity? |
|--------|---------|---------------|-------------------|-------------------|
| **tFoldScore lines** (157 logs) | **1,711,938** | ✅ cdr3b | ✅ peptide | ✅ affinity_logit |
| Episode lines (training logs) | 169,380 | ❌ | ✅ (inferred) | ✅ A, InitA |
| Online pool snapshots | ~10 files | ✅ | ✅ | ✅ |

### Affinity Distribution (tFoldScore records)

| Bin | Count | Percentage |
|-----|-------|------------|
| A ≥ 0 | 2,661 | 0.2% |
| -2 ≤ A < 0 | 276,777 | 16.2% |
| -4 ≤ A < -2 | 158,086 | 9.2% |
| A < -4 | 1,274,414 | 74.4% |

**Key Insight**: We have 279K high-quality samples (A > -2) — more than enough for SFT.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Data Extraction & Preprocessing (2-3 hours)        │
├─────────────────────────────────────────────────────────────┤
│ Input: 157 training logs (1.71M tFoldScore records)         │
│   ↓                                                          │
│ 1. Parse tFoldScore lines → (TCR, peptide, affinity)        │
│ 2. Match InitTCR ↔ FinalTCR (via affinity + peptide)        │
│ 3. Stratified sampling:                                      │
│    - A≥0: 2,661 (keep all)                                  │
│    - -2≤A<0: 276,777 → sample 20K                           │
│    - -4≤A<-2: 158,086 → sample 20K                          │
│   ↓                                                          │
│ 4. Trajectory reconstruction (Levenshtein + random paths):  │
│    - A≥0: 1x shortest path                                  │
│    - -2≤A<0: 2x random paths (data augmentation)            │
│    - -4≤A<-2: 1x shortest path                              │
│   ↓                                                          │
│ Output: data/sft_dataset.json (~80K training samples)       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Phase 2: SFT Training (6-8 hours)                           │
├─────────────────────────────────────────────────────────────┤
│ Input: sft_dataset.json (80K samples)                       │
│   ↓                                                          │
│ Model: Reuse existing ActorCritic (policy.py)               │
│ Training:                                                    │
│   - 50 epochs, batch_size=64                                │
│   - Stratified sampling (3 bins 1:1:1 per batch)            │
│   - Loss: CrossEntropy (action prediction)                  │
│   - Optimizer: Adam, lr=1e-4                                │
│   ↓                                                          │
│ Output: output/sft_checkpoint_best.pt                       │
│         Validation: sample 100 TCRs, mean affinity > -0.5   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Phase 3: RL Fine-tuning (1-2 days)                          │
├─────────────────────────────────────────────────────────────┤
│ Input: sft_checkpoint_best.pt                               │
│   ↓                                                          │
│ Launch PPO:                                                  │
│   - Load SFT checkpoint                                     │
│   - Online pool (min_affinity=-0.5)                         │
│   - 1M steps, n_envs=8                                      │
│   - Reward: trace73 config (target + decoy gate)           │
│   - Learning rate: 3e-5 (10x smaller than from scratch)    │
│   ↓                                                          │
│ Output: output/trace_sft_finetune/checkpoints/final.pt     │
│         Goal: mean affinity = 0.0                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Design Decisions

| Decision Point | Choice | Rationale |
|----------------|--------|-----------|
| **SFT Approach** | Imitation Learning (edit trajectories) | Reuses existing ActorCritic architecture, seamless RL transition |
| **Trajectory Reconstruction** | Hybrid strategy (stratified by affinity) | High-quality: precise imitation; Medium: data augmentation; Low: basic patterns |
| **Loss Design** | Stratified sampling + standard CrossEntropy | Stable, avoids gradient issues from weighting |
| **RL Initialization** | Warm start + online pool | Proven effective in trace73, no KL penalty needed |
| **Data Scale** | Full dataset (80K samples) | Covers all peptides, strong generalization |

---

## 5. Module Specifications

### 5.1 Log Parser (`scripts/extract_sft_data.py`)

**Purpose**: Extract (InitTCR, FinalTCR, peptide, affinity) tuples from all training logs

**Algorithm**:
1. Scan tFoldScore lines, group by timestamp (batches of n_envs=8)
2. First batch = InitTCR (initial scoring)
3. Second batch = FinalTCR (post-edit scoring)
4. Match Episode lines via peptide + affinity value
5. Extract (InitTCR, FinalTCR, peptide, InitA, FinalA)

**Stratified Sampling**:
- A≥0: keep all 2,661 samples
- -2≤A<0: random sample 20K from 276,777
- -4≤A<-2: random sample 20K from 158,086
- **Total**: ~43K samples

**Output Format** (`data/sft_raw_pairs.json`):
```json
{
  "init_tcr": "CASSIDHGSGNEQFF",
  "final_tcr": "CALSIDHGSGNAQYFCCC",
  "peptide": "GLCTLVAML",
  "init_affinity": -6.8148,
  "final_affinity": -0.6653,
  "delta_affinity": 6.1495,
  "source_log": "trace73_curriculum_exploration_train.log",
  "episode_id": 1
}
```

---

### 5.2 Trajectory Reconstructor (`scripts/reconstruct_trajectories.py`)

**Purpose**: Convert (InitTCR → FinalTCR) pairs into edit action sequences

**Strategy Selection** (by affinity):
- **A≥0**: `shortest_path()` — 1x Levenshtein shortest path
- **-2≤A<0**: `random_path()` — 2x random paths (data augmentation)
- **-4≤A<-2**: `shortest_path()` — 1x Levenshtein shortest path

**Algorithms**:

1. **Shortest Path** (Levenshtein):
   ```python
   def shortest_path(s1: str, s2: str) -> List[Action]:
       """
       Dynamic programming edit distance
       Returns: [(op_type, position, token), ...]
       op_type: 0=SUB, 1=INS, 2=DEL, 3=STOP
       """
   ```

2. **Random Path** (data augmentation):
   ```python
   def random_path(s1: str, s2: str, max_steps: int = 8) -> List[Action]:
       """
       Generate random valid path from s1 to s2
       
       Strategy:
       1. Compute Levenshtein distance d
       2. Choose d+k steps (k ∈ [0,2] random)
       3. Each step: random SUB/INS/DEL (ensure reachability to s2)
       4. Final step: STOP
       """
   ```

**Output Format** (`data/sft_dataset.json`):
```json
{
  "init_tcr": "CASSIDHGSGNEQFF",
  "final_tcr": "CALSIDHGSGNAQYFCCC",
  "peptide": "GLCTLVAML",
  "final_affinity": -0.6653,
  "affinity_bin": "A>=0",
  "trajectory": [
    {"step": 0, "op": 0, "pos": 1, "token": 0, "state": "CALSIDHGSGNEQFF"},
    {"step": 1, "op": 1, "pos": 14, "token": 2, "state": "CALSIDHGSGNEQFFC"},
    {"step": 2, "op": 1, "pos": 15, "token": 2, "state": "CALSIDHGSGNEQFFCC"},
    ...
    {"step": 7, "op": 3, "pos": 0, "token": 0, "state": "CALSIDHGSGNAQYFCCC"}
  ]
}
```

**Data Augmentation Result**:
- Input: 43K pairs
- Output: ~80K trajectories (2x for medium-quality samples)

---

### 5.3 SFT Dataset (`tcrppo_v2/data/sft_dataset.py`)

**Purpose**: PyTorch Dataset with stratified sampling support

```python
class StratifiedSFTDataset(Dataset):
    """
    Stratified SFT dataset
    
    Features:
    - Store samples in 3 bins by affinity
    - Support stratified sampling (1:1:1 per batch)
    - Dynamic ESM-2 embedding generation (reuse ESMCache)
    """
    
    def __init__(self, data_path: str, esm_cache, pmhc_loader):
        # Load JSON, group by affinity_bin
        self.bins = {
            'A>=0': [],      # 2,661 samples
            '-2<=A<0': [],   # 40,000 samples (20K pairs × 2 paths)
            '-4<=A<-2': []   # 20,000 samples
        }
        self.esm_cache = esm_cache
        self.pmhc_loader = pmhc_loader
        
    def __getitem__(self, idx):
        """
        Returns:
            obs: [2562] ESM-2 embedding (TCR + pMHC)
            actions: List[(op, pos, token)] edit sequence
            masks: Dict[str, Tensor] action masks per step
            peptide: str
            final_affinity: float
        """
```

**StratifiedBatchSampler**:
```python
class StratifiedBatchSampler(Sampler):
    """
    Ensure 3 bins evenly distributed in each batch
    
    batch_size=64 → 21-22 samples per bin
    """
    
    def __iter__(self):
        # Sample batch_size//3 from each bin
        # Shuffle and return
```

---

### 5.4 SFT Trainer (`scripts/train_sft.py`)

**Configuration** (`configs/sft_default.yaml`):
```yaml
model:
  obs_dim: 2562
  hidden_dim: 512
  
training:
  epochs: 50
  batch_size: 64
  learning_rate: 1e-4
  warmup_steps: 1000
  grad_clip: 1.0
  weight_decay: 0.01
  
data:
  train_split: 0.9
  val_split: 0.1
  stratified_sampling: true
  
validation:
  val_every: 1000
  n_samples: 100
  target_affinity_threshold: -0.5
  
logging:
  log_every: 100
  save_every: 5000
  tensorboard: true
```

**Training Loop**:
```python
class SFTTrainer:
    def __init__(self, policy, dataset, config):
        self.policy = policy  # Reuse ActorCritic
        self.dataset = dataset
        self.optimizer = Adam(policy.parameters(), lr=config.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        
    def train_epoch(self):
        for batch in self.dataloader:
            obs, actions, masks, peptides = batch
            # obs: [B, 2562], actions: [B, max_steps, 3]
            
            total_loss = 0
            current_obs = obs.clone()
            
            # Teacher forcing: predict each step given current state
            for step in range(actions.shape[1]):
                op_target = actions[:, step, 0]
                pos_target = actions[:, step, 1]
                tok_target = actions[:, step, 2]
                
                # Forward pass
                logits_op, logits_pos, logits_tok = self.policy.actor(
                    current_obs, 
                    action_masks=masks[step]
                )
                
                # Loss (cross-entropy with masking)
                loss_op = F.cross_entropy(logits_op, op_target)
                loss_pos = F.cross_entropy(
                    logits_pos, pos_target, 
                    reduction='none'
                ) * masks[step]['pos_valid']
                loss_tok = F.cross_entropy(
                    logits_tok, tok_target,
                    reduction='none'
                ) * masks[step]['tok_valid']
                
                step_loss = loss_op + loss_pos.mean() + loss_tok.mean()
                total_loss += step_loss
                
                # Update state for next step (apply action)
                current_obs = self.apply_action(
                    current_obs, op_target, pos_target, tok_target
                )
            
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            
        self.scheduler.step()
    
    def validate(self):
        """
        Validation: sample 100 TCRs, compute mean affinity
        
        Goal: mean affinity > -0.5
        """
        with torch.no_grad():
            tcrs = []
            peptides_to_test = random.sample(self.val_peptides, 10)
            
            for peptide in peptides_to_test:
                for _ in range(10):
                    # Sample TCR using SFT policy
                    tcr = self.sample_tcr(peptide, max_steps=8)
                    tcrs.append((tcr, peptide))
            
            # Score with tFold
            affinities = self.score_batch(tcrs)
            mean_aff = np.mean(affinities)
            median_aff = np.median(affinities)
            best_aff = np.max(affinities)
            
            print(f"Validation: mean={mean_aff:.4f}, median={median_aff:.4f}, best={best_aff:.4f}")
            
            return mean_aff
    
    def sample_tcr(self, peptide: str, max_steps: int = 8) -> str:
        """
        Sample TCR using SFT policy (autoregressive generation)
        """
        # Initialize from L1 seed or random
        tcr = self.tcr_pool.sample(peptide, level='L1')
        
        for step in range(max_steps):
            obs = self.get_observation(tcr, peptide)
            masks = self.get_action_masks(tcr, step)
            
            # Sample action
            op, pos, tok = self.policy.sample_action(obs, masks)
            
            if op == OP_STOP:
                break
            
            # Apply action
            tcr = self.apply_edit(tcr, op, pos, tok)
        
        return tcr
```

**Early Stopping**:
- Track validation affinity every 1K steps
- If no improvement for 5 consecutive validations, stop training
- Save best checkpoint (highest validation affinity)

---

### 5.5 RL Fine-tuning Launcher (`scripts/launch_sft_finetune.sh`)

**Configuration** (`configs/trace_sft_finetune.yaml`):
```yaml
resume_from: output/sft_checkpoint_best.pt

reward_mode: v2_simple_target_gated_decoy
weights:
  affinity: 1.0
  decoy: 0.3
  naturalness: 0.05
  diversity: 0.02

target_affinity_gate: -0.5  # Stricter than trace73 (-2.0)

online_pool:
  enabled: true
  min_affinity: -0.5
  max_per_target: 256
  min_hamming: 2
  warmup_episodes: 100

curriculum:
  enabled: false  # SFT provides good initialization

training:
  total_timesteps: 1000000
  n_envs: 8
  learning_rate: 3e-5  # 10x smaller than from-scratch
  max_steps: 8
  ban_stop: true
  terminal_reward_only: true
```

**Launch Script**:
```bash
#!/bin/bash
# scripts/launch_sft_finetune.sh

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/trace_sft_finetune.yaml \
    --run_name trace_sft_finetune \
    --resume_from output/sft_checkpoint_best.pt \
    > logs/trace_sft_finetune_train.log 2>&1 &

echo "RL fine-tuning launched on GPU 0"
echo "Monitor: tail -f logs/trace_sft_finetune_train.log"
echo "Expected: mean affinity -0.5 → 0.0 in 500K-1M steps"
```

---

## 6. Success Metrics

| Stage | Metric | Target | Validation Method |
|-------|--------|--------|-------------------|
| **Data Extraction** | Sample count | ~80K | Check JSON file size |
| | Bin ratio | 1:1:1 | Count samples per bin |
| | InitTCR-FinalTCR pairs | 100% matched | Verify affinity alignment |
| **SFT Training** | Validation affinity | > -0.5 | Sample 100 TCRs every 1K steps |
| | Training loss | Converging | TensorBoard curve |
| | Best checkpoint | Saved | File exists + metadata |
| **RL Fine-tuning** | Mean affinity | 0.0 | Evaluate every 100K steps |
| | Episodes > 0.0 | > 10% | Count from episode logs |
| | Online pool size | > 15 targets | Check pool snapshot |

---

## 7. Monitoring & Debugging

### 7.1 Data Extraction Monitoring

```bash
# scripts/monitor_extraction.sh
watch -n 5 'wc -l data/sft_raw_pairs.json && \
            python3 -c "import json; d=json.load(open(\"data/sft_raw_pairs.json\")); \
            print(f\"Bins: A>=0={sum(1 for x in d if x[\"final_affinity\"]>=0)}, \
            -2<=A<0={sum(1 for x in d if -2<=x[\"final_affinity\"]<0)}, \
            -4<=A<-2={sum(1 for x in d if -4<=x[\"final_affinity\"]<-2)}\")"'
```

### 7.2 SFT Training Monitoring

```python
# scripts/monitor_sft_training.py
import re, time, json

log_file = "logs/sft_training.log"

while True:
    with open(log_file) as f:
        lines = f.readlines()
    
    # Extract latest validation affinity
    for line in reversed(lines[-100:]):
        if "Validation:" in line:
            match = re.search(r"mean=([-\d.]+).*median=([-\d.]+).*best=([-\d.]+)", line)
            if match:
                mean_aff = float(match.group(1))
                median_aff = float(match.group(2))
                best_aff = float(match.group(3))
                
                print(f"[{time.strftime('%H:%M:%S')}] Val: mean={mean_aff:.4f}, median={median_aff:.4f}, best={best_aff:.4f}")
                
                if mean_aff > -0.5:
                    print("✅ Target reached! Mean affinity > -0.5")
                break
    
    time.sleep(60)
```

### 7.3 RL Fine-tuning Monitoring

```bash
# Reuse existing monitoring scripts
python3 visualize_alive_traces.py --run_name trace_sft_finetune
python3 plot_all_alive_affinities.py --run_name trace_sft_finetune
```

---

## 8. Risk Mitigation

### Risk 1: SFT Overfitting
**Symptom**: Training loss → 0, but validation affinity stays low  
**Mitigation**:
- Use 90/10 train/val split
- Early stopping on validation affinity
- Data augmentation (random paths for medium-quality samples)

### Risk 2: Policy Collapse in RL
**Symptom**: RL affinity drops below SFT baseline  
**Mitigation**:
- Use smaller learning rate (3e-5 vs 3e-4)
- Monitor online pool size (should grow, not shrink)
- If collapse detected, add KL penalty (optional fallback)

### Risk 3: Trajectory Reconstruction Errors
**Symptom**: Generated trajectories don't reach FinalTCR  
**Mitigation**:
- Unit test: verify all trajectories reach target state
- Log reconstruction failures, inspect manually
- Fallback: skip invalid pairs (should be <1%)

### Risk 4: Insufficient High-Quality Data
**Symptom**: Only 2,661 samples with A≥0  
**Mitigation**:
- Medium-quality samples (-2≤A<0) are still valuable (276K available)
- Data augmentation doubles medium-quality samples
- SFT learns from gradient, not just best samples

---

## 9. Timeline & Resource Allocation

| Phase | Duration | GPU | CPU | Storage |
|-------|----------|-----|-----|---------|
| **Data Extraction** | 2-3 hours | 0 | 1 core | 5 GB (JSON) |
| **Trajectory Reconstruction** | 1-2 hours | 0 | 4 cores | 10 GB (JSON) |
| **SFT Training** | 6-8 hours | 1x A800 | 4 cores | 2 GB (checkpoints) |
| **RL Fine-tuning** | 1-2 days | 1x A800 | 4 cores | 5 GB (checkpoints + logs) |
| **Total** | **3-4 days** | 1x A800 | 4 cores | 22 GB |

**Parallelization Opportunities**:
- Data extraction can run on CPU while other experiments use GPU
- SFT training and RL fine-tuning are sequential (cannot parallelize)

---

## 10. Future Improvements

### 10.1 Curriculum SFT
Instead of uniform sampling, gradually increase difficulty:
- Epochs 1-10: Only A≥0 samples
- Epochs 11-30: A≥0 + (-2≤A<0)
- Epochs 31-50: All 3 bins

**Benefit**: Faster convergence, better final performance

### 10.2 Multi-Task SFT
Train on multiple objectives simultaneously:
- Primary: predict next action
- Auxiliary: predict final affinity (regression head)

**Benefit**: Better state representations

### 10.3 Iterative SFT-RL
After RL fine-tuning, extract new high-quality trajectories and retrain SFT:
- Round 1: SFT (logs) → RL → mean affinity 0.0
- Round 2: SFT (logs + RL trajectories) → RL → mean affinity 0.5

**Benefit**: Continuous improvement loop

---

## 11. Comparison with Alternatives

### Alternative A: Pure RL (Current Baseline)
- **Pros**: No data extraction needed
- **Cons**: Slow convergence, high exploration cost
- **Result**: trace73 achieved -1.172 after 5K episodes

### Alternative B: Behavioral Cloning (BC) Only
- **Pros**: Simpler than SFT+RL
- **Cons**: Cannot improve beyond expert demonstrations
- **Result**: Would plateau at ~-0.5 (best in logs)

### Alternative C: Offline RL (CQL, IQL)
- **Pros**: Learns from offline data like SFT
- **Cons**: More complex, requires value function training
- **Result**: Unclear if better than SFT+RL for this task

**Conclusion**: SFT+RL is the best balance of simplicity and performance.

---

## 12. Appendix: Log Format Reference

### tFoldScore Line Format
```
[tFoldScore] ts=2026-05-28 06:20:18 source=extract_ok path_ms=9046.87 classify_ms=16.69 end_to_end_ms=9114.48 affinity_logit=-6.8148 conf=1.00 cdr3b=CASSIDHGSGNEQFF peptide=GLCTLVAML hla=HLA-A*02:01
```

**Key Fields**:
- `affinity_logit`: tFold affinity score (raw logit)
- `cdr3b`: TCR CDR3β sequence
- `peptide`: Target peptide sequence
- `hla`: HLA allele (always A*02:01 in current data)

### Episode Line Format
```
Episode 1 | Step 644152 | R=-0.341 | Len=8 | A=-0.6653 InitA=-6.8148 DeltaA=6.1495 Nat=0.0000 Div=0.0000  DecViol=2.2508 DecA=-0.7492 TargetShort=0.0000 TargetSat=1.3347 OnlinePool=add
```

**Key Fields**:
- `A`: Final affinity (matches a tFoldScore line)
- `InitA`: Initial affinity (matches a tFoldScore line)
- `DeltaA`: Improvement (FinalA - InitA)
- `OnlinePool=add`: This TCR was added to online pool

### Matching Strategy
1. tFoldScore lines come in batches (n_envs=8)
2. First batch = InitTCR (InitA values)
3. Second batch = FinalTCR (A values)
4. Match via: `(peptide, affinity_logit) == (peptide, InitA or A)`

---

## 13. References

- **TCRPPO v1**: `/share/liuyutian/TCRPPO/`
- **trace73 analysis**: `TRACE73_SUCCESS_ANALYSIS.md`
- **Decoy library**: `/share/liuyutian/pMHC_decoy_library/`
- **ESM-2 paper**: Lin et al., "Language models of protein sequences at the scale of evolution" (2022)
- **RLHF**: Ouyang et al., "Training language models to follow instructions with human feedback" (2022)

---

**End of Design Document**
