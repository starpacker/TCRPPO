# test56_maxstep_curriculum: Trace11-Style Max-Steps Curriculum

**Date:** 2026-05-19  
**Status:** running, stage 1  
**Trace tag:** `trace25_maxstep_curriculum`  
**Controller PID:** `2758321`  
**tFold server PID:** `2758777`  
**Stage 1 trainer PID:** `2774110`  
**GPU:** physical GPU 2  

## Purpose

Run a trace11-style target-delta experiment while gradually increasing the episode horizon:

```text
max_steps=1 for about 2000 episodes
max_steps=2 for about 2000 episodes
max_steps=4 for about 2000 episodes
max_steps=8 for the final long phase
```

The controller resumes each stage from the previous stage checkpoint.

## Base Settings

This follows trace11-style settings:

- `reward_mode=v2_no_decoy_delta`
- `affinity_scorer=tfold`
- `encoder=esm2`
- `n_envs=8`
- `learning_rate=3e-4`
- `entropy_coef=0.02`
- `hidden_dim=512`
- `ban_stop=true`
- `terminal_reward_only=true`
- `n_contrast_decoys=0`
- `w_affinity=1.0`
- `w_decoy=0.0`
- `w_naturalness=0.05`
- `w_diversity=0.02`
- curriculum `L0/L1/L2 = 0.5/0.0/0.5`
- `train_targets=data/tfold_excellent_peptides.txt`

## Files

- Config: `configs/test56_maxstep_curriculum.yaml`
- Controller: `scripts/run_test56_maxstep_curriculum.py`
- Launcher: `scripts/launch_test56_maxstep_curriculum.sh`
- Controller log: `logs/test56_maxstep_curriculum_trace25_maxstep_curriculum_controller.log`
- Stage 1 train log: `logs/test56_maxstep_curriculum_trace25_maxstep_curriculum_ms1_train.log`
- Server log: `logs/test56_maxstep_curriculum_tfold_amp_server_trace25_maxstep_curriculum.log`
- Completion log: `logs/test56_maxstep_curriculum_tfold_completion_trace25_maxstep_curriculum.log`
- Stage 1 output: `output/test56_maxstep_curriculum_trace25_maxstep_curriculum_ms1/`

## Launch Command

```bash
scripts/launch_test56_maxstep_curriculum.sh 2 trace25_maxstep_curriculum
```

The stage 1 trainer command currently running is:

```bash
/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python -u tcrppo_v2/ppo_trainer.py \
  --config configs/test56_maxstep_curriculum.yaml \
  --run_name test56_maxstep_curriculum_trace25_maxstep_curriculum_ms1 \
  --seed 42 \
  --reward_mode v2_no_decoy_delta \
  --affinity_scorer tfold \
  --tfold_server_socket /tmp/tfold_server_trace25_maxstep_curriculum.sock \
  --encoder esm2 \
  --total_timesteps 2048 \
  --n_envs 8 \
  --learning_rate 3e-4 \
  --entropy_coef 0.02 \
  --hidden_dim 512 \
  --max_steps 1 \
  --n_contrast_decoys 0 \
  --w_affinity 1.0 \
  --w_decoy 0.0 \
  --w_naturalness 0.05 \
  --w_diversity 0.02 \
  --curriculum_l0 0.5 \
  --curriculum_l1 0.0 \
  --curriculum_l2 0.5 \
  --train_targets data/tfold_excellent_peptides.txt \
  --tfold_cache_path data/tfold_feature_cache_trace25_maxstep_curriculum.db \
  --decoy_library_path /share/liuyutian/pMHC_decoy_library \
  --ban_stop \
  --terminal_reward_only
```

## SOP Deviation

This run was launched before `docs/experiment_launch_sop.md` was written and uses an isolated tFold cache:

```text
data/tfold_feature_cache_trace25_maxstep_curriculum.db
```

The config and controller defaults have since been changed so future launches use the shared cache:

```text
data/tfold_feature_cache.db
```

The running process was not stopped or relaunched, following the rule that existing experiments should not be interrupted without an explicit user request.

## Monitoring

```bash
ps -eo pid,stat,cmd | rg 'trace25_maxstep_curriculum|test56_maxstep_curriculum' | rg -v 'rg '
tail -f logs/test56_maxstep_curriculum_trace25_maxstep_curriculum_controller.log
tail -f logs/test56_maxstep_curriculum_trace25_maxstep_curriculum_ms1_train.log
tail -f logs/test56_maxstep_curriculum_tfold_amp_server_trace25_maxstep_curriculum.log
```

At the time this note was created, stage 1 had reached about 1000 episodes and was still running.
