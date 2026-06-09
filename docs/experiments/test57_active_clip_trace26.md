# test57 active clipping trace26

## Date and status

- Date: 2026-05-19
- Status: launched and verified

## Purpose

Test active clipping on the trace11 delta settings. The rollout still runs the full 8-edit episode, but PPO trains only on the prefix ending at the best-affinity intermediate TCR. This should reduce cases where the policy edits into a good TCR and then edits it worse before terminal reward assignment.

## Base experiment/settings

Base: trace11_delta settings, matched except for `active_clipping=true`.

- Config: `configs/test51c.yaml`
- Reward mode: `v2_no_decoy_delta`
- Total timesteps: `2000000`
- Env count: `8`
- Learning rate: `3e-4`
- Entropy coefficient: `0.02`
- Hidden dim: `512`
- Max steps: `8`
- STOP: banned
- Terminal reward only: enabled
- Active clipping: enabled
- Affinity scorer: `tfold`
- Encoder: `esm2`

## Reward weights

- `w_affinity=1.0`
- `w_decoy=0.0`
- `w_naturalness=0.05`
- `w_diversity=0.02`
- `n_contrast_decoys=0`

## Dataset and targets

- Training targets: `data/tfold_excellent_peptides.txt`
- Curriculum: `L0=0.5`, `L1=0.0`, `L2=0.5`
- Decoy library: `/share/liuyutian/pMHC_decoy_library`

## Cache and resume

- tFold cache path: `data/tfold_feature_cache.db`
- Resume checkpoint: none

## GPU selection

- Physical GPU: `7`
- Reason: launch preflight showed GPU 7 had 0% utilization and about 16.8/81.9GB used, the least busy option at launch time.

## Names and paths

- Trace tag: `trace26_active_clip`
- Run name: `test57_active_clip_trace26_active_clip`
- Socket: `/tmp/tfold_server_trace26_active_clip.sock`
- Trainer pid: `910379`
- tFold server pid: `897295`
- Train log: `logs/test57_active_clip_trace26_active_clip_train.log`
- Server log: `logs/test57_active_clip_tfold_amp_server_trace26_active_clip.log`
- Completion log: `logs/test57_active_clip_tfold_completion_trace26_active_clip.log`
- Control log: `logs/test57_active_clip_amp_stack_trace26_active_clip.launch.log`
- Output directory: `output/test57_active_clip_trace26_active_clip`

## Launch command

```bash
RUN_NAME_PREFIX=test57_active_clip \
LOG_PREFIX=test57_active_clip \
ACTIVE_CLIPPING=1 \
REWARD_MODE=v2_no_decoy_delta \
ENTROPY_COEF=0.02 \
W_AFFINITY=1.0 \
W_DECOY=0.0 \
W_NATURALNESS=0.05 \
W_DIVERSITY=0.02 \
MAX_STEPS=8 \
BAN_STOP=1 \
SUB_ONLY=0 \
TERMINAL_REWARD_ONLY=1 \
N_CONTRAST_DECOYS=0 \
CURRICULUM_L0=0.5 \
CURRICULUM_L1=0.0 \
CURRICULUM_L2=0.5 \
TRAIN_TARGETS=data/tfold_excellent_peptides.txt \
TFOLD_CACHE_PATH=data/tfold_feature_cache.db \
scripts/manage_test51c_amp_stack.sh start trace26_active_clip 7
```

## Monitoring

```bash
scripts/manage_test51c_amp_stack.sh status trace26_active_clip 7
ps -eo pid,stat,cmd | rg 'trace26_active_clip|test57_active_clip_trace26_active_clip' | rg -v 'rg '
tail -80 logs/test57_active_clip_trace26_active_clip_train.log
tail -80 logs/test57_active_clip_tfold_amp_server_trace26_active_clip.log
```

## SOP deviations

None. Uses a dedicated tFold server and the shared tFold cache.

## Post-launch verification

- Launcher reported server READY and trainer launched.
- `output/test57_active_clip_trace26_active_clip/experiment.json` exists.
- Train log confirms:
  - `run_name=test57_active_clip_trace26_active_clip`
  - `reward_mode=v2_no_decoy_delta`
  - `weights: aff=1.0, decoy=0.0, nat=0.05, div=0.02`
  - `active_clipping=True`
  - `max_steps=8`
  - `ban_stop=True`
  - `terminal_reward_only=True`
  - `train_targets=data/tfold_excellent_peptides.txt`
  - shared cache loaded as `data/tfold_feature_cache.db`
- First active-clipped episode observed:
  `Episode 1 | Step 56 | R=1.713 | Len=3 | A=-4.9829 ... Clip=3/8 BestA=-4.9829 FinalA=-5.0082`
