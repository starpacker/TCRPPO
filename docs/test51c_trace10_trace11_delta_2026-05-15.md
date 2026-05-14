# test51c trace10 / trace11_delta notes (2026-05-15)

## Summary

- `trace10` kept training stably but plateaued early with terminal reward around `-7` to `-8`.
- Root cause analysis pointed to the reward using absolute tFold pre-sigmoid binding logits.
- A follow-up run, `trace11_delta`, was launched with only two intended optimization changes:
  - switch terminal affinity reward from absolute `affinity_raw` to `affinity_delta`
  - reduce PPO `entropy_coef` from `0.05` to `0.02`

## trace10 diagnosis

Run:

- `run_name`: `test51c_amp_server_rerun_detached_trace10`
- trainer log: `logs/test51c_amp_server_rerun_detached_trace10_train.log`
- tFold server log: `logs/test51c_tfold_amp_server_trace10.log`
- tFold completion log: `logs/test51c_tfold_completion_trace10.log`

Observed behavior:

- The run drifted from early rewards around `-4` to a long plateau near `-7.4`.
- The logged `A=` term was the raw tFold binding logit, not a bounded `[0, 1]` score.
- With `ban_stop=true`, `max_steps=8`, and `terminal_reward_only=true`, PPO only receives sparse terminal feedback after eight forced edits.

Interpretation:

- Under absolute-logit reward, many peptides and starting seeds begin from different baselines.
- PPO is pushed to optimize a hard-to-compare raw oracle score instead of relative improvement from the seed TCR.

## trace11_delta launch

Run:

- `run_name`: `test51c_amp_server_rerun_detached_trace11_delta`
- GPU: `4`
- socket: `/tmp/tfold_server_trace11_delta.sock`
- trainer log: `logs/test51c_amp_server_rerun_detached_trace11_delta_train.log`
- tFold server log: `logs/test51c_tfold_amp_server_trace11_delta.log`
- tFold completion log: `logs/test51c_tfold_completion_trace11_delta.log`

Isolation from `trace10`:

- `trace10` remains on GPU `3` with socket `/tmp/tfold_server_trace10.sock`
- `trace11_delta` uses an independent detached launcher, PID files, logs, and socket

Launch overrides relative to `trace10`:

- `reward_mode=v2_no_decoy_delta`
- `entropy_coef=0.02`
- `w_affinity=1.0`
- `w_naturalness=0.05`
- `w_diversity=0.02`

Early signal:

- Early `trace11_delta` episode rewards were centered much closer to zero than `trace10`, including small positive terminal deltas.
- This is expected because the reward is now anchored to improvement over the initial seed affinity.

## Code changes tied to these runs

- Added `v2_no_decoy_delta` reward mode in `tcrppo_v2/reward_manager.py`
- Added CLI override for `entropy_coef` in `tcrppo_v2/ppo_trainer.py`
- Added `latest_checkpoint_interval` support so future runs refresh `latest.pt` every `2k` steps while preserving `20k` milestone checkpoints
- Added a detached stack launcher in `scripts/manage_test51c_amp_stack.sh`
- AMP server/completion logging and tFold wrapper stabilization changes remain part of the stack used by both runs
