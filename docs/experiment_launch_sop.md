# Experiment Launch SOP

This is the required checklist for launching new training/eval experiments in this repo.

## Hard Rules

1. Do not stop, restart, or overwrite any existing experiment unless the user explicitly asks for that exact run to be stopped.
2. Use the shared tFold feature cache by default:
   `data/tfold_feature_cache.db`
3. Launch a dedicated tFold server for each new experiment stack. Do not reuse another experiment's socket.
4. Before launching, scan for orphan tFold servers. It is OK to kill a server only when there is no trainer or eval process using its socket.
5. Every new experiment must have a written record under `docs/experiments/` before or immediately after launch.
6. Every new experiment must use unique names for `run_name`, socket, logs, pid files, and output directory.
7. Record the physical GPU selection and why it was chosen.

## Preflight

Run these from `/share/liuyutian/tcrppo_v2`:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
ps -eo pid,stat,cmd | rg 'ppo_trainer.py|eval_.*tfold|tfold_feature_server.py' | rg -v 'rg '
```

Choose an idle GPU if one exists. If none is idle, choose the least busy GPU and record that fact in the experiment note.

## Zombie Server Cleanup

A tFold server is a zombie only if no live trainer/eval command references the same socket.

Safe scan:

```bash
ps -eo pid,cmd | rg 'tfold_feature_server.py|ppo_trainer.py|eval_.*tfold' | rg -v 'rg '
```

For each `tfold_feature_server.py --socket /tmp/name.sock`, check whether any live trainer/eval process contains `/tmp/name.sock`.

Only then kill the orphan server:

```bash
kill -TERM <server_pid>
sleep 5
kill -0 <server_pid> 2>/dev/null && kill -KILL <server_pid>
rm -f /tmp/name.sock /tmp/name.sock.pid
```

Do not kill a trainer. Do not kill another active user's server. Do not kill a server if the matching trainer/eval process is still alive.

## Launch Rules

Use a unique trace tag:

```text
traceNN_short_description
```

Use a unique run name:

```text
testNN_description_traceNN_short_description
```

The trainer and its dedicated tFold server should be pinned to the same physical GPU via `CUDA_VISIBLE_DEVICES=<gpu>`. The server itself should still receive `--gpu 0`, because the visible device list remaps the selected physical GPU to local CUDA device 0.

The tFold cache path should be:

```bash
--tfold_cache_path data/tfold_feature_cache.db
```

Do not create isolated cache DBs unless the user explicitly requests one.

## Required Experiment Record

Create `docs/experiments/<test_name>.md` with:

- Date and launch status
- Hypothesis or purpose
- Base experiment/settings
- Full launch command or launcher path
- GPU, pid, socket, logs, output directory
- Reward mode and weights
- Dataset/target peptide file
- Cache path
- Resume checkpoint, if any
- Stage schedule, if any
- Monitoring commands
- Any deviations from the SOP

Also make sure `output/<run_name>/experiment.json` exists after trainer setup.

## After Launch

Verify all of the following:

```bash
ps -eo pid,stat,cmd | rg '<trace_tag>|<run_name>' | rg -v 'rg '
tail -80 logs/<train_log>
tail -80 logs/<server_log>
```

The train log must show:

- Correct `run_name`
- Correct `reward_mode`
- Correct `max_steps`
- Correct weights
- Correct target file
- Correct shared cache path
- `VecEnv` setup line
- At least one tFold request or episode line

If the trainer exits early, do not silently relaunch. Record the failure and inspect the log first.

