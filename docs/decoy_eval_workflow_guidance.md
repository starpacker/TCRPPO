# Decoy Reward Evaluation Workflow Guidance

This is the standard workflow for evaluating target-vs-decoy tFold rewards for
TCRPPO checkpoints without touching any running training experiment.

## Goal

For each checkpoint, generate one or more TCRs for the target peptides, score
the generated TCR against the true target peptide and selected decoy peptides
with tFold, then report:

- target reward: higher is better
- decoy mean reward: lower is better
- decoy max reward: lower is better
- decoy top-k mean rewards: lower is better
- target-minus-decoy margins and AUROC

The workflow uses raw tFold binding logits, matching the training scorer
convention.

## Safety Rules

- Do not use or stop any tFold server owned by a training run.
- Use `scripts/run_decoy_eval_workflow.py`; it starts a dedicated eval tFold
  server and shuts it down at the end.
- Snapshot `latest.pt` before evaluation if a training job may keep updating it.
- Only shut down the eval server started by this workflow.
- Use A/B/D decoys by default. Do not use tier C unless the user explicitly asks
  for C fallback.

## Decoy Modes

Named modes are implemented in both workflow and eval scripts:

- `--decoy-mode 1`: `A*1`
- `--decoy-mode 4`: `A*1 + B*2 + D*1`
- `--decoy-mode 16`: `A*4 + B*8 + D*4`

Equivalent aliases:

- `a1`
- `abbd4`
- `a4b8d4`

For custom experiments, `scripts/eval_checkpoint_decoy_reward_tfold.py` also
supports:

```bash
--decoy-mode custom --decoy-plan A:1,B:2,D:1
```

Tier C fallback is off in the standardized workflow. The lower-level eval
script only uses C when `--allow-decoy-c-fallback` is supplied.

## Missing Decoys

`scripts/run_decoy_eval_workflow.py` checks whether each target has enough
A/B/D decoys in `/share/liuyutian/pMHC_decoy_library`.

If not enough decoys exist, it constructs missing hard decoys before evaluation
using the pMHC decoy library. The default construction is fast:

```bash
/home/liuyutian/server/miniconda3/bin/python run_decoy.py TARGET a b \
  --skip-structural --hla HLA-A*02:01
```

To also construct D decoys, pass:

```bash
--construct-strategies a b d
```

This adds:

```bash
--designs 1000 --top-k 10
```

by default.

If decoys are already sufficient, the workflow skips construction and starts
evaluation immediately.

## Checkpoint Preparation

When evaluating a mutable `latest.pt`, snapshot it first:

```bash
mkdir -p results/MY_EVAL/checkpoint_snapshots
cp output/MY_RUN/checkpoints/latest.pt \
  results/MY_EVAL/checkpoint_snapshots/MY_RUN_latest_snapshot.pt
```

Check the actual global step:

```bash
/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python - <<'PY'
import torch
for path in [
    "results/MY_EVAL/checkpoint_snapshots/MY_RUN_latest_snapshot.pt",
]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(path, ckpt.get("global_step"))
PY
```

For an untrained baseline checkpoint, create or reuse a checkpoint with
`global_step=0` and the same model/config shape as the trained checkpoints.

## Standard Command

Run from `/share/liuyutian/tcrppo_v2`:

```bash
/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python \
  scripts/run_decoy_eval_workflow.py \
  --run-name MY_EVAL \
  --checkpoint-dir results/MY_EVAL/checkpoint_snapshots \
  --checkpoints \
    results/MY_EVAL/checkpoint_snapshots/ckpt_a.pt \
    results/MY_EVAL/checkpoint_snapshots/ckpt_b.pt \
  --config configs/test54_delta_minus_decoy.yaml \
  --targets-file data/tfold_excellent_peptides.txt \
  --output-dir results/MY_EVAL \
  --decoy-mode 4 \
  --decoy-top-k 1 3 4 \
  --gpu 6 \
  --score-batch-size 128 \
  --extract-batch-size 64
```

Use `--decoy-mode 1`, `4`, or `16` according to the requested evaluation
strength.

## Server Lifecycle

The workflow starts a dedicated socket named like:

```text
/tmp/tfold_server_eval_MY_EVAL.sock
```

Logs are written to:

```text
logs/tfold_server_eval_MY_EVAL.log
logs/tfold_completion_eval_MY_EVAL.log
```

At completion, the workflow sends `shutdown` to that socket and waits for the
server to stop. If `--keep-server` is supplied, the server remains running and
must be shut down manually later.

## Output Files

The main output directory contains:

- `REPORT.md`: human-readable report
- `summary_by_checkpoint.csv`: checkpoint-level aggregate metrics
- `summary_by_target.csv`: per-target aggregate metrics per checkpoint
- `tcr_level_results.csv`: one row per generated TCR
- `pair_scores.csv`: raw target/decoy tFold scores
- `generated_trajectories.json`: generated TCR trajectories
- `metadata.json`: arguments, checkpoint info, decoy selection, cache stats
- `run.log`: stdout/stderr from the lower-level eval script

## Monitoring

Check whether the dedicated eval workflow is running:

```bash
ps -u "$USER" -o pid,ppid,stat,etime,cmd | \
  rg "MY_EVAL|run_decoy_eval_workflow|eval_checkpoint_decoy_reward_tfold"
```

Watch eval progress:

```bash
tail -n 120 results/MY_EVAL/run.log
tail -n 80 logs/tfold_completion_eval_MY_EVAL.log
```

Check GPU usage:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits
```

After completion, confirm the eval server stopped:

```bash
ps -u "$USER" -o pid,ppid,stat,etime,cmd | rg "tfold_server_eval_MY_EVAL" || true
```

## Interpreting Results

Prefer checkpoints with:

- higher `mean_target_reward`
- lower `mean_decoy_reward`
- lower `mean_decoy_max_reward`
- lower `mean_decoy_topK_mean_reward`
- higher positive target-minus-decoy margins
- higher AUROC

`decoy max` is the hardest individual decoy among selected decoys. Top-k means
are computed from the highest decoy logits, so they measure performance against
the hardest subset.

