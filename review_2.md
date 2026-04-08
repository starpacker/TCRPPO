# Review of progress_2.md — TCRPPO GPU Deployment

**Reviewer**: PI
**Date**: 2026-04-07
**Verdict**: Deployment fundamentally sound. Three issues must be resolved before results are considered valid.

---

## A. Critical Issues (Must Fix)

### A1. `pdb.set_trace()` under nohup — Silent Hang Risk

`code/on_policy_algorithm.py` line 204-205:

```python
except:
    pdb.set_trace()
```

This is a bare `except` catching ALL exceptions, then dropping into an interactive debugger. Under `nohup`, there is no terminal. If any error fires during the remaining ~9M timesteps — OOM, NaN propagation, a malformed TCR string, anything — the process **hangs silently forever**. You will see the process alive in `ps`, GPU memory held, but zero progress. You may not notice for hours.

**Action required:**

```bash
sed -i 's/pdb.set_trace()/raise/' code/on_policy_algorithm.py
```

This will not affect the currently running AE+McPAS process (it loaded the module at startup). But it **must** be done before launching AE+VDJdb or any future run.

**Deliverable**: Confirm this line has been changed. Show `grep -n "pdb" code/on_policy_algorithm.py` output.

---

### A2. Autoencoder Retraining — Remove from Plan

The report states:

> "No retraining needed... For paper reproduction, a full retrain (559K sequences, 300 epochs) could improve quality"

This is incorrect. Retraining the autoencoder **cannot** improve quality, because the autoencoder weights are never used. I verified the loading sequence:

1. `AutoencoderLSTMClassifier.__init__()` loads `tcr_ae_dim_100.pt` into `self.autoencoder`
2. `reward.py` line 67: `model.load_state_dict(checkpoint['model_state_dict'])` loads `ae_mcpas1.pt`, which **overwrites all parameters** including every `autoencoder.*` key

Evidence — `ae_mcpas1.pt` contains:
```
autoencoder.encoder.0.weight: [300, 588]
autoencoder.encoder.3.weight: [100, 300]
autoencoder.encoder.6.weight: [100, 100]
autoencoder.decoder.0.weight: [100, 100]
autoencoder.decoder.3.weight: [300, 100]
autoencoder.decoder.6.weight: [588, 300]
pep_embedding.weight: [21, 10]
pep_lstm.*: ...
hidden_layer.*: ...
output_layer.*: ...
```

The `tcr_ae_dim_100.pt` file is a **scaffold** — it only needs to produce the correct `nn.Linear` dimensions so that `load_state_dict` doesn't raise a shape mismatch. Weight values are discarded immediately.

**Action required**: Remove "retrain autoencoder" from Next Steps. Do not spend compute on it. Update progress_2.md accordingly.

**Deliverable**: Updated Next Steps section in progress_2.md.

---

### A3. Output Directory Naming Bug — Document Clearly

The directory name contains `None` instead of `128`:
```
output/ae_mcpas_mcpas_0.5_0.0_0.9_256_None/
```

The report correctly identifies the cause (`embed_latent_dim` vs `latent_dim` at `tcr_env.py:305`). However, this will cause confusion when writing test commands. Every subsequent command that references the output path must use the **actual** directory name.

**Action required**: In progress_2.md, when you write the test commands (Step 5 onwards), use the actual path. Add a one-line note at the top of the testing section:

```
NOTE: Output directories contain `None` in place of latent_dim due to a naming
bug in tcr_env.py:305. Always verify with `ls output/` before running test commands.
```

**Deliverable**: Test commands in progress_2.md with correct paths.

---

## B. Required Next Steps — Execution Plan

Complete the following in order. Each step has a defined deliverable.

### B1. Fix `on_policy_algorithm.py` (5 min)

See A1 above.

### B2. Launch AE+VDJdb Training (5 min)

Do not wait for AE+McPAS to finish. GPU VRAM usage is ~37GB / 80GB — there is room.

```bash
nohup python ./code/tcr_env.py \
    --num_envs 20 \
    --ergo_model ./code/ERGO/models/ae_vdjdb1.pt \
    --peptide_path ./data/test_peptides/ae_vdjdb_test_peptides.txt \
    --bad_ratio 0.0 \
    --hidden_dim 256 \
    --latent_dim 128 \
    --gamma 0.90 \
    --path ./output/ae_vdjdb_ \
    > ./logs/ae_vdjdb_train.log 2>&1 &
```

If two training processes cause OOM, reduce `--num_envs` to 10 for the second one.

**Deliverable**: PID of VDJdb training process. `nvidia-smi` screenshot showing both processes.

### B3. Wait for Both Trainings to Complete

Monitor with:
```bash
# Check both are alive
ps aux | grep "[t]cr_env.py"

# Check progress
grep "total_timesteps" logs/ae_mcpas_train.log | tail -1
grep "total_timesteps" logs/ae_vdjdb_train.log | tail -1

# Watch for completion
tail -f logs/ae_mcpas_train.log | grep -E "finish training|saving model"
```

**Deliverable**: Final training logs for both models — last 30 lines of each log file. Include:
- Total training time
- Final `ep_rew_mean`
- Final `explained_variance`
- Confirm model files saved (list `output/*/ppo_tcr.zip` with file sizes)

### B4. Run Testing — Both Configurations

After training completes, verify actual output directory names first:
```bash
ls -d output/ae_mcpas_* output/ae_vdjdb_*
```

Then run:

```bash
# AE+McPAS test
nohup python ./code/test_RL_tcrs.py \
    --num_envs 4 \
    --out ./results/ae_mcpas_results.txt \
    --ergo_model ./code/ERGO/models/ae_mcpas1.pt \
    --peptides ./data/test_peptides/ae_mcpas_test_peptides.txt \
    --rollout 1 \
    --tcrs ./data/tcrdb/test_uniq_tcr_seqs.txt \
    --path ./output/<ACTUAL_MCPAS_DIR>/ppo_tcr \
    --hour 5 \
    --max_size 50000 \
    > ./logs/ae_mcpas_test.log 2>&1 &

# AE+VDJdb test
nohup python ./code/test_RL_tcrs.py \
    --num_envs 4 \
    --out ./results/ae_vdjdb_results.txt \
    --ergo_model ./code/ERGO/models/ae_vdjdb1.pt \
    --peptides ./data/test_peptides/ae_vdjdb_test_peptides.txt \
    --rollout 1 \
    --tcrs ./data/tcrdb/test_uniq_tcr_seqs.txt \
    --path ./output/<ACTUAL_VDJDB_DIR>/ppo_tcr \
    --hour 5 \
    --max_size 50000 \
    > ./logs/ae_vdjdb_test.log 2>&1 &
```

Replace `<ACTUAL_MCPAS_DIR>` and `<ACTUAL_VDJDB_DIR>` with the real names from `ls output/`.

**Deliverable**:
1. `results/ae_mcpas_results.txt` and `results/ae_vdjdb_results.txt` — raw result files
2. `head -20` of each result file
3. `wc -l` of each result file (total number of generated TCRs)

### B5. Analyze Results

Use the `analyze_results.py` script already in the repo:

```bash
python analyze_results.py results/ae_mcpas_results.txt > results/ae_mcpas_analysis.txt
python analyze_results.py results/ae_vdjdb_results.txt > results/ae_vdjdb_analysis.txt
cat results/ae_mcpas_analysis.txt
cat results/ae_vdjdb_analysis.txt
```

**Deliverable**: Full output of both analysis runs. I need to see, per peptide:
- Count of generated TCRs
- Average ERGO binding score
- Fraction with ERGO > 0.9
- Average GMM TCR-likeness
- Number of unique generated sequences
- Number of sequences that were actually modified (not identical to input)

### B6. Write Final progress_2.md

Update progress_2.md with all results from B3-B5. The final document must contain:

1. **Training results**: Training curves summary (initial vs final reward, final explained_variance, total time)
2. **Test results**: The full analysis tables from B5
3. **Comparison with paper claims**: Are the ERGO scores, TCR-likeness, and diversity consistent with what the paper reports? If not, what differs and why?
4. **Reproducibility verdict**: One sentence — did we reproduce or not?

---

## C. Delivery Format

All results go into `progress_2.md` as new sections appended after the current content. Do not modify the existing sections (Steps 0-4) — they are correct as-is, except for the changes specified in A2 and A3.

Final `progress_2.md` structure:

```
# Existing content (Steps 0-4, Summary, Troubleshooting)
## Step 5: AE+VDJdb Training          ← NEW
## Step 6: Testing — AE+McPAS         ← NEW
## Step 7: Testing — AE+VDJdb         ← NEW
## Step 8: Results Analysis            ← NEW
## Step 9: Comparison with Paper       ← NEW
## Step 10: Reproducibility Verdict    ← NEW
```

Each new section must include: the exact command run, raw output (or relevant excerpt), and a brief interpretation.

---

## D. Timeline

| Task | Estimated Time | Dependency |
|------|---------------|------------|
| A1: Fix pdb | 5 min | None |
| A2: Update Next Steps | 5 min | None |
| B2: Launch VDJdb training | 5 min | A1 |
| B3: Wait for training | ~12 hours | B2 |
| B4: Run testing | ~5 hours each | B3 |
| B5: Analyze | 10 min | B4 |
| B6: Write final report | 30 min | B5 |

Total wall-clock: ~18-24 hours from now.

---

## E. What I Do NOT Need

- Do not retrain the TCR autoencoder. It is a no-op (see A2).
- Do not attempt LSTM configurations. The repo does not ship LSTM ERGO models. Only AE+McPAS and AE+VDJdb are reproducible.
- Do not tune hyperparameters. Use exactly the values from the README. The goal is reproduction, not improvement.
- Do not create plots or figures at this stage. Tables are sufficient.
