# TCRPPO v2 Archive Plan

## Phase 1: Git Repository (GitHub)
### 1.1 Core Code (push immediately)
- Source code: `tcrppo_v2/`, `evaluation/`, `scripts/`
- Configs: `configs/`
- Documentation: `docs/`, `*.md` files
- Total: ~50MB

### 1.2 What NOT to push to GitHub
- `output/` (13GB checkpoints) → Huggingface
- `logs/` (1.6GB logs) → Huggingface
- `data/tfold_*cache*` (219GB) → Too large, keep local only
- `results/` → Include summary only

## Phase 2: Huggingface Model Repository
### 2.1 Best Checkpoints (Priority)
- trace104_triple_constraint: milestone_5000000.pt (23MB)
- trace98_finetune: milestone_200000.pt (24MB)
- trace99_finetune_nat5: milestone_800000.pt (23MB)
- trace61_fp32_restart: latest.pt (baseline)
Total: ~100MB

### 2.2 Training Logs (Compressed)
- Compress logs/ to logs.tar.gz (~200MB compressed)
- Include key training curves as JSON

### 2.3 Results Summary
- all_traces_qualifying.json
- alive_traces_affinity_summary_v2.csv
- tcrppo_v2_report.html

## Phase 3: Documentation
- README.md with quick start guide
- Model card with training details
- Experiment tracker summary
