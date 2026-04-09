#!/bin/bash
export PYTHONPATH=/share/liuyutian/TCRPPO:/share/liuyutian/TCRPPO/code:$PYTHONPATH
PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo/bin/python

# The first command is already running (PID $(cat /share/liuyutian/TCRPPO/eval_decoy.pid)). Wait for it if you want, but we can just run everything here for clarity or just wait on the PID.
echo "Waiting for eval_decoy.py to finish..."
wait $(cat /share/liuyutian/TCRPPO/eval_decoy.pid)

echo "Step 2: Computing metrics..."
$PYTHON /share/liuyutian/TCRPPO/evaluation/eval_decoy_metrics.py --csv /share/liuyutian/TCRPPO/evaluation/results/decoy/eval_decoy_ae_mcpas.csv > /share/liuyutian/TCRPPO/evaluation/results/decoy/metrics_output.log

echo "Step 3: Generating plots..."
$PYTHON /share/liuyutian/TCRPPO/evaluation/eval_decoy_visualize.py --csv /share/liuyutian/TCRPPO/evaluation/results/decoy/eval_decoy_ae_mcpas.csv > /share/liuyutian/TCRPPO/evaluation/results/decoy/visualize_output.log

echo "Done!"
