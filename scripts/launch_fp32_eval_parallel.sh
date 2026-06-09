#!/bin/bash
# Parallel FP32 evaluation of trace61, trace72, trace73
# Uses 3 GPUs in parallel to speed up evaluation

set -e

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
PROJECT_ROOT=/share/liuyutian/tcrppo_v2
cd $PROJECT_ROOT

OUTPUT_DIR="results/fp32_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

PEPTIDES="data/tfold_excellent_peptides.txt"
N_TCRS=20

echo "=========================================="
echo "FP32 Evaluation of Traces"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Peptides: $PEPTIDES ($(wc -l < $PEPTIDES) peptides)"
echo "TCRs per peptide: $N_TCRS"
echo ""

# Launch evaluations in parallel on different GPUs
echo "Launching trace61 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/eval_traces_fp32_parallel.py \
    --trace-name trace61_dynamic_pool \
    --checkpoint output/trace61_dynamic_pool/checkpoints/latest.pt \
    --peptides $PEPTIDES \
    --n-tcrs $N_TCRS \
    --gpu 0 \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/trace61_log.txt 2>&1 &
PID1=$!

echo "Launching trace72 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/eval_traces_fp32_parallel.py \
    --trace-name trace72_delta_from_trace70 \
    --checkpoint output/trace72_delta_from_trace70/checkpoints/latest.pt \
    --peptides $PEPTIDES \
    --n-tcrs $N_TCRS \
    --gpu 0 \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/trace72_log.txt 2>&1 &
PID2=$!

echo "Launching trace73 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/eval_traces_fp32_parallel.py \
    --trace-name trace73_curriculum_exploration \
    --checkpoint output/trace73_curriculum_exploration/checkpoints/latest.pt \
    --peptides $PEPTIDES \
    --n-tcrs $N_TCRS \
    --gpu 0 \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/trace73_log.txt 2>&1 &
PID3=$!

echo ""
echo "All evaluations launched in parallel"
echo "PIDs: trace61=$PID1, trace72=$PID2, trace73=$PID3"
echo ""
echo "Monitor progress with:"
echo "  tail -f $OUTPUT_DIR/trace61_log.txt"
echo "  tail -f $OUTPUT_DIR/trace72_log.txt"
echo "  tail -f $OUTPUT_DIR/trace73_log.txt"
echo ""

# Wait for all to complete
echo "Waiting for evaluations to complete..."
wait $PID1
echo "trace61 completed"
wait $PID2
echo "trace72 completed"
wait $PID3
echo "trace73 completed"

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo ""

# Generate comparison report
$PYTHON scripts/compare_fp32_results.py --results-dir $OUTPUT_DIR

echo "Results saved to: $OUTPUT_DIR"
