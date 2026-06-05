#!/bin/bash
# trace83 Phase 2: Add specificity constraints while maintaining high affinity
# Hypothesis: Resume from Phase 1 checkpoint (mean aff > 0) + add decoy penalty → maintain mean > 0, AUROC > 0.65
# Key config: w_affinity=1.0, w_decoy=0.8, resume from Phase 1

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

# 先启动tFold server
echo "Starting tFold server for trace83 phase2 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup /home/liuyutian/server/miniconda3/envs/tfold/bin/python \
    scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_trace83_phase2.sock \
    --gpu 2 \
    --use-amp-wrapper \
    --chunk-size 64 \
    --completion-log logs/trace83_phase2_tfold_completion.log \
    > logs/trace83_phase2_tfold_server.log 2>&1 &

sleep 5

# 启动训练（从Phase 1 checkpoint继续）
echo "Launching trace83 Phase 2 training on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup conda run -n tcrppo_v2 --no-capture-output \
    $PYTHON -u -c "import sys; sys.path.insert(0, '/share/liuyutian/tcrppo_v2'); from tcrppo_v2.data.tcr_pool_trace61_patch import *; import tcrppo_v2.ppo_trainer as trainer_module; trainer_module.main()" \
    --config configs/trace83_bootstrap_phase2.yaml \
    --run_name trace83_bootstrap_phase2 \
    --seed 83 \
    --resume_from output/trace83_bootstrap_phase1/checkpoints/latest.pt \
    2>&1 | tee logs/trace83_bootstrap_phase2_train.log &

echo "trace83 Phase 2 launched on GPU 2"
echo "Monitor: tail -f logs/trace83_bootstrap_phase2_train.log"
echo "Hypothesis: Phase 1 high affinity + Phase 2 specificity → mean > 0, AUROC > 0.65"
echo ""
echo "NOTE: Only launch Phase 2 after Phase 1 completes (500K steps)"
