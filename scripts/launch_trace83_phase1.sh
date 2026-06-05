#!/bin/bash
# trace83 Phase 1: Bootstrap seeds + Pure affinity optimization
# Hypothesis: Starting from trace72's best TCRs (mean aff -0.106) + focusing on affinity only → mean aff > 0
# Key config: w_affinity=2.0, w_decoy=0.1, L0=100% bootstrap seeds
# GPU: 3 (most available)

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

# 先启动tFold server (CUDA_VISIBLE_DEVICES=3 使得GPU 3映射为GPU 0)
echo "Starting tFold server for trace83 phase1 on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup /home/liuyutian/server/miniconda3/envs/tfold/bin/python \
    scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_trace83_phase1.sock \
    --gpu 0 \
    --use-amp-wrapper \
    --chunk-size 64 \
    --completion-log logs/trace83_phase1_tfold_completion.log \
    > logs/trace83_phase1_tfold_server.log 2>&1 &

sleep 10

# 启动训练
echo "Launching trace83 Phase 1 training on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup conda run -n tcrppo_v2 --no-capture-output \
    $PYTHON -u -c "import sys; sys.path.insert(0, '/share/liuyutian/tcrppo_v2'); from tcrppo_v2.data.tcr_pool_trace61_patch import *; import tcrppo_v2.ppo_trainer as trainer_module; trainer_module.main()" \
    --config configs/trace83_bootstrap_phase1.yaml \
    --run_name trace83_bootstrap_phase1 \
    --seed 83 \
    2>&1 | tee logs/trace83_bootstrap_phase1_train.log &

echo "trace83 Phase 1 launched on GPU 3"
echo "Monitor: tail -f logs/trace83_bootstrap_phase1_train.log"
echo "Hypothesis: Bootstrap seeds (mean -0.106) + w_aff=2.0 → mean affinity > 0 by 500K steps"
