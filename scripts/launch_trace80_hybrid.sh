#!/bin/bash
# Launch trace80: Hybrid strategy (trace72 stability + trace78 exploration)

cd /share/liuyutian/tcrppo_v2

# Start tFold server first
nohup /home/liuyutian/server/miniconda3/envs/tfold/bin/python \
    scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_trace80_hybrid.sock \
    --gpu 5 \
    --use-amp-wrapper \
    --chunk-size 64 \
    --completion-log logs/trace80_hybrid_tfold_completion.log \
    > logs/trace80_hybrid_tfold_server.log 2>&1 &

echo "tFold server started for trace80, waiting 10s..."
sleep 10

# Start training from trace72's checkpoint
nohup conda run -n tcrppo_v2 --no-capture-output python -c \
  'import sys; sys.path.insert(0, "/share/liuyutian/tcrppo_v2"); from tcrppo_v2.data.tcr_pool_trace61_patch import *; import tcrppo_v2.ppo_trainer as trainer_module; trainer_module.main()' \
  --config configs/trace80_hybrid_72_78.yaml \
  --run_name trace80_hybrid_72_78 \
  --seed 80 \
  --resume_from output/trace72_delta_from_trace70/checkpoints/latest.pt \
  --resume_reset_optimizer \
  2>&1 | tee logs/trace80_hybrid_72_78_train.log &

echo "trace80_hybrid_72_78 launched!"
echo "Strategy: trace72 checkpoint + aggressive exploration (higher LR, entropy, affinity weight)"
echo "Monitor: tail -f logs/trace80_hybrid_72_78_train.log"
