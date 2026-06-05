#!/bin/bash
# Restart trace61_dynamic_pool from latest checkpoint

cd /share/liuyutian/tcrppo_v2

# Start tFold server first
nohup /home/liuyutian/server/miniconda3/envs/tfold/bin/python \
    scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_trace61_dynamic_pool.sock \
    --gpu 4 \
    --use-amp-wrapper \
    --chunk-size 64 \
    --completion-log logs/trace61_dynamic_pool_tfold_completion.log \
    > logs/trace61_dynamic_pool_tfold_server.log 2>&1 &

echo "tFold server started for trace61, waiting 10s..."
sleep 10

# Start training
nohup conda run -n tcrppo_v2 --no-capture-output python -c \
  'import sys; sys.path.insert(0, "/share/liuyutian/tcrppo_v2"); from tcrppo_v2.data.tcr_pool_trace61_patch import *; import tcrppo_v2.ppo_trainer as trainer_module; trainer_module.main()' \
  --config configs/trace61_dynamic_pool.yaml \
  --run_name trace61_dynamic_pool \
  --seed 42 \
  --resume_from output/trace61_dynamic_pool/checkpoints/latest.pt \
  2>&1 | tee logs/trace61_dynamic_pool_train_resume.log &

echo "trace61_dynamic_pool restarted!"
echo "Monitor: tail -f logs/trace61_dynamic_pool_train_resume.log"
