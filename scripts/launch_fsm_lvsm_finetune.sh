#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e 

# Print every command to the log before it is executed for debugging.
set -x 

# Optional: only set these if you haven't already exported them.
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export WANDB_API_KEY=${WANDB_API_KEY:-YOUR_WANDB_KEY_HERE}

# Run the script with distributed training
python -m torch.distributed.run \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  launch_training.py \
  --config fsm/configs/train/fsm_lvsm_upscale.yaml \
  --load_ckpt static/weights/fsm_4dlvsm_patch8_res128.pth \
  --wandb_name fsm_upscale_lvsm \
  --expname fsm_upscale_lvsm_patch8_res256
