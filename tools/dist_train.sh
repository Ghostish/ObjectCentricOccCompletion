#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29503}
export NCCL_DEBUG=INFO 
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#/root/miniconda3/envs/mmdet/bin/python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
