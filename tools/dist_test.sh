#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29501}
#export NCCL_P2P_DISABLE="1"
#export NCCL_IB_DISABLE="1"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#/root/miniconda3/envs/mmdet/bin/python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
