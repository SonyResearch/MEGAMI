#!/bin/bash

# time setup

export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

# main config
conf=conf_blackbox_TencyMastering.yaml

n="MF3wetv6_mapper_blackbox_predictive_fxenc2048AFv3CLAP_paired"

PATH_EXPERIMENT=/data2/eloi/experiments/$n
mkdir -p $PATH_EXPERIMENT


# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
#MASTER_PORT=29500
MASTER_PORT=29500


python train_blackbox.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.resume=True \
  exp.compile=False \
  exp.num_workers=8 \
  exp.optimizer.lr=1e-5 \
  exp.batch_size=4 \
