#!/bin/bash

# time setup

export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

# main config
conf=conf_FxProcessor_TencyMastering.yaml

n="FxProcessor_TencyMastering"

PATH_EXPERIMENT=/data2/eloi/experiments/$n
mkdir -p $PATH_EXPERIMENT


# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
#MASTER_PORT=29500
MASTER_PORT=29500


python train_FxProcessor.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.resume=True \
  exp.compile=False \
  exp.num_workers=8 \
  exp.optimizer.lr=1e-5 \
  dset.validation.num_tracks=0 \
  dset.validation_2.num_tracks=0 \
  exp.batch_size=4 \
