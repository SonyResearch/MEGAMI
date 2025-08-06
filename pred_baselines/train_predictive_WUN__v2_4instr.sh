#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv_clean/bin/activate

conf=baseline_WUN_v2_4instr.yaml

n="baseline_WUN_v2_4instr"

mkdir -p  /data5/eloi/experiments_baselines
PATH_EXPERIMENT=/data5/eloi/experiments_baselines/$n
mkdir -p $PATH_EXPERIMENT

# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=1

python train_predictive_4instr.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.num_workers=4 \
  exp.batch_size=4 \
  exp.resume=True \
  exp.compile=False \
  exp.optimizer.lr=1e-4 \
  dset.validation.num_tracks=8 \
  dset.validation.num_examples=-1 \
  logging=base_logging_1C \
  logging.log=True \




