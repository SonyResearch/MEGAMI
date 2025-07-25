#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

conf=baseline_diffSTFT_dry.yaml

n="baseline_diffSTFT_dry"


mkdir -p  /data2/eloi/experiments_baselines
PATH_EXPERIMENT=/data2/eloi/experiments_baselines/$n
mkdir -p $PATH_EXPERIMENT

# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=1
NUM_GPUS=1
MASTER_PORT=29501

python train_paired_4instr_multitrack.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.num_workers=8 \
  exp.batch_size=6 \
  exp.resume=True \
  exp.compile=False \
  dset.validation_2.num_tracks=8 \
  dset.validation_2.num_examples=16 \
  dset.validation.num_tracks=8 \
  dset.validation.num_examples=16 \
  logging=base_logging_1C \
  logging.log=True \




