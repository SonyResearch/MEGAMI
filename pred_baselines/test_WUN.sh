#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

conf=baseline_WUN.yaml

n="baseline_WUN_14instr"


PATH_EXPERIMENT=/data4/eloi/experiments_baselines/$n

checkpoint=/data4/eloi/experiments_baselines/baseline_WUN/1A_tencymastering_vocals-100000.pt

# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
MASTER_PORT=29501

python test_predictive_14instr.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.resume=True \
  exp.compile=False \
  tester.checkpoint=$checkpoint \
  dset.test.num_tracks=-1 \
  tester=eval_benchmark_14 \
  dset=test_benchmark_14




