#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv_clean/bin/activate

conf=baseline_WUN_v2_4instr.yaml

n="baseline_WUN_v2_4instr"


PATH_EXPERIMENT=/data5/eloi/experiments_baselines/$n

checkpoint=/data5/eloi/experiments_baselines/baseline_WUN_v2_4instr/1A_tencymastering_vocals-120000.pt

# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
MASTER_PORT=29501

python test_predictive_4instr.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.resume=True \
  exp.compile=False \
  tester.checkpoint=$checkpoint \
  tester=eval_benchmark \
  dset=test_benchmark




