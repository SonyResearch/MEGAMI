#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

conf=baseline_DMC.yaml

n="baseline_DMC_14instr"


PATH_EXPERIMENT=/data4/eloi/experiments_baselines/$n

checkpoint=/data4/eloi/experiments_baselines/baseline_DMC/1A_tencymastering_vocals-90000.pt

# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
MASTER_PORT=29501

python test_predictive_14instr_DMC.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.resume=True \
  exp.compile=False \
  tester.checkpoint=$checkpoint \
  tester=eval_benchmark_14 \
  dset=test_benchmark_14 \




