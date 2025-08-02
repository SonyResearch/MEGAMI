#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

conf=baseline_STFT_FM.yaml

n="baseline_STFT_FM"


PATH_EXPERIMENT=/data2/eloi/experiments_baselines/$n

checkpoint=/data2/eloi/experiments_baselines/baseline_STFT_FM/1A_tencymastering_vocals-370000.pt

# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
MASTER_PORT=29501

python test_paired_4instr_multitrack_STFTrefiner.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.resume=True \
  exp.compile=False \
  tester.checkpoint=$checkpoint \
  tester=eval_benchmark \
  dset=test_benchmark




