#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv_clean/bin/activate

# main config
conf=conf_FxGenerator_TencyDB.yaml

n="FxGenerator_TencyDB"

PATH_EXPERIMENT=/data4/eloi/experiments/$n
mkdir -p $PATH_EXPERIMENT


# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=3

python train_FxGenerator.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
