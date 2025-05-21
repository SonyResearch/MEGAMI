#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

# main config
conf=conf_1A_tency1_vocals_LDM.yaml

n="1A_tency1_vocals"

PATH_EXPERIMENT=experiments/$n
mkdir -p $PATH_EXPERIMENT

python train.py --config-name=$conf \
  model_dir=$PATH_EXPERIMENT \

