#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

# main config
conf=conf_1A_tency1_fxnorm_vocals_LDM.yaml

n="1A_tency1_fxnorm_vocals_LDM"

PATH_EXPERIMENT=/data4/eloi/experiments/$n


python train.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  logging=base_logging_debug \
