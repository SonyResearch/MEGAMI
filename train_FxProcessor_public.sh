#!/bin/bash

export HYDRA_FULL_ERROR=1 

module load mamba
source activate /scratch/work/molinee2/conda_envs/automix
#source ~/myenv/bin/activate

# main config
conf=conf_FxProcessor_Public.yaml

n="FxProcessor_public"

PATH_EXPERIMENT=experiments/$n
mkdir -p $PATH_EXPERIMENT


python train_FxProcessor.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
