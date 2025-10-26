#!/bin/bash


export HYDRA_FULL_ERROR=1 

module load mamba
source activate /scratch/work/molinee2/conda_envs/automix


# main config
conf=conf_FxGenerator_Public.yaml

n="FxGenerator_public"

PATH_EXPERIMENT=experiments/$n
mkdir -p $PATH_EXPERIMENT


# Number of GPUs to use
#export CUDA_VISIBLE_DEVICES=30

python train_FxGenerator.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
