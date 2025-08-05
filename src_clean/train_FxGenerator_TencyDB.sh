#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv_clean/bin/activate

# main config
conf=conf_FxGenerator_TencyDB.yaml

n="S9v6"

PATH_EXPERIMENT=/data4/eloi/experiments/$n
mkdir -p $PATH_EXPERIMENT


# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=3

python train_FxGenerator_multitrack.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.num_workers=10 \
  exp.resume=True \
  exp.compile=True \
  dset.validation.num_tracks=16 \
  dset.validation_2.num_tracks=16 \
  exp.batch_size=8 \
  exp.max_tracks=14 \
  exp.skip_first_val=True \
  exp.optimizer.lr=1e-4 \
  logging.log=True \
