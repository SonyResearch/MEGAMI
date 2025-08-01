#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

# main config
conf=conf_MW22_mapper_blackbox_predictive_fxenc2048AFv3CLAP_paired.yaml

n="MW22_mapper_blackbox_predictive_fxenc2048AFv3CLAP_paired"

PATH_EXPERIMENT=/data5/eloi/experiments/$n
mkdir -p $PATH_EXPERIMENT

#python train.py --config-name=$conf \
#  model_dir=$PATH_EXPERIMENT \
#  exp.batch_size=4 \


# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=3
NUM_GPUS=1
#MASTER_PORT=29500
MASTER_PORT=29500

# Launch the training script with torchrun for DDP
#torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_ddp.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  exp.optimizer.lr=1e-4 \
#  exp.batch_size=4 \

python M_train_mapper_blackbox_predictive.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.resume=True \
  exp.compile=False \
  exp.num_workers=4 \
  exp.batch_size=4 \
  logging=base_logging_mapper

  #dset=tencymastering_vocals_server5 \
