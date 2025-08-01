#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

# main config
conf=conf_A2_CLAP_regressor_multitrack.yaml

n="A2_CLAP_regressor_multitrack"

PATH_EXPERIMENT=/data5/eloi/experiments/$n
mkdir -p $PATH_EXPERIMENT

#python train.py --config-name=$conf \
#  model_dir=$PATH_EXPERIMENT \
#  exp.batch_size=4 \


# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=1
NUM_GPUS=1
#MASTER_PORT=29500
MASTER_PORT=29500

# Launch the training script with torchrun for DDP
#torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_ddp.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  exp.optimizer.lr=1e-4 \
#  exp.batch_size=4 \

python train_CLAP_regressor_multitrack.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.resume=False \
  exp.num_workers=12 \
  dset.validation.num_tracks=-1 \
  dset.validation_2.num_tracks=5 \
  logging=base_logging_CLAP_regressor

  #dset=tencymastering_vocals_server5 \
