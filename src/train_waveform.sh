#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

# main config
conf=conf_1A_tency1_fxnorm_vocals_diffSTFT.yaml
#conf=conf_1A_tencymastering_vocals_diffSTFT.yaml

n="1A_tency1_fxnorm_vocals_diffSTFT"
#n="1A_tencymastering_vocals_diffSTFT"

PATH_EXPERIMENT=/data5/eloi/experiments/$n
mkdir -p $PATH_EXPERIMENT

#python train.py --config-name=$conf \
#  model_dir=$PATH_EXPERIMENT \
#  exp.batch_size=4 \


# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
MASTER_PORT=29500
#MASTER_PORT=29500

# Launch the training script with torchrun for DDP
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_ddp.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.optimizer.lr=1e-4 \
  exp.batch_size=4 \
  dset.train.segment_length=262144 \
  dset.validation.segment_length=262144 \
  dset=tency1_vocals_fx_norm_server5 \

#python train.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  dset=tency1_vocals_fx_norm_server5 \
#  logging=base_logging_debug \
#  exp.batch_size=8
#  dset.train.segment_length=262144 \
#  dset.validation.segment_length=262144 \

