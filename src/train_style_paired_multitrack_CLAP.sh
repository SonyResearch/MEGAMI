#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

# main config
conf=conf_tencymastering_multitrack_paired_styleAfxRep_contentCLAP.yaml

n="tencymastering_multitrack_paired_styleAFxRep_contentCLAP"

PATH_EXPERIMENT=/data5/eloi/experiments/$n
mkdir -p $PATH_EXPERIMENT

#python train.py --config-name=$conf \
#  model_dir=$PATH_EXPERIMENT \
#  exp.batch_size=4 \


# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
#MASTER_PORT=29500
MASTER_PORT=29500

# Launch the training script with torchrun for DDP
#torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_ddp.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  exp.optimizer.lr=1e-4 \
#  exp.batch_size=4 \

python train_sim_multitrack.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.resume=False \
  exp.compile=True \
  logging=base_logging_1C \
  logging.log=True \

  #dset=tencymastering_vocals_server5 \
