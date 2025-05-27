#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

# main config

conf=conf_1A_tency1_fxnorm_vocals_LDM.yaml
n="1A_tency1_fxnorm_vocals_LDM_v2"

#conf=conf_1A_tencymastering_vocals_LDM.yaml
#n="1A_tencymastering_vocals_LDM_v2"

PATH_EXPERIMENT=/data4/eloi/experiments/$n
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
#	  model_dir=$PATH_EXPERIMENT \
#	    exp.optimizer.lr=1e-4 \
#	      exp.batch_size=4 \
python train.py --config-name=$conf  \
	  model_dir=$PATH_EXPERIMENT \
	  exp.optimizer.lr=1e-4 \
	  exp.batch_size=8 \
	  exp.compile=True \
	  logging=base_logging_10k \

	      #python train.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  logging=base_logging_debug \
