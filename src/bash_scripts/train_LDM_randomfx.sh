#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

# main config

#conf=conf_1A_tency1_fxnorm_vocals_LDM_SAO.yaml
#n="1A_tency1_fxnorm_vocals_LDM_SAO"

conf=conf_1A_tencymastering_vocals_LDM_M2L4_random_fx.yaml
n="1A_tencymastering_vocals_LDM_M2L4_randomfx"

#conf=conf_1A_tencymastering_vocals_LDM_M2L4.yaml
#n="1A_tencymastering_vocals_LDM_M2L4_preprocessed"

#conf=conf_1A_tencymastering_vocals_LDM_M2L4.yaml
#n="1A_tencymastering_vocals_LDM_M2L4"

PATH_EXPERIMENT=/data5/eloi/experiments/$n
mkdir -p $PATH_EXPERIMENT

#python train.py --config-name=$conf \
#  model_dir=$PATH_EXPERIMENT \
#  exp.batch_size=4 \


# Number of GPUs to use
#export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=3
NUM_GPUS=1
#MASTER_PORT=29501
MASTER_PORT=29500

# Launch the training script with torchrun for DDP
#torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_ddp.py --config-name=$conf  \
#		model_dir=$PATH_EXPERIMENT \
##		exp.optimizer.lr=1e-4 \
#		exp.resume=False \
#		exp.batch_size=8 \
#	  	exp.compile=True \
#	  	logging=base_logging_debug \

python train.py --config-name=$conf  \
	  model_dir=$PATH_EXPERIMENT \
	  exp.optimizer.lr=1e-4 \
	  exp.resume=False \
	  exp.batch_size=16 \
	  exp.compile=True \
	  logging=base_logging \

#
	      #python train.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  logging=base_logging_debug \
