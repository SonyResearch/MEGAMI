#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

# main config
conf=conf_tencymastering_multitrack_simulated_styleAFxrep_contentCLAP_fxnorm_CLAPadaptor.yaml

n="tencymastering_multitrack_simulated_styleAFxRep_contentCLAP_fxnorm_CLAPadaptor_nonoise"

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
  diff_params=edm_style_AFxRep_CLAP_simulated_multitrack_fxnorm_CLAPadaptor_nonoise \
  model_dir=$PATH_EXPERIMENT \
  exp.resume=True \
  exp.compile=False \
  logging=base_logging_1C \
  logging.log=True \

  #dset=tencymastering_vocals_server5 \
