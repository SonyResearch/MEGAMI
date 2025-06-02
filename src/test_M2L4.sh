#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=0

source ~/myenv/bin/activate

# main config

conf=conf_1A_tencymastering_vocals_LDM_M2L4.yaml
n="1A_tencymastering_vocals_LDM_M2L4_preprocessed"
ckpt=1A_tencymastering_vocals_LDM_M2L4-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_preprocessed_server5 \
  dset.test.segment_length=525312 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  tester.wandb.run_name="cond:dry12s_model:M2L4DiT_traindata:TMdry_its:350k_cfg:1" \
  tester.wandb.tags="[\"cond:dry12s\",\"model:M2L4DiT\",\"traindata:TMdry\",\"its:350k\",\"cfg:1\"]"

