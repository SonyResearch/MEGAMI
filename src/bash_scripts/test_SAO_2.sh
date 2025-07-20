#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=1

source ~/myenv/bin/activate

# main config


conf=conf_1A_tencymastering_vocals_LDM_SAO.yaml
n="1A_tencymastering_vocals_LDM_SAO"
ckpt=1A_tencymastering_vocals-300000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_preprocessed_server5 \
  dset.validation.segment_length=524288 \
  dset.validation.x_as_mono=False \
  tester=evaluate_conditional_dry_vocals_LDM_SAO \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  tester.wandb.run_name="cond:dry12s_model:SAODiT_traindata:TMdry_its:300k_cfg:1" \
  tester.wandb.tags="[\"cond:dry12s\",\"model:SAODiT\",\"traindata:TMdry\",\"its:300k\",\"cfg:1\"]" \
  tester.wandb.project="project_test_05_06"
