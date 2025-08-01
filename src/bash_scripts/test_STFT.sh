#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=0

source ~/myenv/bin/activate

# main config

conf=conf_1A_tencymastering_vocals_diffSTFT.yaml
n="1A_tencymastering_vocals_diffSTFT"
ckpt=1A_tencymastering_vocals-200000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_preprocessed_server5 \
  dset.validation.segment_length=524288 \
  dset.validation.x_as_mono=False \
  diff_params.default_shape=[1,2,524288] \
  tester=evaluate_conditional_dry_vocals \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  tester.wandb.run_name="cond:dry12s_model:diffSTFT_traindata:TMdry_its:200k_cfg:1" \
  tester.wandb.tags="[\"cond:dry12s\",\"model:diffSTFT\",\"traindata:TMdry\",\"its:200k\",\"cfg:1\"]" \
  tester.wandb.project="project_test_05_06"

