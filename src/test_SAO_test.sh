#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=2

source ~/myenv/bin/activate

# main config
conf=conf_1A_tencymastering_vocals_LDM_SAO.yaml
n="1A_tencymastering_vocals_LDM_SAO"
ckpt=1A_tencymastering_vocals-300000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


  
python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_preprocessed_server5 \
  dset.test.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_SAO \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=100 \
  tester.modes="[\"baseline_dry\"]" \
  tester.wandb.run_name="baseline_dry_input" \
  tester.wandb.tags="[\"baseline:dry_input\"]"
