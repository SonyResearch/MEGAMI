#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=3

source ~/myenv/bin/activate

# main config
conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n

python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.test.segment_length=525312 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=100 \
  dset.validation_2.num_examples=100 \
  tester.modes="[\"baseline_dry\"]" \
  tester.wandb.run_name="baseline_fxnorm" \
  tester.wandb.tags="[\"baseline:fxnorm\"]"

