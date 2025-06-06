#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=2

source ~/myenv/bin/activate

conf=conf_1A_tencymastering_vocals_LDM_M2L4.yaml
n="1A_tencymastering_vocals_LDM_M2L4_preprocessed"
ckpt=1A_tencymastering_vocals_LDM_M2L4-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n

python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_preprocessed_server5 \
  dset.validation.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  tester.modes="[\"baseline_autoencoder\"]" \
  tester.wandb.run_name="baseline_M2L4_autoencoder" \
  tester.wandb.tags="[\"baseline:M2L4_autoencoder\"]" \
  tester.wandb.project="project_test_05_06"

