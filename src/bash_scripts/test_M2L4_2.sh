#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=3

source ~/myenv/bin/activate


conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_dr_noisy0dB.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_dr_noisy_0dB"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm_dr-300000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_dr_server5 \
  dset.validation.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  tester.modes="[\"baseline_dry\"]" \
  tester.wandb.run_name="baseline_fxnorm_dr" \
  tester.wandb.tags="[\"baseline:fxnorm_dr\"]" \
  tester.wandb.project="project_test_09_06"

python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_dr_server5 \
  dset.validation.segment_length=524288 \
  dset.validation_2.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  tester.wandb.run_name="cond:fxnorm_mdr_12s_model:M2L4DiT_traindata:TMwetfxnorm_mdr_its:300k_cfg:1" \
  tester.wandb.tags="[\"cond:fxnorm_mdr_12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm_mdr\",\"its:300k\",\"cfg:1\"]" \
  tester.wandb.project="project_test_09_06"




