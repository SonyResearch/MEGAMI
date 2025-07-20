#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=2

source ~/myenv/bin/activate

conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_randomreverb_noisy10dB_v2.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_randomreverb_noisy10dB_v2"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm_randomreverb-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset.validation.segment_length=524288 \
  dset.validation_2.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  tester.modes="[\"baseline_dry\"]" \
  tester.wandb.run_name="baseline_fnnorm" \
  tester.wandb.tags="[\"baseline:fxnorm\"]" \
  tester.wandb.project="project_test_09_06"


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset.validation.segment_length=524288 \
  dset.validation_2.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.batch_size=8 \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  tester.wandb.run_name="cond:fxnorm_randomreverb_12s_model:M2L4DiT_traindata:TMwetfxnorm_randomreverb_noisy10dB_its:350k_cfg:1" \
  tester.wandb.tags="[\"cond:fxnorm_randomreverb\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm_randomreverb_noisy10dB\",\"its:350k\",\"cfg:1\"]" \
  tester.wandb.project="project_test_09_06"

conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_randomreverb_noisy0dB_v2.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_randomreverb_noisy0dB_v2"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm_randomreverb-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n

python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset.validation.segment_length=524288 \
  dset.validation_2.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.batch_size=8 \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  tester.wandb.run_name="cond:fxnorm_randomreverb_12s_model:M2L4DiT_traindata:TMwetfxnorm_randomreverb_noisy0dB_its:350k_cfg:1" \
  tester.wandb.tags="[\"cond:fxnorm_randomreverb\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm_randomreverb_noisy0dB\",\"its:350k\",\"cfg:1\"]" \
  tester.wandb.project="project_test_09_06"

