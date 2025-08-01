#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=2

source ~/myenv/bin/activate

conf="conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_randomreverb_compression_noisy0dB.yaml"
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_randomreverb_compression_noisy0dB"
ckpt="1A_tencymastering_vocals_LDM_M2L4_fxnorm_randomreverb-500000.pt"
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  diff_params=edm_LDM_M2L4_randomreverb_compression_testrandom \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.validation.segment_length=524288 \
  dset.validation_2.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=2.0 \
  tester.batch_size=8 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  diff_params.context_preproc.SNR_mean=0 \
  tester.wandb.run_name="cond:randomfx_model:M2L4DiT_traindata:TMrandomfx_its:500k_cfg:1" \
  tester.wandb.tags="[\"cond:randomfx\",\"model:M2L4DiT\",\"traindata:TMrandomfx\",\"its:500k\",\"cfg:2\",\"T60:random\",\"compressed_random\",\"SNR:random\"]" \
  tester.wandb.project="project_test_randomfx_12_06"

