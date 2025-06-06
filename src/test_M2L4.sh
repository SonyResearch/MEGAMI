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

# main config
conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n

python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.validation.segment_length=524288 \
  dset.validation_2.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  tester.modes="[\"baseline_dry\"]" \
  tester.wandb.run_name="baseline_fxnorm" \
  tester.wandb.tags="[\"baseline:fxnorm\"]" \
  tester.wandb.project="project_test_05_06"



python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_preprocessed_server5 \
  dset.validation.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  tester.wandb.run_name="cond:dry12s_model:M2L4DiT_traindata:TMdry_its:350k_cfg:1" \
  tester.wandb.tags="[\"cond:dry12s\",\"model:M2L4DiT\",\"traindata:TMdry\",\"its:350k\",\"cfg:1\"]" \
  tester.wandb.project="project_test_05_06"



conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.validation.segment_length=524288 \
  dset.validation_2.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:1" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:1\"]" \
  tester.wandb.project="project_test_05_06"



conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy0dB.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy_0dB"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.validation.segment_length=524288 \
  dset.validation_2.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:1_noisy:0db" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:1\",\"noisy:0db\"]" \
  tester.wandb.project="project_test_05_06"


conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy-10dB.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy_-10dB"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.validation.segment_length=524288 \
  dset.validation_2.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:1_noisy:-10db" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:1\",\"noisy:-10db\"]" \
  tester.wandb.project="project_test_05_06"

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
  tester.cfg_scale=2.0 \
  dset.validation.num_examples=300 \
  tester.wandb.run_name="cond:dry12s_model:M2L4DiT_traindata:TMdry_its:350k_cfg:2" \
  tester.wandb.tags="[\"cond:dry12s\",\"model:M2L4DiT\",\"traindata:TMdry\",\"its:350k\",\"cfg:2\"]" \
  tester.wandb.project="project_test_05_06"


conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.validation.segment_length=524288 \
  dset.validation_2.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=2.0 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:2" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:2\"]" \
  tester.wandb.project="project_test_05_06"


conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy0dB.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy_0dB"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.validation.segment_length=524288 \
  dset.validation_2.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=2.0 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:2_noisy:0db" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:2\",\"noisy:0db\"]" \
  tester.wandb.project="project_test_05_06"


conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy-10dB.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy_-10dB"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.validation.segment_length=524288 \
  dset.validation_2.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=2.0 \
  dset.validation.num_examples=300 \
  dset.validation_2.num_examples=300 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:2_noisy:-10db" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:2\",\"noisy:-10db\"]" \
  tester.wandb.project="project_test_05_06"




