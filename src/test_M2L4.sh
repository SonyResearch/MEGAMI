#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=1

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
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:1" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:1\"]"


conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy0dB.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy_0dB"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.test.segment_length=525312 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:1_noisy:0db" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:1\",\"noisy:0db\"]"


conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy-10dB.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy_-10dB"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.test.segment_length=525312 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:1_noisy:-10db" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:1\",\"noisy:-10db\"]"

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
  tester.cfg_scale=2.0 \
  tester.wandb.run_name="cond:dry12s_model:M2L4DiT_traindata:TMdry_its:350k_cfg:2" \
  tester.wandb.tags="[\"cond:dry12s\",\"model:M2L4DiT\",\"traindata:TMdry\",\"its:350k\",\"cfg:2\"]"


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
  tester.cfg_scale=2.0 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:2" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:2\"]"


conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy0dB.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy_0dB"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.test.segment_length=525312 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=2.0 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:2_noisy:0db" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:2\",\"noisy:0db\"]"


conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy-10dB.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy_-10dB"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.test.segment_length=525312 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=2.0 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:2_noisy:-10db" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:2\",\"noisy:-10db\"]"


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
  tester.cfg_scale=4.0 \
  tester.wandb.run_name="cond:dry12s_model:M2L4DiT_traindata:TMdry_its:350k_cfg:4" \
  tester.wandb.tags="[\"cond:dry12s\",\"model:M2L4DiT\",\"traindata:TMdry\",\"its:350k\",\"cfg:4\"]"


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
  tester.cfg_scale=4.0 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:4" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:4\"]"


conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy0dB.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy_0dB"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.test.segment_length=525312 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=4.0 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:4_noisy:0db" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:4\",\"noisy:0db\"]"


conf=conf_1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy-10dB.yaml
n="1A_tencymastering_vocals_LDM_M2L4_fxnorm_noisy_-10dB"
ckpt=1A_tencymastering_vocals_LDM_M2L4_fxnorm-350000.pt
PATH_EXPERIMENT=/data5/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_fxnorm_server5 \
  dset.test.segment_length=525312 \
  tester=evaluate_conditional_dry_vocals_LDM_M2L4 \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=4.0 \
  tester.wandb.run_name="cond:fxnorm12s_model:M2L4DiT_traindata:TMwetfxnorm_its:350k_cfg:4_noisy:-10db" \
  tester.wandb.tags="[\"cond:fxnorm12s\",\"model:M2L4DiT\",\"traindata:TMwetfxnorm\",\"its:350k\",\"cfg:4\",\"noisy:-10db\"]"
