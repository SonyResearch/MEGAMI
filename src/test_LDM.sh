#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=1

source ~/myenv/bin/activate

# main config

conf=conf_1A_tencymastering_vocals_LDM_SAO.yaml
n="1A_tencymastering_vocals_LDM_SAO"
ckpt=1A_tencymastering_vocals-300000.pt
PATH_EXPERIMENT=/data4/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_test \
  dset.test.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  tester.wandb.run_name="cond:dry12s_model:SAODiT_traindata:tencymastering_its:300k_cfg:1" \

conf=conf_1A_tencymastering_vocals_LDM_SAO.yaml
n="1A_tencymastering_vocals_LDM_SAO"
ckpt=1A_tencymastering_vocals-250000.pt
PATH_EXPERIMENT=/data4/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_test \
  dset.test.segment_length=524288 \
  tester=evaluate_conditional_dry_vocals_LDM \
  tester.checkpoint=$ckpt \
  tester.cfg_scale=1.0 \
  tester.wandb.run_name="cond:dry12s_model:SAODiT_traindata:tencymastering_its:250k_cfg:1" \
