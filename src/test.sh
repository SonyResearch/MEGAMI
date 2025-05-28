#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=0

source ~/myenv/bin/activate

# main config

conf=conf_1A_tencymastering_vocals_diffSTFT.yaml
n="1A_tencymastering_vocals_diffSTFT"
ckpt=1A_tencymastering_vocals-200000.pt
PATH_EXPERIMENT=/data4/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_test \
  dset.test.segment_length=262144 \
  tester=evaluate_conditional_dry_vocals \
  tester.checkpoint=$ckpt \
  tester.wandb.run_name="cond:dry_model:waveformSTFT_traindata:tencymastering_its:200k" \

conf=conf_1A_tencymastering_vocals_diffSTFT.yaml
n="1A_tencymastering_vocals_diffSTFT"
ckpt=1A_tencymastering_vocals-100000.pt
PATH_EXPERIMENT=/data4/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_test \
  dset.test.segment_length=262144 \
  tester=evaluate_conditional_dry_vocals \
  tester.checkpoint=$ckpt \
  tester.wandb.run_name="cond:dry_model:waveformSTFT_traindata:tencymastering_its:100k" \


conf=conf_1A_tencymastering_vocals_diffSTFT.yaml
n="1A_tencymastering_vocals_diffSTFT"
ckpt=1A_tencymastering_vocals-150000.pt
PATH_EXPERIMENT=/data4/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_test \
  dset.test.segment_length=262144 \
  tester=evaluate_conditional_dry_vocals \
  tester.checkpoint=$ckpt \
  tester.wandb.run_name="cond:dry_model:waveformSTFT_traindata:tencymastering_its:150k" \

conf=conf_1A_tencymastering_vocals_diffSTFT.yaml
n="1A_tencymastering_vocals_diffSTFT"
ckpt=1A_tencymastering_vocals-50000.pt
PATH_EXPERIMENT=/data4/eloi/experiments/$n


python test.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  dset=tencymastering_vocals_test \
  dset.test.segment_length=262144 \
  tester=evaluate_conditional_dry_vocals \
  tester.checkpoint=$ckpt \
  tester.wandb.run_name="cond:dry_model:waveformSTFT_traindata:tencymastering_its:50k" \