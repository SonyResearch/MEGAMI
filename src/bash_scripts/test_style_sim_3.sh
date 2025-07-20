#!/bin/bash

# time setup


export HYDRA_FULL_ERROR=1 

source ~/myenv/bin/activate

# main config
#conf=conf_1C_tencymastering_vocals_simulated_style_MERT_dryAFxRep_abspos.yaml
#python train.py --config-name=$conf \
#  model_dir=$PATH_EXPERIMENT \
#  exp.batch_size=4 \


# Number of GPUs to use
export CUDA_VISIBLE_DEVICES=1
NUM_GPUS=1
#MASTER_PORT=29500
MASTER_PORT=29500


#conf=conf_1C_tencymastering_vocals_simulated_style_MERT.yaml
#ckpt="/data5/eloi/experiments/1C_tencymastering_vocals_style_sim_2306_MERT_drywet/1C_tencymastering_vocals-100000.pt"
#n="1C_tencymastering_vocals_style_sim_2306_MERT_drywet"
#PATH_EXPERIMENT=/data5/eloi/experiments/$n

#python test_sim.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  exp.compile=True \
#  tester.checkpoint=$ckpt \
#  tester.wandb.run_name="MERT" \
#  tester.wandb.tags="[\"cond:MERT\",\"network:DiT\"]" \
#  tester.wandb.project="project_1C_test_2506" \


#conf=conf_1C_tencymastering_vocals_simulated_style_MERT_abspos.yaml
#ckpt="/data5/eloi/experiments/1C_tencymastering_vocals_style_sim_2306_MERT_abspos_drywet/1C_tencymastering_vocals-100000.pt"
#n="1C_tencymastering_vocals_style_sim_2306_MERT_abspos_drywet"
#PATH_EXPERIMENT=/data5/eloi/experiments/$n
#
#python test_sim.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  exp.compile=True \
#  tester.checkpoint=$ckpt \
#  tester.wandb.run_name="MERT_abspos" \
#  tester.wandb.tags="[\"cond:MERT\",\"network:DiTv2\"]" \
#  tester.wandb.project="project_1C_test_2506" \

#conf=conf_1C_tencymastering_vocals_simulated_style_MERT_dryAFxRep_abspos.yaml
#ckpt="/data5/eloi/experiments/1C_tencymastering_vocals_style_sim_2306_MERT_dryAFxRep_abspos_drywet/1C_tencymastering_vocals-100000.pt"
#n="1C_tencymastering_vocals_style_sim_2306_MERT_dryAFxRep_abspos_drywet"
#PATH_EXPERIMENT=/data5/eloi/experiments/$n
#
#python test_sim.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  exp.compile=True \
#  tester.checkpoint=$ckpt \
#  tester.wandb.run_name="MERT_dryAFxRep_abspos" \
#  tester.wandb.tags="[\"cond:MERT_dryAFxRep\",\"network:DiTv2\"]" \
#  tester.wandb.project="project_1C_test_2506" \


#conf=conf_1C_tencymastering_vocals_simulated_style.yaml
#ckpt="/data5/eloi/experiments/1C_tencymastering_vocals_style_sim_2306_M2L4_drywet/1C_tencymastering_vocals-100000.pt"
#n="1C_tencymastering_vocals_style_sim_2306_M2L4_drywet"
#PATH_EXPERIMENT=/data5/eloi/experiments/$n
#
#python test_sim.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  exp.compile=True \
#  tester.checkpoint=$ckpt \
#  tester.wandb.run_name="M2L4" \
#  tester.wandb.tags="[\"cond:M2L4\",\"network:DiT\"]" \
#  tester.wandb.project="project_1C_test_2506" \
#
#conf=conf_1C_tencymastering_vocals_simulated_style_CLAP.yaml
#ckpt="/data5/eloi/experiments/1C_tencymastering_vocals_style_sim_2306_CLAP_drywet/1C_tencymastering_vocals-100000.pt"
#n="1C_tencymastering_vocals_style_sim_2306_CLAP_drywet"
#PATH_EXPERIMENT=/data5/eloi/experiments/$n
#
#python test_sim.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  exp.compile=True \
#  tester.checkpoint=$ckpt \
#  tester.wandb.run_name="CLAP" \
#  tester.wandb.tags="[\"cond:CLAP\",\"network:DiT\"]" \
#  tester.wandb.project="project_1C_test_2506" \

#conf=conf_1C_tencymastering_vocals_simulated_style_CLAP_dryAFxRep_abspos.yaml
#ckpt="/data5/eloi/experiments/1C_tencymastering_vocals_style_sim_2306_CLAP_dryAFxRep_abspos_drywet/1C_tencymastering_vocals-100000.pt"
#n="1C_tencymastering_vocals_style_sim_2306_CLAP_dryAFxRep_abspos_drywet"
#PATH_EXPERIMENT=/data5/eloi/experiments/$n
#
#python test_sim.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  exp.compile=True \
#  tester.checkpoint=$ckpt \
#  tester.wandb.run_name="CLAP_AFxRep_abspos" \
#  tester.wandb.tags="[\"cond:CLAP_AFxRep\",\"network:DiTv2\"]" \
#  tester.wandb.project="project_1C_test_2506" \

#conf=conf_1C_tencymastering_vocals_simulated_style_MERT_abspos_FM.yaml
#ckpt="/data5/eloi/experiments/1C_tencymastering_vocals_style_sim_2306_MERT_abspos_FM_drywet/1C_tencymastering_vocals-100000.pt"
#n="1C_tencymastering_vocals_style_sim_2306_MERT_abspos_FM_drywet"
#PATH_EXPERIMENT=/data5/eloi/experiments/$n
#
#python test_sim.py --config-name=$conf  \
#  model_dir=$PATH_EXPERIMENT \
#  exp.compile=True \
#  tester.checkpoint=$ckpt \
#  tester.wandb.run_name="MERT_abspos" \
#  tester.wandb.tags="[\"cond:MERT\",\"network:DiTv2\",\"objective:FM\"]" \
#  tester.wandb.project="project_1C_test_2506" \

conf=conf_1C_tencymastering_vocals_simulated_style_oracle_abspos.yaml
ckpt="/data5/eloi/experiments/1C_tencymastering_vocals_style_sim_2306_oracle_abspos_drywet/1C_tencymastering_vocals-100000.pt"
n="1C_tencymastering_vocals_style_sim_2306_oracle_abspos_drywet"
PATH_EXPERIMENT=/data5/eloi/experiments/$n

python test_sim.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
  exp.compile=True \
  tester.checkpoint=$ckpt \
  tester.wandb.run_name="oracle_abspos" \
  tester.wandb.tags="[\"cond:oracle\",\"network:DiTv2\"]" \
  tester.wandb.project="project_1C_test_2506" \

