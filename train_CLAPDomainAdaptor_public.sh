#!/bin/bash
# time setup
##SBATCH  --time=0-23:29:59
#SBATCH  --time=0-12:59:59
##SBATCH  --time=01:59:59

# CPU, GPU, memory setup
#SBATCH --mem=100G

#SBATCH --cpus-per-task=8
#SBATCH  --gres=gpu:h200:1
##SBATCH  --gres=gpu:1 --constraint=volta
#SBATCH --exclude=gpu17
#SBATCH --array=[1]
#SBATCH --job-name=Fxprocessor
#SBATCH --output=/scratch/work/%u/projects/project_mfm_automix/experiments/train_%j.out

# time setup


export HYDRA_FULL_ERROR=1 


source activate /scratch/work/molinee2/conda_envs/automix
# main config
conf=conf_CLAPDomainAdaptor_public.yaml

n="CLAPDomainAdaptorPublic"

PATH_EXPERIMENT=experiments/$n
mkdir -p $PATH_EXPERIMENT


# Launch the training script with torchrun for DDP

python train_CLAPDomainAdaptor.py --config-name=$conf  \
  model_dir=$PATH_EXPERIMENT \
