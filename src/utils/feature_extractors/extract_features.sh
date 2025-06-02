#!/bin/bash
GPU=0
gpu_stat="--using_gpu ${GPU}"
conda_python_path='/home/tony/miniconda3/envs/tony/bin/python' # conda env python path
aud_ext='--audio_extension wav'
# aud_ext='--audio_extension flac'

tgt_dir='--target_dir /home/tony/tda/test_samples/fsat_yuk/generated_tracks_mar31/'
output_base_dir='/home/tony/tda/test_samples/fsat_yuk/generated_tracks_mar31/features/'


# # ST-ITO
# cur_output_dir="--output_dir ${output_base_dir}STITO/"
# CUDA_VISIBLE_DEVICES=$GPU $conda_python_path feature_extraction.py --feature_type stito $tgt_dir $cur_output_dir $gpu_stat $aud_ext

# # FXencoder
# cur_output_dir="--output_dir ${output_base_dir}FXenc/"
# CUDA_VISIBLE_DEVICES=$GPU $conda_python_path feature_extraction.py --feature_type fxenc --fxenc_ckpt_name default $tgt_dir $cur_output_dir $gpu_stat $aud_ext

# # CLAP
# cur_output_dir="--output_dir ${output_base_dir}CLAP/"
# CUDA_VISIBLE_DEVICES=$GPU $conda_python_path feature_extraction.py --feature_type clap $tgt_dir $cur_output_dir $gpu_stat $aud_ext

# # PANN
# cur_output_dir="--output_dir ${output_base_dir}PANN/"
# CUDA_VISIBLE_DEVICES=$GPU $conda_python_path feature_extraction.py --feature_type pann $tgt_dir $cur_output_dir $gpu_stat $aud_ext

# # VGGish
# cur_output_dir="--output_dir ${output_base_dir}VGGish/"
# CUDA_VISIBLE_DEVICES=$GPU $conda_python_path feature_extraction.py --feature_type vggish $tgt_dir $cur_output_dir $gpu_stat $aud_ext

# # MERT
# cur_output_dir="--output_dir ${output_base_dir}MERT/"
# CUDA_VISIBLE_DEVICES=$GPU $conda_python_path feature_extraction.py --feature_type mert $tgt_dir $cur_output_dir $gpu_stat $aud_ext

# # DSP
# cur_output_dir="--output_dir ${output_base_dir}DSP/"
# CUDA_VISIBLE_DEVICES=$GPU $conda_python_path feature_extraction.py --feature_type dsp $tgt_dir $cur_output_dir $gpu_stat $aud_ext

# # HPCP
# cur_output_dir="--output_dir ${output_base_dir}HPCP/"
# CUDA_VISIBLE_DEVICES=$GPU $conda_python_path feature_extraction.py --feature_type hpcp $tgt_dir $cur_output_dir $gpu_stat $aud_ext


