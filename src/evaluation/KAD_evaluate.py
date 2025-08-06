
import pandas as pd

import torch
import os
import omegaconf
import numpy as np
import glob
import pyloudnorm as pyln
import sys
#move to the parent directory to import datasets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.eval_benchmark import load_audio
from evaluation.dist_metrics import KADFeatures

# see device 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path_test_set="/data5/eloi/test_set/MDX_TM_benchmark"

set="MDX_TM"

#features=[
#    "AFxRep",
#    "FxEncoder",
#    "FxEncoder++",
#    "CLAP",
#    "MERT",
#    "spectral",
#    "panning",
#    "dynamic",
#    "vggish",
#]
features=[
    "AFxRep",
    "FxEncoder",
    "FxEncoder++",
    "CLAP",
    "bark",
    "dynamic",
]




KAD_metrics={}

KAD_args = { 
  "do_PCA_figure": False, #if True, the FAD figure will be computed
  "do_TSNE_figure": False, #if True, the FAD figure will be computed
  "kernel": "gaussian", #kernel to use for the KAD metric
   "PCA_fit_mode": "all"
}
KAD_args = omegaconf.OmegaConf.create(KAD_args)
print("KAD_args:", KAD_args)

for feature in features:
    if feature=="AFxRep":
        AFxRep_args = {
              "distance_type": "cosine",  # not used
              "ckpt_path": "/home/eloi/projects/project_mfm_eloi/src/tmp/afx-rep.ckpt"
        }
        AFxRep_args = omegaconf.OmegaConf.create(AFxRep_args)
        KAD_metrics[feature] = KADFeatures(type="AFxRep", sample_rate=44100, AFxRep_args=AFxRep_args, KAD_args=KAD_args)
    elif feature=="FxEncoder":
        fx_encoder_args = {
            "distance_type": "cosine",  # not used
            "ckpt_path": "utils/feature_extractors/ckpt/fxenc_default.pt"
        }
        fx_encoder_args = omegaconf.OmegaConf.create(fx_encoder_args)
        KAD_metrics[feature] = KADFeatures(type="fx_encoder", sample_rate=44100, fx_encoder_args=fx_encoder_args, KAD_args=KAD_args)
    elif feature=="FxEncoder++":
        fx_encoder_plusplus_args = {
            "distance_type": "cosine",  # not used
            "ckpt_path": "utils/feature_extractors/ckpt/fxenc_plusplus_default.pt"
        }
        fx_encoder_plusplus_args = omegaconf.OmegaConf.create(fx_encoder_plusplus_args)
        KAD_metrics[feature] = KADFeatures(type="fx_encoder_++", sample_rate=44100, fx_encoder_plusplus_args=fx_encoder_plusplus_args, KAD_args=KAD_args)

    elif feature=="CLAP":
        clap_args = {
            "ckpt_path": "/data5/eloi/checkpoints/laion_clap/music_audioset_epoch_15_esc_90.14.patched.pt",
            "distance_type": "cosine",
            "normalize": True,  # if True, the features will be normalized
            "use_adaptor": False,  # if True, the features will be adapted to the CLAP space
            "adaptor_checkpoint":None,
            "adaptor_type":None,
            "add_noise": False,  #   if True, the features will be augmented with orthogonal noise
            "noise_sigma": 0  # sigma of the orthogonal noise to
        }

        clap_args = omegaconf.OmegaConf.create(clap_args)
        KAD_metrics[feature] = KADFeatures(type="CLAP", sample_rate=44100, CLAP_args=clap_args, KAD_args=KAD_args)

    elif feature=="MERT":
        mert_args = {
            "distance_type": "cosine",  # not used
            "layer": 8, 
            "average": True,
            "normalize": True,  # if True, the features will be normalized
        }
        mert_args = omegaconf.OmegaConf.create(mert_args)
        KAD_metrics[feature] = KADFeatures(type="MERT", sample_rate=44100, MERT_args=mert_args, KAD_args=KAD_args)
    elif feature=="bark":
        bark_args = {
            "distance_type": "cosine",  # not used
            "normalize": True,  # if True, the features will be normalized
        }
        bark_args = omegaconf.OmegaConf.create(bark_args)
        KAD_metrics[feature] = KADFeatures(type="bark", sample_rate=44100, bark_args=bark_args, KAD_args=KAD_args, normalize=True)
    elif feature=="spectral":
        spectral_args = {
            "distance_type": "cosine",  # not used
            "normalize": True,  # if True, the features will be normalized
        }
        spectral_args = omegaconf.OmegaConf.create(spectral_args)
        KAD_metrics[feature] = KADFeatures(type="spectral", sample_rate=44100, spectral_args=spectral_args, KAD_args=KAD_args, normalize=True)
    elif feature=="panning":
        panning_args = {
            "distance_type": "cosine",  # not used
            "normalize": True,  # if True, the features will be normalized
        }
        panning_args = omegaconf.OmegaConf.create(panning_args)
        KAD_metrics[feature] = KADFeatures(type="panning", sample_rate=44100, panning_args=panning_args, KAD_args=KAD_args, normalize=True)
    elif feature=="dynamic":
        dynamic_args = {
            "distance_type": "cosine",  # not used
            "normalize": True,  # if True, the features will be normalized
        }
        dynamic_args = omegaconf.OmegaConf.create(dynamic_args)
        KAD_metrics[feature] = KADFeatures(type="dynamic", sample_rate=44100, dynamic_args=dynamic_args, KAD_args=KAD_args, normalize=True)

    else:
        raise ValueError(f"Unknown feature: {feature}")


def get_ref_file_path(dir):
    ref_mixture = os.path.join(dir,"mix", "mix.wav")
    return ref_mixture




pylnmeter = pyln.Meter(44100)  # Create a meter for 44100 Hz sampling rate

def loudness_normalize(audio, target_loudness=-23.0):
    """
    Normalize the loudness of the audio to a target level.
    """
    audio= np.array(audio, dtype=np.float32).T
    loudness = pylnmeter.integrated_loudness(audio)

    # loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(audio, loudness, -14.0)

    return  torch.tensor(loudness_normalized_audio.T, dtype=torch.float32)



song_dirs= sorted(glob.glob(os.path.join(path_test_set, "*")))

if set=="TM":
    song_dirs = [d for d in song_dirs if "TM" in os.path.basename(d)]
elif set=="MDX":
    song_dirs = [d for d in song_dirs if "MDX" in os.path.basename(d)]


reference_dict = {}

for song_dir in song_dirs:

    song_name = os.path.basename(song_dir)
    segment_subdirs = sorted(glob.glob(os.path.join(song_dir, "*")))

    song_id=os.path.basename(song_dir)

    for segment_dir in segment_subdirs:

        segment= os.path.basename(segment_dir)
        id= f"{song_id}_{segment}"

        ref_file_path= get_ref_file_path(segment_dir)
        mix_ref, fs=load_audio(str(ref_file_path), stereo=True)
        assert fs==44100, "Expected sampling rate of 44100 Hz"
        mix_ref = loudness_normalize(mix_ref)
        reference_dict[id] = mix_ref


# create dataframe colums are features, rows are methods
dataframe= pd.DataFrame(columns=["method"] +list(KAD_metrics.keys()))

filename_results="results_eval/results_KAD"+f"_{set}_308.csv"

#methods=[ "fxnorm_automix_S_Lb", "fxnorm_automix_L_Lb",]
methods=[ "proposed_random_churn","S3_random_churn", "S4_random_churn", "only_rms_S4_random_churn",  "only_rms_random_churn" ]
#methods=["mst_oracle",  "equal_loudness", "diff_baseline",  "proposed_oracle", "proposed_centroid_close", "proposed_centroid_far", "proposed_random", "only_rms_centroid_close", "only_rms_centroid_far", "only_rms_random" ]
#methods=["mst_oracle",  "equal_loudness", "diff_baseline",  "proposed_oracle", "proposed_random", "only_rms_random" , "proposed_centroid_close", "proposed_centroid_far", "only_rms_centroid_close", "only_rms_centroid_far"]
#methods=["cfg2_random", "cfg2_random_only_rms", "cfg1_T100_random", "cfg1_T100_onlyrms", "S3_random", "S3_onlyrms", "S4_random", "S4_onlyrms"]
#methods=["prop_random_indep", "only_rms_indep"]
#methods=["WUN_4instr", "WUN_14instr", "DMC_14instr"]
#methods=["WUN_4instr", "WUN_14instr", "DMC_14instr"]


for method in methods:

        #try:

        if method=="mst_oracle":
            def get_pred_file_path(dir):
                pred_mixture= os.path.join(dir,"mst_oracle", "mixture_output.wav")
                return pred_mixture
        elif method=="mst_oracle_multi":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"mst_oracle_multi", "mixture_output.wav")
               return pred_mixture
        elif method=="equal_loudness":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"anchor_equal_loudness", "mix.wav")
               return pred_mixture
        elif method=="diff_baseline":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"diff_baseline", "pred_mixture.wav")
               return pred_mixture
        elif method=="fxnorm_automix_S_Lb":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"fxnorm_automix_S_Lb_v2", "mixture_output.wav")
               return pred_mixture
        elif method=="fxnorm_automix_L_Lb":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"fxnorm_automix_L_Lb_v2", "mixture_output.wav")
               return pred_mixture
        elif method=="proposed_oracle":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"oracle_proposed", "mix.wav")
               return pred_mixture
        elif method=="proposed_centroid_close":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_cfg_1", "centroid_close.wav")
               return pred_mixture
        elif method=="proposed_centroid_far":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_cfg_1", "centroid_far.wav")
               return pred_mixture
        elif method=="proposed_random":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_cfg_1", "random.wav")
               return pred_mixture
        elif method=="proposed_random_churn":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_cfg_1_T50_churn5", "random.wav")
               return pred_mixture
        elif method=="only_rms_centroid_close":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_cfg_1", "only_rms_centroid_close.wav")
               return pred_mixture
        elif method=="only_rms_centroid_far":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_cfg_1", "only_rms_centroid_far.wav")
               return pred_mixture
        elif method=="only_rms_random":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_cfg_1", "only_rms_random.wav")
               return pred_mixture
        elif method=="only_rms_random_churn":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_cfg_1_T50_churn5", "only_rms_random.wav")
               return pred_mixture
        elif method=="cfg2_random":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_cfg_2", "random.wav")
               return pred_mixture
        elif method=="cfg2_random_only_rms":
              def get_pred_file_path(dir):
                pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_cfg_2", "only_rms_random.wav")
                return pred_mixture
        elif method=="cfg1_T100_random":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_cfg_1_T100", "random.wav")
               return pred_mixture
        elif method=="cfg1_T100_onlyrms":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_cfg_1_T100", "only_rms_random.wav")
               return pred_mixture
        elif method=="S3_random":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S3v6_MF3wetv6_MDX_TM_benchmark_cfg_1", "random.wav")
               return pred_mixture
        elif method=="S3_random_churn":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S3v6_MF3wetv6_MDX_TM_benchmark_cfg_1_T50_churn5", "random.wav")
               return pred_mixture
        elif method=="S3_onlyrms":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S3v6_MF3wetv6_MDX_TM_benchmark_cfg_1", "only_rms_random.wav")
               return pred_mixture
        elif method=="S4_random_churn":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S4v6_MF3wetv6_MDX_TM_benchmark_cfg_1_T50_churn5", "random.wav")
               return pred_mixture
        elif method=="only_rms_S4_random_churn":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S4v6_MF3wetv6_MDX_TM_benchmark_cfg_1_T50_churn5", "only_rms_random.wav")
               return pred_mixture
        elif method=="S4_onlyrms":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S4v6_MF3wetv6_MDX_TM_benchmark_cfg_1", "only_rms_random.wav")
               return pred_mixture
        elif method=="prop_random_indep":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_indep", "random.wav")
               return pred_mixture
        elif method=="only_rms_indep":
           def get_pred_file_path(dir):
               pred_mixture= os.path.join(dir,"stylediffpipeline_S9v6_MF3wetv6_MDX_TM_benchmark_indep", "only_rms_random.wav")
               return pred_mixture
        elif method=="WUN_4instr":
            def get_pred_file_path(dir):
                pred_mixture= os.path.join(dir,"WUN_4instr", "pred_mixture.wav")
                return pred_mixture
        elif method=="WUN_14instr":
            def get_pred_file_path(dir):
                pred_mixture= os.path.join(dir,"WUN_14instr", "pred_mixture.wav")
                return pred_mixture
        elif method=="DMC_14instr":
            def get_pred_file_path(dir):
                pred_mixture= os.path.join(dir,"DMC_14instr", "pred_mixture.wav")
                return pred_mixture
        else:
           raise ValueError(f"Unknown method: {method}")
    
    
        method_dict = {}
    
        for song_dir in song_dirs:
    
            song_name = os.path.basename(song_dir)
            segment_subdirs = sorted(glob.glob(os.path.join(song_dir, "*")))
        
            song_id=os.path.basename(song_dir)
        
            for segment_dir in segment_subdirs:
        
                segment= os.path.basename(segment_dir)
                id= f"{song_id}_{segment}"
        
                pred_file_path= get_pred_file_path(segment_dir)
                pred_mixture, fs=load_audio(str(pred_file_path), stereo=True)
                assert fs==44100, "Expected sampling rate of 44100 Hz"
                pred_mixture = loudness_normalize(pred_mixture)
                assert not np.isnan(pred_mixture).any(), f"NaN values found in predicted mixture for {id}"
                method_dict[id] = pred_mixture
    
    
        #create new empty row with key as "method"
        dataframe.loc[method] = [None] * (len(KAD_metrics)+1)
    
        for feature, metric in KAD_metrics.items():
    
            KAD_distance, dict_output=metric.compute(reference_dict, method_dict, None)

            print(f"KAD distance for method {method} and feature {feature}: {KAD_distance}")
        
            #write the KAD distance to the dataframe
            dataframe.at[method, "method"] = method
            dataframe.at[method, feature] = KAD_distance
        
            for output_key, output_value in dict_output.items():
                if "figure" in output_key:
                    output_value.savefig(f"results_eval/{output_key}_{set}_{method}_{feature}.png")
                    print(f"Saved figure: {output_key}_{set}_{method}_{feature}.png")
    
    
    
        dataframe.to_csv(filename_results, index=False)
        print(f"Results saved to {filename_results}")

        #except Exception as e:

        #    print(f"Error processing method {method}: {e}")

        #    dataframe.to_csv(filename_results, index=False)
        #    print(f"Probably incompleted Results saved to {filename_results}")
        #    continue




