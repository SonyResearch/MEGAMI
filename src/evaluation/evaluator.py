
#load generator model
import hydra
import torch
import os
import sys
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import utils.training_utils as tr_utils

import torch
from datasets.tency_mastering_multitrack_paired import TencyMastering_Test
from datasets.eval_benchmark import Eval_Benchmark
import omegaconf
import math

import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE   
from utils.log import make_PCA_figure
from utils.data_utils import apply_RMS_normalization

import pandas as pd

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


### Loading EffectsDiT ###

class Evaluator:
    def __init__(self, method="stylediffpipeline", sampling_strategies=["random","centroid_close","centroid_far"] , method_args=None, dataset_code=None, extra_id=None, path_results="/data5/eloi/results", num_tracks_to_load=1):

        sampling_strategies=["random"]
        self.method=method
        self.method_args = method_args

        self.results_df = pd.DataFrame(columns=["track", "method","metric", "value" ])

        self.sampling_strategies=sampling_strategies

        if method=="stylediffpipeline":
            self.S1_code = method_args.S1_code
            self.S2_code = method_args.S2_code
            self.dataset_code = dataset_code
        
            if extra_id is not None:
                self.experiment_name = f"{self.method}_{self.S1_code}_{self.S2_code}_{self.dataset_code}_{extra_id}"
            else:
                self.experiment_name = f"{self.method}_{self.S1_code}_{self.S2_code}_{self.dataset_code}"
        
            self.path_results = path_results
            self.path_results = f"{self.path_results}/{self.experiment_name}"
            os.makedirs(self.path_results, exist_ok=True)

            self.csv_path = f"{self.path_results}/results.csv"

        
            self.config_file_rel="../conf"
            #self.config_path="/home/eloi/projects/project_mfm_eloi/src/conf"
        
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            self.load_S1_model()  # Load the S1 model
            self.load_S2_model()  # Load the S2 model
            self.prepare_feature_extractors()  # Prepare the feature extractors
        else:
            raise ValueError(f"Unknown method: {method}")
    

        self.load_dataset(num_tracks_to_load=num_tracks_to_load)  # Load the dataset
        
        self.anchors=["equal_loudness", "only_rms"]
        self.metrics=[
            "pairwise-spectral",
            "pairwise-panning",
            "pairwise-dynamic",
            "pairwise-loudness",
            "pairwise-fxenc++",
            "pairwise-MSS",
            "lh-spectral",
            "lh-panning",
            "lh-dynamic",
            "lh-loudness",
            "lh-fxenc++",
            "lh-MSS",
            "percentile-spectral",
            "percentile-panning",
            "percentile-dynamic",
            "percentile-loudness",
            "percentile-fxenc++",
            "percentile-MSS",
            "lh-genfxenc++",
            "percentile-genfxenc++",
            ]

        self.anchor_fns = {}
        for anchor in self.anchors:
            if anchor == "equal_loudness":
                self.anchor_fns[anchor] = self.prepare_equal_loudness_anchor(target_lufs_dB=-48.0)
            elif anchor == "only_rms":
                self.anchor_fns[anchor] = self.prepare_only_rms_anchor()


    def load_S1_model(self):
        if self.S1_code == "S9":
            #config_name="conf_S9gate_tencymastering_multitrack_paired_stylefxenc2048AF_contentCLAP_CLAPadaptor.yaml"
            config_name="conf_S9_tencymastering_multitrack_paired_stylefxenc2048AF_contentCLAP_CLAPadaptor.yaml"
            #model_dir="/data2/eloi/checkpoints/S9gate"
            model_dir="/data5/eloi/checkpoints/S9"
            ckpt="1C_tencymastering_vocals-160000.pt"
            #ckpt="1C_tencymastering_vocals-290000.pt"
        elif self.S1_code == "S9v6":
            config_name="conf_S9v6_tencymastering_multitrack_paired_stylefxenc2048AF_contentCLAP_CLAPadaptor.yaml"
            model_dir="/data5/eloi/checkpoints/S9v6"
            ckpt="1C_tencymastering_vocals-200000.pt"
        elif self.S1_code == "S4v6":
            config_name="conf_S4v6_tencymastering_multitrack_paired_stylefxenc2048AF_contentCLAP.yaml"
            model_dir="/data5/eloi/checkpoints/S4v6"
            ckpt="1C_tencymastering_vocals-50000.pt"
        elif self.S1_code == "S3v6":
            config_name="conf_S3v6_tencymastering_multitrack_paired_stylefxenc2048AF_contentCLAP.yaml"
            model_dir="/data5/eloi/checkpoints/S3v6"
            ckpt="1C_tencymastering_vocals-50000.pt"
        else:
            raise ValueError(f"Unknown S1_code: {self.S1_code}")
        

        overrides = [
            f"model_dir={model_dir}",
            f"tester.checkpoint={ckpt}",
        ]

        with initialize(version_base=None, config_path=self.config_file_rel):
            args = compose(config_name=config_name, overrides=overrides)

        if not os.path.exists(args.model_dir):
            raise Exception(f"Model directory {args.model_dir} does not exist")


        diff_params=hydra.utils.instantiate(args.diff_params)

        network=hydra.utils.instantiate(args.network)
        network=network.to(self.device)
        state_dict = torch.load(os.path.join(args.model_dir,args.tester.checkpoint), map_location=self.device, weights_only=False)

        tr_utils.load_state_dict(state_dict, ema=network)

        self.sampler = hydra.utils.instantiate(args.tester.sampler, network, diff_params, args, )
    
    def load_S2_model(self):

        ### Loading effects model ###

        if self.S2_code == "MF3wet":
            #config_name="conf_MF3gatewet_mapper_blackbox_predictive_fxenc2048AFv3CLAP_paired.yaml"
            config_name="conf_MF3wet_mapper_blackbox_predictive_fxenc2048AFv3CLAP_paired.yaml"
            #model_dir="/data2/eloi/checkpoints/MF3gatewet"
            model_dir="/data5/eloi/checkpoints/MF3wet"
            #ckpt="mapper_blackbox_TCN-300000.pt"
            ckpt="mapper_blackbox_TCN-180000.pt"
        if self.S2_code == "MF3wetv6":
            #config_name="conf_MF3gatewet_mapper_blackbox_predictive_fxenc2048AFv3CLAP_paired.yaml"
            config_name="conf_MF3wetv6_mapper_blackbox_predictive_fxenc2048AFv3CLAP_paired.yaml"
            #model_dir="/data2/eloi/checkpoints/MF3gatewet"
            model_dir="/data5/eloi/checkpoints/MF3wetv6"
            #ckpt="mapper_blackbox_TCN-300000.pt"
            ckpt="mapper_blackbox_TCN-270000.pt"
        else:
            raise ValueError(f"Unknown S2_code: {self.S2_code}")

        overrides = [
            f"model_dir={model_dir}",
            f"tester.checkpoint={ckpt}",
        ]

        with initialize(version_base=None, config_path=self.config_file_rel):
            args = compose(config_name=config_name, overrides=overrides)

        if not os.path.exists(args.model_dir):
            raise Exception(f"Model directory {args.model_dir} does not exist")


        fx_model=hydra.utils.instantiate(args.network)
        self.fx_model=fx_model.to(self.device)
        state_dict = torch.load(os.path.join(args.model_dir,args.tester.checkpoint), map_location=self.device, weights_only=False)

        tr_utils.load_state_dict(state_dict, network=fx_model)

        if args.exp.apply_fxnorm:
            print("Applying fx_normalizer")
            self.fx_normalizer= hydra.utils.instantiate(args.exp.fxnorm)
        else:
            print("No fx_normalizer specified, using identity function")
            self.fx_normalizer= lambda x: x  # identity function if no fx_normalizer is specified


    def load_dataset(self, num_tracks_to_load=1):
        ### Loading dataset ###

        normalize_params=omegaconf.OmegaConf.create(
            {
            "normalize_mode": "rms_dry",
            "rms_dry": -25.0
            }
        )


        if self.dataset_code == "TencyMastering_train":
            raise NotImplementedError("TencyMastering_train is not implemented yet")
            num_examples= 64  # use all examples
            num_tracks= num_tracks_to_load
            self.dataset= TencyMastering_Test(
               mode= "dry-wet",
               segment_length= 525312,
               fs= 44100,
               stereo= True,
               num_tracks= num_tracks,
               tracks="all",
               path_csv= "/data5/eloi/TencyMastering/PANNs_country_pop/train_split.csv",
               normalize_params=normalize_params,
               num_examples= num_examples, #use all examples
               RMS_threshold_dB= -40.0,
               seed= 42
            )
        elif self.dataset_code == "TencyMastering_val":
            raise NotImplementedError("TencyMastering_val is not implemented yet")
            num_examples= 64  # use all examples
            num_tracks= num_tracks_to_load
            self.dataset= TencyMastering_Test(
               mode= "dry-wet",
               segment_length= 525312,
               fs= 44100,
               stereo= True,
               num_tracks=num_tracks,
               tracks="all",
               path_csv= "/data5/eloi/TencyMastering/PANNs_country_pop/train_split.csv",
               normalize_params=normalize_params,
               num_examples= num_examples, #use all examples
               RMS_threshold_dB= -40.0,
               seed= 42
            )
        elif self.dataset_code == "TencyMastering_val_randomFx":
            raise NotImplementedError("TencyMastering_val_randomFx is not implemented yet")
            num_examples= 64  # use all examples
            num_tracks= num_tracks_to_load
            self.dataset= TencyMastering_Test(
               mode= "dry-wet",
               segment_length= 525312,
               fs= 44100,
               stereo= True,
               num_tracks=num_tracks,
               tracks="all",
               path_csv= "/data5/eloi/TencyMastering/PANNs_country_pop/train_split.csv",
               normalize_params=normalize_params,
               num_examples= num_examples, #use all examples
               RMS_threshold_dB= -40.0,
               seed= 42,
               apply_randomFx_to_dry= True, #if True, random effects are applied to the samples
               RIR_path_csv="/data5/eloi/ImpulseResponses/rir_files_val.csv",
            )
        elif self.dataset_code == "MDX_TM_benchmark":
            self.dataset= Eval_Benchmark(
               mode= "dry-mixture",
               segment_length= 525312,
               fs= 44100,
               num_tracks=-1,
               num_examples=-1,
               num_segments_per_track=-1,
               path="/data5/eloi/test_set/MDX_TM_benchmark",
            )
        else:
            raise ValueError(f"Unknown dataset_code: {self.dataset_code}")

        #batch_size = 1
        #self.val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, num_workers=1, collate_fn=lambda x: x) 

    def prepare_equal_loudness_anchor(self, target_lufs_dB=-48.0):
        """
        Prepare the equal loudness anchor function.
        This function will be used to normalize the audio to a reference loudness level.
        """
        import pyloudnorm as pyln
        meter=pyln.Meter(44100)  # create a meter for the reference loudness level

        def equal_loudness_anchor(x, *args, **kwargs):
            """
            Normalize all tracks to have equal loudness.
            x: tensor of shape [N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
            """
            N, C, L = x.shape  # N is the number of tracks, C is the number of channels and L is the length of the audio

            norm_tracks = []

            for track_idx in range(x.shape[0]):
                track= x[track_idx].unsqueeze(0)  # Get the track tensor of shape [1, C, L]
                lufs_dB = meter.integrated_loudness(track.squeeze(0).permute(1, 0).cpu().numpy())

                if lufs_dB < -80.0:
                    print(f"Skipping track {track_idx} with {lufs_dB:.2f} LUFS.")
                    continue
        
                lufs_delta_db = target_lufs_dB - lufs_dB
                track *= 10 ** (lufs_delta_db / 20)
                norm_tracks.append(track) #each track is of shape [1, C, L]
        
            norm_tracks = torch.cat(norm_tracks, dim=0)  #shape [N, C, L]
            # create a sum mix with equal loudness
            sum_mix = torch.sum(norm_tracks, dim=0, keepdim=False)

            #peak normalization
            peak = torch.max(torch.abs(sum_mix))
            sum_mix /= peak

            return sum_mix
        
        return equal_loudness_anchor
    
    def prepare_only_rms_anchor(self):

        def only_rms_anchor(x, z, *args, **kwargs):
            """
            Normalize all tracks to have the same RMS level.
            x: tensor of shape [N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
            """
            N, C, L = x.shape
            B=1

            #x_norm= x.mean(dim=1, keepdim=True)  # Stereo to mono
            x=apply_RMS_normalization(x,-25.0, device=self.device)

            y_final=self.apply_rms(x, z)  # Apply RMS normalization to the generated audio


            y_final=y_final.sum(dim=0, keepdim=False)  # Sum the tracks to get a single output

            return y_final

        return only_rms_anchor


    def prepare_feature_extractors(self):

        ### preparing feature extractor ###

        Fxencoder_kwargs=omegaconf.OmegaConf.create(
            {
                "ckpt_path": "/home/eloi/projects/project_mfm_eloi/src/utils/feature_extractors/ckpt/fxenc_plusplus_default.pt"
            }
        )

        from evaluation.feature_extractors import load_fx_encoder_plusplus_2048
        feat_extractor = load_fx_encoder_plusplus_2048(Fxencoder_kwargs, self.device)

        from utils.AF_features_embedding_v2 import AF_fourier_embedding
        AFembedding= AF_fourier_embedding(device=self.device)

        def FxEnc(x):
            """
            x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
            """
            z=feat_extractor(x)
            z= torch.nn.functional.normalize(z, dim=-1, p=2)  # normalize to unit variance
            z= z* math.sqrt(z.shape[-1])  # rescale to keep the same scale
        
            z_af,_=AFembedding.encode(x)
            z_af=z_af* math.sqrt(z_af.shape[-1])  # rescale to keep the same scale
        
            z_all= torch.cat([z, z_af], dim=-1)
        
            #now L2 normalize
        
            norm_z= z_all/ math.sqrt(z_all.shape[-1])  # normalize by dividing by sqrt(dim) to keep the same scale
        
            return norm_z
        
        self.FxEnc=FxEnc

        def embedding_post_processing(z):
            """
            L2 normalize each of the features in z
            """
            z_fxenc=z[..., :2048]  # assuming the FxEncoder features are the first 2048 dimensions
            z_af=z[..., 2048:]  # assuming the AF features are the last 2048 dimensions
        
            z_fxenc=torch.nn.functional.normalize(z_fxenc, dim=-1, p=2)  # normalize to unit variance
            z_af=torch.nn.functional.normalize(z_af, dim=-1, p=2)
        
            z_fxenc=z_fxenc * math.sqrt(z_fxenc.shape[-1])  # rescale to keep the same scale
            z_af=z_af * math.sqrt(z_af.shape[-1])  # rescale to
        
            z_all= torch.cat([z_fxenc, z_af], dim=-1)
        
            return z_all/ math.sqrt(z_all.shape[-1])  # normalize by dividing by sqrt(dim) to keep the same scale

        self.embedding_post_processing=embedding_post_processing


        def get_log_rms_from_z(z):

            z = z * math.sqrt(z.shape[-1])  # rescale to keep the same scale
            AF=z[...,2048:]  # assuming the AF features are the last 2048 dimensions
            AF=AF/ math.sqrt(AF.shape[-1])  # normalize to unit variance
        
            features= AFembedding.decode(AF)
            log_rms=features[0]
        
            return log_rms


        def generate_Fx(x, input_type="dry",num_samples=1, T=30, cfg_scale=1.0, Schurn=10):
            N, C, L = x.shape  # B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
            B=1
        
        
            shape=self.sampler.diff_params.default_shape
            shape= [num_samples, N,*shape[2:]]  # B is the batch size, we want to sample B samples
        
            masks_fwd= torch.ones((B, N), dtype=torch.bool, device=self.device)  # Create masks for all tracks, assuming all tracks are present
            masks_diff= torch.ones((num_samples, N), dtype=torch.bool, device=self.device)  # Create masks for all tracks, assuming all tracks are present
        
            self.sampler.T=T
            self.sampler.Schurn=Schurn  # Set the Schurn parameter for the sampler
        
            with torch.no_grad():
                is_wet= "wet" in input_type
                cond, x_preprocessed=self.sampler.diff_params.transform_forward(x.unsqueeze(0),  is_condition=True, is_test=True, masks=masks_fwd, is_wet=is_wet)
                cond=cond.expand(shape[0], -1, -1,-1)  # Expand the condition to match the batch size
                preds, noise_init = self.sampler.predict_conditional(shape, cond=cond.contiguous(), cfg_scale=cfg_scale, device=self.device,  masks=masks_diff)
        
            return preds

        self.generate_Fx=lambda x, num_samples : generate_Fx(x, input_type="wet", num_samples=num_samples, T=self.method_args.T, cfg_scale=self.method_args.cfg_scale, Schurn=self.method_args.Schurn)


        from utils.feature_extractors.dsp_features import compute_log_rms_gated_v2, compute_log_rms

        def apply_rms(y_hat, z_pred):
            """
            Apply RMS normalization to the generated audio y_hat based on the predicted features z_pred.
            """
            pred_logrms=get_log_rms_from_z(z_pred)  # get the log RMS from the generated features
            pred_rms= 10 ** (pred_logrms / 20)  # convert log RMS to linear scale

            #log_rms_y_hat= compute_log_rms_gated(y_hat)  # compute the log RMS of the generated audio
            if "v6" in self.S1_code:
                log_rms_y_hat= compute_log_rms_gated_v2(y_hat, sample_rate=44100, threshold=-60)
            else:
                log_rms_y_hat= compute_log_rms(y_hat)  # compute the log RMS of the generated audio

            rms_y_hat= 10 ** (log_rms_y_hat / 20)  # convert log RMS to linear scale
            
            gain= pred_rms / (rms_y_hat + 1e-6)  # Compute the gain to apply to the generated audio

            print("pred_rms", pred_rms.shape, "rms_y_hat", rms_y_hat.shape, "gain", gain.shape)
            print("y_hat shape", y_hat.shape)

            y_final= y_hat * gain.unsqueeze(-1)

            return y_final
        
        self.apply_rms=apply_rms

        def apply_effects(x, z_pred):

            x_norm= x.mean(dim=1, keepdim=True)  # Normalize the input audio by its mean across the tracks

            if "v6" in self.S2_code:
                x_norm=apply_RMS_normalization(x_norm,-25.0, device=self.device, use_gate=True)  # Apply RMS normalization with gating
                x_norm=self.fx_normalizer(x_norm, use_gate=True)  # Apply the fx_normalizer if specified
                #x_norm=self.fx_normalizer(x_norm)  # Apply the fx_normalizer if specified
            else:
                x_norm=apply_RMS_normalization(x_norm,-25.0, device=self.device)
                x_norm=self.fx_normalizer(x_norm)  # Apply the fx_normalizer if specified
         
            with torch.no_grad():
                y_hat=self.fx_model(x_norm, z_pred)

            y_final=apply_rms(y_hat, z_pred)

            return y_final
        
        self.apply_effects=apply_effects
        

    def run_evaluation_paired(self):

            """

            Run the evaluation on the validation set

            """

            ### Run evaluation on the validation set ###
            for i in range(len(self.dataset)):

                datav=self.dataset[i]

                x_dry, y_wet , mixture, track_id, segment_id, path_segment= datav 

                print(f"Processing track {i} with path {path_segment}")

                x_dry=x_dry.to(self.device)
                mixture=mixture.to(self.device)

                try:
                    y_wet=y_wet.to(self.device)
                    z_ref=self.FxEnc(y_wet)  # z_y is a tensor of shape [B, N, D] where D is the dimension of the features (2048 + 2048 = 4096)
                    y_final=self.apply_effects(x_dry, z_ref)  # Apply the effects to the input audio
                    y_hat_mixture=y_final.sum(dim=0, keepdim=False)

                    #sf.write(f"{self.path_results}/oracle_{i}.wav", y_hat_mixture.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')  # Save the generated audio
                    os.makedirs(f"{path_segment}/oracle_proposed", exist_ok=True)
                    #peak normalization of y_hat_mixture
                    peak = torch.max(torch.abs(y_hat_mixture))
                    y_hat_mixture /= peak  # Normalize the audio to [-1, 1]
                    sf.write(f"{path_segment}/oracle_proposed/mix.wav", y_hat_mixture.cpu().numpy().T, 44100, subtype='PCM_16')  # Save the generated audio

                except Exception as e:
                    print(f"Error processing track {i} with path {path_segment}: {e}")
                    pass
                    

                if "equal_loudness" in self.anchor_fns:
                    y_equal_loudness=self.anchor_fns["equal_loudness"](x_dry)  # Apply the equal loudness anchor

                    #peak normalization of y_equal_loudness
                    peak = torch.max(torch.abs(y_equal_loudness))
                    y_equal_loudness /= peak  # Normalize the audio to [-1, 1]

                    os.makedirs(f"{path_segment}/anchor_equal_loudness", exist_ok=True)
                    sf.write(f"{path_segment}/anchor_equal_loudness/mix.wav", y_equal_loudness.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')
            
                preds=self.generate_Fx(x_dry, 1) 
                z_pred=self.embedding_post_processing(preds)  # post-process the generated features

                for s in self.sampling_strategies:
                    if s=="random":
                        #take a random sample from the generated featuresjj
                        index=torch.randint(0, z_pred.shape[0], (1,)).item()  # Randomly sample an index from the generated features
                        z_i=z_pred[index] # Randomly sample 100 features from the generated features
                        y_final=self.apply_effects(x_dry.clone(), z_i)  # Apply the effects to the input audio
                        y_hat_mixture=y_final.sum(dim=0, keepdim=False)

                        #peak normalization of y_hat_mixture
                        peak = torch.max(torch.abs(y_hat_mixture))
                        y_hat_mixture /= peak  # Normalize the audio to [-1, 1]

                        os.makedirs(f"{path_segment}/{self.experiment_name}", exist_ok=True)
                        sf.write(f"{path_segment}/{self.experiment_name}/random.wav", y_hat_mixture.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')

                        if "only_rms" in self.anchor_fns:
                            y_rms_anchor=self.anchor_fns["only_rms"](x_dry.clone(), z_i)
                            #peak normalization of y_rms_anchor
                            peak = torch.max(torch.abs(y_rms_anchor))
                            y_rms_anchor /= peak  # Normalize the audio to [-1, 1]
                            #sf.write(f"{self.path_results}/anchor_only_rms_{i}_{j}.wav", y_rms_anchor.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')
                            sf.write(f"{path_segment}/{self.experiment_name}/only_rms_random.wav", y_rms_anchor.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')
                    
                    if s=="centroid_close":
                        #calculate the centroid of the generated features and take the closest one
                        centroid=z_pred.mean(dim=0, keepdim=True)  # Calculate the centroid of
                        #l2 normalize the centroid (pytorch)
                        # L2 normalize the centroid
                        centroid_norm = centroid / torch.norm(centroid, p=2, dim=-1, keepdim=True)

                        # Calculate similarities (dot products)
                        #print("centroid_norm shape", centroid_norm.shape, "z_pred_norm shape", z_pred_norm.shape)
                        similarities = torch.matmul(z_pred.view(z_pred.shape[0],-1), centroid_norm.view(1,-1).T).squeeze()
                        
                        # Get the index of the sample closest to the centroid
                        index = torch.argmax(similarities).item()
                        
                        # Use the closest sample
                        z_i = z_pred[index]
                        y_final = self.apply_effects(x_dry.clone(), z_i)
                        y_hat_mixture = y_final.sum(dim=0, keepdim=False)

                        #peak normalization of y_hat_mixture
                        peak = torch.max(torch.abs(y_hat_mixture))
                        y_hat_mixture /= peak  # Normalize the audio to [-1, 1]
                        
                        os.makedirs(f"{path_segment}/{self.experiment_name}", exist_ok=True)
                        sf.write(f"{path_segment}/{self.experiment_name}/centroid_close.wav", y_hat_mixture.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')

                        if "only_rms" in self.anchor_fns:
                            y_rms_anchor=self.anchor_fns["only_rms"](x_dry.clone(), z_i)
                            #peak normalization of y_rms_anchor
                            peak = torch.max(torch.abs(y_rms_anchor))
                            y_rms_anchor /= peak  # Normalize the audio to [-1, 1]
                            #sf.write(f"{self.path_results}/anchor_only_rms_{i}_{j}.wav", y_rms_anchor.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')
                            sf.write(f"{path_segment}/{self.experiment_name}/only_rms_centroid_close.wav", y_rms_anchor.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')
                    
                    if s=="centroid_far":
                        # Calculate the centroid of the generated features
                        centroid = z_pred.mean(dim=0, keepdim=True)
                        
                        # L2 normalize the centroid
                        centroid_norm = centroid / torch.norm(centroid, p=2, dim=-1, keepdim=True)
                        
                        # Calculate cosine similarity between each sample and the centroid
                        similarities = torch.matmul(z_pred.view(z_pred.shape[0],-1), centroid_norm.view(1,-1).T).squeeze()
                        
                        # Get the index of the sample furthest from the centroid
                        index = torch.argmin(similarities).item()
                        
                        # Use the furthest sample
                        z_i = z_pred[index]
                        y_final = self.apply_effects(x_dry.clone(), z_i)
                        y_hat_mixture = y_final.sum(dim=0, keepdim=False)

                        #peak normalization of y_hat_mixture
                        peak = torch.max(torch.abs(y_hat_mixture))
                        y_hat_mixture /= peak  # Normalize the audio to [-1, 1]
                        
                        os.makedirs(f"{path_segment}/{self.experiment_name}", exist_ok=True)
                        sf.write(f"{path_segment}/{self.experiment_name}/centroid_far.wav", y_hat_mixture.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')

                        if "only_rms" in self.anchor_fns:
                            y_rms_anchor=self.anchor_fns["only_rms"](x_dry.clone(), z_i)
                            #peak normalization of y_rms_anchor
                            peak = torch.max(torch.abs(y_rms_anchor))
                            y_rms_anchor /= peak  # Normalize the audio to [-1, 1]
                            #sf.write(f"{self.path_results}/anchor_only_rms_{i}_{j}.wav", y_rms_anchor.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')
                            sf.write(f"{path_segment}/{self.experiment_name}/only_rms_centroid_far.wav", y_rms_anchor.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')
                        
            
                #for j in range(z_pred.shape[0]):
                #    y_final=self.apply_effects(x_dry.clone(), z_pred[j])  # Apply the effects to the input audio
                
                #    y_hat_mixture=y_final.sum(dim=0, keepdim=False)
                
                #    sf.write(f"{self.path_results}/pred_blackbox_{i}_{j}.wav", y_hat_mixture.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')

                #    if "only_rms" in self.anchor_fns:
                #        y_rms_anchor=self.anchor_fns["only_rms"](x_dry.clone(), z_pred[j])
                #        sf.write(f"{self.path_results}/anchor_only_rms_{i}_{j}.wav", y_rms_anchor.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')

                #preds=self.generate_Fx(x_dry,  500 ) 
                #preds=self.embedding_post_processing(preds)  # post-process the generated features
            
                # Step 1: Optional PCA preprocessing (recommended for high dimensions)
                # If you want to add PCA preprocessing:
                #from sklearn.decomposition import PCA
                #pca = PCA(n_components=2)
                #transform_data = pca.fit_transform(preds.view(preds.shape[0], -1).cpu().numpy())

                #transform_reference= pca.transform(z_ref.view(1, -1).cpu().numpy())
            
                #data_dict = {
                #   "predicted": transform_data,
                #   "reference": transform_reference,  # Use the first element as reference
                #    } 
            
                #fig = make_PCA_figure(data_dict, title="PCA")
                #fig.savefig(f"{self.path_results}/pca_blackbox_{i}.png")
