
#load generator model
import sys
import omegaconf
import hydra
import torch
import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import math
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils.training_utils as tr_utils
from utils.data_utils import apply_RMS_normalization
from utils.data_utils import read_wav_segment, get_audio_length
from utils.feature_extractors.dsp_features import compute_log_rms_gated_max, compute_log_rms




class MusicMixer():
    def __init__(self, method_args=None, path_input=None,  path_output=None):

        self.method_args = method_args

        self.S1_code = method_args.S1_code
        self.S2_code = method_args.S2_code

        self.path_output = path_output
        os.makedirs(self.path_output, exist_ok=True)

        assert self.path_output is not None, "path_output must be specified"
        assert os.path.exists(self.path_output), f"path_output {self.path_output} does not exist"
        self.path_input = path_input
        assert self.path_input is not None, "path_input must be specified"
        assert os.path.exists(self.path_input), f"path_input {self.path_input} does not exist"


        self.config_file_rel="conf"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.load_FxGenerator()  # Load the S1 model
        self.load_FxProcessor()  # Load the S2 model
        self.prepare_feature_extractors()  # Prepare the feature extractors

        self.load_data()  # Load the dataset
        

    def load_FxGenerator(self):

        if self.S1_code == "S9v6":
            config_name="conf_FxGenerator_TencyDB.yaml"
            model_dir="checkpoints"
            ckpt="S9v6_1C_tencymastering_vocals-200000.pt"
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
    
    def load_FxProcessor(self):

        ### Loading effects model ###

        if self.S2_code == "MF3wetv6":
            #config_name="conf_MF3gatewet_mapper_blackbox_predictive_fxenc2048AFv3CLAP_paired.yaml"
            config_name="conf_FxProcessor_TencyMastering.yaml"
            #model_dir="/data2/eloi/checkpoints/MF3gatewet"
            model_dir="checkpoints"
            #ckpt="mapper_blackbox_TCN-300000.pt"
            ckpt="MF3wetv6_mapper_blackbox_TCN-270000.pt"
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


    def load_data(self):
        ### Loading data ###

        #search audio files in the path_input directory
        audio_files = [f for f in os.listdir(self.path_input) if f.endswith('.wav') or f.endswith('.mp3') or f.endswith('.flac')]
        if not audio_files:
            raise Exception(f"No audio files found in {self.path_input}")
        else:
            print(f"Found {len(audio_files)} audio files in {self.path_input}")
            print("Input tracks:", audio_files)
        
        tracks= []
        track_names= []

        for i in range(len(audio_files)):
            file_path = os.path.join(self.path_input, audio_files[i])

            x, fs=read_wav_segment(file_path, None,  None)
            
            x=torch.from_numpy(x)

            #convert to mono if stereo
            if x.ndim == 2 and x.shape[-1] > 1:
                x = x.mean(dim=-1, keepdim=True)  # Convert stereo to mono by averaging channels
            elif x.ndim == 1:
                x = x.unsqueeze(-1)
            
            x=x.permute(1, 0)  # Change shape to [C, L] where C is the number of channels and L is the length of the audio

            basename= os.path.basename(file_path)

            tracks.append(x)
            track_names.append(basename)
        
        self.tracks= torch.stack(tracks, dim=0)  # shape [N, C, L] where N is the number of tracks, C is the number of channels and L is the length of the audio
        self.track_names= track_names  # list of track names


    def prepare_feature_extractors(self):

        ### preparing feature extractor ###

        Fxencoder_kwargs=omegaconf.OmegaConf.create(
            {
                "ckpt_path": "/home/eloi/projects/project_mfm_eloi/src/utils/feature_extractors/ckpt/fxenc_plusplus_default.pt"
            }
        )

        from utils.feature_extractors.load_features import load_fx_encoder_plusplus_2048
        feat_extractor = load_fx_encoder_plusplus_2048(Fxencoder_kwargs, self.device)

        from utils.feature_extractors.AF_features_embedding import AF_fourier_embedding
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



        def apply_rms(y_hat, z_pred):
            """
            Apply RMS normalization to the generated audio y_hat based on the predicted features z_pred.
            """
            pred_logrms=get_log_rms_from_z(z_pred)  # get the log RMS from the generated features
            pred_rms= 10 ** (pred_logrms / 20)  # convert log RMS to linear scale

            #log_rms_y_hat= compute_log_rms_gated(y_hat)  # compute the log RMS of the generated audio
            if "v6" in self.S1_code:
                log_rms_y_hat= compute_log_rms_gated_max(y_hat, sample_rate=44100, threshold=-60)
            else:
                log_rms_y_hat= compute_log_rms(y_hat)  # compute the log RMS of the generated audio

            rms_y_hat= 10 ** (log_rms_y_hat / 20)  # convert log RMS to linear scale
            
            gain= pred_rms / (rms_y_hat + 1e-6)  # Compute the gain to apply to the generated audio

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

        def apply_effects_full_track(x, z_pred,  seg_size=16384, pad_size=16384):


            x_norm= x.mean(dim=1, keepdim=True)  # Normalize the input audio by its mean across the tracks


            for i in range(x_norm.shape[0]):
                x_i= x_norm[i:i+1]  # Select the i-th track

                # discard the parts without activity (RMS < -60 dBFS)

                # (bs, c, num_frames, seg_size)
                B, C, L = x_i.size()
                assert B==1
                assert C==1

                #pad x_i to have a length multiple of seg_size
                x_i = torch.nn.functional.pad(x_i, (0, seg_size - (L % seg_size)), mode='constant', value=0)
            
                x_frames = x_i.unfold(2, seg_size, seg_size)  # (bs, c, num_frames, seg_size)

                # RMS over last dimension (seg_size)
                rms = torch.sqrt((x_frames ** 2).mean(dim=-1))  # (bs, c, num_frames)

                #mean over channelos
                rms = rms.mean(dim=1, keepdim=True) # (bs, 1, num_frames)

                # dB conversion
                rms_db = 20 * torch.log10(rms.clamp(min=1e-8))  # (bs, 1, num_frames)


                # discard frames with RMS < -60 dBFS
                valid_frames = rms_db > -80


                #valid_frames= valid_frames.unsqueeze(-1).expand(x_frames.shape)  # Expand valid_frames to match x_frames shape

                assert valid_frames.shape[0]==1
                assert valid_frames.shape[1]==1

                valid_frames = valid_frames.view(-1)  # Remove the channel and batch dimension


                x_frames_filtered = x_frames[:, : , valid_frames, :]  # Keep only valid frames


                x_filtered= x_frames_filtered.contiguous().view(B, C, -1)  # Reshape back to (bs, c, L)


                if "v6" in self.S2_code:
                    x_filtered=apply_RMS_normalization(x_filtered,-25.0, device=self.device, use_gate=True)  # Apply RMS normalization with gating
                    x_filtered=self.fx_normalizer(x_filtered, use_gate=True)  # Apply the fx_normalizer if specified
                    #x_norm=self.fx_normalizer(x_norm)  # Apply the fx_normalizer if specified
                else:
                    x_filtered=apply_RMS_normalization(x_filtered,-25.0, device=self.device)
                    x_filtered=self.fx_normalizer(x_filtered)  # Apply the fx_normalizer if specified
                
         
                #pad some zeros to the beginning and end of the audio to avoid edge effects


                x_filtered = torch.nn.functional.pad(x_filtered, (pad_size, pad_size), mode='reflect')  # Pad with the last value to avoid edge effects



                with torch.no_grad():
                    y_hat_i=self.fx_model(x_filtered, z_pred[i:i+1])

                y_hat_i=y_hat_i[:, :, pad_size:-pad_size]  # Remove the padding

                #apply fade in and fade out to avoid clicks
                window_left= torch.hann_window(pad_size, device=self.device).view(1, 1, -1)[:, :, :pad_size//2]  # (1, 1, pad_size)
                window_right= torch.hann_window(pad_size, device=self.device).view(1, 1, -1)[:, :, pad_size//2:]  # (1, 1, pad_size)
                y_hat_i[:, :, :pad_size//2] *= window_left  # Apply fade in
                y_hat_i[:, :, -pad_size//2:] *= window_right  # Apply fade out


                #add the removed frames back to the output
                y_hat_i_frames= y_hat_i.unfold(2, seg_size, seg_size)  # (bs, c, num_frames, seg_size)

                #add frames with zero activity back to the output
                shape_zeros=(1, 2, x_i.shape[-1])
                zeros=torch.zeros(shape_zeros, device=self.device)
                y_hat_i_frames_full= zeros.unfold(2, seg_size, seg_size)  # (bs, c, num_frames, seg_size)

                y_hat_i_frames_full[0,:, valid_frames, :] = y_hat_i_frames  # Fill valid frames

                y_hat_i = y_hat_i_frames_full.contiguous().view(1, 2, -1)


                assert y_hat_i.shape[-1] >= L, f"y_hat_i shape {y_hat_i.shape} does not match input shape {x_i.shape}"

                y_hat_i = y_hat_i[:, :, :L]  # Trim to the original length
                

                if i == 0:
                    y_hat=y_hat_i
                else:
                    y_hat=torch.cat([y_hat, y_hat_i], dim=0)

            y_final=apply_rms(y_hat, z_pred)

            return y_final
        
        self.apply_effects=apply_effects
        self.apply_effects_full_track=apply_effects_full_track
        
    def select_high_energy_segments(self, x_dry, seq_length=525312):
            N, C, L = x_dry.shape
            highest_energy_segments = []
            
            for track_idx in range(N):
                track = x_dry[track_idx]
                
                # Calculate energy for windows of size seq_length
                num_windows = L - seq_length + 1
                max_energy = 0
                max_energy_start = 0
                
                for i in range(0, num_windows, 1000):  # Step by 1000 for efficiency
                    segment = track[..., i:i+seq_length]
                    energy = (segment ** 2).sum()
                    
                    if energy > max_energy:
                        max_energy = energy
                        max_energy_start = i
                
                # Fine-tune search around the best region
                fine_start = max(0, max_energy_start - 1000)
                fine_end = min(L - seq_length + 1, max_energy_start + 1000)
                
                for i in range(fine_start, fine_end):
                    segment = track[..., i:i+seq_length]
                    energy = (segment ** 2).sum()
                    
                    if energy > max_energy:
                        max_energy = energy
                        max_energy_start = i
                
                highest_energy_segments.append(x_dry[track_idx, :, max_energy_start:max_energy_start+seq_length])
            
            # Stack the highest energy segments back together
            return torch.stack(highest_energy_segments, dim=0).to(self.device)

    def run_automix_single_segment(self):

            """
            Process one single segment (Default mode)

            """

            x_dry=self.tracks  # Use the loaded tracks as the input dry audio
            track_names=self.track_names  # Use the loaded track names

            x_dry_original= x_dry.clone().to(self.device)  # Keep a copy of the original dry audio for later use

            seq_length= 525312
            assert x_dry.shape[-1] == seq_length, f"Input audio length {x_dry.shape[-1]} is not equal to the expected sequence length {seq_length}"

            rms_dry=compute_log_rms_gated_max(x_dry, sample_rate=44100)  # Compute the log RMS of the dry audio
            silent_tracks = rms_dry < -60  # Identify silent tracks
            silent_tracks = silent_tracks.squeeze()  # Remove singleton dimensions


            #remove silent tracks from x_dry and track_names
            if silent_tracks.any():
                print(f"Removing {silent_tracks.sum()} silent tracks from the input audio")
                #shape before removing silent tracks is (N, C, L)
                x_dry = x_dry[~silent_tracks]  # Remove silent tracks
                track_names = [name for i, name in enumerate(track_names) if not silent_tracks[i]]
                x_dry_original = x_dry_original[~silent_tracks]  # Keep the original dry audio for the remaining tracks
        
            x_dry=x_dry.to(self.device)


            preds=self.generate_Fx(x_dry, 1) 

            z_pred=self.embedding_post_processing(preds)  # post-process the generated features

            y_final=self.apply_effects(x_dry_original, z_pred[0], batch=False)  # Apply the effects to the input audio
            

            y_hat_mixture=y_final.sum(dim=0, keepdim=False)
            if y_hat_mixture.abs().max() > 1.0:
                y_hat_mixture = y_hat_mixture / y_hat_mixture.abs().max()  # Normalize the mixture to [-1, 1] if necessary


            #save all the outputs (every track with its name and the final mixture)
            output_dir = self.path_output

            for i in range(x_dry.shape[0]):
                track_name = track_names[i]
                output_path = os.path.join(output_dir, f"{track_name}_dry.wav")
                sf.write(output_path, x_dry[i].cpu().numpy().T, 44100)
                print(f"Saved dry track {track_name} to {output_path}")

            for i in range(y_final.shape[0]):
                track_name = track_names[i]
                output_path = os.path.join(output_dir, f"{track_name}_processed.wav")
                sf.write(output_path, y_final[i].cpu().numpy().T, 44100)
                print(f"Saved processed track {track_name} to {output_path}")

            #save the final mixture
            mixture_output_path = os.path.join(output_dir, "mixture_processed.wav")
            sf.write(mixture_output_path, y_hat_mixture.cpu().numpy().T, 44100)
            print(f"Saved final mixture to {mixture_output_path}")

    def run_automix_full_track(self):

            """

            Process the full track. 
            The script does:
                1. Load the input audio tracks from the specified path
                2. Select the segments of seq_length size that have the highest energy for each track
                3. Generate the features using the diffusion model, using features extracted from the high-energy segments
                4. Apply the effects using the processor, dis
                5. Save the processed tracks and the final mixture to the output path


            """

            x_dry=self.tracks  # Use the loaded tracks as the input dry audio
            track_names=self.track_names  # Use the loaded track names

            print("x_dry shape:", x_dry.shape)

            x_dry_original= x_dry.clone().to(self.device)  # Keep a copy of the original dry audio for later use

            seq_length= 525312

            if x_dry.shape[-1] > seq_length:
                # search for each track  the segment of seq_length size that has highest energy
                # x_dry shape is (N, C, L) where N is the number of tracks, C is the number of channels and L is the length of the audio
                x_dry = self.select_high_energy_segments(x_dry, seq_length=seq_length)

            #first check if all tracks in x_dry have activity (RMS > -60 dBFS)

            rms_dry=compute_log_rms_gated_max(x_dry, sample_rate=44100)  # Compute the log RMS of the dry audio
            silent_tracks = rms_dry < -60  # Identify silent tracks
            silent_tracks = silent_tracks.squeeze()  # Remove singleton dimensions


            #remove silent tracks from x_dry and track_names
            if silent_tracks.any():
                print(f"Removing {silent_tracks.sum()} silent tracks from the input audio")
                #shape before removing silent tracks is (N, C, L)
                x_dry = x_dry[~silent_tracks]  # Remove silent tracks
                track_names = [name for i, name in enumerate(track_names) if not silent_tracks[i]]
                x_dry_original = x_dry_original[~silent_tracks]  # Keep the original dry audio for the remaining tracks
        
            x_dry=x_dry.to(self.device)


            preds=self.generate_Fx(x_dry, 1) 

            z_pred=self.embedding_post_processing(preds)  # post-process the generated features

            y_final=self.apply_effects_full_track(x_dry_original, z_pred[0] )  # Apply the effects to the input audio
            print("y_final shape:", y_final.shape)

            y_hat_mixture=y_final.sum(dim=0, keepdim=False)
            if y_hat_mixture.abs().max() > 1.0:
                y_hat_mixture = y_hat_mixture / y_hat_mixture.abs().max()  # Normalize the mixture to [-1, 1] if necessary


            #save all the outputs (every track with its name and the final mixture)
            output_dir = self.path_output


            for i in range(y_final.shape[0]):
                track_name = track_names[i]
                output_path = os.path.join(output_dir, f"{track_name}_processed.wav")
                sf.write(output_path, y_final[i].cpu().numpy().T, 44100)
                print(f"Saved processed track {track_name} to {output_path}")

            #save the final mixture
            mixture_output_path = os.path.join(output_dir, "mixture_processed.wav")
            sf.write(mixture_output_path, y_hat_mixture.cpu().numpy().T, 44100)
            print(f"Saved final mixture to {mixture_output_path}")



if __name__ == "__main__":


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    method_args= {
        "S1_code": "S9v6",
        "S2_code": "MF3wetv6",
        "T": 30,
        "Schurn": 5,
        "cfg_scale": 1.0,
    }

    method_args=omegaconf.OmegaConf.create(method_args)

    automixer=MusicMixer(method_args=method_args, path_input="examples/full_song", path_output="results/full_song")

    automixer.run_automix_full_track()
