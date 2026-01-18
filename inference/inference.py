# Copyright (c) 2025 Sony Research
# Licensed under CC BY-NC-SA 4.0
# See LICENSE file for details

import sys
import os
from utils.feature_extractors.dsp_features import compute_log_rms_gated_max

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import pyloudnorm as pyln

import hydra
import torch
import torchaudio
from hydra import initialize, compose
import utils.training_utils as tr_utils

import torch
import omegaconf
import math

import soundfile as sf
from utils.data_utils import apply_RMS_normalization

import pandas as pd


import numpy as np
import glob

from utils.data_utils import read_wav_segment


def load_audio( file, start=None, end=None, stereo=True):

            x, fs=read_wav_segment(file, start, end)
            if stereo:
                if len(x.shape)==1:
                    #print( "dry not stereo , doubling channels", x_dry.shape)
                    x=x[:,np.newaxis]
                    x= np.concatenate((x, x), axis=-1)
                elif len(x.shape)==2 and x.shape[-1]==1:
                    #print( "dry not stereo , doubling channels", x_dry.shape)
                    x = np.concatenate((x, x), axis=-1)

            x=torch.from_numpy(x).permute(1,0)

            return x, fs


class Inference:
    def __init__(
        self,
        method_args=None,
        path_benchmark="/add/the/path/to/the/benchmark/data/here",
        load_segment_length=525312, #segment length used for loading and extract CLAP embeddings
        processor_segment_length=525312, #segment length used for loading and extract CLAP embeddings
        processor_overlap=8192,
    ):

        self.method_args = method_args
        self.path_benchmark = path_benchmark
        self.load_segment_length=load_segment_length
        self.processor_segment_length=processor_segment_length
        self.processor_overlap=processor_overlap

        self.results_df = pd.DataFrame(columns=["track", "method", "metric", "value"])

        self.FxGenerator_code = method_args.FxGenerator_code
        self.FxProcessor_code = method_args.FxProcessor_code

        self.config_file_rel = "../conf"
        # self.config_path="/home/eloi/projects/project_mfm_eloi/src/conf"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_FxGenerator()  # Load the S1 model
        self.load_FxProcessor()  # Load the S2 model
        self.prepare_feature_extractors()  # Prepare the feature extractors

    def load_FxGenerator(self):
        if self.FxGenerator_code == "public":
            config_name = "conf_FxGenerator_Public.yaml"
            model_dir = "checkpoints"
            ckpt = "FxGenerator_public.pt"
        else:
            raise ValueError(f"Unknown FxGenerator_code: {self.FxGenerator_code}")

        overrides = [
            f"model_dir={model_dir}",
            f"tester.checkpoint={ckpt}",
        ]

        with initialize(version_base=None, config_path=self.config_file_rel):
            args = compose(config_name=config_name, overrides=overrides)

        if not os.path.exists(args.model_dir):
            raise Exception(f"Model directory {args.model_dir} does not exist")

        diff_params = hydra.utils.instantiate(args.diff_params)

        network = hydra.utils.instantiate(args.network)
        network = network.to(self.device)
        state_dict = torch.load(
            os.path.join(args.model_dir, args.tester.checkpoint),
            map_location=self.device,
            weights_only=False,
        )

        tr_utils.load_state_dict(state_dict, ema=network)

        self.sampler = hydra.utils.instantiate(
            args.tester.sampler,
            network,
            diff_params,
            args,
        )

    def load_FxProcessor(self):

        ### Loading effects model ###

        if self.FxProcessor_code == "public":
            config_name = "conf_FxProcessor_Public.yaml"
            model_dir = "checkpoints"
            ckpt = "FxProcessor_public.pt"
        else:
            raise ValueError(f"Unknown FxProcessor_code: {self.FxProcessor_code}")

        overrides = [
            f"model_dir={model_dir}",
            f"tester.checkpoint={ckpt}",
        ]

        with initialize(version_base=None, config_path=self.config_file_rel):
            args = compose(config_name=config_name, overrides=overrides)

        if not os.path.exists(args.model_dir):
            raise Exception(f"Model directory {args.model_dir} does not exist")

        fx_model = hydra.utils.instantiate(args.network)
        self.fx_model = fx_model.to(self.device)
        state_dict = torch.load(
            os.path.join(args.model_dir, args.tester.checkpoint),
            map_location=self.device,
            weights_only=False,
        )

        tr_utils.load_state_dict(state_dict, network=fx_model)

        if args.exp.apply_fxnorm:
            print("Applying fx_normalizer")
            if "public" in self.FxProcessor_code:
                fx_normalizer = hydra.utils.instantiate(args.exp.fxnorm)
                self.fx_normalizer = lambda x: fx_normalizer(
                    x, use_gate=args.exp.use_gated_RMSnorm, RMS=args.exp.RMS_norm
                )
            else:
                self.fx_normalizer = hydra.utils.instantiate(args.exp.fxnorm)

        else:
            print("No fx_normalizer specified, using identity function")
            self.fx_normalizer = (
                lambda x: x
            )  # identity function if no fx_normalizer is specified


    def prepare_feature_extractors(self):

        ### preparing feature extractor ###

        Fxencoder_kwargs = omegaconf.OmegaConf.create(
            {
                "ckpt_path": "checkpoints/fxenc_plusplus_default.pt"
            }
        )

        from utils.feature_extractors.load_features import load_fx_encoder_plusplus_2048

        feat_extractor = load_fx_encoder_plusplus_2048(Fxencoder_kwargs, self.device)

        from utils.feature_extractors.AF_features_embedding import AF_fourier_embedding

        AFembedding = AF_fourier_embedding(device=self.device)

        def FxEnc(x):
            """
            x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
            """
            z = feat_extractor(x)
            z = torch.nn.functional.normalize(
                z, dim=-1, p=2
            )  # normalize to unit variance
            z = z * math.sqrt(z.shape[-1])  # rescale to keep the same scale

            z_af, _ = AFembedding.encode(x)
            z_af = z_af * math.sqrt(z_af.shape[-1])  # rescale to keep the same scale

            z_all = torch.cat([z, z_af], dim=-1)

            # now L2 normalize

            norm_z = z_all / math.sqrt(
                z_all.shape[-1]
            )  # normalize by dividing by sqrt(dim) to keep the same scale

            return norm_z

        self.FxEnc = FxEnc

        def embedding_post_processing(z):
            """
            L2 normalize each of the features in z
            """
            z_fxenc = z[
                ..., :2048
            ]  # assuming the FxEncoder features are the first 2048 dimensions
            z_af = z[
                ..., 2048:
            ]  # assuming the AF features are the last 2048 dimensions

            z_fxenc = torch.nn.functional.normalize(
                z_fxenc, dim=-1, p=2
            )  # normalize to unit variance
            z_af = torch.nn.functional.normalize(z_af, dim=-1, p=2)

            z_fxenc = z_fxenc * math.sqrt(
                z_fxenc.shape[-1]
            )  # rescale to keep the same scale
            z_af = z_af * math.sqrt(z_af.shape[-1])  # rescale to

            z_all = torch.cat([z_fxenc, z_af], dim=-1)

            return z_all / math.sqrt(
                z_all.shape[-1]
            )  # normalize by dividing by sqrt(dim) to keep the same scale

        self.embedding_post_processing = embedding_post_processing

        def get_log_rms_from_z(z):

            z = z * math.sqrt(z.shape[-1])  # rescale to keep the same scale
            AF = z[..., 2048:]  # assuming the AF features are the last 2048 dimensions
            AF = AF / math.sqrt(AF.shape[-1])  # normalize to unit variance

            features = AFembedding.decode(AF)
            log_rms = features[0]

            return log_rms

        def generate_Fx(
            x, input_type="dry", num_samples=1, T=30, cfg_scale=1.0, Schurn=10
        ):
            N, C, L = (
                x.shape
            )  # B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
            B = 1

            shape = self.sampler.diff_params.default_shape
            shape = [
                num_samples,
                N,
                *shape[2:],
            ]  # B is the batch size, we want to sample B samples

            masks_fwd = torch.ones(
                (B, N), dtype=torch.bool, device=self.device
            )  # Create masks for all tracks, assuming all tracks are present
            masks_diff = torch.ones(
                (num_samples, N), dtype=torch.bool, device=self.device
            )  # Create masks for all tracks, assuming all tracks are present

            self.sampler.T = T
            self.sampler.Schurn = Schurn  # Set the Schurn parameter for the sampler

            with torch.no_grad():
                is_wet = "wet" in input_type
                cond, x_preprocessed = self.sampler.diff_params.transform_forward(
                    x.unsqueeze(0),
                    is_condition=True,
                    is_test=True,
                    masks=masks_fwd,
                    is_wet=is_wet,
                )
                cond = cond.expand(
                    shape[0], -1, -1, -1
                )  # Expand the condition to match the batch size
                preds, noise_init = self.sampler.predict_conditional(
                    shape,
                    cond=cond.contiguous(),
                    cfg_scale=cfg_scale,
                    device=self.device,
                    masks=masks_diff,
                )

            return preds

        self.generate_Fx = lambda x, num_samples: generate_Fx(
            x,
            input_type="wet",
            num_samples=num_samples,
            T=self.method_args.T,
            cfg_scale=self.method_args.cfg_scale,
            Schurn=self.method_args.Schurn,
        )


        def apply_rms(y_hat, z_pred):
            """
            Apply RMS normalization to the generated audio y_hat based on the predicted features z_pred.
            """
            pred_logrms = get_log_rms_from_z(
                z_pred
            )  # get the log RMS from the generated features
            pred_rms = 10 ** (pred_logrms / 20)  # convert log RMS to linear scale

            log_rms_y_hat = compute_log_rms_gated_max(
                y_hat, sample_rate=44100, threshold=-60
            )

            rms_y_hat = 10 ** (log_rms_y_hat / 20)  # convert log RMS to linear scale

            gain = pred_rms / (
                rms_y_hat + 1e-6
            )  # Compute the gain to apply to the generated audio

            y_final = y_hat * gain.unsqueeze(-1)

            return y_final

        self.apply_rms = apply_rms

        def apply_effects(x, z_pred):
            segment_length = self.processor_segment_length
            overlap = self.processor_overlap
            total_length = x.shape[-1]
            batch_size = x.shape[0]
        
            # Normalize input and conditioning outside block loop
            x_norm = x.mean(dim=1, keepdim=True)
        
            if total_length > segment_length:
                y_final = torch.zeros((batch_size, 2, total_length), device=x.device, dtype=x.dtype)
        
                hann = torch.hann_window(overlap * 2, device=x.device, dtype=x.dtype)
                hann_left = hann[:overlap].view(1, 1, -1)
                hann_right = hann[overlap:].view(1, 1, -1)

                step = segment_length - overlap
                positions = list(range(0, total_length - overlap, step))
                for i, start in enumerate(positions):
                    end = min(start + segment_length, total_length)

                    seg_x_norm = x_norm[..., start:end]

                    #check activity in seg_x_norm

                    rms_dry_segment=compute_log_rms_gated_max(seg_x_norm, sample_rate=44100)  # Compute the log RMS of the dry audio
                    indices_non_silent = torch.where(rms_dry_segment > -45)[0]  # Identify silent tracks


                    seg_x_norm_non_silent = seg_x_norm[indices_non_silent]
                    z_pred_non_silent = z_pred[indices_non_silent]

                    if "public" in self.FxProcessor_code:
                        seg_x_norm_non_silent = self.fx_normalizer(seg_x_norm_non_silent)
                    else:
                        seg_x_norm_non_silent = apply_RMS_normalization(seg_x_norm_non_silent, -25.0, device=self.device, use_gate=True)
                        seg_x_norm_non_silent = self.fx_normalizer(seg_x_norm_non_silent, use_gate=True)
        
                    with torch.no_grad():
                        seg_y_hat_non_silent=torch.zeros((seg_x_norm_non_silent.shape[0],2, seg_x_norm_non_silent.shape[2]), device=x.device, dtype=x.dtype)
                        #I thought it may be better (but less efficient) to run it like this instead of in parallel. To avoid OOM issues.
                        for i in range(seg_x_norm_non_silent.shape[0]):
                            seg_y_hat_non_silent[i] = self.fx_model(seg_x_norm_non_silent[i].unsqueeze(0), z_pred_non_silent[i].unsqueeze(0)).squeeze(0)

                    seg_y_hat_non_silent = apply_rms(seg_y_hat_non_silent, z_pred_non_silent)
                    
                    #fill with zeros the silent segments
                    seg_y_hat= torch.zeros((seg_x_norm.shape[0], seg_y_hat_non_silent.shape[1], seg_y_hat_non_silent.shape[2]), device=x.device, dtype=x.dtype)
                    seg_y_hat[indices_non_silent]=seg_y_hat_non_silent

        
                    seg_len = end - start
        
                    if i == 0:
                        # First segment
                        y_final[..., start:end-overlap] += seg_y_hat[..., :seg_len-overlap]
                        y_final[..., end-overlap:end] += seg_y_hat[..., seg_len-overlap:] * hann_right
                    elif end == total_length:
                        # Last segment
                        y_final[..., start:start+overlap] += seg_y_hat[..., :overlap] * hann_left
                        y_final[..., start+overlap:end] += seg_y_hat[..., overlap:]
                    else:
                        # Middle segments
                        y_final[..., start:start+overlap] += seg_y_hat[..., :overlap] * hann_left
                        y_final[..., start+overlap:end-overlap] += seg_y_hat[..., overlap:seg_len-overlap]
                        y_final[..., end-overlap:end] += seg_y_hat[..., seg_len-overlap:] * hann_right
        
                return y_final
        
            else:
                with torch.no_grad():
                    y_hat=torch.zeros((x_norm.shape[0], 2, x_norm.shape[2]), device=x.device, dtype=x.dtype)
                    for i in range(x_norm.shape[0]):
                        y_hat[i] = self.fx_model(x_norm[i].unsqueeze(0), z_pred[i].unsqueeze(0)).squeeze(0)
                y_final = apply_rms(y_hat, z_pred)
                return y_final        

        self.apply_effects = apply_effects

    def select_high_energy_segment(self, x_dry, seq_length=525312):
            C, L = x_dry.shape
            
            track = x_dry
                
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
                
            return x_dry[:, max_energy_start:max_energy_start+seq_length]


    def run_inference_single_song(self, exp_name="test_Sep28", directory=None, num_samples=1):
        """
        Run the inference on a single example
        """

        dry_files=glob.glob(os.path.join(directory, "*.wav"))

        assert len(dry_files) > 0, f"No .wav files found in {directory}"
        print(f"Found {len(dry_files)} dry files in {directory}")
        print(dry_files)

        dry_tracks=[]
        dry_tracks_segments=[]

        for f in dry_files:
            x_dry_i, fs=load_audio(str(f), stereo=True)
            x_dry_i=x_dry_i.to(self.device)

            if fs!=self.sampler.diff_params.sample_rate:
                x_dry_i=torchaudio.functional.resample(x_dry_i, orig_sr=fs, target_sr=self.sampler.diff_params.sample_rate)
            #stereo to mono
            x_dry_i = x_dry_i.mean(dim=0, keepdim=True)
            dry_tracks.append(x_dry_i )

            if x_dry_i.shape[-1] >= self.load_segment_length:
                # search for each track  the segment of seq_length size that has highest energy
                # x_dry_i shape is (N, C, L) where N is the number of tracks, C is the number of channels and L is the length of the audio
                x_dry_i_segment = self.select_high_energy_segment(x_dry_i, seq_length=self.load_segment_length)
            else:
                raise ValueError(f"Input audio {f} is too short, needs to be at least {self.load_segment_length/self.sampler.diff_params.sample_rate:.2f} seconds.")

            dry_tracks_segments.append(x_dry_i_segment)
        
        x_dry=torch.stack(dry_tracks, dim=0)  # shape (N, C, L) where N is the number of tracks, C is the number of channels and L is the length of the audio
        x_dry_segments= torch.stack(dry_tracks_segments, dim=0)  # shape (N, C, L) where N is the number of tracks, C is the number of channels and L is the length of the audio

        #first check if all tracks in x_dry have activity (RMS > -60 dBFS)
        rms_dry=compute_log_rms_gated_max(x_dry_segments, sample_rate=44100)  # Compute the log RMS of the dry audio
        silent_tracks = rms_dry < -60  # Identify silent tracks
        silent_tracks = silent_tracks.squeeze()  # Remove singleton dimensions

        if silent_tracks.any():
            print(f"Removing {silent_tracks.sum()} silent tracks from the input audio.")
            #shape before removing silent tracks is (N, C, L)
            x_dry = x_dry[~silent_tracks]  # Remove silent tracks
            x_dry_segments = x_dry_segments[~silent_tracks]  # Remove silent tracks

        preds = self.generate_Fx(x_dry_segments, num_samples)

        z_pred = self.embedding_post_processing(
            preds
        )  # post-process the generated features

        del self.sampler

        for i in range(num_samples):
            z_i = z_pred[
                i
            ]  # Randomly sample 100 features from the generated features

            y_final = self.apply_effects(
                    x_dry.clone(), z_i
            )  # Apply the effects to the input audio
            y_hat_mixture = y_final.sum(dim=0, keepdim=False)
    
            # peak normalization of y_hat_mixture
            peak = torch.max(torch.abs(y_hat_mixture))
            y_hat_mixture /= peak  # Normalize the audio to [-1, 1]
    
            filename="MEGAMI_inference"+f"_sample{i}.wav"
            os.makedirs(f"{directory}/{exp_name}", exist_ok=True)
            sf.write(
                f"{directory}/{exp_name}/{filename}",
                y_hat_mixture.cpu().clamp(-1, 1).numpy().T,
                44100,
                subtype="PCM_16",
            )


