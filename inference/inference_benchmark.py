# load generator model
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import pyloudnorm as pyln

import hydra
import torch
from hydra import initialize, compose
import utils.training_utils as tr_utils

import torch
from datasets.eval_benchmark import Eval_Benchmark
import omegaconf
import math

import soundfile as sf
from utils.data_utils import apply_RMS_normalization

import pandas as pd
from tqdm import tqdm
from utils.evaluation.dist_metrics import KADFeatures
from utils.data_utils import loudness_normalize


class InferenceBenchmark:
    def __init__(
        self,
        method_args=None,
        dataset_code=None,
        extra_id=None,
        path_results="/data5/eloi/results",
        num_tracks_to_load=1,
        KAD_features=["AFxRep", "FxEncoder", "FxEncoder++", "CLAP"],
        path_benchmark="/scratch/elec/t412-asp/automix/MDX_TM_benchmark",
    ):

        self.method_args = method_args
        self.path_benchmark = path_benchmark

        self.results_df = pd.DataFrame(columns=["track", "method", "metric", "value"])

        self.FxGenerator_code = method_args.FxGenerator_code
        self.FxProcessor_code = method_args.FxProcessor_code
        self.dataset_code = dataset_code

        self.experiment_name = (
            f"{self.FxGenerator_code}_{self.FxProcessor_code}_{self.dataset_code}"
        )
        if extra_id is not None:
            self.experiment_name = f"{self.FxGenerator_code}_{self.FxProcessor_code}_{self.dataset_code}_{extra_id}"

        self.path_results = path_results
        self.path_results = f"{self.path_results}/{self.experiment_name}"
        os.makedirs(self.path_results, exist_ok=True)

        self.csv_path = f"{self.path_results}/results.csv"

        self.config_file_rel = "../conf"
        # self.config_path="/home/eloi/projects/project_mfm_eloi/src/conf"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_FxGenerator()  # Load the S1 model
        self.load_FxProcessor()  # Load the S2 model
        self.prepare_feature_extractors()  # Prepare the feature extractors

        self.load_dataset(num_tracks_to_load=num_tracks_to_load)  # Load the dataset

        # self.anchors=["equal_loudness", "only_rms"]
        self.anchors = ["only_rms"]

        self.anchor_fns = {}
        for anchor in self.anchors:
            if anchor == "equal_loudness":
                self.anchor_fns[anchor] = self.prepare_equal_loudness_anchor(
                    target_lufs_dB=-48.0
                )
            elif anchor == "only_rms":
                self.anchor_fns[anchor] = self.prepare_only_rms_anchor()

        self.KAD_metrics = self.prepare_KAD_feature_extractors(KAD_features)

    def prepare_KAD_feature_extractors(self, KAD_features):

        KAD_metrics = {}

        KAD_args = {
            "do_PCA_figure": False,  # if True, the FAD figure will be computed
            "do_TSNE_figure": False,  # if True, the FAD figure will be computed
            "kernel": "gaussian",  # kernel to use for the KAD metric
            "PCA_fit_mode": "all",
        }
        KAD_args = omegaconf.OmegaConf.create(KAD_args)

        for feature in KAD_features:
            if feature == "AFxRep":
                AFxRep_args = {
                    "distance_type": "cosine",  # not used
                    "ckpt_path": "utils/st_ito/afx-rep.ckpt",
                }
                AFxRep_args = omegaconf.OmegaConf.create(AFxRep_args)
                KAD_metrics[feature] = KADFeatures(
                    type="AFxRep",
                    sample_rate=44100,
                    AFxRep_args=AFxRep_args,
                    KAD_args=KAD_args,
                )
            elif feature == "FxEncoder":
                fx_encoder_args = {
                    "distance_type": "cosine",  # not used
                    "ckpt_path": "checkpoints/fxenc_default.pt",
                }
                fx_encoder_args = omegaconf.OmegaConf.create(fx_encoder_args)
                KAD_metrics[feature] = KADFeatures(
                    type="fx_encoder",
                    sample_rate=44100,
                    fx_encoder_args=fx_encoder_args,
                    KAD_args=KAD_args,
                )
            elif feature == "FxEncoder++":
                fx_encoder_plusplus_args = {
                    "distance_type": "cosine",  # not used
                    "ckpt_path": "checkpoints/fxenc_plusplus_default.pt",
                }
                fx_encoder_plusplus_args = omegaconf.OmegaConf.create(
                    fx_encoder_plusplus_args
                )
                KAD_metrics[feature] = KADFeatures(
                    type="fx_encoder_++",
                    sample_rate=44100,
                    fx_encoder_plusplus_args=fx_encoder_plusplus_args,
                    KAD_args=KAD_args,
                )

            elif feature == "CLAP":
                clap_args = {
                    "ckpt_path": "checkpoints/music_audioset_epoch_15_esc_90.14.patched.pt",
                    "distance_type": "cosine",
                    "normalize": True,  # if True, the features will be normalized
                    "use_adaptor": False,  # if True, the features will be adapted to the CLAP space
                    "adaptor_checkpoint": None,
                    "adaptor_type": None,
                    "add_noise": False,  #   if True, the features will be augmented with orthogonal noise
                    "noise_sigma": 0,  # sigma of the orthogonal noise to
                }

                clap_args = omegaconf.OmegaConf.create(clap_args)
                KAD_metrics[feature] = KADFeatures(
                    type="CLAP",
                    sample_rate=44100,
                    CLAP_args=clap_args,
                    KAD_args=KAD_args,
                )

            elif feature == "bark":
                bark_args = {
                    "distance_type": "cosine",  # not used
                    "normalize": True,  # if True, the features will be normalized
                }
                bark_args = omegaconf.OmegaConf.create(bark_args)
                KAD_metrics[feature] = KADFeatures(
                    type="bark",
                    sample_rate=44100,
                    bark_args=bark_args,
                    KAD_args=KAD_args,
                    normalize=True,
                )
            elif feature == "spectral":
                spectral_args = {
                    "distance_type": "cosine",  # not used
                    "normalize": True,  # if True, the features will be normalized
                }
                spectral_args = omegaconf.OmegaConf.create(spectral_args)
                KAD_metrics[feature] = KADFeatures(
                    type="spectral",
                    sample_rate=44100,
                    spectral_args=spectral_args,
                    KAD_args=KAD_args,
                    normalize=True,
                )
            elif feature == "panning":
                panning_args = {
                    "distance_type": "cosine",  # not used
                    "normalize": True,  # if True, the features will be normalized
                }
                panning_args = omegaconf.OmegaConf.create(panning_args)
                KAD_metrics[feature] = KADFeatures(
                    type="panning",
                    sample_rate=44100,
                    panning_args=panning_args,
                    KAD_args=KAD_args,
                    normalize=True,
                )
            elif feature == "dynamic":
                dynamic_args = {
                    "distance_type": "cosine",  # not used
                    "normalize": True,  # if True, the features will be normalized
                }
                dynamic_args = omegaconf.OmegaConf.create(dynamic_args)
                KAD_metrics[feature] = KADFeatures(
                    type="dynamic",
                    sample_rate=44100,
                    dynamic_args=dynamic_args,
                    KAD_args=KAD_args,
                    normalize=True,
                )

            else:
                raise ValueError(f"Unknown feature: {feature}")

        return KAD_metrics

    def load_FxGenerator(self):
        if self.FxGenerator_code == "internal_TencyDB":
            config_name = "conf_FxGenerator_TencyDB.yaml"
            model_dir = "checkpoints"
            ckpt = "FxGenerator_TencyDB-200000.pt"
        elif self.FxGenerator_code == "internal_TencyMastering":
            config_name = "conf_FxGenerator_TencyMastering.yaml"
            model_dir = "checkpoints"
            ckpt = "FxGenerator_TencyMastering-50000.pt"
        elif self.FxGenerator_code == "internal_TencyMastering_paired":
            config_name = "conf_FxGenerator_TencyMastering_paired.yaml"
            model_dir = "checkpoints"
            ckpt = "FxGenerator_TencyMastering_paired-50000.pt"
        elif self.FxGenerator_code == "public":
            config_name = "conf_FxGenerator_public.yaml"
            model_dir = "checkpoints"
            # ckpt="styleDiT_multitrack_public-170000.pt"
            ckpt = "FxGenerator_public-50000.pt"
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

        if self.FxProcessor_code == "internal_TencyMastering":
            config_name = "conf_FxProcessor_TencyMastering.yaml"
            model_dir = "checkpoints"
            ckpt = "FxProcessor_internal_blackbox_TCN-270000.pt"
        elif self.FxProcessor_code == "public":
            config_name = "conf_FxProcessor_Public.yaml"
            model_dir = "checkpoints"
            ckpt = "FxProcessor_public_blackbox_TCN_340000.pt"
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

    def load_dataset(self, num_tracks_to_load=1):
        ### Loading dataset ###

        if self.dataset_code == "MDX_TM_benchmark":
            self.dataset = Eval_Benchmark(
                mode="dry-mixture",
                segment_length=525312,
                fs=44100,
                num_tracks=-1,
                num_examples=-1,
                num_segments_per_track=-1,
                path=self.path_benchmark,
            )
        else:
            raise ValueError(f"Unknown dataset_code: {self.dataset_code}")

    def prepare_equal_loudness_anchor(self, target_lufs_dB=-48.0):
        """
        Prepare the equal loudness anchor function.
        This function will be used to normalize the audio to a reference loudness level.
        """
        meter = pyln.Meter(44100)  # create a meter for the reference loudness level

        def equal_loudness_anchor(x, *args, **kwargs):
            """
            Normalize all tracks to have equal loudness.
            x: tensor of shape [N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
            """
            N, C, L = (
                x.shape
            )  # N is the number of tracks, C is the number of channels and L is the length of the audio

            norm_tracks = []

            for track_idx in range(x.shape[0]):
                track = x[track_idx].unsqueeze(
                    0
                )  # Get the track tensor of shape [1, C, L]
                lufs_dB = meter.integrated_loudness(
                    track.squeeze(0).permute(1, 0).cpu().numpy()
                )

                if lufs_dB < -80.0:
                    print(f"Skipping track {track_idx} with {lufs_dB:.2f} LUFS.")
                    continue

                lufs_delta_db = target_lufs_dB - lufs_dB
                track *= 10 ** (lufs_delta_db / 20)
                norm_tracks.append(track)  # each track is of shape [1, C, L]

            norm_tracks = torch.cat(norm_tracks, dim=0)  # shape [N, C, L]
            # create a sum mix with equal loudness
            sum_mix = torch.sum(norm_tracks, dim=0, keepdim=False)

            # peak normalization
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
            B = 1

            # x_norm= x.mean(dim=1, keepdim=True)  # Stereo to mono
            x = apply_RMS_normalization(x, -25.0, device=self.device)

            y_final = self.apply_rms(
                x, z
            )  # Apply RMS normalization to the generated audio

            y_final = y_final.sum(
                dim=0, keepdim=False
            )  # Sum the tracks to get a single output

            return y_final

        return only_rms_anchor

    def prepare_feature_extractors(self):

        ### preparing feature extractor ###

        Fxencoder_kwargs = omegaconf.OmegaConf.create(
            {
                "ckpt_path": "/scratch/work/molinee2/projects/project_mfm_eloi/src_clean_internal/checkpoints/fxenc_plusplus_default.pt"
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

        from utils.feature_extractors.dsp_features import compute_log_rms_gated_max

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

            x_norm = x.mean(
                dim=1, keepdim=True
            )  # Normalize the input audio by its mean across the tracks

            # x_norm=self.fx_normalizer(x_norm)  # Apply the fx_normalizer if specified
            if "public" in self.FxProcessor_code:
                x_norm = self.fx_normalizer(
                    x_norm
                )  # Apply the fx_normalizer if specified
            else:
                x_norm = apply_RMS_normalization(
                    x_norm, -25.0, device=self.device, use_gate=True
                )  # Apply RMS normalization with gating
                x_norm = self.fx_normalizer(
                    x_norm, use_gate=True
                )  # Apply the fx_normalizer if specified

            with torch.no_grad():
                y_hat = self.fx_model(x_norm, z_pred)

            y_final = apply_rms(y_hat, z_pred)

            return y_final

        self.apply_effects = apply_effects

    def run_evaluation_oracle(self, exp_name="oracle_test_Oct25"):

        ### Run evaluation on the validation set ###
        for i in range(len(self.dataset)):

            datav = self.dataset[i]

            x_dry, y_wet, mixture, track_id, segment_id, path_segment = datav

            print(f"Processing track {i} with path {path_segment}")

            x_dry = x_dry.to(self.device)
            mixture = mixture.to(self.device)

            y_wet = y_wet.to(self.device)
            z_ref = self.FxEnc(
                y_wet
            )  # z_y is a tensor of shape [B, N, D] where D is the dimension of the features (2048 + 2048 = 4096)
            y_final = self.apply_effects(
                x_dry, z_ref
            )  # Apply the effects to the input audio
            y_hat_mixture = y_final.sum(dim=0, keepdim=False)

            # peak normalization of y_hat_mixture
            peak = torch.max(torch.abs(y_hat_mixture))
            y_hat_mixture /= peak  # Normalize the audio to [-1, 1]

            filename = path_segment.replace("/", "_").replace(" ", "_") + ".wav"
            os.makedirs(f"{self.path_results}/{exp_name}", exist_ok=True)
            sf.write(
                f"{self.path_results}/{exp_name}/{filename}",
                y_hat_mixture.cpu().clamp(-1, 1).numpy().T,
                44100,
                subtype="PCM_16",
            )

    def run_evaluation_equal_loudness(self, exp_name="equal_loudness_test_Sep28"):
        """

        Run the evaluation on the validation set

        """

        ### Run evaluation on the validation set ###
        for i in range(len(self.dataset)):

            datav = self.dataset[i]

            x_dry, y_wet, mixture, track_id, segment_id, path_segment = datav

            print(f"Processing track {i} with path {path_segment}")

            x_dry = x_dry.to(self.device)
            mixture = mixture.to(self.device)

            y_equal_loudness = self.anchor_fns["equal_loudness"](
                x_dry
            )  # Apply the equal loudness anchor

            # peak normalization of y_equal_loudness
            peak = torch.max(torch.abs(y_equal_loudness))
            y_equal_loudness /= peak  # Normalize the audio to [-1, 1]

            filename = path_segment.replace("/", "_").replace(" ", "_") + ".wav"
            os.makedirs(f"{self.path_results}/{exp_name}", exist_ok=True)
            sf.write(
                f"{self.path_results}/{exp_name}/{filename}",
                y_equal_loudness.cpu().clamp(-1, 1).numpy().T,
                44100,
                subtype="PCM_16",
            )

    def run_evaluation(self, exp_name="test_Sep28", num_samples=1, compute_KAD=True):
        """

        Run the evaluation on the validation set

        """

        if compute_KAD:
            reference_dict = {}
            method_dict = {}

        ### Run evaluation on the validation set ###
        # for i in tqdm(range(10)):
        for i in tqdm(range(len(self.dataset))):

            x_dry, y_wet, mixture, track_id, segment_id, path_segment = self.dataset[i]

            if compute_KAD:
                id = f"{track_id}_seg{segment_id}"
                reference_dict[id] = loudness_normalize(mixture.cpu())

            print(f"Processing track {i} with path {path_segment}")

            x_dry = x_dry.to(self.device)
            mixture = mixture.to(self.device)

            preds = self.generate_Fx(x_dry, num_samples)
            z_pred = self.embedding_post_processing(
                preds
            )  # post-process the generated features

            if num_samples > 1:
                # take a random sample from the generated features
                # more intelligent approaches, such as taking the centroid, or the sample with higher likelihood could be implemented here
                index = torch.randint(
                    0, z_pred.shape[0], (1,)
                ).item()  # Randomly sample an index from the generated features
            else:
                index = 0

            z_i = z_pred[
                index
            ]  # Randomly sample 100 features from the generated features
            y_final = self.apply_effects(
                x_dry.clone(), z_i
            )  # Apply the effects to the input audio
            y_hat_mixture = y_final.sum(dim=0, keepdim=False)

            # peak normalization of y_hat_mixture
            peak = torch.max(torch.abs(y_hat_mixture))
            y_hat_mixture /= peak  # Normalize the audio to [-1, 1]

            if compute_KAD:
                method_dict[id] = loudness_normalize(y_hat_mixture.cpu())

            filename = path_segment.replace("/", "_").replace(" ", "_") + ".wav"
            os.makedirs(f"{self.path_results}/{exp_name}", exist_ok=True)
            sf.write(
                f"{self.path_results}/{exp_name}/{filename}",
                y_hat_mixture.cpu().clamp(-1, 1).numpy().T,
                44100,
                subtype="PCM_16",
            )

        if compute_KAD:
            print("Computing KAD metrics...")
            for feature, metric in self.KAD_metrics.items():
                KAD_distance, dict_output = metric.compute(
                    reference_dict, method_dict, None
                )
                print(
                    f"KAD distance for experiment {exp_name} and feature {feature}: {KAD_distance}"
                )


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # method_args= {
    #    "FxGenerator_code": "internal_TencyDB",
    #    "FxProcessor_code": "internal_TencyMastering",
    #    "T": 30,
    #    "Schurn": 0,
    #    "cfg_scale": 1.0,
    # }

    # method_args=omegaconf.OmegaConf.create(method_args)
    # evaluator=InferenceBenchmark( method_args=method_args, dataset_code="MDX_TM_benchmark", extra_id="internal_test_Oct25", path_results="/scratch/elec/t412-asp/automix/results")
    # evaluator.run_evaluation()

    method_args = {
        "FxGenerator_code": "public",
        "FxProcessor_code": "public",
        "T": 30,
        "Schurn": 0,
        "cfg_scale": 1.0,
    }

    method_args = omegaconf.OmegaConf.create(method_args)
    evaluator = InferenceBenchmark(
        method_args=method_args,
        dataset_code="MDX_TM_benchmark",
        extra_id="public_test_Oct25",
        path_results="/scratch/elec/t412-asp/automix/results",
    )
    evaluator.run_evaluation()
