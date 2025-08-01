import torch
import numpy as np
import torchcomp

from fx_model.processors.compexp import prepare_compexp_parameters, compexp_functional

from fx_model.processors.peq import prepare_PEQ_parameters, peq_functional, prepare_PEQ_FDN_parameters, peq_FDN_functional

from fx_model.processors.STFT_diffused_reverb import STFTMaskedNoiseReverb_filterbank, STFTMaskedNoiseReverb, STFTEQ
from utils.training_utils import Gauss_smooth, Gauss_smooth_vectorized, create_music_mean_spectrum_curve, prepare_smooth_filter

import utils.training_utils as utils
from utils.distributions import Normal, NormalRamp, Uniform, sample_from_distribution_dict, UniformRamp, LogUniform

def T602logmag(t60, sample_rate=44100, hop_length=512):

    return 6.908 / (t60 * (sample_rate / hop_length))  # Convert T60 to delta log magnitude

from utils.data_utils import taxonomy2track


class FxNormAug:

    def __init__(self,
                sample_rate=44100,  # Sample rate of the audio
                device="cuda" if torch.cuda.is_available() else "cpu",
                mode="train",  # Mode can be "train" or "eval"
                seed=42,
                features_path="/home/eloi/projects/project_mfm_eloi/src/fx_model/features/features_tency1_8instr_v4.npy",  # Path to the features file
                ):

        torch.random.manual_seed(seed)

        self.features_path = features_path

        self.sample_rate = sample_rate
        self.device = device
        if mode=="train":
            self.train_setup()
        elif mode=="inference":
            self.inference_setup()

        self.EQ_normalize_setup()  # Initialize EQ normalization function


    def inference_setup(self):
        #hardcoded setuo used for training

        self.distribution_SNR=Normal(mean=5, std=0, shape=(1,))  # SNR in dB
    
        self.distribution_RMS=Normal(mean=-25, std=0, shape=(1,))  # RMS in dB

        self.RMS_norm=-25  # Target RMS level in dB, used for normalization

        self.reverberator= STFTMaskedNoiseReverb(fixed_noise=False, ir_len=88000)

        self.distribution_init_log_magnitude_1=Normal(mean=0, std=0, shape=(self.reverberator.num_bins,))  # Range for log magnitude randomization
        self.distribution_delta_log_magnitude_1 = NormalRamp(mean_low=1.2, mean_high=0.2, std_low=0.01, std_high=0.01, log=True, shape=(self.reverberator.num_bins,), transformation=lambda x: T602logmag(x.clamp(min=0.001), sample_rate=self.sample_rate, hop_length=self.reverberator.hop_length))  # Range for delta log magnitude randomization

        self.distribution_drywet_ratio_1= Normal(mean=0.4, std=0, shape=(1,), transformation=lambda x: torch.clamp(x, min=0, max=1))  # Dry/wet ratio for reverb


        _, self.params_CompExp_non_optimizable, _ = prepare_compexp_parameters(self.sample_rate, device=self.device)

        self.distributions_CompExp_1 = {
            "comp_ratio": Normal(mean=5.0, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # compression ratio
            "exp_ratio": Normal(mean=0.2, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.01, max=0.999)),
            "at_coef": Normal(mean=15.0, std=0.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x,min=0.01), self.sample_rate)),  # attack time in milliseconds
            "rt_coef": Normal(mean=100.0, std=0.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), self.sample_rate)),  # release time in milliseconds
            "avg_coef": Normal(mean=0.95, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, 0.01, 0.99)),  # average coefficient
            "make_up": Normal(mean=3.0, std=0.0, shape=(1,)),  # make-up gain in dB
            "comp_th": Normal(mean=-25.0, std=0.0, shape=(1,)),  # compression threshold in dB
            "exp_th": Normal(mean=-31.0, std=0.0, shape=(1,)),  # expansion threshold in dB
            "lookahead": Uniform(low=0 ,high=0, shape=(1,)),  # lookahead in milliseconds
        }


        self.distributions_CompExp_2 = {
            "comp_ratio": Normal(mean=5.0, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # compression ratio
            "exp_ratio": Normal(mean=0.8, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.01, max=0.999)),
            "at_coef": Normal(mean=15.0, std=0.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x,min=0.01), self.sample_rate)),  # attack time in milliseconds
            "rt_coef": Normal(mean=100.0, std=0.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), self.sample_rate)),  # release time in milliseconds
            "avg_coef": Normal(mean=0.95, std=0., shape=(1,), transformation=lambda x: torch.clamp(x, 0.01, 0.99)),  # average coefficient
            "make_up": Normal(mean=3.0, std=0.0, shape=(1,)),  # make-up gain in dB
            "comp_th": Normal(mean=-32.0, std=0.0, shape=(1,)),  # compression threshold in dB
            "exp_th": Normal(mean=-45.0, std=0.0, shape=(1,)),  # expansion threshold in dB
            "lookahead": Uniform(low=0 ,high=10, shape=(1,)),  # lookahead in milliseconds
        }

        self.EQ_std=0
    def train_setup(self):
        #hardcoded setuo used for training

        self.distribution_SNR=Normal(mean=5, std=2, shape=(1,))  # SNR in dB
    
        self.distribution_RMS=Normal(mean=-25, std=3, shape=(1,))  # RMS in dB

        self.RMS_norm=-25  # Target RMS level in dB, used for normalization

        self.reverberator= STFTMaskedNoiseReverb(fixed_noise=False, ir_len=88000)

        self.distribution_init_log_magnitude_1=Normal(mean=0, std=2, shape=(self.reverberator.num_bins,))  # Range for log magnitude randomization
        self.distribution_delta_log_magnitude_1 = NormalRamp(mean_low=1.2, mean_high=0.1, std_low=0.3, std_high=0.05, log=True, shape=(self.reverberator.num_bins,), transformation=lambda x: T602logmag(x.clamp(min=0.001), sample_rate=self.sample_rate, hop_length=self.reverberator.hop_length))  # Range for delta log magnitude randomization

        self.distribution_drywet_ratio_1= Normal(mean=0.4, std=0.15, shape=(1,), transformation=lambda x: torch.clamp(x, min=0, max=1))  # Dry/wet ratio for reverb


        _, self.params_CompExp_non_optimizable, _ = prepare_compexp_parameters(self.sample_rate, device=self.device)

        self.distributions_CompExp_1 = {
            "comp_ratio": Normal(mean=5.0, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # compression ratio
            "exp_ratio": Normal(mean=0.2, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.01, max=0.999)),
            "at_coef": Normal(mean=15.0, std=1.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x,min=0.01), self.sample_rate)),  # attack time in milliseconds
            "rt_coef": Normal(mean=100.0, std=10.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), self.sample_rate)),  # release time in milliseconds
            "avg_coef": Normal(mean=0.95, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, 0.01, 0.99)),  # average coefficient
            "make_up": Normal(mean=3.0, std=0.0, shape=(1,)),  # make-up gain in dB
            "comp_th": Normal(mean=-25.0, std=0.0, shape=(1,)),  # compression threshold in dB
            "exp_th": Normal(mean=-31.0, std=0.0, shape=(1,)),  # expansion threshold in dB
            "lookahead": Uniform(low=0 ,high=0, shape=(1,)),  # lookahead in milliseconds
        }


        self.distributions_CompExp_2 = {
            "comp_ratio": Normal(mean=5.0, std=2.0, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # compression ratio
            "exp_ratio": Normal(mean=0.8, std=0.3, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.01, max=0.999)),
            "at_coef": Normal(mean=15.0, std=3.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x,min=0.01), self.sample_rate)),  # attack time in milliseconds
            "rt_coef": Normal(mean=100.0, std=20.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), self.sample_rate)),  # release time in milliseconds
            "avg_coef": Normal(mean=0.95, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, 0.01, 0.99)),  # average coefficient
            "make_up": Normal(mean=3.0, std=1.0, shape=(1,)),  # make-up gain in dB
            "comp_th": Normal(mean=-32.0, std=5.0, shape=(1,)),  # compression threshold in dB
            "exp_th": Normal(mean=-45.0, std=5.0, shape=(1,)),  # expansion threshold in dB
            "lookahead": Uniform(low=0 ,high=10, shape=(1,)),  # lookahead in milliseconds
        }

        self.EQ_std=0.05
    
    def inference_setup_unused(self):
        #at inference time , parameters are fixed to a specific value
        self.distribution_SNR=Normal(mean=30, std=0, shape=(1,))  # SNR in dB
    
        self.distribution_RMS=Normal(mean=-25, std=0, shape=(1,))  # RMS in dB

        self.RMS_norm=-25  # Target RMS level in dB, used for normalization

        self.reverberator= STFTMaskedNoiseReverb_filterbank(fixed_noise=False, ir_len=88000)
        self.distribution_init_log_magnitude=Uniform(low=0, high=0, shape=(1,self.reverberator.filterbank.num_filters))  # Range for log magnitude randomization

        #self.distribution_delta_log_magnitude = UniformRamp(low_low=1.5, low_high=0.5, high_low=1.5, high_high=0.5, log=False, shape=(self.reverberator.filterbank.num_filters,), transformation=lambda x: T602logmag(x, sample_rate=self.sample_rate, hop_length=self.reverberator.hop_length))  # Range for delta log magnitude randomization
        self.distribution_delta_log_magnitude = NormalRamp(mean_low=1.5, mean_high=0.2, std_low=0.0, std_high=0.0, log=False, shape=(self.reverberator.filterbank.num_filters,), transformation=lambda x: T602logmag(x.clamp(min=0.01), sample_rate=self.sample_rate, hop_length=self.reverberator.hop_length))  # Range for delta log magnitude randomization

        self.distribution_drywet_ratio= Normal(mean=0.1, std=0.00, shape=(1,), transformation=lambda x: torch.clamp(x, min=0, max=1))  # Dry/wet ratio for reverb

        _, self.params_CompExp_non_optimizable, _ = prepare_compexp_parameters(self.sample_rate, device=self.device)

        self.distributions_CompExp = {
            "comp_ratio": Normal(mean=4.0, std=0, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # compression ratio
            "exp_ratio": Normal(mean=0.25, std=0, shape=(1,), transformation=lambda x: torch.clamp(x, min=0, max=0.999)),
            "at_coef": Normal(mean=35.0, std=0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x,min=0.01), self.sample_rate)),  # attack time in milliseconds
            "rt_coef": Normal(mean=120.0, std=0.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), self.sample_rate)),  # release time in milliseconds
            "avg_coef": Normal(mean=0.95, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, 0.01, 0.99)),  # average coefficient
            "make_up": Normal(mean=3.0, std=0.0, shape=(1,)),  # make-up gain in dB
            "comp_th": Normal(mean=-25.0, std=0.0, shape=(1,)),  # compression threshold in dB
            "exp_th": Normal(mean=-31.0, std=0.0, shape=(1,)),  # expansion threshold in dB
            "lookahead": Uniform(low=0 ,high=0, shape=(1,)),  # lookahead in milliseconds
        }


    def apply_RMS_normalization(self, x, random=True):

        if random:
            RMS=self.distribution_RMS.sample(x.shape[0]).view(-1, 1, 1).to(x.device)  # Ensure RMS is broadcastable
        else:
            RMS= torch.tensor(self.RMS_norm, device=x.device).view(1, 1, 1).repeat(x.shape[0],1,1)  # Use fixed RMS for evaluation


        x_RMS=20*torch.log10(torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True).mean(dim=-2, keepdim=True)))

        gain= RMS - x_RMS
        gain_linear = 10 ** (gain / 20 + 1e-6)  # Convert dB gain to linear scale, adding a small value to avoid division by zero
        x=x* gain_linear.view(-1, 1, 1)

        return x
    
    def add_noise(self,x):
        SNR=self.distribution_SNR.sample(x.shape[0]).to(x.device)
        x= utils.add_pink_noise(x, SNR)
        return x
        
    def apply_reverb_1(self,x):
        log_magnitude = self.distribution_init_log_magnitude_1.sample(x.shape[0]).to(x.device).unsqueeze(1)  # Sample log magnitude for each filterbank channel
        delta_log_magnitude = self.distribution_delta_log_magnitude_1.sample(x.shape[0]).to(x.device).unsqueeze(1)
        drywet_ratio = self.distribution_drywet_ratio_1.sample(x.shape[0]).to(x.device)

        self.reverberator.to(x.device)  # Ensure reverberator is on the same device as x
        assert not torch.isnan(log_magnitude).any(), "NaN detected in log_magnitude after sampling"
        assert not torch.isnan(delta_log_magnitude).any(), "NaN detected in delta_log_magnitude after sampling"
        assert not torch.isnan(drywet_ratio).any(), "NaN detected in drywet_ratio after sampling"

        y=self.reverberator(x, log_magnitude, delta_log_magnitude)
        assert not torch.isnan(y).any(), "NaN detected in y after reverb 1 self.reverberator"

        return  x*(drywet_ratio).view(-1, 1, 1) + y *(1- drywet_ratio.view(-1, 1, 1))
    
    def apply_EQ(self, x):

        params_PEQ = sample_from_distribution_dict(self.distributions_PEQ, x.shape[0], device=self.device)
        #print("Time to sample PEQ parameters:", time.time() - a)
        y= peq_functional(x, **params_PEQ, **self.params_PEQ_non_optimizable)

        return y

    def apply_compexp_2(self, x):

        params_CompExp = sample_from_distribution_dict(self.distributions_CompExp_2, x.shape[0], device=self.device)
        x= compexp_functional(x, **params_CompExp, **self.params_CompExp_non_optimizable)

        return x

    def apply_compexp_1(self, x):

        params_CompExp = sample_from_distribution_dict(self.distributions_CompExp_1, x.shape[0], device=self.device)
        x= compexp_functional(x, **params_CompExp, **self.params_CompExp_non_optimizable)

        return x

    def EQ_normalize_setup(self ):

        #FEATURES_LOAD="fx_model/features/features_tency1_8instr_v4.npy"
        #FEATURES_LOAD="fx_model/features/features_tency1_10instr.npy"

        #features_mean = np.load(FEATURES_LOAD, allow_pickle='TRUE')[()]
        features_mean = np.load(self.features_path, allow_pickle='TRUE')[()]

        target_cuves_original= {
            "vocals": torch.tensor(features_mean["eq"]["vocals"]).to(torch.float32).to(self.device),
            "drums": torch.tensor(features_mean["eq"]["drums"]).to(torch.float32).to(self.device),
            "bass": torch.tensor(features_mean["eq"]["bass"]).to(torch.float32).to(self.device),
            "other": torch.tensor(features_mean["eq"]["other"]).to(torch.float32).to(self.device),
            "piano": torch.tensor(features_mean["eq"]["piano"]).to(torch.float32).to(self.device),
            "guitar": torch.tensor(features_mean["eq"]["guitar"]).to(torch.float32).to(self.device),  
            "strings": torch.tensor(features_mean["eq"]["strings"]).to(torch.float32).to(self.device),
            "brass": torch.tensor(features_mean["eq"]["brass"]).to(torch.float32).to(self.device),
        }


        nfft=4096 # FFT size hardcoded
        nfft_orig = 65536  # FFT size for the smoothing filter

        win_length=2048  # Window length hardcoded
        hop_length=1024  # Hop length hardcoded

        window = torch.sqrt(torch.hann_window(win_length, device=self.device))
        window_energy = window.pow(2).sum().sqrt()  # Energy of the window

        freqs = torch.fft.rfftfreq(nfft, d=1.0).to(self.device)
        freqs_Hz=torch.fft.rfftfreq(nfft, d=1.0).to(self.device) * self.sample_rate

        smooth_filter = prepare_smooth_filter(freqs_Hz, Noct=3).to(self.device)  # Prepare the smoothing filter

        freqs_Hz_orig=torch.fft.rfftfreq(nfft_orig, d=1.0).to(self.device) * self.sample_rate
        smooth_filter_orig = prepare_smooth_filter(freqs_Hz_orig, Noct=3).to(self.device)  # Prepare the smoothing filter

        def downsample_curve(x):
            return torch.nn.functional.interpolate(
                x.unsqueeze(0).unsqueeze(0),
                size=(nfft // 2 + 1,),
                mode='linear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        target_curves = {
            "vocals": downsample_curve(Gauss_smooth_vectorized(target_cuves_original["vocals"], freqs_Hz_orig, Noct=3, smooth_filter=smooth_filter_orig)),
            "drums": downsample_curve(Gauss_smooth_vectorized(target_cuves_original["drums"], freqs_Hz_orig, Noct=3, smooth_filter=smooth_filter_orig)),
            "bass": downsample_curve(Gauss_smooth_vectorized(target_cuves_original["bass"], freqs_Hz_orig, Noct=3, smooth_filter=smooth_filter_orig)),
            "other": downsample_curve(Gauss_smooth_vectorized(target_cuves_original["other"], freqs_Hz_orig, Noct=3, smooth_filter=smooth_filter_orig)),
            "piano": downsample_curve(Gauss_smooth_vectorized(target_cuves_original["piano"], freqs_Hz_orig, Noct=3, smooth_filter=smooth_filter_orig)),
            "guitar": downsample_curve(Gauss_smooth_vectorized(target_cuves_original["guitar"], freqs_Hz_orig, Noct=3, smooth_filter=smooth_filter_orig)), 
            "strings": downsample_curve(Gauss_smooth_vectorized(target_cuves_original["strings"], freqs_Hz_orig, Noct=3 , smooth_filter=smooth_filter_orig)),
            "brass": downsample_curve(Gauss_smooth_vectorized(target_cuves_original["brass"], freqs_Hz_orig, Noct=3, smooth_filter=smooth_filter_orig)),
        }

        #target_curve = create_music_mean_spectrum_curve(freqs_Hz, device=self.device)

        def EQ_normalize_fn(x, taxonomy, random=False):
            shape= x.shape
            assert len(taxonomy) == shape[0], "taxonomy length must match batch size"

            target_curves_tensor=torch.zeros((len(taxonomy), nfft // 2 + 1), device=self.device, dtype=torch.float32)
            for i in range(len(taxonomy)):
                track_class = taxonomy2track(taxonomy[i], num_instr=8)
                assert track_class in target_curves, f"track_class {track_class} not found in target_curves"
                target_curves_tensor[i] = target_curves[track_class] 
                #target_curves[i] = target_curves[track_class] if track_class in target_curves else target_curves["other"]

            x=x.view(-1, shape[-1])
            X=torch.stft(x, n_fft=nfft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)/ window_energy
            X_pow=X.abs().pow(2)
            X_mean= torch.sqrt(X_pow.mean(dim=-1, keepdim=False))  # Mean power spectrum


            ratio= target_curves_tensor  / (X_mean + 1e-6)
            if random:
                ratio = ratio + torch.randn_like(ratio) * self.EQ_std  # Add some noise to the ratio for randomization

            ratio = torch.clamp(ratio, max=10.0**(20.0/20.0))

            ratio_smooth = Gauss_smooth_vectorized(ratio, freqs_Hz, Noct=3, smooth_filter=smooth_filter)

            X= X * ratio_smooth.unsqueeze(-1)

            X_unnormalized=X* window_energy

            x_reconstructed = torch.istft(X_unnormalized, 
                             n_fft=nfft, 
                             hop_length=hop_length, 
                             win_length=win_length, 
                             window=window,
                             return_complex=False)  # Set to True if you want complex outpu

            x_reconstructed = x_reconstructed.view(shape)
            return x_reconstructed
        
        self.EQ_normalize = EQ_normalize_fn


    def forward(self, x, taxonomy):


        #first stereo to mono (if needed)
        B, C, T = x.shape
        if C > 1:
            x = x.mean(dim=1, keepdim=True)



        #then apply random compressor
        x= self.apply_RMS_normalization(x, random=False)
        x = self.apply_compexp_1(x)
        assert not torch.isnan(x).any(), "NaN detected in x after compression"

        x= self.apply_RMS_normalization(x, random=False)
        x=self.EQ_normalize(x, taxonomy)
        assert not torch.isnan(x).any(), "NaN detected in x after EQ normalization"

        x=self.apply_reverb_1(x)

        assert not torch.isnan(x).any(), "NaN detected in x after reverb 1"
        #stereo to mono (if needed)
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        x= self.apply_RMS_normalization(x, random=True)
        assert not torch.isnan(x).any(), "NaN detected in x after RMS normalization"

        x= self.apply_compexp_2(x)

        assert not torch.isnan(x).any(), "NaN detected in x after compression"

        x=self.apply_reverb_1(x)

        assert not torch.isnan(x).any(), "NaN detected in x after reverb 2"
        #stereo to mono (if needed)
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        #x= self.apply_RMS_normalization(x, random=True)

        #assert not torch.isnan(x).any(), "NaN detected in x after RMS normalization"

        #x=self.EQ_normalize(x, taxonomy, random=True)

        #assert not torch.isnan(x).any(), "NaN detected in x after EQ normalization"

        #if x.shape[1]==2:
        #    x=x.mean(dim=1, keepdim=True)

        #then apply random EQ
        #x= self.apply_EQ(x)


        #then add noise

        x = self.add_noise(x)

        #then apply random gain

        x = self.apply_RMS_normalization(x, random=True)


        return x