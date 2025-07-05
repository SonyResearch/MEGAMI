import torch
import torchcomp

from fx_model.processors.compexp import prepare_compexp_parameters, compexp_functional

from fx_model.processors.STFT_diffused_reverb import STFTMaskedNoiseReverb_filterbank
from utils.training_utils import Gauss_smooth, Gauss_smooth_vectorized, create_music_mean_spectrum_curve, prepare_smooth_filter

import utils.training_utils as utils
from utils.distributions import Normal, NormalRamp, Uniform, sample_from_distribution_dict, UniformRamp

def T602logmag(t60, sample_rate=44100, hop_length=512):

    return 6.908 / (t60 * (sample_rate / hop_length))  # Convert T60 to delta log magnitude


class FxNormAug:

    def __init__(self,
                sample_rate=44100,  # Sample rate of the audio
                device="cuda" if torch.cuda.is_available() else "cpu",
                mode="train",  # Mode can be "train" or "eval"
                seed=42
                ):

        torch.random.manual_seed(seed)

        self.sample_rate = sample_rate
        self.device = device
        if mode=="train":
            self.train_setup()
        elif mode=="inference":
            self.inference_setup()

        self.EQ_normalize_setup()  # Initialize EQ normalization function

    def inference_setup(self):
        #hardcoded setuo used for training

        self.distribution_SNR=Normal(mean=5, std=2, shape=(1,))  # SNR in dB
    
        self.distribution_RMS=Normal(mean=-25, std=3, shape=(1,))  # RMS in dB

        self.RMS_norm=-25  # Target RMS level in dB, used for normalization


        self.reverberator= STFTMaskedNoiseReverb_filterbank(fixed_noise=False, ir_len=88000)
        self.distribution_init_log_magnitude=Normal(mean=0, std=2, shape=(self.reverberator.filterbank.num_filters,))  # Range for log magnitude randomization

        #min_T60=0.25  # Minimum reverberation time in seconds
        #max_T60=1.25  # Maximum reverberation time in seconds
        self.distribution_delta_log_magnitude = NormalRamp(mean_low=1.5, mean_high=0.2, std_low=0.5, std_high=0.1, log=False, shape=(self.reverberator.filterbank.num_filters,), transformation=lambda x: T602logmag(x.clamp(min=0.01), sample_rate=self.sample_rate, hop_length=self.reverberator.hop_length))  # Range for delta log magnitude randomization

        self.distribution_drywet_ratio= Normal(mean=0.1, std=0.02, shape=(1,), transformation=lambda x: torch.clamp(x, min=0, max=1))  # Dry/wet ratio for reverb

        _, self.params_CompExp_non_optimizable, _ = prepare_compexp_parameters(self.sample_rate, device=self.device)

        self.distributions_CompExp = {
            "comp_ratio": Normal(mean=4.0, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # compression ratio
            "exp_ratio": Normal(mean=0.25, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.01, max=0.999)),
            "at_coef": Normal(mean=35.0, std=0.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x,min=0.01), self.sample_rate)),  # attack time in milliseconds
            "rt_coef": Normal(mean=120.0, std=0.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), self.sample_rate)),  # release time in milliseconds
            "avg_coef": Normal(mean=0.95, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, 0.01, 0.99)),  # average coefficient
            "make_up": Normal(mean=3.0, std=0.0, shape=(1,)),  # make-up gain in dB
            "comp_th": Normal(mean=-25.0, std=0.0, shape=(1,)),  # compression threshold in dB
            "exp_th": Normal(mean=-31.0, std=0.0, shape=(1,)),  # expansion threshold in dB
            "lookahead": Uniform(low=0 ,high=0, shape=(1,)),  # lookahead in milliseconds
        }
    def train_setup(self):
        #hardcoded setuo used for training

        self.distribution_SNR=Normal(mean=5, std=2, shape=(1,))  # SNR in dB
    
        self.distribution_RMS=Normal(mean=-25, std=3, shape=(1,))  # RMS in dB

        self.RMS_norm=-25  # Target RMS level in dB, used for normalization


        self.reverberator= STFTMaskedNoiseReverb_filterbank(fixed_noise=False, ir_len=88000)
        self.distribution_init_log_magnitude=Normal(mean=0, std=2, shape=(self.reverberator.filterbank.num_filters,))  # Range for log magnitude randomization

        #min_T60=0.25  # Minimum reverberation time in seconds
        #max_T60=1.25  # Maximum reverberation time in seconds
        self.distribution_delta_log_magnitude = NormalRamp(mean_low=1.5, mean_high=0.2, std_low=0.5, std_high=0.1, log=False, shape=(self.reverberator.filterbank.num_filters,), transformation=lambda x: T602logmag(x.clamp(min=0.01), sample_rate=self.sample_rate, hop_length=self.reverberator.hop_length))  # Range for delta log magnitude randomization

        self.distribution_drywet_ratio= Normal(mean=0.1, std=0.02, shape=(1,), transformation=lambda x: torch.clamp(x, min=0, max=1))  # Dry/wet ratio for reverb

        _, self.params_CompExp_non_optimizable, _ = prepare_compexp_parameters(self.sample_rate, device=self.device)

        self.distributions_CompExp = {
            "comp_ratio": Normal(mean=4.0, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # compression ratio
            "exp_ratio": Normal(mean=0.25, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.01, max=0.999)),
            "at_coef": Normal(mean=35.0, std=0.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x,min=0.01), self.sample_rate)),  # attack time in milliseconds
            "rt_coef": Normal(mean=120.0, std=0.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), self.sample_rate)),  # release time in milliseconds
            "avg_coef": Normal(mean=0.95, std=0.0, shape=(1,), transformation=lambda x: torch.clamp(x, 0.01, 0.99)),  # average coefficient
            "make_up": Normal(mean=3.0, std=0.0, shape=(1,)),  # make-up gain in dB
            "comp_th": Normal(mean=-25.0, std=0.0, shape=(1,)),  # compression threshold in dB
            "exp_th": Normal(mean=-31.0, std=0.0, shape=(1,)),  # expansion threshold in dB
            "lookahead": Uniform(low=0 ,high=0, shape=(1,)),  # lookahead in milliseconds
        }
    
    def inference_setup_unused(self):
        #at inference time , parameters are fixed to a specific value
        self.distribution_SNR=Normal(mean=5, std=0, shape=(1,))  # SNR in dB
    
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
        gain_linear = 10 ** (gain / 20)
        x=x* gain_linear.view(-1, 1, 1)

        return x
    
    def add_noise(self,x):
        SNR=self.distribution_SNR.sample(x.shape[0]).to(x.device)
        x= utils.add_pink_noise(x, SNR)
        return x
        
    def apply_reverb(self,x):
        log_magnitude = self.distribution_init_log_magnitude.sample(x.shape[0]).to(x.device).unsqueeze(1)  # Sample log magnitude for each filterbank channel
        delta_log_magnitude = self.distribution_delta_log_magnitude.sample(x.shape[0]).to(x.device).unsqueeze(1)
        drywet_ratio = self.distribution_drywet_ratio.sample(x.shape[0]).to(x.device)

        self.reverberator.to(x.device)  # Ensure reverberator is on the same device as x
        y=self.reverberator(x, log_magnitude, delta_log_magnitude)

        return  x*(drywet_ratio).view(-1, 1, 1) + y *(1- drywet_ratio.view(-1, 1, 1))

    def apply_compexp(self, x, fixed=False):

        params_CompExp = sample_from_distribution_dict(self.distributions_CompExp, x.shape[0], device=self.device)
        x= compexp_functional(x, **params_CompExp, **self.params_CompExp_non_optimizable)

        return x

    def EQ_normalize_setup(self):

        nfft=4096 # FFT size hardcoded
        win_length=2048  # Window length hardcoded
        hop_length=1024  # Hop length hardcoded

        window = torch.hann_window(win_length, device=self.device)
        window_energy = window.pow(2).sum().sqrt()  # Energy of the window

        freqs = torch.fft.rfftfreq(nfft, d=1.0).to(self.device)
        freqs_Hz=torch.fft.rfftfreq(nfft, d=1.0).to(self.device) * self.sample_rate

        smooth_filter = prepare_smooth_filter(freqs_Hz, Noct=3).to(self.device)  # Prepare the smoothing filter

        target_curve = create_music_mean_spectrum_curve(freqs_Hz, device=self.device)

        def EQ_normalize_fn(x):
            shape= x.shape
            x=x.view(-1, shape[-1])
            X=torch.stft(x, n_fft=nfft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)/ window_energy
            X_pow=X.abs().pow(2)
            X_mean= torch.sqrt(X_pow.mean(dim=-1, keepdim=False))  # Mean power spectrum

            ratio= target_curve / (X_mean + 1e-6)

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


    def forward(self, x):


        #first stereo to mono (if needed)
        B, C, T = x.shape
        if C > 1:
            x = x.mean(dim=1, keepdim=True)

        x=self.apply_reverb(x)
        #stereo to mono (if needed)

        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)


        x= self.apply_RMS_normalization(x, random=False)

        #First EQ normalize
        x=self.EQ_normalize(x)

        #then apply random compressor
        x= self.apply_RMS_normalization(x, random=False)

        x = self.apply_compexp(x, fixed=True)



        if x.shape[1]==2:
            x=x.mean(dim=1, keepdim=True)

        #then apply random reverbo

        #x=self.apply_reverb(x)


        #then add noise

        x = self.add_noise(x)

        #then apply random gain

        x = self.apply_RMS_normalization(x, random=True)


        return x