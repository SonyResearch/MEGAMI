import torch
import numpy as np

from utils.training_utils import  Gauss_smooth_vectorized,  prepare_smooth_filter

def T602logmag(t60, sample_rate=44100, hop_length=512):
    return 6.908 / (t60 * (sample_rate / hop_length))  # Convert T60 to delta log magnitude

from utils.data_utils import apply_RMS_normalization


class FxNormAug:

    def __init__(self,
                sample_rate=44100,  # Sample rate of the audio
                device="cuda" if torch.cuda.is_available() else "cpu",
                mode="train",  # Mode can be "train" or "eval"
                seed=42,
                features_path="features_tency1_4instr_v4.npy",  # Path to the features file
                ):

        torch.random.manual_seed(seed)

        #the path is in the same directory as this file
        self.features_path = features_path

        self.sample_rate = sample_rate
        self.device = device

        self.EQ_normalize_setup()  # Initialize EQ normalization function



    def EQ_normalize_setup(self ):

        features_mean = np.load(self.features_path, allow_pickle='TRUE')[()]

        target_cuves_original= {
            "vocals": torch.tensor(features_mean["eq"]["vocals"]).to(torch.float32).to(self.device),
            "drums": torch.tensor(features_mean["eq"]["drums"]).to(torch.float32).to(self.device),
            "bass": torch.tensor(features_mean["eq"]["bass"]).to(torch.float32).to(self.device),
            "other": torch.tensor(features_mean["eq"]["other"]).to(torch.float32).to(self.device),
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
        }


        def EQ_normalize_fn(x):

            shape= x.shape

            target_curves_tensor=torch.zeros((shape[0], nfft // 2 + 1), device=self.device, dtype=torch.float32)

            for i in range(shape[0]):
                track_class = "other"

                assert track_class in target_curves, f"track_class {track_class} not found in target_curves"
                target_curves_tensor[i] = target_curves[track_class] 

            x=x.view(-1, shape[-1])
            #ensure x.shape[-1] is divisible by hop_length
            if x.shape[-1] % hop_length != 0:
                # Pad the input to make it divisible by hop_length
                pad_length = hop_length - (x.shape[-1] % hop_length)
                x = torch.nn.functional.pad(x, (0, pad_length), mode='constant', value=0)
            X=torch.stft(x, n_fft=nfft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)/ window_energy
            X_pow=X.abs().pow(2)
            X_mean= torch.sqrt(X_pow.mean(dim=-1, keepdim=False))  # Mean power spectrum

            ratio= target_curves_tensor  / (X_mean + 1e-6)

            ratio = torch.clamp(ratio, max=10.0**(40.0/20.0))


            ratio_smooth = Gauss_smooth_vectorized(ratio, freqs_Hz, Noct=3, smooth_filter=smooth_filter)


            X= X * ratio_smooth.unsqueeze(-1)

            X_unnormalized=X* window_energy

            x_reconstructed = torch.istft(X_unnormalized, 
                             n_fft=nfft, 
                             hop_length=hop_length, 
                             win_length=win_length, 
                             window=window,
                             return_complex=False)  # Set to True if you want complex outpu

            #remove the padding if it was added
            if x_reconstructed.shape[-1] > shape[-1]:
                x_reconstructed = x_reconstructed[..., :shape[-1]]
            x_reconstructed = x_reconstructed.view(shape)
            return x_reconstructed
        
        self.EQ_normalize = EQ_normalize_fn


    def __call__(self, x, use_gate=False, RMS=-25):

        B, C, T = x.shape
        if C > 1:
            x = x.mean(dim=1, keepdim=True)

        x=x/x.max()

        x=self.EQ_normalize(x)

        x= apply_RMS_normalization(x, RMS, use_gate=use_gate)
        assert not torch.isnan(x).any(), "NaN detected in x after EQ normalization"

        return x
