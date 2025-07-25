"""
    Implementation of objective functions used in the task 'ITO-Master'
"""
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import auraloss
import torchaudio
import warnings


import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(currentdir))

class FrontEnd(nn.Module):
    def __init__(self, channel='stereo', \
                        n_fft=2048, \
                        n_mels=128, \
                        sample_rate=44100, \
                        hop_length=None, \
                        win_length=None, \
                        window="hann", \
                        eps=1e-7, \
                        device=torch.device("cpu")):
        super(FrontEnd, self).__init__()
        self.channel = channel
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = n_fft//4 if hop_length==None else hop_length
        self.win_length = n_fft if win_length==None else win_length
        self.eps = eps
        if window=="hann":
            self.window = torch.hann_window(window_length=self.win_length, periodic=True).to(device)
        elif window=="hamming":
            self.window = torch.hamming_window(window_length=self.win_length, periodic=True).to(device)
        self.melscale_transform = torchaudio.transforms.MelScale(n_mels=self.n_mels, \
                                                                    sample_rate=self.sample_rate, \
                                                                    n_stft=self.n_fft//2+1).to(device)


    def forward(self, input, mode):
        # front-end function which channel-wise combines all demanded features
        # input shape : batch x channel x raw waveform
        # output shape : batch x channel x frequency x time
        phase_output = None

        front_output_list = []
        for cur_mode in mode:
            # Real & Imaginary
            if cur_mode=="cplx":
                if self.channel=="mono":
                    output = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                elif self.channel=="stereo":
                    output_l = torch.stft(input[:,0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    output_r = torch.stft(input[:,1], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    output = torch.cat((output_l, output_r), axis=-1)
                if input.shape[-1] % round(self.n_fft/4) == 0:
                    output = output[:, :, :-1]
                if self.n_fft % 2 == 0:
                    output = output[:, :-1]
                front_output_list.append(output.permute(0, 3, 1, 2))
            # Magnitude & Phase or Mel
            elif "mag" in cur_mode or "mel" in cur_mode:
                if self.channel=="mono":
                    cur_cplx = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, return_complex=True)
                    output = self.mag(cur_cplx).unsqueeze(-1)[..., 0:1]
                    if "mag_phase" in cur_mode:
                        phase = self.phase(cur_cplx)
                    if "mel" in cur_mode:
                        output = self.melscale_transform(output.squeeze(-1)).unsqueeze(-1)
                elif self.channel=="stereo":
                    cplx_l = torch.stft(input[:,0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, return_complex=True)
                    cplx_r = torch.stft(input[:,1], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, return_complex=True)
                    mag_l = self.mag(cplx_l).unsqueeze(-1)
                    mag_r = self.mag(cplx_r).unsqueeze(-1)
                    output = torch.cat((mag_l, mag_r), axis=-1)
                    if "mag_phase" in cur_mode:
                        phase_l = self.phase(cplx_l).unsqueeze(-1)
                        phase_r = self.phase(cplx_r).unsqueeze(-1)
                        output = torch.cat((mag_l, phase_l, mag_r, phase_r), axis=-1)
                    if "mel" in cur_mode:
                        output = torch.cat((self.melscale_transform(mag_l.squeeze(-1)).unsqueeze(-1), self.melscale_transform(mag_r.squeeze(-1)).unsqueeze(-1)), axis=-1)

                if "log" in cur_mode:
                    output = torch.log(output+self.eps)

                if input.shape[-1] % round(self.n_fft/4) == 0:
                    output = output[:, :, :-1]
                if cur_mode!="mel" and self.n_fft % 2 == 0: # discard highest frequency
                    output = output[:, 1:]
                front_output_list.append(output.permute(0, 3, 1, 2))

        # combine all demanded features
        if not front_output_list:
            raise NameError("NameError at FrontEnd: check using features for front-end")
        elif len(mode)!=1:
            for i, cur_output in enumerate(front_output_list):
                if i==0:
                    front_output = cur_output
                else:
                    front_output = torch.cat((front_output, cur_output), axis=1)
        else:
            front_output = front_output_list[0]
            
        return front_output


    def mag(self, cplx_input, eps=1e-07):
        # mag_summed = cplx_input.pow(2.).sum(-1) + eps
        mag_summed = cplx_input.real.pow(2.) + cplx_input.imag.pow(2.) + eps
        return mag_summed.pow(0.5)


    def phase(self, cplx_input, ):
        return torch.atan2(cplx_input.imag, cplx_input.real)
        # return torch.angle(cplx_input)



# Root Mean Squared Loss
#   penalizes the volume factor with non-linearlity
class RMSLoss(nn.Module):
    def __init__(self, reduce, loss_type="l2"):
        super(RMSLoss, self).__init__()
        self.weight_factor = 100.
        if loss_type=="l2":
            self.loss = nn.MSELoss(reduce=None)


    def forward(self, est_targets, targets):
        est_targets = est_targets.reshape(est_targets.shape[0]*est_targets.shape[1], est_targets.shape[2])
        targets = targets.reshape(targets.shape[0]*targets.shape[1], targets.shape[2])
        normalized_est = torch.sqrt(torch.mean(est_targets**2, dim=-1))
        normalized_tgt = torch.sqrt(torch.mean(targets**2, dim=-1))

        weight = torch.clamp(torch.abs(normalized_tgt-normalized_est), min=1/self.weight_factor) * self.weight_factor

        return torch.mean(weight**1.5 * self.loss(normalized_est, normalized_tgt))



# Multi-Scale Spectral Loss proposed at the paper "DDSP: DIFFERENTIABLE DIGITAL SIGNAL PROCESSING" (https://arxiv.org/abs/2001.04643)
#   we extend this loss by applying it to mid/side channels
class MultiScale_Spectral_Loss_MidSide_DDSP(nn.Module):
    def __init__(self, mode='midside', \
                        reduce=True, \
                        n_filters=None, \
                        windows_size=None, \
                        hops_size=None, \
                        window="hann", \
                        eps=1e-7, \
                        device=torch.device("cpu")):
        super(MultiScale_Spectral_Loss_MidSide_DDSP, self).__init__()
        self.mode = mode
        self.eps = eps
        self.mid_weight = 0.5   # value in the range of 0.0 ~ 1.0
        self.logmag_weight = 0.1

        if n_filters is None:
            n_filters = [4096, 2048, 1024, 512]
        if windows_size is None:
            windows_size = [4096, 2048, 1024, 512]
        if hops_size is None:
            hops_size = [1024, 512, 256, 128]

        self.multiscales = []
        for i in range(len(windows_size)):
            cur_scale = {'window_size' : float(windows_size[i])}
            if self.mode=='midside':
                cur_scale['front_end'] = FrontEnd(channel='mono', \
                                                    n_fft=n_filters[i], \
                                                    hop_length=hops_size[i], \
                                                    win_length=windows_size[i], \
                                                    window=window, \
                                                    device=device)
            elif self.mode=='ori':
                cur_scale['front_end'] = FrontEnd(channel='stereo', \
                                                    n_fft=n_filters[i], \
                                                    hop_length=hops_size[i], \
                                                    win_length=windows_size[i], \
                                                    window=window, \
                                                    device=device)
            self.multiscales.append(cur_scale)

        self.reduce=reduce
        self.objective_l1 = nn.L1Loss(reduce=reduce)
        self.objective_l2 = nn.MSELoss(reduce=reduce)


    def forward(self, est_targets, targets):
        if self.mode=='midside':
            return self.forward_midside(est_targets, targets)
        elif self.mode=='ori':
            return self.forward_ori(est_targets, targets)


    def forward_ori(self, est_targets, targets):
        if self.reduce:
            total_mag_loss = 0.0
            total_logmag_loss = 0.0
        else:
            total_mag_loss=torch.zeros(est_targets.shape[0], 1).to(est_targets.device)
            total_logmag_loss=torch.zeros(est_targets.shape[0], 1).to(est_targets.device)
        for cur_scale in self.multiscales:
            est_mag = cur_scale['front_end'](est_targets, mode=["mag"])
            tgt_mag = cur_scale['front_end'](targets, mode=["mag"])

            mag_loss = self.magnitude_loss(est_mag, tgt_mag)
            logmag_loss = self.log_magnitude_loss(est_mag, tgt_mag)
            if self.reduce:
                total_mag_loss += mag_loss
                total_logmag_loss += logmag_loss
            else:
                total_logmag_loss += logmag_loss.mean((1, 2, 3)).unsqueeze(-1)
                total_mag_loss += mag_loss.mean((1, 2, 3)).unsqueeze(-1)
        # return total_loss
        return (1-self.logmag_weight)*total_mag_loss + \
                (self.logmag_weight)*total_logmag_loss


    def forward_midside(self, est_targets, targets):
        est_mid, est_side = self.to_mid_side(est_targets)
        tgt_mid, tgt_side = self.to_mid_side(targets)
        if self.reduce:
            total_mag_loss = 0.0
            total_logmag_loss = 0.0
        else:
            total_logmag_loss=torch.zeros(est_targets.shape[0], 1).to(est_targets.device)
            total_mag_loss=torch.zeros(est_targets.shape[0], 1).to(est_targets.device)

        for cur_scale in self.multiscales:
            est_mid_mag = cur_scale['front_end'](est_mid, mode=["mag"])
            est_side_mag = cur_scale['front_end'](est_side, mode=["mag"])
            tgt_mid_mag = cur_scale['front_end'](tgt_mid, mode=["mag"])
            tgt_side_mag = cur_scale['front_end'](tgt_side, mode=["mag"])

            mag_loss = self.mid_weight*self.magnitude_loss(est_mid_mag, tgt_mid_mag) + \
                        (1-self.mid_weight)*self.magnitude_loss(est_side_mag, tgt_side_mag)
            logmag_loss = self.mid_weight*self.log_magnitude_loss(est_mid_mag, tgt_mid_mag) + \
                        (1-self.mid_weight)*self.log_magnitude_loss(est_side_mag, tgt_side_mag)

                #take mean over all dimensions except batch
            if self.reduce:
                total_mag_loss += mag_loss
                total_logmag_loss += logmag_loss
            else:
                total_mag_loss += mag_loss.mean((1, 2, 3)).unsqueeze(-1)
                #mean over dims 1, 2, 3
                total_logmag_loss += logmag_loss.mean((1, 2, 3)).unsqueeze(-1)
        # return total_loss
        return (1-self.logmag_weight)*total_mag_loss + \
                (self.logmag_weight)*total_logmag_loss


    def to_mid_side(self, stereo_in):
        mid = stereo_in[:,0] + stereo_in[:,1]
        side = stereo_in[:,0] - stereo_in[:,1]
        return mid, side


    def magnitude_loss(self, est_mag_spec, tgt_mag_spec):
        if self.reduce:
            return torch.norm(self.objective_l1(est_mag_spec, tgt_mag_spec))
        else:
            return self.objective_l1(est_mag_spec, tgt_mag_spec)


    def log_magnitude_loss(self, est_mag_spec, tgt_mag_spec):
        est_log_mag_spec = torch.log10(est_mag_spec+self.eps)
        tgt_log_mag_spec = torch.log10(tgt_mag_spec+self.eps)
        return self.objective_l2(est_log_mag_spec, tgt_log_mag_spec)



# Class of available loss functions
class Loss:
    def __init__(self, args, reduce=True):
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
        self.l1 = nn.L1Loss(reduce=reduce)
        self.mse = nn.MSELoss(reduce=reduce)
        self.ce = nn.CrossEntropyLoss()
        self.triplet = nn.TripletMarginLoss(margin=1., p=2)
        self.cos = nn.CosineSimilarity(eps=args.eps)
        self.cosemb = nn.CosineEmbeddingLoss()

        self.multi_scale_spectral_midside = MultiScale_Spectral_Loss_MidSide_DDSP(mode='midside', eps=args.eps, device=device)
        self.multi_scale_spectral_ori = MultiScale_Spectral_Loss_MidSide_DDSP(mode='ori', eps=args.eps, device=device)
        self.gain = RMSLoss(reduce=reduce)
        # perceptual weighting with mel scaled spectrograms
        self.mrs_mel_perceptual = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 8192],
            hop_sizes=[256, 512, 2048],
            win_lengths=[1024, 2048, 8192],
            scale="mel",
            n_bins=128,
            sample_rate=args.sample_rate,
            perceptual_weighting=True,
        )


# CLAP feature loss
class CLAPFeatureLoss(nn.Module):
    def __init__(self):
        super(CLAPFeatureLoss, self).__init__()
        import laion_clap
        import torchaudio
        self.target_sample_rate = 48000  # CLAP expects 48kHz audio
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()  # download the default pretrained checkpoint
        self.model.eval()

    def forward(self, input_audio, target, sample_rate, distance_fn='cosine'):
        # Process input audio
        input_embed = self.process_audio(input_audio, sample_rate)

        # Process target (audio or text)
        if isinstance(target, torch.Tensor):
            target_embed = self.process_audio(target, sample_rate)
        elif isinstance(target, str) or (isinstance(target, list) and isinstance(target[0], str)):
            target_embed = self.process_text(target)
        else:
            raise ValueError("Target must be either audio tensor or text (string or list of strings)")

        # Compute loss using the specified distance function
        loss = self.compute_distance(input_embed, target_embed, distance_fn)

        return loss

    def process_audio(self, audio, sample_rate):
        # Ensure input is in the correct shape (N, C, T)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Convert to mono if stereo
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            audio = self.resample(audio, sample_rate)
        audio = audio.squeeze(1)
        
        # Get CLAP embeddings
        embed = self.model.get_audio_embedding_from_data(x=audio, use_tensor=True)
        return embed

    def process_text(self, text):
        # Get CLAP embeddings for text
        # ensure input is a list of strings
        if not isinstance(text, list):
            text = [text]
        embed = self.model.get_text_embedding(text, use_tensor=True)
        return embed

    def compute_distance(self, x, y, distance_fn):
        if distance_fn == 'mse':
            return F.mse_loss(x, y)
        elif distance_fn == 'l1':
            return F.l1_loss(x, y)
        elif distance_fn == 'cosine':
            return 1 - F.cosine_similarity(x, y).mean()
        else:
            raise ValueError(f"Unsupported distance function: {distance_fn}")

    def resample(self, audio, input_sample_rate):
        resampler = torchaudio.transforms.Resample(
            orig_freq=input_sample_rate, new_freq=self.target_sample_rate
        ).to(audio.device)
        return resampler(audio)



"""
    Audio Feature Loss implementation 
        copied from https://github.com/sai-soum/Diff-MST/blob/main/mst/loss.py
"""

import librosa


from typing import List
def _create_triangular_filterbank(
    all_freqs: torch.Tensor,
    f_pts: torch.Tensor,
) -> torch.Tensor:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb



def _hz_to_bark(freqs: float, bark_scale: str = "traunmuller") -> float:
    r"""Convert Hz to Barks.

    Args:
        freqs (float): Frequencies in Hz
        bark_scale (str, optional): Scale to use: ``traunmuller``, ``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        barks (float): Frequency in Barks
    """

    if bark_scale not in ["schroeder", "traunmuller", "wang"]:
        raise ValueError(
            'bark_scale should be one of "schroeder", "traunmuller" or "wang".'
        )

    if bark_scale == "wang":
        return 6.0 * math.asinh(freqs / 600.0)
    elif bark_scale == "schroeder":
        return 7.0 * math.asinh(freqs / 650.0)
    # Traunmuller Bark scale
    barks = ((26.81 * freqs) / (1960.0 + freqs)) - 0.53
    # Bark value correction
    if barks < 2:
        barks += 0.15 * (2 - barks)
    elif barks > 20.1:
        barks += 0.22 * (barks - 20.1)

    return barks


def _bark_to_hz(barks: torch.Tensor, bark_scale: str = "traunmuller") -> torch.Tensor:
    """Convert bark bin numbers to frequencies.

    Args:
        barks (torch.Tensor): Bark frequencies
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        freqs (torch.Tensor): Barks converted in Hz
    """

    if bark_scale not in ["schroeder", "traunmuller", "wang"]:
        raise ValueError(
            'bark_scale should be one of "traunmuller", "schroeder" or "wang".'
        )

    if bark_scale == "wang":
        return 600.0 * torch.sinh(barks / 6.0)
    elif bark_scale == "schroeder":
        return 650.0 * torch.sinh(barks / 7.0)
    # Bark value correction
    if any(barks < 2):
        idx = barks < 2
        barks[idx] = (barks[idx] - 0.3) / 0.85
    elif any(barks > 20.1):
        idx = barks > 20.1
        barks[idx] = (barks[idx] + 4.422) / 1.22

    # Traunmuller Bark scale
    freqs = 1960 * ((barks + 0.53) / (26.28 - barks))

    return freqs




def barkscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_barks: int,
    sample_rate: int,
    bark_scale: str = "traunmuller",
) -> torch.Tensor:
    r"""Create a frequency bin conversion matrix.

    .. devices:: CPU

    .. properties:: TorchScript

    .. image:: https://download.pytorch.org/torchaudio/doc-assets/bark_fbanks.png
        :alt: Visualization of generated filter bank

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_barks (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        torch.Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_barks``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * barkscale_fbanks(A.size(-1), ...)``.

    """

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate bark freq bins
    m_min = _hz_to_bark(f_min, bark_scale=bark_scale)
    m_max = _hz_to_bark(f_max, bark_scale=bark_scale)

    m_pts = torch.linspace(m_min, m_max, n_barks + 2)
    f_pts = _bark_to_hz(m_pts, bark_scale=bark_scale)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one bark filterbank has all zero values. "
            f"The value for `n_barks` ({n_barks}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb

def compute_mid_side(x: torch.Tensor):
    x_mid = x[:, 0, :] + x[:, 1, :]
    x_side = x[:, 0, :] - x[:, 1, :]
    return x_mid, x_side


def compute_melspectrum(
    x: torch.Tensor,
    sample_rate: int = 44100,
    fft_size: int = 32768,
    n_bins: int = 128,
    **kwargs,
):
    """Compute mel-spectrogram.

    Args:
        x: (bs, 2, seq_len)
        sample_rate: sample rate of audio
        fft_size: size of fft
        n_bins: number of mel bins

    Returns:
        X: (bs, n_bins)

    """
    fb = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_bins)
    fb = torch.tensor(fb).unsqueeze(0).type_as(x)

    x = x.mean(dim=1, keepdim=True)
    X = torch.fft.rfft(x, n=fft_size, dim=-1)
    X = torch.abs(X)
    X = torch.mean(X, dim=1, keepdim=True)  # take mean over time
    X = X.permute(0, 2, 1)  # swap time and freq dims
    X = torch.matmul(fb, X)
    X = torch.log(X + 1e-8)

    return X


def compute_barkspectrum(
    x: torch.Tensor,
    fft_size: int = 32768,
    n_bands: int = 24,
    sample_rate: int = 44100,
    f_min: float = 20.0,
    f_max: float = 20000.0,
    mode: str = "mid-side",
    **kwargs,
):
    """Compute bark-spectrogram.

    Args:
        x: (bs, 2, seq_len)
        fft_size: size of fft
        n_bands: number of bark bins
        sample_rate: sample rate of audio
        f_min: minimum frequency
        f_max: maximum frequency
        mode: "mono", "stereo", or "mid-side"

    Returns:
        X: (bs, 24)

    """
    # compute filterbank
    fb = barkscale_fbanks((fft_size // 2) + 1, f_min, f_max, n_bands, sample_rate)
    fb = fb.unsqueeze(0).type_as(x)
    fb = fb.permute(0, 2, 1)

    if mode == "mono":
        x = x.mean(dim=1)  # average over channels
        signals = [x]
    elif mode == "stereo":
        signals = [x[:, 0, :], x[:, 1, :]]
    elif mode == "mid-side":
        x_mid = x[:, 0, :] + x[:, 1, :]
        x_side = x[:, 0, :] - x[:, 1, :]
        signals = [x_mid, x_side]
    else:
        raise ValueError(f"Invalid mode {mode}")

    outputs = []
    for signal in signals:
        X = torch.stft(
            signal,
            n_fft=fft_size,
            hop_length=fft_size // 4,
            return_complex=True,
            window=torch.hann_window(fft_size).to(x.device),
        )  # compute stft
        X = torch.abs(X)  # take magnitude
        X = torch.mean(X, dim=-1, keepdim=True)  # take mean over time
        # X = X.permute(0, 2, 1)  # swap time and freq dims
        X = torch.matmul(fb, X)  # apply filterbank
        X = torch.log(X + 1e-8)
        # X = torch.cat([X, X_log], dim=-1)
        outputs.append(X)

    # stack into tensor
    X = torch.cat(outputs, dim=-1)

    return X


def compute_rms(x: torch.Tensor, **kwargs):
    """Compute root mean square energy.

    Args:
        x: (bs, 1, seq_len)

    Returns:
        rms: (bs, )
    """
    rms = torch.sqrt(torch.mean(x**2, dim=-1).clamp(min=1e-8))
    return rms

import loudness

def compute_loudness(x: torch.Tensor, sample_rate=44100):
    B, C, T = x.shape
    lufs_out=torch.zeros((x.shape[0], x.shape[1]), device=x.device)
    for b in range(B):
        x_i=x[b].cpu().numpy().T
        lufs_in=loudness.integrated_loudness(x_i, sample_rate)
        lufs_out[b] = torch.tensor(lufs_in, device=x.device)
    
    return lufs_out

def compute_log_rms(x: torch.Tensor, **kwargs):
    """Compute root mean square energy.

    Args:
        x: (bs, 1, seq_len)

    Returns:
        rms: (bs, )
    """
    rms=compute_rms(x)
    return 20 * torch.log10(rms.clamp(min=1e-8))

def compute_crest_factor(x: torch.Tensor, **kwargs):
    """Compute crest factor as ratio of peak to rms energy in dB.

    Args:
        x: (bs, 2, seq_len)

    """
    num = torch.max(torch.abs(x), dim=-1)[0]
    den = compute_rms(x).clamp(min=1e-8)
    cf = 20 * torch.log10((num / den).clamp(min=1e-8))
    return cf

def compute_log_spread(x: torch.Tensor, **kwargs):
    """Compute log spread as the mean difference between log magnitude of samples and log RMS.
    
    Args:
        x: (bs, 1, seq_len)
        
    Returns:
        log_spread: (bs, )
    """
    # Compute log RMS
    log_rms = compute_log_rms(x).unsqueeze(-1)  # (bs, 1, 1)
    
    # Compute log magnitude of each sample
    log_magnitude = 20 * torch.log10(torch.abs(x).clamp(min=1e-8))  # (bs, 1, seq_len)
    
    # Compute the difference and take the mean
    log_spread = torch.mean(log_magnitude - log_rms, dim=-1).squeeze(1)  # (bs, )
    
    return log_spread


def compute_stereo_width(x: torch.Tensor, **kwargs):
    """Compute stereo width as ratio of energy in sum and difference signals.

    Args:
        x: (bs, 2, seq_len)

    """
    bs, chs, seq_len = x.size()

    assert chs == 2, "Input must be stereo"

    # compute sum and diff of stereo channels
    x_sum = x[:, 0, :] + x[:, 1, :]
    x_diff = x[:, 0, :] - x[:, 1, :]

    # compute power of sum and diff
    sum_energy = torch.mean(x_sum**2, dim=-1)
    diff_energy = torch.mean(x_diff**2, dim=-1)

    # compute stereo width as ratio
    stereo_width = diff_energy / sum_energy.clamp(min=1e-8)

    return stereo_width


def compute_stereo_imbalance(x: torch.Tensor, **kwargs):
    """Compute stereo imbalance as ratio of energy in left and right channels.

    Args:
        x: (bs, 2, seq_len)

    Returns:
        stereo_imbalance: (bs, )

    """
    left_energy = torch.mean(x[:, 0, :] ** 2, dim=-1)
    right_energy = torch.mean(x[:, 1, :] ** 2, dim=-1)

    stereo_imbalance = (right_energy - left_energy) / (
        right_energy + left_energy
    ).clamp(min=1e-8)

    return stereo_imbalance


class AudioFeatureLoss(torch.nn.Module):
    def __init__(
        self,
        weights: List[float],
        sample_rate: int,
        stem_separation: bool = False,
        use_clap: bool = False,
    ) -> None:
        """Compute loss using a set of differentiable audio features.

        Args:
            weights: weights for each feature
            sample_rate: sample rate of audio
            stem_separation: whether to compute loss on stems or mix

        Based on features proposed in:

        Man, B. D., et al.
        "An analysis and evaluation of audio features for multitrack music mixtures."
        (2014).

        """
        super().__init__()
        self.weights = weights
        self.sample_rate = sample_rate
        self.stem_separation = stem_separation
        self.sources_list = ["mix"]
        self.source_weights = [1.0]
        self.use_clap = use_clap

        self.transforms = [
            compute_rms,
            compute_crest_factor,
            compute_stereo_width,
            compute_stereo_imbalance,
            compute_barkspectrum,
        ]

        assert len(self.transforms) == len(weights)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        losses = {}

        # reshape for example stem dim
        input_stems = input.unsqueeze(1)
        target_stems = target.unsqueeze(1)

        n_stems = input_stems.shape[1]

        # iterate over each stem compute loss for each transform
        for stem_idx in range(n_stems):
            input_stem = input_stems[:, stem_idx, ...]
            target_stem = target_stems[:, stem_idx, ...]

            for transform, weight in zip(self.transforms, self.weights):
                transform_name = "_".join(transform.__name__.split("_")[1:])
                key = f"{self.sources_list[stem_idx]}-{transform_name}"
                input_transform = transform(input_stem, sample_rate=self.sample_rate)
                target_transform = transform(target_stem, sample_rate=self.sample_rate)
                val = torch.nn.functional.mse_loss(input_transform, target_transform)
                losses[key] = weight * val * self.source_weights[stem_idx]

        return losses

