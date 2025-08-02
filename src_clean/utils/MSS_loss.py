"""
    Implementation of objective functions used in the task 'ITO-Master'
    https://github.com/SonyResearch/ITO-Master/blob/master/ito_master/modules/loss.py
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



