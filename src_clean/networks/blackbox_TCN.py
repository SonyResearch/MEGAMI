""" 
    Adapted from: https://github.com/SonyResearch/ITO-Master
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import os
import sys

import omegaconf
import torchaudio

# 1-dimensional convolutional layer
# in the order of conv -> norm -> activation
class Conv1d_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                                    stride=1, \
                                    padding="SAME", dilation=1, bias=True, \
                                    norm="batch", activation="relu", \
                                    mode="conv"):
        super(Conv1d_layer, self).__init__()
        
        self.conv1d = nn.Sequential()

        ''' padding '''
        if mode=="deconv":
            padding = int(dilation * (kernel_size-1) / 2)
            out_padding = 0 if stride==1 else 1
        elif mode=="conv" or "alias_free" in mode:
            if padding == "SAME":
                pad = int((kernel_size-1) * dilation)
                l_pad = int(pad//2)
                r_pad = pad - l_pad
                padding_area = (l_pad, r_pad)
            elif padding == "VALID":
                padding_area = (0, 0)
            else:
                pass

        ''' convolutional layer '''
        if mode=="deconv":
            self.conv1d.add_module("deconv1d", nn.ConvTranspose1d(in_channels, out_channels, kernel_size, \
                                                            stride=stride, padding=padding, output_padding=out_padding, \
                                                            dilation=dilation, \
                                                            bias=bias))
        elif mode=="conv":
            self.conv1d.add_module(f"{mode}1d_pad", nn.ReflectionPad1d(padding_area))
            self.conv1d.add_module(f"{mode}1d", nn.Conv1d(in_channels, out_channels, kernel_size, \
                                                            stride=stride, padding=0, \
                                                            dilation=dilation, \
                                                            bias=bias))
        elif "alias_free" in mode:
            if "up" in mode:
                up_factor = stride * 2
                down_factor = 2
            elif "down" in mode:
                up_factor = 2
                down_factor = stride * 2
            else:
                raise ValueError("choose alias-free method : 'up' or 'down'")
            # procedure : conv -> upsample -> lrelu -> low-pass filter -> downsample
            # the torchaudio.transforms.Resample's default resampling_method is 'sinc_interpolation' which performs low-pass filter during the process
            # details at https://pytorch.org/audio/stable/transforms.html
            self.conv1d.add_module(f"{mode}1d_pad", nn.ReflectionPad1d(padding_area))
            self.conv1d.add_module(f"{mode}1d", nn.Conv1d(in_channels, out_channels, kernel_size, \
                                                            stride=1, padding=0, \
                                                            dilation=dilation, \
                                                            bias=bias))
            self.conv1d.add_module(f"{mode}upsample", torchaudio.transforms.Resample(orig_freq=1, new_freq=up_factor))
            self.conv1d.add_module(f"{mode}lrelu", nn.LeakyReLU())
            self.conv1d.add_module(f"{mode}downsample", torchaudio.transforms.Resample(orig_freq=down_factor, new_freq=1))

        ''' normalization '''
        if norm=="batch":
            self.conv1d.add_module("batch_norm", nn.BatchNorm1d(out_channels))
            # self.conv1d.add_module("batch_norm", nn.SyncBatchNorm(out_channels))

        ''' activation '''
        if 'alias_free' not in mode:
            if activation=="relu":
                self.conv1d.add_module("relu", nn.ReLU())
            elif activation=="lrelu":
                self.conv1d.add_module("lrelu", nn.LeakyReLU())


    def forward(self, input):
        # input shape should be : batch x channel x height x width
        output = self.conv1d(input)
        return output


# compute receptive field
def compute_receptive_field(kernels, strides, dilations):
    rf = 0
    for i in range(len(kernels)):
        rf += rf * strides[i] + (kernels[i]-strides[i]) * dilations[i]
    return rf

# Feature-wise Linear Modulation
class FiLM(nn.Module):
    def __init__(self, condition_len=2048, feature_len=1024):
        super(FiLM, self).__init__()
        self.film_fc = nn.Linear(condition_len, feature_len*2)
        self.feat_len = feature_len

    def forward(self, feature, condition):
        film_factor = self.film_fc(condition).unsqueeze(-1)
        r, b = torch.split(film_factor, self.feat_len, dim=1)
        return r*feature + b


class ConvBlock(nn.Module):
    def __init__(self, dimension, layer_num, \
                        in_channels, out_channels, \
                        kernel_size, \
                        stride=1, padding="SAME", \
                        dilation=1, \
                        bias=True, \
                        norm="batch", \
                        activation="relu", last_activation="relu", \
                        mode="conv"):
        super(ConvBlock, self).__init__()

        conv_block = []
        if dimension==1:
            for i in range(layer_num-1):
                conv_block.append(Conv1d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation))
            conv_block.append(Conv1d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, mode=mode))
        elif dimension==2:
            for i in range(layer_num-1):
                conv_block.append(Conv2d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation))
            conv_block.append(Conv2d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, mode=mode))
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, input):
        return self.conv_block(input)


class TCNBlock(torch.nn.Module):
    def __init__(self, 
                in_ch, 
                out_ch, 
                kernel_size=3, 
                stride=1, 
                dilation=1, 
                cond_dim=2048, 
                grouped=False, 
                causal=False,
                conditional=False, 
                **kwargs):
        super(TCNBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.grouped = grouped
        self.causal = causal
        self.conditional = conditional

        groups = out_ch if grouped and (in_ch % out_ch == 0) else 1

        self.pad_length = ((kernel_size-1)*dilation) if self.causal else ((kernel_size-1)*dilation)//2
        self.conv1 = torch.nn.Conv1d(in_ch, 
                                     out_ch, 
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.pad_length,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=False)
        if grouped:
            self.conv1b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)

        if conditional:
            self.film = FiLM(cond_dim, out_ch)
        self.bn = torch.nn.BatchNorm1d(out_ch)

        self.relu = torch.nn.LeakyReLU()

        if out_ch % in_ch == 0:
            self.res = torch.nn.Conv1d(in_ch, 
                                   out_ch, 
                                   kernel_size=1,
                                   stride=stride,
                                   groups=in_ch,
                                   bias=False)
        else:
            self.res = torch.nn.Conv1d(in_ch, 
                                   out_ch, 
                                   kernel_size=1,
                                   stride=stride,
                                   groups=1,
                                   bias=False)

    def forward(self, x, p):
        x_in = x

        x = self.relu(self.bn(self.conv1(x)))
        #print("p", p.shape)
        x = self.film(x, p)

        x_res = self.res(x_in)

        if self.causal:
            x = x[..., :-self.pad_length]
        x += x_res

        return x

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(
            [out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class TCNModel(nn.Module):
    """ Temporal convolutional network with conditioning module.
        Args:
            nparams (int): Number of conditioning parameters.
            ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
            noutputs (int): Number of output channels (mono = 1, stereo 2). Default: 1
            nblocks (int): Number of total TCN blocks. Default: 10
            kernel_size (int): Width of the convolutional kernels. Default: 3
            dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 1
            channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 2
            channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 64
            stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
            grouped (bool): Use grouped convolutions to reduce the total number of parameters. Default: False
            causal (bool): Causal TCN configuration does not consider future input values. Default: False
            skip_connections (bool): Skip connections from each block to the output. Default: False
        """
    def __init__(self, 
                 ninputs=1,
                 noutputs=2,
                 nblocks=14, 
                 kernel_size=15, 
                 stride=1,
                 dilation_growth=2, 
                 channel_growth=1, 
                 channel_width=128, 
                 stack_size=15,
                 cond_dim=2048,
                 grouped=False,
                 causal=False,
                 skip_connections=False,
                 use_CLAP=False,
                 CLAP_args=None,
                 ):
        super(TCNModel, self).__init__()

        self.use_CLAP = use_CLAP
        if self.use_CLAP:
            assert CLAP_args is not None, "CLAP_args must be provided for CLAP AE"
            from evaluation.feature_extractors import load_CLAP
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            CLAP_encoder= load_CLAP(CLAP_args, device=device)
            cond_dim+= 512
            def merge_CLAP_embeddings(x, emb):

                clap_embedding = CLAP_encoder(x, type="dry")
                #l2 normalize the clap embedding
                clap_embedding = F.normalize(clap_embedding, p=2, dim=-1)

                return torch.cat((emb, clap_embedding), dim=-1)
            
            self.merge_CLAP_embeddings = merge_CLAP_embeddings


        self.hparams = {
            "ninputs": ninputs,
            "noutputs": noutputs,
            "nblocks": nblocks,
            "kernel_size": kernel_size,
            "stride": stride,
            "dilation_growth": dilation_growth,
            "channel_growth": channel_growth,
            "channel_width": channel_width,
            "stack_size": stack_size,
            "cond_dim": cond_dim,
            "grouped": grouped,
            "causal": causal,
            "skip_connections": skip_connections,
        }
        self.hparams= omegaconf.OmegaConf.create(self.hparams)

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            
            if self.hparams.channel_growth > 1:
                out_ch = in_ch * self.hparams.channel_growth 
            else:
                out_ch = self.hparams.channel_width

            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            cur_stride = stride[n] if isinstance(stride, list) else stride
            self.blocks.append(TCNBlock(in_ch, 
                                        out_ch, 
                                        kernel_size=self.hparams.kernel_size, 
                                        stride=cur_stride, 
                                        dilation=dilation,
                                        padding="same" if self.hparams.causal else "valid",
                                        causal=self.hparams.causal,
                                        cond_dim=cond_dim, 
                                        grouped=self.hparams.grouped,
                                        conditional=True ))

        self.output = torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)

    def forward(self, x, cond):
        # iterate over blocks passing conditioning
        if self.use_CLAP:
            with torch.no_grad():
                cond = self.merge_CLAP_embeddings(x, cond)

        for idx, block in enumerate(self.blocks):
            # for SeFa
            if isinstance(cond, list):
                x = block(x, cond[idx])
            else:
                x = block(x, cond)
            skips = 0

        # out = torch.tanh(self.output(x + skips))
        out = torch.clamp(self.output(x + skips), min=-1, max=1)

        return out

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.hparams.kernel_size
        for n in range(1,self.hparams.nblocks):
            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            rf = rf + ((self.hparams.kernel_size-1) * dilation)
        return rf


