import torch
import math
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

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


class DownsamplingBlock(torch.nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 15,
        cond_dim: int = 2048,
    ):
        super().__init__()

        assert kernel_size % 2 != 0  # kernel must be odd length
        padding = kernel_size // 2  # calculate same padding

        self.conv1 = torch.nn.Conv1d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = torch.nn.BatchNorm1d(ch_out)
        self.prelu = torch.nn.PReLU(ch_out)
        self.conv2 = torch.nn.Conv1d(
            ch_out,
            ch_out,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
        )
        self.film = FiLM(cond_dim, ch_out)

    def forward(self, x, p):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.prelu(x)
        x = self.film(x, p)  # assuming condition is the same as input for simplicity
        x_ds = self.conv2(x)
        return x_ds, x


class UpsamplingBlock(torch.nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 5,
        skip: str = "add",
        cond_dim: int = 2048,
    ):
        super().__init__()

        assert kernel_size % 2 != 0  # kernel must be odd length
        padding = kernel_size // 2  # calculate same padding

        self.skip = skip
        self.conv = torch.nn.Conv1d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = torch.nn.BatchNorm1d(ch_out)
        self.prelu = torch.nn.PReLU(ch_out)
        self.us = torch.nn.Upsample(scale_factor=2)

        self.film = FiLM(cond_dim, ch_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, p: torch.Tensor):
        x = self.us(x)  # upsample by x2

        # handle skip connections
        if self.skip == "add":
            x = x + skip
        elif self.skip == "concat":
            x = torch.cat((x, skip), dim=1)
        elif self.skip == "none":
            pass
        else:
            raise NotImplementedError()

        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)

        x=self.film(x, p)  # apply FiLM modulation

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

class MixWaveUNet(torch.nn.Module):
    """

    Martínez Ramírez M. A., Stoller, D. and Moffat, D., “A deep learning approach to intelligent drum mixing with the Wave-U-Net”
    Journal of the Audio Engineering Society, vol. 69, no. 3, pp. 142-151, March 2021
    """

    def __init__(
        self,
        ninputs: int,
        noutputs: int,
        ds_kernel: int = 13,
        us_kernel: int = 13,
        out_kernel: int = 5,
        layers: int = 12,
        ch_growth: int = 24,
        skip: str = "concat",
        cond_dim=2112,
        use_CLAP=False,
        CLAP_args=None,
        temb_dim=64,
    ):
        super().__init__()

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

        self.timestep_features = FourierFeatures(1, temb_dim)
        cond_dim += temb_dim  # Add time embedding dimension

        self.encoder = torch.nn.ModuleList()
        for n in np.arange(layers):

            if n == 0:
                ch_in = ninputs
                ch_out = ch_growth
            else:
                ch_in = ch_out
                ch_out = ch_in + ch_growth

            self.encoder.append(DownsamplingBlock(ch_in, ch_out, kernel_size=ds_kernel, cond_dim=cond_dim))

        self.embedding = torch.nn.Conv1d(ch_out, ch_out, kernel_size=1)

        self.decoder = torch.nn.ModuleList()
        for n in np.arange(layers, stop=0, step=-1):

            ch_in = ch_out
            ch_out = ch_in - ch_growth

            if ch_out < ch_growth:
                ch_out = ch_growth

            if skip == "concat":
                ch_in *= 2

            self.decoder.append(
                UpsamplingBlock(
                    ch_in,
                    ch_out,
                    kernel_size=us_kernel,
                    skip=skip,
                    cond_dim=cond_dim
                )
            )

        self.output_conv = torch.nn.Conv1d(
            ch_out + ninputs,
            noutputs,
            kernel_size=out_kernel,
            padding=out_kernel // 2,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, time_cond=None, input_concat_cond=None):

        #if self.use_CLAP:
        #    with torch.no_grad():
        #        cond = self.merge_CLAP_embeddings(x, cond)

        if input_concat_cond is not None:
            x = torch.cat((x, input_concat_cond), dim=1)
        x_in=x
        
        timestep_embed = self.timestep_features(time_cond) # (b, embed_dim)

        cond= torch.cat((cond, timestep_embed), dim=-1) 

        skips = []

        for enc in self.encoder:
            x, skip = enc(x, cond)
            skips.append(skip)

        x = self.embedding(x)

        for dec in self.decoder:
            skip = skips.pop()
            x = dec(x, skip, cond)

        x = torch.cat((x_in, x), dim=1)

        y = self.output_conv(x)

        return y  # return dummy parameters