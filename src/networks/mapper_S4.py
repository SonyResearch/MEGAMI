
# Code adapted from: https://github.com/mcomunita/nablafx/blob/master/nablafx/s4.py

import math
from einops import repeat
import os
import torch
from einops import rearrange
from torch import Tensor

import torch.nn.functional as F


import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)


# -----------------------------------------------------------------------------
# FiLM layer
# -----------------------------------------------------------------------------


class FiLM(torch.nn.Module):
    """
    Given an input sequence x and conditioning parameters cond, modulates x.

    nfeatures: number of features (i.e., convolution channels)
    cond_dim: number of conditioning features
    """

    def __init__(self, nfeatures, cond_dim):
        super(FiLM, self).__init__()
        self.nfeatures = nfeatures
        self.cond_dim = cond_dim

        self.bn = torch.nn.BatchNorm1d(nfeatures, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, nfeatures * 2)

    def forward(self, x, cond):
        # x = [batch, channels, length]
        # cond = [batch, cond_dim]
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.unsqueeze(-1)  # [batch, nfeatures, 1]
        b = b.unsqueeze(-1)  # [batch, nfeatures, 1]
        x = self.bn(x)  # apply BatchNorm without affine
        x = (x * g) + b  # apply conditional affine
        return x


# -----------------------------------------------------------------------------
# S4 Conditional Block
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Diagonal State-Space Model Block
# -----------------------------------------------------------------------------

c2r = torch.view_as_real
r2c = torch.view_as_complex

from torchaudio.functional import fftconvolve


class DSSM(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float = 1e-3,
        use_fft_convolve: bool = False,
    ):
        super().__init__()

        H = input_dim
        self.H = H
        N = state_dim
        self.N = N

        self.use_fft_convolve = use_fft_convolve

        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.register("log_dt", log_dt, lr)

        C = torch.randn(H, N, dtype=torch.cfloat)
        self.C = torch.nn.Parameter(c2r(C))

        log_A_real = torch.log(0.5 * torch.ones(H, N))
        A_imag = math.pi * repeat(torch.arange(N), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

        self.D = torch.nn.Parameter(torch.randn(H))

    def get_kernel(self, length: int):  # `length` is `L`
        dt = torch.exp(self.log_dt)  # (H)
        C = r2c(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        P = dtA.unsqueeze(-1) * torch.arange(length, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(P)).real
        return K

    def register(self, name: str, tensor: Tensor, lr: float = None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, torch.nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}  # Never use weight decay
            if lr is not None:  # Use custom learning rate when a learning rate is given
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def forward(self, u: Tensor, length: Tensor = None):
        # Input and output shape (B H L)
        assert u.dim() == 3
        #print(("u", u.shape))
        B, H, L = u.size()
        assert H == self.H

        # length shape (L)
        if length is None:
            length = torch.empty(B).fill_(L)
        assert length.dim() == 1
        assert length.size(0) == B
        assert torch.all(length <= L)
        length = length.to(torch.long).cpu()

        l_s, i_s = length.sort(stable=True, descending=True)
        l_s, i_s = l_s.tolist(), i_s.tolist()
        prev_i = 0
        pair: list[tuple[int, list[int]]] = []
        for i in range(1, B):
            if l_s[i] == l_s[i - 1]:
                continue
            pair.append((l_s[prev_i], i_s[prev_i:i]))
            prev_i = i
        pair.append((l_s[prev_i], i_s[prev_i:]))

        kernel = self.get_kernel(L)  # (H L)
        out = torch.zeros_like(u)  # (B H L)

        for l, idxs in pair:
            _k = kernel[:, :l]
            _u = u[idxs, :, :l]

            if self.use_fft_convolve:
                y = fftconvolve(_u, _k.unsqueeze(0), mode="same")  # (B H l + L - 1)
            else:
                k_f = torch.fft.rfft(_k, n=2 * l)  # (H l)
                #print(("u_f", _u.shape, "k_f", k_f.shape))
                u_f = torch.fft.rfft(_u, n=2 * l)  # (B H l)
                y = torch.fft.irfft(u_f * k_f, n=2 * l)[..., :l]  # (B H l)

            y += _u * self.D.unsqueeze(-1)  # (B H l)

            out[idxs, :, :l] = y[...]

        return out


class S4CondBlock(torch.nn.Module):
    def __init__(
        self,
        channel_width: int,
        batchnorm: bool,
        residual: bool,
        s4_state_dim: int,
        s4_learning_rate: float,
        cond_type: str,
        cond_dim: int,
        act_type: str,
        use_fft_convolve: bool = False,
    ):
        super(S4CondBlock, self).__init__()
        assert cond_dim >= 0
        assert cond_type in [None, "film"]
        assert act_type in ["tanh", "prelu", "rational", "gelu"]

        self.channel_width = channel_width
        self.batchnorm = batchnorm
        self.residual = residual
        self.s4_state_dim = s4_state_dim
        self.s4_learning_rate = s4_learning_rate
        self.cond_type = cond_type
        self.cond_dim = cond_dim
        self.act_type = act_type

        # LINEAR
        self.linear = torch.nn.Linear(channel_width, channel_width)

        # S4
        self.s4 = DSSM(input_dim=channel_width, state_dim=s4_state_dim, lr=s4_learning_rate, use_fft_convolve=use_fft_convolve)

        # CONDITIONING/MODULATION
        if cond_type == "film":
            self.film = FiLM(nfeatures=channel_width, cond_dim=cond_dim)
        elif cond_type is None and batchnorm:
            self.bn = torch.nn.BatchNorm1d(channel_width)

        # ACTIVATIONS
        if act_type == "tanh":
            self.act1 = torch.nn.Tanh()
            self.act2 = torch.nn.Tanh()
        elif act_type == "prelu":
            self.act1 = torch.nn.PReLU(num_parameters=channel_width)
            self.act2 = torch.nn.PReLU(num_parameters=channel_width)
        elif act_type == "gelu":
            self.act1 = torch.nn.GELU()
            self.act2 = torch.nn.GELU()
        elif act_type == "rational":
            raise NotImplementedError("Rational activation is not implemented in this block.")
            self.act1 = Rational(approx_func="tanh", degrees=[4, 3], version="A")
            self.act2 = Rational(approx_func="tanh", degrees=[4, 3], version="A")

        # RESIDUAL
        if residual:
            self.res = torch.nn.Conv1d(
                channel_width,
                channel_width,
                kernel_size=1,
                groups=channel_width,
                bias=False,
            )

    def forward(self, x: Tensor, cond: Tensor = None) -> Tensor:

        x_in = x

        # LINEAR
        x = rearrange(x, "B H L -> B L H")
        x = self.linear(x)
        x = rearrange(x, "B L H -> B H L")

        # ACTIVATION
        x = self.act1(x)

        # S4
        x = self.s4(x)

        # CONDITIONING/MODULATION
        if self.cond_type is not None:
            x = self.film(x, cond)
        elif self.cond_type is None and self.batchnorm:
            x = self.bn(x)

        # ACTIVATION
        x = self.act2(x)

        # OUTPUT
        if self.residual:
            x = x + self.res(x_in)

        return x

class S4(torch.nn.Module):
    def __init__(
        self,
        num_inputs: int = 1,
        num_outputs: int = 2,
        num_blocks: int = 4,
        channel_width: int = 32,
        s4_state_dim: int = 4,
        batchnorm: bool = False,
        residual: bool = False,
        direct_path: bool = False,
        cond_dim: int = 0,
        cond_type: str = "film",
        act_type: str = "tanh",
        s4_learning_rate: float = 0.0005,
        use_CLAP=False,
        CLAP_args=None,
    ):
        super().__init__()
        assert num_inputs == 1, f"implemented only for 1 input channels"
        #assert num_outputs == 1, f"implemented only for 1 output channels"
        assert cond_type in [None, "film"]
        assert act_type in ["tanh", "prelu", "rational", "gelu"]
        assert channel_width >= 1, f"The inner audio channel is expected to be one or greater, but got {channel_width}."
        assert s4_state_dim >= 1, f"The S4 hidden size is expected to be one or greater, but got {s4_state_dim}."
        assert num_blocks >= 0, f"The model depth is expected to be zero or greater, but got {num_blocks}."

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

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_blocks = num_blocks
        self.channel_width = channel_width
        self.s4_state_dim = s4_state_dim
        self.batchnorm = batchnorm
        self.residual = residual
        self.direct_path = direct_path
        self.cond_type = cond_type

        self.cond_dim= cond_dim

        self.act_type = act_type

        #if cond_type == "film":  # conditioning MLP
        #    self.cond_nn = MLP(input_dim=num_controls, output_dim=self.cond_dim)

        # DIRECT PATH
        if direct_path:
            self.direct_gain = torch.nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)

        # INPUT
        self.expand = torch.nn.Linear(num_inputs, channel_width)

        # BLOCKS
        self.blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                S4CondBlock(
                    channel_width=channel_width,
                    batchnorm=batchnorm,
                    residual=residual,
                    s4_state_dim=s4_state_dim,
                    s4_learning_rate=s4_learning_rate,
                    cond_type=cond_type,
                    cond_dim=self.cond_dim,
                    act_type=act_type,
                    use_fft_convolve=False
                )
            )

        # OUTPUT
        self.contract = torch.nn.Linear(channel_width, num_outputs)



    def forward(self, x: Tensor, cond: Tensor = None) -> Tensor:
        # x = input : (batch, channels, seq)
        # p = params : (batch, params)
        bs, chs, seq_len = x.size()
        assert chs == 1, "The input tensor is expected to have one channel."

        if self.use_CLAP:
            with torch.no_grad():
                cond = self.merge_CLAP_embeddings(x, cond)

        # CONDITIONING
        #if self.cond_type is None:
        #    cond = None
        #elif self.cond_type == "film":
        #    cond = self.cond_nn(p)
        #cond=...

        # DIRECT PATH
        #if self.direct_path:
        #    y_direct = self.direct_gain(x)

        #x = x.view(bs, seq_len)
        #y_proc = self._pass_blocks(x, cond)

        x = rearrange(x, "B C L -> B L C")
        x = self.expand(x)
        x = rearrange(x, "B L H -> B H L")

        for block in self.blocks:
            x = block(x, cond)

        x = rearrange(x, "B H L -> B L H")
        x = self.contract(x)
        x = rearrange(x, "B H C -> B C H")

        #y_proc = y_proc.view(bs, 1, seq_len)  # add channel dimension

        return x
