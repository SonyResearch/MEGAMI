import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

from torch_fftconv import fft_conv1d
from torchcomp import db2amp
from time import time

from fx_model.processors.transformations import Identity, UniLossLess, MinMax





def prepare_FDN_parameters(
        sample_rate: int,
        ir_duration: float = 12.0,
        delays=(997, 1153, 1327, 1559, 1801, 2099),
        num_decay_freq=49,
        device="cpu",
    ):

        num_delays = len(delays)


        ir_length=int(sample_rate * ir_duration)

        T_60=ir_duration*0.75
        delays = torch.tensor(delays)
        gamma_max = -60 / sample_rate / T_60 * delays.min()
        gamma_max= 10 ** (gamma_max / 20)

        dict_non_optimizable = {
            "ir_length": ir_length,  # shape (1,)
            "delays": delays.to(device),
        }

        dict_transformations= {
            "b": MinMax(-1,1),
            "c": MinMax(-1,1),
            "U": UniLossLess(),
            "gamma": MinMax(0.0, gamma_max),
        }

        b=torch.ones(num_delays, 2) / num_delays
        c=torch.zeros(2, num_delays)
        U=torch.randn(num_delays, num_delays) / num_delays**0.5
        gamma=dict_transformations["gamma"].right_inverse(torch.rand(num_decay_freq, 1)* 0.2+ 0.4)

        dict_optimizable = {
            "b": b.to(device),  # shape (num_delays, 2)
            "c": c.to(device),  # shape (2, num_delays)
            "U": U.to(device),  # shape (num_delays, num_delays)
            "gamma": gamma.to(device),  # shape (num_decay_freq, 1)
        }

        return dict_optimizable, dict_non_optimizable, dict_transformations


def fdn_functional(
    x: torch.Tensor, # shape (batch, num_channels, num_samples)
    b: torch.Tensor, # shape (batch, num_delays, 2)
    c: torch.Tensor, # shape (batch,2, num_delays)
    U: torch.Tensor, # shape (batch, num_delays, num_delays)
    gamma: torch.Tensor, # shape (batch, num_decay_freq, 1)  decays are delay independent
    ir_length: torch.Tensor, # shape (1,)
    delays: torch.Tensor,  # shape (num_delays, )
    eq=None,
    ):

    c=c+ 0j
    b=b+ 0j

    if gamma.size(1) > 1:
        gamma = F.interpolate(
            gamma.transpose(1,2),
            size=ir_length // 2 + 1,
            align_corners=True,
            mode="linear",
        ).transpose(1, 2).unsqueeze(2)

    #gamma shape now is (batch, fft_freqs, 1, 1)

    if gamma.ndim==3:
        gamma = gamma.unsqueeze(-1)

    if gamma.size(3)==1:
        gamma= gamma ** (delays / delays.min())

    #shape of gamma is (batch, num_decay_freq, 1, num_delays)

    A= U.unsqueeze(1) * gamma

    #shape of A is (batch, fft_freqs, num_delays, num_delays)
    
    freqs = (
        torch.arange(ir_length // 2 + 1, device=x.device)
        / ir_length
        * 2
        * torch.pi
    )

    invD=torch.exp(1j * freqs[:,None] * delays)
    #shape of invD is (fft_freqs, num_delays)

    H = c.unsqueeze(1) @ torch.linalg.solve(torch.diag_embed(invD) - A, b.unsqueeze(1))


    h = torch.fft.irfft(H.permute(0,2, 3, 1), n=ir_length)

    if eq is not None:
        h = eq(h)

    return parallel_batch_fft_conv1d(x, h)  # Flip h for convolution
    #return loop_based_batch_fft_conv1d(x, h)  # Flip h for convolution

 



def parallel_batch_fft_conv1d(x, h):
    """
    Perform batched 1D convolution over each sample in a batch with separate filters for each.
    
    Args:
        x: Tensor of shape [B, C_in, L]
        h: Tensor of shape [B, C_out, C_in, filter_length]

    Returns:
        Tensor of shape [B, C_out, L]
    """
    B, C_in, L = x.shape
    _, C_out, _, filter_length = h.shape

    # Flip filters to match conv1d behavior (correlation -> convolution)
    h_flipped = h.flip(-1)  # [B, C_out, C_in, filter_length]

    # Pad input at the beginning for causal convolution
    x_padded = F.pad(x, (filter_length - 1, 0))  # [B, C_in, L + filter_length - 1]

    # Reshape for grouped conv1d
    x_reshaped = x_padded.view(1, B * C_in, -1)  # [1, B*C_in, L + filter_length - 1]
    h_reshaped = h_flipped.view(B * C_out, C_in, filter_length)  # [B*C_out, C_in, filter_length]

    # Apply grouped conv1d
    y = fft_conv1d(x_reshaped, h_reshaped, groups=B)  # [1, B*C_out, L]
    y = y.view(B, C_out, -1)  # [B, C_out, L]
    return y

def loop_based_batch_fft_conv1d(x, h):
    B, C_in, L = x.shape
    _, C_out, _, filter_length = h.shape
    output = torch.zeros(B, C_out, L, device=x.device, dtype=x.dtype)
    for b in range(B):
        x_b = x[b:b+1]  # Shape: [1, C_in, L]
        h_b = h[b]      # Shape: [C_out, C_in, filter_length]
        result_b = fft_conv1d(F.pad(x_b, (filter_length - 1, 0)), h_b.flip(-1))
        output[b] = result_b
    return output

