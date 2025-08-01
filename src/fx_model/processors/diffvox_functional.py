import torch
import torch.nn.functional as F
from torchcomp import compexp_gain, db2amp
from torchlpc import sample_wise_lpc
from typing import List, Tuple, Union, Any, Optional
import math


def inv_22(a, b, c, d):
    return torch.stack([d, -b, -c, a]).view(2, 2) / (a * d - b * c)


def eig_22(a, b, c, d):
    # https://croninprojects.org/Vince/Geodesy/FindingEigenvectors.pdf
    T = a + d
    D = a * d - b * c
    half_T = T * 0.5
    root = torch.sqrt(half_T * half_T - D)  # + 0j)
    L = torch.stack([half_T + root, half_T - root])

    y = (L - a) / b
    # y = c / L
    V = torch.stack([torch.ones_like(y), y])
    return L, V / V.abs().square().sum(0).sqrt()


def fir(x, b):
    padded = F.pad(x.reshape(-1, 1, x.size(-1)), (b.size(0) - 1, 0))
    return F.conv1d(padded, b.flip(0).view(1, 1, -1)).view(*x.shape)


def allpole(x: torch.Tensor, a: torch.Tensor):
    h = x.reshape(-1, x.shape[-1])
    return sample_wise_lpc(
        h,
        a.broadcast_to(h.shape + a.shape),
    ).reshape(*x.shape)

def biquad_parallel(x: torch.Tensor, b0, b1, b2, a0, a1, a2):

    B,Ch, T = x.shape

    if Ch>1:
        #handle the multi-channel case. We just assume that the coefficients are the same for all channels
        b0 = b0.view(-1, 1, 1).repeat(1, Ch, 1).view(-1, 1)
        b1 = b1.view(-1, 1, 1).repeat(1, Ch, 1).view(-1, 1)
        b2= b2.view(-1, 1, 1).repeat(1, Ch, 1).view(-1, 1)
        a0 = a0.view(-1, 1, 1).repeat(1, Ch, 1).view(-1,1)
        a1 = a1.view(-1, 1, 1).repeat(1, Ch, 1).view(-1,1)
        a2 = a2.view(-1, 1, 1).repeat(1, Ch, 1).view(-1,1)
        x = x.view(-1,1, T)

    b0 = b0 / a0
    try:
        b1 = b1 / a0
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
    b2 = b2 / a0
    a1 = a1 / a0
    a2 = a2 / a0

    beta1 = b1 - b0 * a1
    beta2 = b2 - b0 * a2

    beta1 = beta1.view(-1, 1, 1)
    beta2 = beta2.view(-1, 1, 1)

    b0 = b0.view(-1, 1, 1)

    tmp = a1.square() - 4 * a2

    sep= (tmp >= 0).squeeze(-1)

    h= torch.zeros_like(x[..., :-1])


    if (sep).any():
        indexes = torch.where(sep)[0]

        # TODO: parallelize this part

        for i in indexes:
            #print("Processing index:", i)
            a1_i= a1[i].view(-1)
            a2_i= a2[i].view(-1)
            beta1_i = beta1[i].view(-1)
            beta2_i = beta2[i].view(-1)
            

            L, V = eig_22(-a1_i, -a2_i, torch.ones_like(a1_i), torch.zeros_like(a1_i))
            #L.shape torch.Size([2]) V.shape torch.Size([2, 2])
            L=L.squeeze(-1)
            V = V.squeeze(-1)

            inv_V = inv_22(*V.view(-1))
            #inv_V.shape torch.Size([2, 2])
    
            C = torch.stack([beta1_i, beta2_i]).squeeze(-1) @ V #shape: torch.Size([2])
    
            # project input to eigen space
            x_i = x[i]
            h_i = x_i[..., :-1].unsqueeze(-2) * inv_V[:, :1]
            L = L.unsqueeze(-1).broadcast_to(h_i.shape)
    
            h[i] = (
                sample_wise_lpc(h_i.reshape(-1, h_i.shape[-1]), -L.reshape(-1, L.shape[-1], 1))
                .reshape(*h_i.shape)
                .transpose(-2, -1)
            ) @ C
    
    if (~sep).any():

        pole = 0.5 * (-a1[~sep] + 1j * torch.sqrt(-tmp[~sep]))
        pole=pole.view(-1, 1, 1)
        x_i = x[~sep]
        u = -1j * x_i[..., :-1]
        h_i = sample_wise_lpc(
            u.reshape(-1, u.shape[-1]),
            -pole.broadcast_to(u.shape).reshape(-1, u.shape[-1], 1),
        ).reshape(*u.shape)
        h[~sep] = (
            h_i.real * (beta1[~sep] * pole.real / pole.imag + beta2[~sep] / pole.imag)
            - beta1[~sep] * h_i.imag
        )

    tmp = b0 * x
    y = torch.cat([tmp[..., :1], h + tmp[..., 1:]], -1)
    return y.view(B, Ch, T) 

def biquad(x: torch.Tensor, b0, b1, b2, a0, a1, a2):
    b0 = b0 / a0
    b1 = b1 / a0
    b2 = b2 / a0
    a1 = a1 / a0
    a2 = a2 / a0

    beta1 = b1 - b0 * a1
    beta2 = b2 - b0 * a2

    tmp = a1.square() - 4 * a2
    if tmp < 0:
        #print("tmp is negative, using pole-zero decomposition")
        pole = 0.5 * (-a1 + 1j * torch.sqrt(-tmp))
        u = -1j * x[..., :-1]
        h = sample_wise_lpc(
            u.reshape(-1, u.shape[-1]),
            -pole.broadcast_to(u.shape).reshape(-1, u.shape[-1], 1),
        ).reshape(*u.shape)
        h = (
            h.real * (beta1 * pole.real / pole.imag + beta2 / pole.imag)
            - beta1 * h.imag
        )
    else:
        #print("tmp is positive, using eigen decomposition")
        L, V = eig_22(-a1, -a2, torch.ones_like(a1), torch.zeros_like(a1))
        inv_V = inv_22(*V.view(-1))

        C = torch.stack([beta1, beta2]) @ V 

        # project input to eigen space
        h = x[..., :-1].unsqueeze(-2) * inv_V[:, :1]
        L = L.unsqueeze(-1).broadcast_to(h.shape)

        h = (
            sample_wise_lpc(h.reshape(-1, h.shape[-1]), -L.reshape(-1, L.shape[-1], 1))
            .reshape(*h.shape)
            .transpose(-2, -1)
        ) @ C
    tmp = b0 * x
    y = torch.cat([tmp[..., :1], h + tmp[..., 1:]], -1)
    return y


def highpass_biquad_coef(
    sample_rate: int,
    cutoff_freq: torch.Tensor,
    Q: torch.Tensor,
):
    w0 = 2 * torch.pi * cutoff_freq / sample_rate
    alpha = torch.sin(w0) / 2.0 / Q

    b0 = (1 + torch.cos(w0)) / 2
    b1 = -1 - torch.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha
    return b0, b1, b2, a0, a1, a2


def apply_biquad(bq, parallel=False):
    if not parallel:
        return lambda waveform, *args, **kwargs: biquad(waveform, *bq(*args, **kwargs))
    else:
        #def apply_biquad_parallel(waveform, *args, **kwargs):
        #    b0, b1, b2, a0, a1, a2 = bq(*args, **kwargs)
        #    return torch.cat(
        #        [
        #            biquad(waveform[i].unsqueeze(0), b0[i], b1[i], b2[i], a0[i], a1[i], a2[i])
        #            for i in range(b0.shape[0])
        #        ],
        #        dim=0,
        #    )
        #return apply_biquad_parallel

        #for some reaason, the above version is faster
        return lambda waveform, *args, **kwargs: biquad_parallel(waveform, *bq(*args, **kwargs))


highpass_biquad = apply_biquad(highpass_biquad_coef)


def lowpass_biquad_coef(
    sample_rate: int,
    cutoff_freq: torch.Tensor,
    Q: torch.Tensor,
):
    w0 = 2 * torch.pi * cutoff_freq / sample_rate
    alpha = torch.sin(w0) / 2 / Q

    b0 = (1 - torch.cos(w0)) / 2
    b1 = 1 - torch.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha
    return b0, b1, b2, a0, a1, a2


def equalizer_biquad_coef(
    sample_rate: int,
    center_freq: torch.Tensor,
    gain: torch.Tensor,
    Q: torch.Tensor,
):

    w0 = 2 * torch.pi * center_freq / sample_rate
    A = torch.exp(gain / 40.0 * math.log(10))
    alpha = torch.sin(w0) / 2 / Q

    b0 = 1 + alpha * A
    b1 = -2 * torch.cos(w0)
    b2 = 1 - alpha * A

    a0 = 1 + alpha / A
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha / A
    return b0, b1, b2, a0, a1, a2


def lowshelf_biquad_coef(
    sample_rate: int,
    cutoff_freq: torch.Tensor,
    gain: torch.Tensor,
    Q: torch.Tensor,
):

    w0 = 2 * torch.pi * cutoff_freq / sample_rate
    A = torch.exp(gain / 40.0 * math.log(10))
    alpha = torch.sin(w0) / 2 / Q
    cosw0 = torch.cos(w0)
    sqrtA = torch.sqrt(A)

    b0 = A * (A + 1 - (A - 1) * cosw0 + 2 * alpha * sqrtA)
    b1 = 2 * A * (A - 1 - (A + 1) * cosw0)
    b2 = A * (A + 1 - (A - 1) * cosw0 - 2 * alpha * sqrtA)

    a0 = A + 1 + (A - 1) * cosw0 + 2 * alpha * sqrtA
    a1 = -2 * (A - 1 + (A + 1) * cosw0)
    a2 = A + 1 + (A - 1) * cosw0 - 2 * alpha * sqrtA

    return b0, b1, b2, a0, a1, a2


def highshelf_biquad_coef(
    sample_rate: int,
    cutoff_freq: torch.Tensor,
    gain: torch.Tensor,
    Q: torch.Tensor,
):

    w0 = 2 * torch.pi * cutoff_freq / sample_rate
    A = torch.exp(gain / 40.0 * math.log(10))
    alpha = torch.sin(w0) / 2 / Q
    cosw0 = torch.cos(w0)
    sqrtA = torch.sqrt(A)

    b0 = A * (A + 1 + (A - 1) * cosw0 + 2 * alpha * sqrtA)
    b1 = -2 * A * (A - 1 + (A + 1) * cosw0)
    b2 = A * (A + 1 + (A - 1) * cosw0 - 2 * alpha * sqrtA)

    a0 = A + 1 - (A - 1) * cosw0 + 2 * alpha * sqrtA
    a1 = 2 * (A - 1 - (A + 1) * cosw0)
    a2 = A + 1 - (A - 1) * cosw0 - 2 * alpha * sqrtA

    return b0, b1, b2, a0, a1, a2


highpass_biquad = apply_biquad(highpass_biquad_coef)
lowpass_biquad = apply_biquad(lowpass_biquad_coef)
highshelf_biquad = apply_biquad(highshelf_biquad_coef)
lowshelf_biquad = apply_biquad(lowshelf_biquad_coef)
equalizer_biquad = apply_biquad(equalizer_biquad_coef)

highpass_biquad_parallel = apply_biquad(highpass_biquad_coef, parallel=True)
lowpass_biquad_parallel = apply_biquad(lowpass_biquad_coef, parallel=True)
highshelf_biquad_parallel = apply_biquad(highshelf_biquad_coef, parallel=True)
lowshelf_biquad_parallel = apply_biquad(lowshelf_biquad_coef, parallel=True)
equalizer_biquad_parallel = apply_biquad(equalizer_biquad_coef, parallel=True)


def avg(rms: torch.Tensor, avg_coef: torch.Tensor):
    assert torch.all(avg_coef > 0) and torch.all(avg_coef <= 1)

    h = rms * avg_coef

    return sample_wise_lpc(
        h,
        (avg_coef - 1).broadcast_to(h.shape).unsqueeze(-1),
    )


def avg_rms(audio: torch.Tensor, avg_coef) -> torch.Tensor:
    return avg(audio.square().clamp_min(1e-8), avg_coef).sqrt()


def compressor_expander(
    x: torch.Tensor,
    avg_coef: Union[torch.Tensor, float],
    cmp_th: Union[torch.Tensor, float],
    cmp_ratio: Union[torch.Tensor, float],
    exp_th: Union[torch.Tensor, float],
    exp_ratio: Union[torch.Tensor, float],
    at: Union[torch.Tensor, float],
    rt: Union[torch.Tensor, float],
    make_up: torch.Tensor,
    lookahead_func=lambda x: x,
):
    #print("avg_coef",avg_coef)
    #print("is there a nan in x?", torch.isnan(x).any())
    rms = avg_rms(x, avg_coef=avg_coef)
    if (rms < 0).any():
        print("nan in x:", torch.isnan(x).any())
        print("avg_coef", avg_coef)
        print("nan in rms:", torch.isnan(rms).any())
        raise ValueError("RMS value is negative, which is not expected.")
    
    try:
        assert torch.all(rms > 0)
    except AssertionError:
        print("RMS values:", rms)
        print("nan in rms",torch.isnan(rms).any())
        raise

    gain = compexp_gain(rms, cmp_th, cmp_ratio, exp_th, exp_ratio, at, rt)
    gain = lookahead_func(gain)
    return x * gain * db2amp(make_up).broadcast_to(x.shape[0], 1)
