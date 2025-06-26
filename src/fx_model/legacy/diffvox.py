
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Any, Optional
from torchcomp import compexp_gain, db2amp
from torchcomp import ms2coef, coef2ms, db2amp
from torchlpc import sample_wise_lpc

from dasp_pytorch import Processor

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
    #print("rms", rms)
    gain = compexp_gain(rms, cmp_th, cmp_ratio, exp_th, exp_ratio, at, rt)
    gain = lookahead_func(gain)
    return x * gain * db2amp(make_up).broadcast_to(x.shape[0], 1)

float2param = lambda x: nn.Parameter(
    torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
)

class FX(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.params = nn.ParameterDict({k: float2param(v) for k, v in kwargs.items()})

class CompressorExpander(Processor):
    cmp_ratio_min: float = 1
    cmp_ratio_max: float = 20

    def __init__(
        self,
        sr: int,
        cmp_ratio: float = 2.0,
        exp_ratio: float = 0.5,
        at_ms: float = 50.0,
        rt_ms: float = 50.0,
        avg_coef: float = 0.3,
        cmp_th: float = -18.0,
        exp_th: float = -54.0,
        make_up: float = 0.0,
        delay: int = 0,
        use_lookahead: bool = False,
        max_lookahead: float = 15.0,
    ):
        #super().__init__(
        #    cmp_th=cmp_th,
        #    exp_th=exp_th,
        #    make_up=make_up,
        #    avg_coef=avg_coef,
        #    cmp_ratio=cmp_ratio,
        #    exp_ratio=exp_ratio,
        #)
        super().__init__()
        # deprecated, please use lookahead instead
        self.delay = delay
        self.sr = sr

        self.params["at"] = nn.Parameter(ms2coef(torch.tensor(at_ms), sr))
        self.params["rt"] = nn.Parameter(ms2coef(torch.tensor(rt_ms), sr))

        self.use_lookahead = use_lookahead
        if self.use_lookahead:
            self.params["lookahead"] = nn.Parameter(torch.ones(1) / sr * 1000)
            register_parametrization(
                self.params, "lookahead", WrappedPositive(max_lookahead)
            )
            sinc_length = int(sr * (max_lookahead + 1) * 0.001) + 1
            left_pad_size = int(sr * 0.001)
            self._pad_size = (left_pad_size, sinc_length - left_pad_size - 1)
            self.register_buffer(
                "_arange",
                torch.arange(sinc_length) - left_pad_size,
                persistent=False,
            )
        self.lookahead = lookahead

        register_parametrization(self.params, "at", SmoothingCoef())
        register_parametrization(self.params, "rt", SmoothingCoef())
        register_parametrization(self.params, "avg_coef", SmoothingCoef())
        register_parametrization(
            self.params, "cmp_ratio", MinMax(self.cmp_ratio_min, self.cmp_ratio_max)
        )
        register_parametrization(self.params, "exp_ratio", SmoothingCoef())


    def parameter_size(self):
        r"""
        Returns:
            :python:`Dict[str, Tuple[int, ...]]`: A dictionary that contains each parameter tensor's shape.
        """
        size = {"at": 1, "rt": 1, "avg_coef": 1, "cmp_th": 1, "exp_th": 1, "make_up": 1, "cmp_ratio": 1, "exp_ratio": 1}
        if self.use_lookahead:
            size["lookahead"] = 1

        return size

    def forward(self, inout_signals, at, rt, avg_coef, cmp_tu, exp_th, cmp_ratio, exp_ratio, make_up, lookahead=None):

        if self.lookahead:
            lookahead_in_samples = self.params.lookahead * 0.001 * self.sr
            sinc_filter = torch.sinc(self._arange - lookahead_in_samples)
            lookahead_func = lambda gain: F.conv1d(
                F.pad(
                    gain.view(-1, 1, gain.size(-1)), self._pad_size, mode="replicate"
                ),
                sinc_filter[None, None, :],
            ).view(*gain.shape)
        else:
            lookahead_func = lambda x: x

        return compressor_expander(
            x.reshape(-1, x.shape[-1]),
            lookahead_func=lookahead_func,
            **{k: v for k, v in self.params.items() if k != "lookahead"},
        ).view(*x.shape)
