import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

from typing import Union, Optional
from torch_fftconv import fft_conv1d
from torchcomp import db2amp
from time import time



float2param = lambda x: nn.Parameter(
    torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
)

def broadcast2stereo(m, args):
    x, *_ = args
    return x.expand(-1, 2, -1) if x.shape[1] == 1 else x


class FX(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.params = nn.ParameterDict({k: float2param(v) for k, v in kwargs.items()})

    # randomize params values
    def randomize_parameters(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        for k, v in self.params.items():
            self.params[k].data = torch.rand_like(v)


class SmoothingCoef(nn.Module):
    def forward(self, x):
        return x.sigmoid()

    def right_inverse(self, y):
        return (y / (1 - y)).log()


class MinMax(nn.Module):
    def __init__(self, min=0.0, max: Union[float, torch.Tensor] = 1.0):
        super().__init__()
        if isinstance(min, torch.Tensor):
            self.register_buffer("min", min, persistent=False)
        else:
            self.min = min

        if isinstance(max, torch.Tensor):
            self.register_buffer("max", max, persistent=False)
        else:
            self.max = max

        self._m = SmoothingCoef()

    def forward(self, x):
        return self._m(x) * (self.max - self.min) + self.min

    def right_inverse(self, y):
        return self._m.right_inverse((y - self.min) / (self.max - self.min))
    

class UniLossLess(nn.Module):
    def forward(self, x):
        tri = x.triu(1)
        return torch.linalg.matrix_exp(tri - tri.T)
    

class FDN(FX):
    max_delay = 100

    def __init__(
        self,
        sr: int,
        ir_duration: float = 2.0,
        delays=(997, 1153, 1327, 1559, 1801, 2099),
        trainable_delay=False,
        num_decay_freq=49,
        delay_independent_decay=False,
        eq: Optional[nn.Module] = None,
    ):
        # beta = torch.distributions.Beta(1.1, 6)
        num_delays = len(delays)
        super().__init__(
            b=torch.ones(num_delays, 2) / num_delays,
            c=torch.ones(2, num_delays),
            U=torch.randn(num_delays, num_delays) / num_delays**0.5,
            gamma=torch.rand(
                num_decay_freq, num_delays if not delay_independent_decay else 1
            )
            * 0.2
            + 0.4,
            # delays=beta.sample((num_delays,)) * 64,
        )
        self.sr = sr
        self.ir_length = int(sr * ir_duration)

        # ir_duration = T_60
        T_60 = ir_duration * 0.75
        delays = torch.tensor(delays)
        if delay_independent_decay:
            gamma_max = db2amp(-60 / sr / T_60 * delays.min())
        else:
            gamma_max = db2amp(-60 / sr / T_60 * delays)

        register_parametrization(self.params, "gamma", MinMax(0, gamma_max))
        register_parametrization(self.params, "U", UniLossLess())

        # additional codes
        self.num_delays = num_delays
        self.delay_independent_decay = delay_independent_decay
        self.gamma_max = gamma_max
        self.num_decay_freq = num_decay_freq

        if not trainable_delay:
            self.register_buffer(
                "delays",
                delays,
            )
        else:
            self.params["delays"] = nn.Parameter(delays / sr * 1000)
            register_parametrization(self.params, "delays", MinMax(0, self.max_delay))

        self.register_forward_pre_hook(broadcast2stereo)

        self.eq = eq

        # get total size of parameters
        self.num_param = 0
        for k, v in self.params.items():
            self.num_param += v.numel()

    def forward(self, x):
        conv1d = F.conv1d if x.size(-1) > 44100 * 20 else fft_conv1d

        c = self.params.c + 0j
        b = self.params.b + 0j

        gamma = self.params.gamma
        delays = self.delays if hasattr(self, "delays") else self.params.delays

        if gamma.size(0) > 1:
            gamma = F.interpolate(
                gamma.T.unsqueeze(1),
                size=self.ir_length // 2 + 1,
                align_corners=True,
                mode="linear",
            ).transpose(0, 2)

        if gamma.size(2) == 1:
            gamma = gamma ** (delays / delays.min())

        A = self.params.U * gamma

        freqs = (
            torch.arange(self.ir_length // 2 + 1, device=x.device)
            / self.ir_length
            * 2
            * torch.pi
        )
        invD = torch.exp(1j * freqs[:, None] * delays)
        # H = c @ torch.linalg.inv(torch.diag_embed(invD) - A) @ b
        H = c @ torch.linalg.solve(torch.diag_embed(invD) - A, b)

        h = torch.fft.irfft(H.permute(1, 2, 0), n=self.ir_length)

        if self.eq is not None:
            h = self.eq(h)

        # return fft_conv1d(
        return conv1d(
            F.pad(x, (self.ir_length - 1, 0)),
            h.flip(-1),
        )

    def randomize_parameters(self,):
        # call super function first
        super().randomize_parameters()
        # then randomize the registered parameters
        U=torch.randn(self.num_delays, self.num_delays) / self.num_delays**0.5
        gamma=torch.rand(
            self.num_decay_freq, self.num_delays if not self.delay_independent_decay else 1
        )
        # uni = UniLossLess()
        # cur_gamma_max = MinMax(0, self.gamma_max)
        # with torch.no_grad():
        #     self.params.parametrizations['U'].original.copy_(uni(U))
        #     self.params.parametrizations['gamma'].original.copy_(cur_gamma_max(gamma))
        with torch.no_grad():
            self.params.parametrizations['U'].original.copy_(U)
            self.params.parametrizations['gamma'].original.copy_(gamma)


    def process_normalized(self, x, params):
        '''
            Process for number of batch size with given parameters.
            x: (batch_size, channels, time)
            params: (batch_size, num_params)
        '''
        assert x.shape[0]==params.shape[0], "Batch size of x and params must match."
        
        conv1d = F.conv1d if x.size(-1) > 44100 * 20 else fft_conv1d

        outputs = []
        for cur_batch_idx in range(x.shape[0]):
            cur_x = x[cur_batch_idx:cur_batch_idx+1]

            # map params values to class parameters
            cur_param_idx = 0
            cur_params = {}
            for k, v in self.params.items():
                cur_param_num = v.numel()
                if k=='U':
                    uni = UniLossLess()
                    cur_params[k] = uni(params[cur_batch_idx, cur_param_idx:cur_param_idx+cur_param_num].view(v.shape))
                elif k=='gamma':
                    cur_gamma_max = MinMax(0, self.gamma_max)
                    cur_params[k] = cur_gamma_max(params[cur_batch_idx, cur_param_idx:cur_param_idx+cur_param_num].view(v.shape))
                else:
                    cur_params[k] = params[cur_batch_idx, cur_param_idx:cur_param_idx+cur_param_num].view(v.shape)
                cur_param_idx += cur_param_num

            c = cur_params['c'] + 0j
            b = cur_params['b'] + 0j

            gamma = cur_params['gamma']
            delays = self.delays if hasattr(self, "delays") else cur_params['delays']

            if gamma.size(0) > 1:
                gamma = F.interpolate(
                    gamma.T.unsqueeze(1),
                    size=self.ir_length // 2 + 1,
                    align_corners=True,
                    mode="linear",
                ).transpose(0, 2)

            if gamma.size(2) == 1:
                gamma = gamma ** (delays / delays.min())

            A = cur_params['U'] * gamma

            freqs = (
                torch.arange(self.ir_length // 2 + 1, device=x.device)
                / self.ir_length
                * 2
                * torch.pi
            )
            invD = torch.exp(1j * freqs[:, None] * delays)
            # H = c @ torch.linalg.inv(torch.diag_embed(invD) - A) @ b
            H = c @ torch.linalg.solve(torch.diag_embed(invD) - A, b)

            h = torch.fft.irfft(H.permute(1, 2, 0), n=self.ir_length)

            if self.eq is not None:
                h = self.eq(h)

            cur_output = conv1d(
                F.pad(cur_x, (self.ir_length - 1, 0)),
                h.flip(-1),
            )
            outputs.append(cur_output.squeeze(0))

        return torch.stack(outputs, dim=0)





if __name__ == '__main__':

    batch_size = 8

    import soundfile as sf

    sample_path = '/home/tony/mastering_transfer/testing/data_check/precomputed/0/A_in.flac'

    aud_in, sr = sf.read(sample_path)
    aud_in = torch.Tensor(aud_in).transpose(1,0).unsqueeze(0).repeat(batch_size, 1, 1)
    print(f"audio input: {aud_in.shape}   sample rate: {sr}")

    ''' reverb '''
    reverb = FDN(sr=sr, 
                 ir_duration=2.0, 
                 delays=(997, 1153, 1327, 1559, 1801, 2099), 
                 trainable_delay=False, 
                 num_decay_freq=49, 
                 delay_independent_decay=False)
    
    rand_params = torch.rand((batch_size, reverb.num_param))
    aud_out = reverb.process_normalized(aud_in, rand_params)
 