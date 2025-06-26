
import time

import torch
import torch.nn as nn

from fx_model.diffvox.modules.fx import Peak, LowShelf, HighShelf, LowPass, HighPass, CompressorExpander, SurrogateDelay, FDN, SendFXsAndSum


import torch.nn.functional as F


from fx_model.diffvox.modules.fx import FX, Panning, SmoothingCoef
from torch.nn.utils.parametrize import register_parametrization
from functools import reduce

class MySendFXsAndSum(FX):
    def __init__(self, *args, cross_send=True, pan_direct=False):
        super().__init__(
            **(
                {
                    f"sends_{i}": torch.full([len(args) - i - 1], 0.01)
                    for i in range(len(args) - 1)
                }
                if cross_send
                else {}
            )
        )
        self.effects = nn.ModuleList(args)
        if pan_direct:
            self.pan = Panning()

        if cross_send:
            for i in range(len(args) - 1):
                register_parametrization(self.params, f"sends_{i}", SmoothingCoef())

    def forward(self, x):
        if hasattr(self, "pan"):
            di = self.pan(x)
        else:
            di = x

        if len(self.params) == 0:
            return reduce(
                lambda x, y: x[..., : y.shape[-1]] + y[..., : x.shape[-1]],
                map(lambda f: f(x), self.effects),
                di,
            )

        def f(states, ps):
            x, cum_sends = states
            m, send_gains = ps
            h = m(cum_sends[0])
            return (
                x[..., : h.shape[-1]] + h[..., : x.shape[-1]],
                (
                    None
                    if cum_sends.size(0) == 1
                    else cum_sends[1:, ..., : h.shape[-1]]
                    + send_gains[:, None, None, None] * h[..., : cum_sends.shape[-1]]
                ),
            )

        return reduce(
            f,
            zip(
                self.effects,
                [self.params[f"sends_{i}"] for i in range(len(self.effects) - 1)]
                + [None],
            ),
            (di, x.unsqueeze(0).expand(len(self.effects), -1, -1, -1)),
        )[0]


class MyModel(nn.Module):
    """
    GRAFx model class for managing and processing graphical data.
    """

    def __init__(self, 
        sample_rate=44100,
        device='cpu',
        ):
        """
        Initialize the GRAFx model with the provided data.

        :param data: The data to be processed by the GRAFx model.
        """
        super(MyModel, self).__init__()


        self.sample_rate = sample_rate

        #define EQ
        self.PEQ=torch.nn.Sequential(
           Peak(sr=sample_rate, freq=800,min_freq=33, max_freq=5400), 
           Peak(sr=sample_rate, freq=4000,min_freq=200, max_freq=17500), 
           LowShelf(sr=sample_rate, freq=115,min_freq=30, max_freq=200), 
           HighShelf(sr=sample_rate, freq=6000,min_freq=750, max_freq=8300), 
           LowPass(sr=sample_rate, freq=17500,min_freq=200, max_freq=18000), 
           HighPass(sr=sample_rate, freq=200,min_freq=16, max_freq=5300), 
        )

        #self.PEQ_2=torch.nn.Sequential(
        #   Peak(sr=sample_rate, freq=800,min_freq=33, max_freq=5400), 
        #   Peak(sr=sample_rate, freq=4000,min_freq=200, max_freq=17500), 
        #   LowShelf(sr=sample_rate, freq=115,min_freq=30, max_freq=200), 
        #   HighShelf(sr=sample_rate, freq=6000,min_freq=750, max_freq=8300), 
        #   LowPass(sr=sample_rate, freq=17500,min_freq=200, max_freq=18000), 
        #   HighPass(sr=sample_rate, freq=200,min_freq=16, max_freq=5300), 
        #)

        self.PEQ_FDN=torch.nn.Sequential(
           Peak(sr=sample_rate, freq=800,min_freq=200, max_freq=2500, min_Q=0.1, max_Q=3), 
           Peak(sr=sample_rate, freq=4000,min_freq=600, max_freq=7000, min_Q=0.1, max_Q=3), 
           LowShelf(sr=sample_rate, freq=115,min_freq=30, max_freq=450), 
           HighShelf(sr=sample_rate, freq=8000,min_freq=1500, max_freq=16000), 
           LowPass(sr=sample_rate, freq=17500,min_freq=200, max_freq=18000), 
        )

        #self.CCR= CubicCatmullRomSpline(
        #    mu=10,
        #    G=5,
        #    fix_zero=True
        #)



        self.CompExp=CompressorExpander(
            sr=sample_rate,
            cmp_ratio= 2.0,
            exp_ratio= 0.5,
            at_ms= 50.0,
            rt_ms= 50.0,
            avg_coef= 0.3,
            cmp_th= -18.0,
            exp_th= -48.0,
            make_up= 0.0,
            lookahead= True,
            max_lookahead= 15
            )
        
        #self.delay=SurrogateDelay(
        #    sr=sample_rate,
        #    delay=400,  # in milliseconds
        #    dropout=0.0,  # no dropout
        #    straight_through=True,  # use straight-through estimator
        #    recursive_eq=True,  # use recursive EQ
        #    ir_duration=4,  # impulse response duration in seconds
        #    eq=LowPass(sr=sample_rate, freq=8000, min_freq=200, max_freq=16000, min_Q=0.5, max_Q=2)
        #)


        self.fdn=FDN(
            sr=sample_rate,
            delays=[997, 1153, 1327, 1559, 1801, 2099],  # in samples
            num_decay_freq=49,
            delay_independent_decay= True,  # use delay-independent decay
            ir_duration=12,
            eq=self.PEQ_FDN,  # use the defined EQ for FDN
        )

        #self.fdn=FLAMOFDN(
        #    device="cuda" if torch.cuda.is_available() else "cpu",
        #)

        self.sendFx=MySendFXsAndSum( self.fdn, cross_send=True, pan_direct=True)

        self.pre_gain= nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.post_gain_dry= nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.post_gain_wet= nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.pan= nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x):
        """
        Process the data using GRAFx-specific algorithms.
        x: shape (batch_size, num_channels, num_samples)
        """
        #x= x * self.pre_gain

        start= time.time()
        x=self.PEQ(x)
        #x=self.CCR(x)

        x=self.CompExp(x)

        #x=self.PEQ_2(x)

        #print(f"CompExp time: {time.time()-start:.4f} seconds")

        #pan=torch.sigmoid(self.pan).view(-1,1,1).repeat(x.shape[0], 1, 1)  # Ensure pan is in the range [0, 1] and has the correct shape
        #x= stereo_panner(x, sample_rate=self.sample_rate, pan=pan).squeeze(2)

        
        #x_fdn= self.fdn(x)
#
        #x= self.post_gain_wet*x_fdn + self.post_gain_dry*x
        x=self.sendFx(x)


        return x