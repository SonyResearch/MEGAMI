

import torch
import torch.nn as nn

from fx_model.diffvox.modules.fx import Peak, LowShelf, HighShelf, LowPass, HighPass, CompressorExpander, SurrogateDelay, FDN, SendFXsAndSum

import time

from dasp_pytorch.functional import stereo_panner
#from torch_fftconv import fft_conv1d

from flamo.processor import dsp, system

from collections import OrderedDict

import torch.nn.functional as F

def fast_apply_RIR(y, filter, rm_delay=False, zero_pad=False):

    if rm_delay:
        filter = filter[ torch.argmax(filter): ]

    #filter = filter.unsqueeze(0).unsqueeze(0)
    B = filter.to(y.device)
    #y = y.unsqueeze(1)
    
    # Get the size of the input signal and filter
    N = y.size(2)
    M = filter.size(2)
    
    # Compute the size of the FFT
    if zero_pad:
        fft_size=torch.tensor(2*N+2*M-1)
    else:
        fft_size=torch.tensor(N+M-1)
    fft_size=int(2**torch.ceil(torch.log2(fft_size)))
    
    # Perform FFT on the input signal and filter
    Y = torch.fft.fft(y, fft_size, dim=2)
    H = torch.fft.fft(B, fft_size, dim=2)
    
    # Perform element-wise multiplication in the frequency domain
    Y_conv = Y * H
    
    # Perform inverse FFT to get the convolution result
    y_conv = torch.fft.ifft(Y_conv, fft_size, dim=2)
    
    # Take the real part of the result
    y_conv = y_conv[:, :, :N].real
    
    # Squeeze the unnecessary dimensions
    y_conv = y_conv

    return y_conv

class FLAMOFDN(nn.Module):
    """
    GRAFx model class for managing and processing graphical data.
    """

    def __init__(self, 
        sample_rate=44100,
        nfft=44100,
        samplerate=44100,
        device='cpu',
        ):
        """
        Initialize the GRAFx model with the provided data.

        :param data: The data to be processed by the GRAFx model.
        """
        super(FLAMOFDN, self).__init__()


        self.sample_rate = sample_rate


        N = 6  # number of delays
        alias_decay_db = 30  # alias decay in dB
        delay_lengths = torch.tensor([593, 743, 929, 1153, 1399, 1699])

        self.ir_length=nfft

        ## ---------------- CONSTRUCT FDN ---------------- ##

        # Input and output gains
        input_gain = dsp.Gain(
            size=(N, 2),
            nfft=nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
        )
        output_gain = dsp.Gain(
            size=(2, N),
            nfft=nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
        )
         
        #  Feedback loop with delays
        delays = dsp.parallelDelay(
            size=(N,),
            max_len=delay_lengths.max(),
            nfft=nfft,
            isint=True,
            requires_grad=False,
            alias_decay_db=alias_decay_db,
            device=device,
        )
         
        delays.assign_value(delays.sample2s(delay_lengths))
        #  Feedback path with orthogonal matrix
        mixing_matrix = dsp.Matrix(
            size=(N, N),
            nfft=nfft,
            matrix_type="orthogonal",
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
        )
         
        attenuation = dsp.parallelGEQ(
            size=(N,),
            octave_interval=1,
            nfft=nfft,
            fs=samplerate,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
        )
         
        attenuation.map = lambda x: 20 * torch.log10(torch.sigmoid(x))
        feedback = system.Series(
            OrderedDict({"mixing_matrix": mixing_matrix, "attenuation": attenuation})
        )
         

        #  Recursion
        feedback_loop = system.Recursion(fF=delays, fB=feedback)

        #  Full FDN
        FDN = system.Series(
            OrderedDict(
                {
                    "input_gain": input_gain,
                    "feedback_loop": feedback_loop,
                    "output_gain": output_gain,
                }
            )
        )
         

        #  Create the model with Shell
        input_layer = dsp.FFT(nfft)
        #  Since time aliasing mitigation is enabled, we use the iFFTAntiAlias layer
        #  to undo the effect of the anti aliasing modulation introduced by the system's layers
        output_layer = dsp.iFFTAntiAlias(
            nfft=nfft, alias_decay_db=alias_decay_db, device=device
        )
         
        self.model = system.Shell(core=FDN, input_layer=input_layer, output_layer=output_layer)

    def forward(self, x):
        """
        Process the data using GRAFx-specific algorithms.
        x: shape (batch_size, num_channels, num_samples)
        """
        # Ensure input is in the correct shape
        h=self.model.get_time_response().permute(0, 2, 1)  # (N, i, ir_length)
        h=h.repeat(x.shape[0], 1, 1)  # Repeat for batch size

        return fast_apply_RIR(x, h, rm_delay=False, zero_pad=False)  # Apply the filter to the input signal

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

        self.PEQ_FDN=torch.nn.Sequential(
           Peak(sr=sample_rate, freq=800,min_freq=200, max_freq=2500, min_Q=0.1, max_Q=3), 
           Peak(sr=sample_rate, freq=4000,min_freq=600, max_freq=7000, min_Q=0.1, max_Q=3), 
           LowShelf(sr=sample_rate, freq=115,min_freq=30, max_freq=450), 
           HighShelf(sr=sample_rate, freq=8000,min_freq=1500, max_freq=16000), 
           LowPass(sr=sample_rate, freq=17500,min_freq=200, max_freq=18000), 
        )


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
            num_decay_freq=1,
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
        x= x * self.pre_gain

        #start= time.time()
        x=self.PEQ(x)
        #print(f"PEQ time: {time.time()-start:.4f} seconds")
        x=self.CompExp(x)
        #print(f"CompExp time: {time.time()-start:.4f} seconds")

        #pan=torch.sigmoid(self.pan).view(-1,1,1).repeat(x.shape[0], 1, 1)  # Ensure pan is in the range [0, 1] and has the correct shape
        #x= stereo_panner(x, sample_rate=self.sample_rate, pan=pan).squeeze(2)

        
        #x_fdn= self.fdn(x)
#
        #x= self.post_gain_wet*x_fdn + self.post_gain_dry*x
        x=self.sendFx(x)


        return x