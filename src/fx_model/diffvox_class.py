

import torch
import torch.nn as nn

from fx_model.diffvox.modules.fx import Peak, LowShelf, HighShelf, LowPass, HighPass, CompressorExpander, SurrogateDelay, FDN, SendFXsAndSum


class DiffVox(nn.Module):
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
        super(DiffVox, self).__init__()


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
        
        self.delay=SurrogateDelay(
            sr=sample_rate,
            delay=400,  # in milliseconds
            dropout=0.0,  # no dropout
            straight_through=True,  # use straight-through estimator
            recursive_eq=True,  # use recursive EQ
            ir_duration=4,  # impulse response duration in seconds
            eq=LowPass(sr=sample_rate, freq=8000, min_freq=200, max_freq=16000, min_Q=0.5, max_Q=2)
        )


        self.fdn=FDN(
            sr=sample_rate,
            delays=[997, 1153, 1327, 1559, 1801, 2099],  # in samples
            num_decay_freq=49,
            delay_independent_decay= True,  # use delay-independent decay
            ir_duration=12,
            eq=self.PEQ_FDN,  # use the defined EQ for FDN
        )

        self.sendFx=SendFXsAndSum(self.delay, self.fdn, cross_send=True, pan_direct=True)


    def forward(self, x):
        """
        Process the data using GRAFx-specific algorithms.
        x: shape (batch_size, num_channels, num_samples)
        """

        x=self.PEQ(x)
        x=self.CompExp(x)
        x=self.sendFx(x)

        

        return x