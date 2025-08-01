
import torch

from utils.distributions import Uniform, Normal, unilossless, NormalRamp, LogUniform, UniformRamp
import torchcomp

T60_max = 4  # maximum reverberation time in seconds. It cannot be longer because the FFT length of the FDN is 4s, and longer RIRs may cause temporal aliasing, which sounds bad.

def T60togamma(T_60, sample_rate: int = 44100):
    gamma = -60 / sample_rate / T_60 * 997
    gamma= 10 ** (gamma / 20)
    return gamma


def get_distributions_uniform(sample_rate: int = 44100):
    """
    Returns the distributions for the parameters of the model.
    """

    distributions_PEQ = {
        "peak1_gain": Uniform(low=-10.0, high=10, shape=(1,)),  # Lower gain for a more subtle effect
        "peak1_freq": LogUniform(low=33, high=5400, shape=(1,), transformation=lambda x: torch.clamp(x, min=33, max=5400)),  # Focus on lower mids
        "peak1_Q": Uniform(low=0.21, high=19.99, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.2, max=20)),  # Slightly broader Q for warmth
    
        "peak2_gain": Uniform(low=-10.0, high=10.0, shape=(1,)),  # Lower gain for a more natural sound
        "peak2_freq": LogUniform(low=201, high=17499, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=17500)),  # Focus on upper mids
        "peak2_Q": Uniform(low=0.21, high=19.99, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.2, max=20)),  # Standard Q for balance
    
        "lowshelf_gain": Uniform(low=-10.0, high=10, shape=(1,)),  # Less aggressive cut for lows
        "lowshelf_freq": LogUniform(low=31, high=199, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=200)),  # Focus on lower frequencies
    
        "highshelf_gain": Uniform(low=-10.0, high=10, shape=(1,)),  # Moderate boost for highs
        "highshelf_freq": LogUniform(low=751, high=8299, shape=(1,), transformation=lambda x: torch.clamp(x, min=750, max=8300)),  # Focus on warmth rather than brightness
    
        "lowpass_freq": LogUniform(low=201, high=17999, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=18000)),  # Slightly lower cutoff for a warmer sound
        "lowpass_Q": Uniform(low=0.51, high=9.9, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.5, max=10)),  # Standard Q for balance
    
        "highpass_freq": LogUniform(low=17, high=5299, shape=(1,), transformation=lambda x: torch.clamp(x, min=16, max=5300)),  # Lower cutoff for retaining warmth
        "highpass_Q": Uniform(low=0.51, high=9.9, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.5, max=10)),  # Standard Q for balance
    }

    #strong compression with fast attack and release
    distributions_CompExp = {
        "comp_ratio": Uniform(low=1.01, high=20, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # compression ratio
        "exp_ratio": Uniform(low=0.05, high=0.95, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.001, max=0.999)),
        "at_coef": LogUniform(low=0.1, high=300, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # attack time in milliseconds
        "rt_coef": LogUniform(low=5.0, high=1500.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # release time in milliseconds
        "avg_coef": Uniform(low=0.5, high=0.995, shape=(1,)),  # average coefficient
        "make_up": Uniform(low=-6.0, high=12.0, shape=(1,)),  # make-up gain in dB
        "comp_th": Uniform(low=-30, high=-5, shape=(1,)),  # compression threshold in dB
        "exp_th": Uniform(low=-50.0, high=-15, shape=(1,)),  # expansion threshold in dB
        "lookahead": LogUniform(low=0.1, high=15, shape=(1,)),  # lookahead in milliseconds
    }

    distributions_FDN = {
        "b": Normal(mean=0, std=1, shape=(6,2), transformation=lambda x: torch.clamp(x, min=-1/(6**0.5) , max= 1/(6**0.5)  )),  # in milliseconds (use uniform distribution in +- 1/sqrt(6) range)
        "c": Normal(mean=0, std=1, shape=(2,6),  transformation=lambda x: torch.clamp(x, min=-1/(6**0.5) , max= 1/(6**0.5)  )),  # in milliseconds
        "U": Normal(mean=0, std=(1/(6**0.5)), shape=(6,6), transformation= lambda x:unilossless(x) ),  # in milliseconds
        "gamma": UniformRamp(low_low=0.1, low_high=0.01, high_low=2, high_high=0.8, log=True, shape=(49, 1), transformation= lambda x: T60togamma(torch.clamp(x, min=0, max=T60_max), sample_rate=sample_rate)),  # decay frequencies
    }

    distributions_PEQ_FDN = {
        "peak1_gain": Uniform(low=-10.0, high=10, shape=(1,)),  # Lower gain for a more subtle effect
        "peak1_freq": Uniform(low=201, high=2499, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=2500)),
        "peak1_Q": Uniform(low=0.101, high=2.99, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.1, max=3)),
    
        "peak2_gain": Uniform(low=-10.0, high=10.0, shape=(1,)),  # Lower gain for a more natural sound
        "peak2_freq": Uniform(low=601, high=6999, shape=(1,), transformation=lambda x: torch.clamp(x, min=600, max=7000)),
        "peak2_Q": Uniform(low=0.101, high=2.99, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.1, max=3)),
    
        "lowshelf_gain": Uniform(low=-10.0, high=10.0, shape=(1,)),  # Lower gain for a more natural sound
        "peak2_freq": Uniform(low=601, high=6999, shape=(1,), transformation=lambda x: torch.clamp(x, min=600, max=7000)),
        "lowshelf_freq": Uniform(low=31, high=449, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=450)),
    
        "highshelf_gain": Uniform(low=-10.0, high=10.0, shape=(1,)),  # Lower gain for a more natural sound
        "highshelf_freq": Uniform(low=1501, high=15999, shape=(1,), transformation=lambda x: torch.clamp(x, min=1500, max=16000)),
    }

    distributions_pan = {

        "pan_param": Uniform(low=0, high=1, shape=(1,), transformation=lambda x: torch.clamp(x, 0, 1) ),  # Panning parameter between 0 and 1
    }

    distributions_RMSnorm = {
        "RMSnorm": Normal(mean=-25, std=0, shape=(1,),transformation=lambda x: torch.clamp(x, -40, -10) ),  # Panning parameter between 0 and 1
    }

    return {
        "PEQ": distributions_PEQ,
        "CompExp": distributions_CompExp,
        "FDN": distributions_FDN,
        "PEQ_FDN": distributions_PEQ_FDN,
        "pan": distributions_pan,
        "RMSnorm": distributions_RMSnorm,
    }

