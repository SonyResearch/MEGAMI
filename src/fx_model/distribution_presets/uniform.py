
import torch

from utils.distributions import Uniform, Normal, unilossless, NormalRamp, LogUniform
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
        "peak1_gain": Uniform(low=-12.0, max=12, shape=(1,)),  # Lower gain for a more subtle effect
        "peak1_freq": LogUniform(low=33, max=5400, shape=(1,), transformation=lambda x: torch.clamp(x, min=33, max=5400)),  # Focus on lower mids
        "peak1_Q": Uniform(low=0.21, max=19.99, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.2, max=20)),  # Slightly broader Q for warmth
    
        "peak2_gain": Uniform(low=-12.0, max=12.0, shape=(1,)),  # Lower gain for a more natural sound
        "peak2_freq": LogUniform(low=201, max=17499, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=17500)),  # Focus on upper mids
        "peak2_Q": Uniform(low=0.21, max=19.99, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.2, max=20)),  # Standard Q for balance
    
        "lowshelf_gain": Uniform(low=-18.0, max=18, shape=(1,)),  # Less aggressive cut for lows
        "lowshelf_freq": LogUniform(low=31, max=199, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=200)),  # Focus on lower frequencies
    
        "highshelf_gain": Uniform(low=-18.0, max=18, shape=(1,)),  # Moderate boost for highs
        "highshelf_freq": LogUniform(low=751, max=8299, shape=(1,), transformation=lambda x: torch.clamp(x, min=750, max=8300)),  # Focus on warmth rather than brightness
    
        "lowpass_freq": LogUniform(low=201, high=17999, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=18000)),  # Slightly lower cutoff for a warmer sound
        "lowpass_Q": Uniform(low=0.51, max=9.9, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.5, max=10)),  # Standard Q for balance
    
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
        "comp_th": Uniform(mean=-30, high=-5, shape=(1,)),  # compression threshold in dB
        "exp_th": Uniform(mean=-50.0, std=-15, shape=(1,)),  # expansion threshold in dB
        "lookahead": LogUniform(low=0.1, high=15, shape=(1,)),  # lookahead in milliseconds
    }

    distributions_FDN = {
        "b": Normal(mean=1/6, std=0.01, shape=(6,2)),  # in milliseconds
        "c": Normal(mean=1, std=0.01, shape=(2,6)),  # in milliseconds
        "U": Normal(mean=0, std=(1/(6**0.5)), shape=(6,6), transformation= lambda x:unilossless(x) ),  # in milliseconds
        "gamma": NormalRamp(mean_low=1.5, mean_high=0.1, std_low=0.5, std_high=0.1, log=True, shape=(49, 1), transformation= lambda x: T60togamma(torch.clamp(x, min=0, max=T60_max), sample_rate=sample_rate)),  # decay frequencies
    }

    distributions_PEQ_FDN = {
        "peak1_gain": Normal(mean=0.0, std=2, shape=(1,)),
        "peak1_freq": Normal(mean=1000, std=100, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=2500)),
        "peak1_Q": Normal(mean=0.707, std=0.3, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),
    
        "peak2_gain": Normal(mean=0.0, std=2, shape=(1,)),
        "peak2_freq": Normal(mean=3000, std=100, shape=(1,), transformation=lambda x: torch.clamp(x, min=600, max=7000)),
        "peak2_Q": Normal(mean=0.707, std=0.3, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.6, max=20)),
    
        "lowshelf_gain": Normal(mean=1.0, std=2, shape=(1,)),
        "lowshelf_freq": Normal(mean=200, std=50, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=450)),
    
        "highshelf_gain": Normal(mean=0.0, std=2, shape=(1,)),
        "highshelf_freq": Normal(mean=8000, std=1000, shape=(1,), transformation=lambda x: torch.clamp(x, min=1500, max=16000)),
    }

    distributions_pan = {

        "pan_param": Uniform(low=0.25, high=0.75, shape=(1,), transformation=lambda x: torch.clamp(x, 0, 1) ),  # Panning parameter between 0 and 1
    }

    distributions_RMSnorm = {
        "RMSnorm": Normal(mean=-25, std=3, shape=(1,),transformation=lambda x: torch.clamp(x, -40, -10) ),  # Panning parameter between 0 and 1
    }

    return {
        "PEQ": distributions_PEQ,
        "CompExp": distributions_CompExp,
        "FDN": distributions_FDN,
        "PEQ_FDN": distributions_PEQ_FDN,
        "pan": distributions_pan,
        "RMSnorm": distributions_RMSnorm,
    }

