
import torch

from utils.distributions import Uniform, Normal, unilossless, NormalRamp
import torchcomp

T60_max = 4  # maximum reverberation time in seconds. It cannot be longer because the FFT length of the FDN is 4s, and longer RIRs may cause temporal aliasing, which sounds bad.

def T60togamma(T_60, sample_rate: int = 44100):
    gamma = -60 / sample_rate / T_60 * 997
    gamma= 10 ** (gamma / 20)
    return gamma

def get_distributions_Cluster1_vocals(sample_rate: int = 44100):
    """
    Returns the distributions for the parameters of the model.
    """

    distributions_PEQ = {
        "peak1_gain": Normal(mean=4.0, std=2, shape=(1,)),  # Lower gain for a more subtle effect
        "peak1_freq": Normal(mean=800, std=200, shape=(1,), transformation=lambda x: torch.clamp(x, min=33, max=5400)),  # Focus on lower mids
        "peak1_Q": Normal(mean=1.2, std=0.3, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Slightly broader Q for warmth
    
        "peak2_gain": Normal(mean=4.0, std=3, shape=(1,)),  # Lower gain for a more natural sound
        "peak2_freq": Normal(mean=2500, std=800, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=17500)),  # Focus on upper mids
        "peak2_Q": Normal(mean=0.707, std=0.3, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Standard Q for balance
    
        "lowshelf_gain": Normal(mean=-3.0, std=3, shape=(1,)),  # Less aggressive cut for lows
        "lowshelf_freq": Normal(mean=150, std=50, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=200)),  # Focus on lower frequencies
    
        "highshelf_gain": Normal(mean=4.0, std=3, shape=(1,)),  # Moderate boost for highs
        "highshelf_freq": Normal(mean=6000, std=800, shape=(1,), transformation=lambda x: torch.clamp(x, min=2000, max=17500)),  # Focus on warmth rather than brightness
    
        "lowpass_freq": Normal(mean=14000, std=1000, shape=(1,), transformation=lambda x: torch.clamp(x, min=2000, max=18000)),  # Slightly lower cutoff for a warmer sound
        "lowpass_Q": Normal(mean=0.707, std=0.3, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=10)),  # Standard Q for balance
    
        "highpass_freq": Normal(mean=100, std=30, shape=(1,), transformation=lambda x: torch.clamp(x, min=20, max=300)),  # Lower cutoff for retaining warmth
        "highpass_Q": Normal(mean=0.707, std=0.3, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=10)),  # Standard Q for balance
    }

    #strong compression with fast attack and release
    distributions_CompExp = {
        "comp_ratio": Normal(mean=1.2, std=0.5, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # compression ratio
        "exp_ratio": Normal(mean=0.5, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.001, max=0.999)),
        "at_coef": Normal(mean=20.0, std=5.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # attack time in milliseconds
        "rt_coef": Normal(mean=75.0, std=15.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # release time in milliseconds
        "avg_coef": Normal(mean=0.9, std=0.01, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.001, max=0.999)),  # Extremely high average coefficient for maximum control
        "make_up": Normal(mean=3.0, std=1.0, shape=(1,)),  # make-up gain in dB
        "comp_th": Normal(mean=-5, std=2.5, shape=(1,)),  # compression threshold in dB
        "exp_th": Normal(mean=-54.0, std=3.0, shape=(1,)),  # expansion threshold in dB
        "lookahead": Uniform(low=0.1, high=5, shape=(1,)),  # lookahead in milliseconds
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

def get_distributions_Cluster0_vocals(sample_rate: int = 44100):
    """
    Returns the distributions for the parameters of the model.
    """
    distributions_PEQ = {
        "peak1_gain": Normal(mean=8.0, std=2, shape=(1,)),
        "peak1_freq": Normal(mean=2000, std=500, shape=(1,), transformation=lambda x: torch.clamp(x, min=33, max=5400)),
        "peak1_Q": Normal(mean=1.0, std=0.3, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.6, max=20)),
    
        "peak2_gain": Normal(mean=8.0, std=5, shape=(1,)),
        "peak2_freq": Normal(mean=4000, std=1000, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=17500)),
        "peak2_Q": Normal(mean=0.707, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.6, max=20)),
    
        "lowshelf_gain": Normal(mean=-10.0, std=4, shape=(1,)),
        "lowshelf_freq": Normal(mean=200, std=50, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=200)),
    
        "highshelf_gain": Normal(mean=10.0, std=4, shape=(1,)),
        "highshelf_freq": Normal(mean=8000, std=1000, shape=(1,), transformation=lambda x: torch.clamp(x, min=2000, max=17500)),
    
        "lowpass_freq": Normal(mean=16000, std=1000, shape=(1,), transformation=lambda x: torch.clamp(x, min=2000, max=18000)),
        "lowpass_Q": Normal(mean=0.707, std=0.3, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.6, max=10)),
    
        "highpass_freq": Normal(mean=150, std=50, shape=(1,), transformation=lambda x: torch.clamp(x, min=20, max=300)),
        "highpass_Q": Normal(mean=0.707, std=0.3, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.6, max=10)),
    }

    #strong compression with fast attack and release

    distributions_CompExp = {
        "comp_ratio": Normal(mean=7.0, std=1, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # compression ratio
        "exp_ratio": Normal(mean=0.5, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.001, max=0.999)),
        "at_coef": Normal(mean=5.0, std=2, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x,min=0.01), sample_rate)),  # attack time in milliseconds
        "rt_coef": Normal(mean=25.0, std=10.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # release time in milliseconds
        "avg_coef": Normal(mean=0.98, std=0.01, shape=(1,), transformation=lambda x: torch.clamp(x, 0.01, 0.99)),  # average coefficient
        "make_up": Normal(mean=6.0, std=1.0, shape=(1,)),  # make-up gain in dB
        "comp_th": Normal(mean=-20.0, std=3.0, shape=(1,)),  # compression threshold in dB
        "exp_th": Normal(mean=-54.0, std=3.0, shape=(1,)),  # expansion threshold in dB
        "lookahead": Uniform(low=0.1, high=5, shape=(1,)),  # lookahead in milliseconds
    }

    distributions_FDN = {
        "b": Normal(mean=1/6, std=0.01, shape=(6,2)),  # in milliseconds
        "c": Normal(mean=1, std=0.01, shape=(2,6)),  # in milliseconds
        "U": Normal(mean=0, std=(1/6**0.5), shape=(6,6), transformation= lambda x:unilossless(x) ),  # in milliseconds
        "gamma": NormalRamp(mean_low=0.5, mean_high=0.1, std_low=0.3, std_high=0.01, log=True, shape=(49, 1), transformation= lambda x: T60togamma(torch.clamp(x, min=0, max=T60_max), sample_rate=sample_rate)),  # decay frequencies
    }

    distributions_PEQ_FDN = {
        "peak1_gain": Normal(mean=0.0, std=2, shape=(1,)),
        "peak1_freq": Normal(mean=1000, std=500, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=2500)),
        "peak1_Q": Normal(mean=0.707, std=0.3, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.6, max=20)),
    
        "peak2_gain": Normal(mean=0.0, std=2, shape=(1,)),
        "peak2_freq": Normal(mean=4000, std=1000, shape=(1,), transformation=lambda x: torch.clamp(x, min=600, max=7000)),
        "peak2_Q": Normal(mean=0.707, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.6, max=20)),
    
        "lowshelf_gain": Normal(mean=0.0, std=2, shape=(1,)),
        "lowshelf_freq": Normal(mean=200, std=50, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=450)),
    
        "highshelf_gain": Normal(mean=1.0, std=2, shape=(1,)),
        "highshelf_freq": Normal(mean=8000, std=1000, shape=(1,), transformation=lambda x: torch.clamp(x, min=1500, max=16000)),
    }

    #distributions_PEQ_FDN = {
    #    "freq": Uniform(20, 20000),
    #    "gain": Normal(0, 10),
    #    "Q": Uniform(0.1, 10),
    #}

    distributions_pan = {
        "pan_param": Uniform(low=0.45, high=0.55, shape=(1,),transformation=lambda x: torch.clamp(x, 0, 1) ),  # Panning parameter between 0 and 1
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
def get_distributions_Cluster0_drums(sample_rate: int = 44100):
    """
    Returns the distributions for drum processing parameters with:
    - EQ: Boosted highs and lows, aggressive mid cut
    - Compression: Heavy with fast attack/release
    - Reverb: Short to medium, bright with enhanced early reflections
    """

    distributions_PEQ = {
        # Mid cut - first peak focused on lower mids
        "peak1_gain": Normal(mean=-6.0, std=2, shape=(1,)),  # Aggressive cut in lower mids
        "peak1_freq": Normal(mean=400, std=100, shape=(1,), transformation=lambda x: torch.clamp(x, min=33, max=5400)),  # Lower mid frequency
        "peak1_Q": Normal(mean=1.0, std=0.2, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Moderate Q for broader cut
        
        # Mid cut - second peak focused on upper mids
        "peak2_gain": Normal(mean=-6.0, std=2, shape=(1,)),  # Aggressive cut in upper mids
        "peak2_freq": Normal(mean=1200, std=200, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=17500)),  # Upper mid frequency
        "peak2_Q": Normal(mean=1.2, std=0.3, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Moderate Q for focused cut
        
        # Low boost
        "lowshelf_gain": Normal(mean=8.0, std=2, shape=(1,)),  # Significant low boost
        "lowshelf_freq": Normal(mean=120, std=30, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=200)),  # Bass frequency
        
        # High boost
        "highshelf_gain": Normal(mean=6.0, std=2, shape=(1,)),  # Significant high boost
        "highshelf_freq": Normal(mean=5000, std=500, shape=(1,), transformation=lambda x: torch.clamp(x, min=2000, max=17500)),  # Presence/air frequency
        
        # Low-pass to control extreme highs
        "lowpass_freq": Normal(mean=18000, std=500, shape=(1,), transformation=lambda x: torch.clamp(x, min=16000, max=18000)),  # Allow most highs through
        "lowpass_Q": Normal(mean=0.7, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=10)),  # Gentle slope
        
        # High-pass to control extreme lows
        "highpass_freq": Normal(mean=40, std=10, shape=(1,), transformation=lambda x: torch.clamp(x, min=20, max=300)),  # Remove sub-bass rumble
        "highpass_Q": Normal(mean=0.7, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=10)),  # Gentle slope
    }

    # Heavy compression with very fast attack/release
    distributions_CompExp= {
        "comp_ratio": Normal(mean=10.0, std=0.5, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # Extreme compression ratio
        "exp_ratio": Normal(mean=0.8, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.001, max=0.999)),  # More aggressive expansion
        "at_coef": Normal(mean=10.0, std=1.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # Ultra-fast attack (ms)
        "rt_coef": Normal(mean=40.0, std=5.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # Very fast release (ms)
        "avg_coef": Normal(mean=0.97, std=0.01, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.001, max=0.999)),  # Extremely high average coefficient for maximum control
        "make_up": Normal(mean=8.0, std=1.0, shape=(1,)),  # Heavy make-up gain for powerful sound
        "comp_th": Normal(mean=-32, std=1.0, shape=(1,)),  # Very low threshold for maximum compression
        "exp_th": Normal(mean=-50.0, std=3.0, shape=(1,)),  # Higher expansion threshold for noise control
        "lookahead": Uniform(low=0.1, high=1.0, shape=(1,)),  # Minimal lookahead for maximum punch
    }

    # Short to medium bright reverb with increased early reflections
    distributions_FDN = {
        "b": Normal(mean=1/4, std=0.01, shape=(6,2)),  # Increased input gain for stronger early reflections
        "c": Normal(mean=1.2, std=0.05, shape=(2,6)),  # Increased output gain for brighter character
        "U": Normal(mean=0, std=(1/(5**0.5)), shape=(6,6), transformation=lambda x: unilossless(x)),  # Slightly denser matrix for brighter sound
        "gamma": NormalRamp(mean_low=0.9, mean_high=0.05, std_low=0.3, std_high=0.05, log=True, shape=(49, 1), 
                           transformation=lambda x: T60togamma(torch.clamp(x, min=0, max=T60_max), sample_rate=sample_rate)),  # Shorter decay times
    }

    # EQ for the reverb - brighter character
    distributions_PEQ_FDN = {
        "peak1_gain": Normal(mean=2.0, std=1, shape=(1,)),  # Slight boost in mids for reverb
        "peak1_freq": Normal(mean=1500, std=200, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=2500)),  # Mid presence
        "peak1_Q": Normal(mean=0.8, std=0.2, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Broader Q
    
        "peak2_gain": Normal(mean=3.0, std=1, shape=(1,)),  # Boost in upper mids for brightness
        "peak2_freq": Normal(mean=4000, std=500, shape=(1,), transformation=lambda x: torch.clamp(x, min=600, max=7000)),  # Upper presence
        "peak2_Q": Normal(mean=0.9, std=0.2, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.6, max=20)),  # Moderate Q
    
        "lowshelf_gain": Normal(mean=-2.0, std=1, shape=(1,)),  # Slight cut in lows for clearer reverb
        "lowshelf_freq": Normal(mean=250, std=50, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=450)),  # Low-mid frequency
    
        "highshelf_gain": Normal(mean=0.0, std=1, shape=(1,)),  # Boost in highs for bright character
        "highshelf_freq": Normal(mean=6000, std=500, shape=(1,), transformation=lambda x: torch.clamp(x, min=1500, max=16000)),  # Air frequency
    }

    distributions_pan = {
        "pan_param": Uniform(low=0.4, high=0.6, shape=(1,), transformation=lambda x: torch.clamp(x, 0., 1)),  # Slightly wider stereo image
    }

    distributions_RMSnorm = {
        "RMSnorm": Normal(mean=-25, std=3, shape=(1,), transformation=lambda x: torch.clamp(x, -40, -10)),  # Louder output level for drums
    }

    return {
        "PEQ": distributions_PEQ,
        "CompExp": distributions_CompExp,
        "FDN": distributions_FDN,
        "PEQ_FDN": distributions_PEQ_FDN,
        "pan": distributions_pan,
        "RMSnorm": distributions_RMSnorm,
    }

def get_distributions_Cluster1_drums(sample_rate: int = 44100):
    """
    Returns the distributions for country-style drum processing parameters with:
    - EQ: Moderate boost in lows and mids, cut in highs
    - Compression: Light with medium attack/release
    - Reverb: Medium to long, warm character with longer tail
    """

    distributions_PEQ = {
        # Mid boost - first peak focused on lower mids
        "peak1_gain": Normal(mean=4.0, std=1.5, shape=(1,)),  # Moderate boost in lower mids
        "peak1_freq": Normal(mean=350, std=50, shape=(1,), transformation=lambda x: torch.clamp(x, min=33, max=5400)),  # Lower mid frequency
        "peak1_Q": Normal(mean=0.8, std=0.2, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Broader Q for natural sound
        
        # Mid boost - second peak focused on upper mids
        "peak2_gain": Normal(mean=3.0, std=1.5, shape=(1,)),  # Moderate boost in upper mids
        "peak2_freq": Normal(mean=800, std=100, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=17500)),  # Mid frequency
        "peak2_Q": Normal(mean=0.9, std=0.2, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Moderate Q for natural sound
        
        # Low boost
        "lowshelf_gain": Normal(mean=5.0, std=1.5, shape=(1,)),  # Moderate low boost
        "lowshelf_freq": Normal(mean=150, std=30, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=200)),  # Bass frequency
        
        # High cut
        "highshelf_gain": Normal(mean=-4.0, std=1.5, shape=(1,)),  # Cut in highs for warmer sound
        "highshelf_freq": Normal(mean=4500, std=500, shape=(1,), transformation=lambda x: torch.clamp(x, min=2000, max=17500)),  # Presence frequency
        
        # Low-pass to further control highs
        "lowpass_freq": Normal(mean=10000, std=1000, shape=(1,), transformation=lambda x: torch.clamp(x, min=2000, max=18000)),  # More aggressive high cut
        "lowpass_Q": Normal(mean=0.8, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=10)),  # Gentle slope
        
        # High-pass to control extreme lows
        "highpass_freq": Normal(mean=50, std=10, shape=(1,), transformation=lambda x: torch.clamp(x, min=20, max=300)),  # Allow more low end
        "highpass_Q": Normal(mean=0.7, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=10)),  # Gentle slope
    }

    # Light compression with medium attack/release

    distributions_CompExp = {
        "comp_ratio": Normal(mean=1.2, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # Barely-there compression ratio
        "exp_ratio": Normal(mean=0.95, std=0.03, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.001, max=0.999)),  # Minimal expansion
        "at_coef": Normal(mean=40.0, std=5.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # Very slow attack (ms)
        "rt_coef": Normal(mean=250.0, std=25.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # Very slow release (ms)
        "avg_coef": Normal(mean=0.85, std=0.02, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.001, max=0.999)),  # Extremely high average coefficient for maximum control
        "make_up": Normal(mean=1.0, std=0.5, shape=(1,)),  # Minimal make-up gain to preserve dynamics
        "comp_th": Normal(mean=-10, std=2.0, shape=(1,)),  # Very high threshold for minimal compression
        "exp_th": Normal(mean=-60.0, std=3.0, shape=(1,)),  # Lower expansion threshold
        "lookahead": Uniform(low=2.5, high=3.0, shape=(1,)),  # Maximum lookahead for natural transients
    }

    # Medium to long warm reverb with longer tail
    distributions_FDN = {
        "b": Normal(mean=1/6, std=0.01, shape=(6,2)),  # Standard input gain
        "c": Normal(mean=0.9, std=0.05, shape=(2,6)),  # Slightly reduced output gain for warmer character
        "U": Normal(mean=0, std=(1/(6**0.5)), shape=(6,6), transformation=lambda x: unilossless(x)),  # Standard density
        "gamma": NormalRamp(mean_low=1.5, mean_high=0.2, std_low=0.5, std_high=0.1, log=True, shape=(49, 1), 
                           transformation=lambda x: T60togamma(torch.clamp(x, min=0, max=T60_max), sample_rate=sample_rate)),  # Longer decay times
    }

    # EQ for the reverb - warmer character
    distributions_PEQ_FDN = {
        "peak1_gain": Normal(mean=1.0, std=0.5, shape=(1,)),  # Slight boost in lower mids for warmth
        "peak1_freq": Normal(mean=600, std=100, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=2500)),  # Lower mid frequency
        "peak1_Q": Normal(mean=0.7, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Broader Q for warmth
    
        "peak2_gain": Normal(mean=-1.0, std=0.5, shape=(1,)),  # Slight cut in upper mids
        "peak2_freq": Normal(mean=3000, std=300, shape=(1,), transformation=lambda x: torch.clamp(x, min=600, max=7000)),  # Upper mid frequency
        "peak2_Q": Normal(mean=0.8, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.6, max=20)),  # Moderate Q
    
        "lowshelf_gain": Normal(mean=2.0, std=0.5, shape=(1,)),  # Boost in lows for warmth
        "lowshelf_freq": Normal(mean=200, std=30, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=450)),  # Low frequency
    
        "highshelf_gain": Normal(mean=-3.0, std=1, shape=(1,)),  # Cut in highs for warmer reverb
        "highshelf_freq": Normal(mean=5000, std=500, shape=(1,), transformation=lambda x: torch.clamp(x, min=1500, max=16000)),  # High frequency
    }

    distributions_pan = {
        "pan_param": Uniform(low=0.45, high=0.55, shape=(1,), transformation=lambda x: torch.clamp(x, 0., 1)),  # Narrower stereo image for more natural sound
    }

    distributions_RMSnorm = {
        "RMSnorm": Normal(mean=-25, std=3, shape=(1,), transformation=lambda x: torch.clamp(x, -40, -10)),  # More dynamic level for country drums
    }

    return {
        "PEQ": distributions_PEQ,
        "CompExp": distributions_CompExp,
        "FDN": distributions_FDN,
        "PEQ_FDN": distributions_PEQ_FDN,
        "pan": distributions_pan,
        "RMSnorm": distributions_RMSnorm,
    }


def get_distributions_Cluster0_bass(sample_rate: int = 44100):
    """
    Returns the distributions for pop bass processing parameters with:
    - EQ: Significant low boost, aggressive low-mid cut, significant high boost
    - Compression: Moderate to heavy with slow attack/fast release
    - Reverb: Very short or none
    """

    distributions_PEQ = {
        # Strong low-mid scoop for modern sound
        "peak1_gain": Normal(mean=-12.0, std=2, shape=(1,)),  # Strong cut in low-mids
        "peak1_freq": Normal(mean=300, std=50, shape=(1,), transformation=lambda x: torch.clamp(x, min=33, max=5400)),  # Low-mid frequency
        "peak1_Q": Normal(mean=1.3, std=0.2, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Focused Q for targeted cut
        
        # Pronounced upper-mid presence for attack
        "peak2_gain": Normal(mean=4.5, std=1.5, shape=(1,)),  # Noticeable boost for definition
        "peak2_freq": Normal(mean=1100, std=150, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=17500)),  # Upper-mid frequency for clarity
        "peak2_Q": Normal(mean=1.1, std=0.2, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Moderate Q
        
        # Strong low boost for power and impact
        "lowshelf_gain": Normal(mean=10.0, std=2, shape=(1,)),  # Strong low boost
        "lowshelf_freq": Normal(mean=75, std=15, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=200)),  # Deep bass frequency
        
        # Strong high boost for presence
        "highshelf_gain": Normal(mean=9.0, std=2, shape=(1,)),  # Strong high boost for presence
        "highshelf_freq": Normal(mean=3200, std=400, shape=(1,), transformation=lambda x: torch.clamp(x, min=2000, max=17500)),  # Presence frequency
        
        # Higher low-pass for brightness
        "lowpass_freq": Normal(mean=9000, std=1000, shape=(1,), transformation=lambda x: torch.clamp(x, min=2000, max=18000)),  # Allow more highs
        "lowpass_Q": Normal(mean=0.7, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=10)),  # Gentle slope
        
        # Lower high-pass for sub-bass
        "highpass_freq": Normal(mean=32, std=5, shape=(1,), transformation=lambda x: torch.clamp(x, min=20, max=300)),  # Allow more sub-bass
        "highpass_Q": Normal(mean=0.7, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=10)),  # Gentle slope
    }



    # Moderate to heavy compression with slow attack/fast release
    distributions_CompExp = {
        "comp_ratio": Normal(mean=10.0, std=0.5, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # Maximum compression ratio
        "exp_ratio": Normal(mean=0.2, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.001, max=0.999)),  # Extreme expansion for noise control
        "at_coef": Normal(mean=13.0, std=10.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # Slow attack to preserve initial transient
        "rt_coef": Normal(mean=50.0, std=2.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # Ultra-fast release for maximum punch
        "avg_coef": Normal(mean=0.98, std=0.01, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.001, max=0.999)),  # Extremely high average coefficient for maximum control
        "make_up": Normal(mean=9.0, std=1.0, shape=(1,)),  # Maximum make-up gain for aggressive presence
        "comp_th": Normal(mean=-30, std=2.0, shape=(1,)),  # Extremely low threshold for maximum compression
        "exp_th": Normal(mean=-60.0, std=3.0, shape=(1,)),  # Very low expansion threshold for maximum noise control
        "lookahead": Uniform(low=0.1, high=0.5, shape=(1,)),  # Minimal lookahead for tightest response
    }


    # Very short or minimal reverb
    distributions_FDN = {
        "b": Normal(mean=1/10, std=0.01, shape=(6,2)),  # Reduced input gain for minimal reverb
        "c": Normal(mean=0.7, std=0.05, shape=(2,6)),  # Reduced output gain for minimal reverb
        "U": Normal(mean=0, std=(1/(6**0.5)), shape=(6,6), transformation=lambda x: unilossless(x)),  # Standard density
        "gamma": NormalRamp(mean_low=0.3, mean_high=0.02, std_low=0.1, std_high=0.01, log=True, shape=(49, 1), 
                           transformation=lambda x: T60togamma(torch.clamp(x, min=0, max=T60_max), sample_rate=sample_rate)),  # Very short decay times
    }

    # EQ for the minimal reverb
    distributions_PEQ_FDN = {
        "peak1_gain": Normal(mean=0.0, std=0.5, shape=(1,)),  # Neutral mids
        "peak1_freq": Normal(mean=800, std=100, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=2500)),  # Mid frequency
        "peak1_Q": Normal(mean=0.7, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Broad Q
    
        "peak2_gain": Normal(mean=0.0, std=0.5, shape=(1,)),  # Neutral upper mids
        "peak2_freq": Normal(mean=2000, std=200, shape=(1,), transformation=lambda x: torch.clamp(x, min=600, max=7000)),  # Upper mid frequency
        "peak2_Q": Normal(mean=0.7, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.6, max=20)),  # Broad Q
    
        "lowshelf_gain": Normal(mean=-3.0, std=1, shape=(1,)),  # Cut lows in reverb
        "lowshelf_freq": Normal(mean=200, std=30, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=450)),  # Low frequency
    
        "highshelf_gain": Normal(mean=-2.0, std=1, shape=(1,)),  # Cut highs in reverb
        "highshelf_freq": Normal(mean=4000, std=500, shape=(1,), transformation=lambda x: torch.clamp(x, min=1500, max=16000)),  # High frequency
    }

    distributions_pan = {
        "pan_param": Uniform(low=0.48, high=0.52, shape=(1,), transformation=lambda x: torch.clamp(x, 0, 1)),  # Nearly centered for bass
    }

    distributions_RMSnorm = {
        "RMSnorm": Normal(mean=-25, std=3, shape=(1,), transformation=lambda x: torch.clamp(x, -40, -10)),  # Prominent level for pop bass
    }

    return {
        "PEQ": distributions_PEQ,
        "CompExp": distributions_CompExp,
        "FDN": distributions_FDN,
        "PEQ_FDN": distributions_PEQ_FDN,
        "pan": distributions_pan,
        "RMSnorm": distributions_RMSnorm,
    }

def get_distributions_Cluster1_bass(sample_rate: int = 44100):
    """
    Returns the distributions for country bass processing parameters with:
    - EQ: Moderate low boost, low-mid cut, moderate mid boost
    - Compression: Light with slow attack/release
    - Reverb: Short with increased early reflections
    """
    distributions_PEQ = {
        # Moderate low-mid dip for clarity
        "peak1_gain": Normal(mean=-5.0, std=1.0, shape=(1,)),  # Moderate cut in low-mids
        "peak1_freq": Normal(mean=230, std=30, shape=(1,), transformation=lambda x: torch.clamp(x, min=33, max=5400)),  # Low-mid frequency
        "peak1_Q": Normal(mean=0.9, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Moderate Q for natural sound
        
        # Enhanced mid boost for warmth and character
        "peak2_gain": Normal(mean=5.5, std=1.5, shape=(1,)),  # Enhanced boost in mids
        "peak2_freq": Normal(mean=650, std=80, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=17500)),  # Mid frequency for warmth
        "peak2_Q": Normal(mean=0.8, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Broad Q for natural warmth
        
        # Moderate low boost for fullness
        "lowshelf_gain": Normal(mean=4.0, std=1.0, shape=(1,)),  # Moderate low boost
        "lowshelf_freq": Normal(mean=110, std=20, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=200)),  # Bass frequency
        
        # Slight high cut for warmth
        "highshelf_gain": Normal(mean=-1.5, std=1.0, shape=(1,)),  # Slight cut in highs for warmth
        "highshelf_freq": Normal(mean=2800, std=300, shape=(1,), transformation=lambda x: torch.clamp(x, min=2000, max=17500)),  # Presence frequency
        
        # Lower low-pass for vintage character
        "lowpass_freq": Normal(mean=5500, std=500, shape=(1,), transformation=lambda x: torch.clamp(x, min=2000, max=18000)),  # Control extreme highs
        "lowpass_Q": Normal(mean=0.7, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=10)),  # Gentle slope
        
        # Moderate high-pass for controlled low end
        "highpass_freq": Normal(mean=50, std=8, shape=(1,), transformation=lambda x: torch.clamp(x, min=20, max=300)),  # Control extreme sub-bass
        "highpass_Q": Normal(mean=0.7, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=10)),  # Gentle slope
    }

    # Light compression with slow attack/release
    distributions_CompExp = {
        "comp_ratio": Normal(mean=1.1, std=0.05, shape=(1,), transformation=lambda x: torch.clamp(x, min=1.0001)),  # Barely perceptible compression ratio
        "exp_ratio": Normal(mean=0.9, std=0.05, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.001, max=0.999)),  # Minimal expansion
        "at_coef": Normal(mean=100.0, std=10.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # Extremely slow attack
        "rt_coef": Normal(mean=500.0, std=30.0, shape=(1,), transformation=lambda x: torchcomp.ms2coef(torch.clamp(x, min=0.01), sample_rate)),  # Extremely slow release
        "avg_coef": Normal(mean=0.8, std=0.2, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.001, max=0.999)),  # Extremely high average coefficient for maximum control
        "make_up": Normal(mean=0.5, std=0.3, shape=(1,)),  # Almost no make-up gain to preserve natural dynamics
        "comp_th": Normal(mean=-8, std=1.0, shape=(1,)),  # Extremely high threshold for minimal compression
        "exp_th": Normal(mean=-35.0, std=2.0, shape=(1,)),  # Higher expansion threshold
        "lookahead": Uniform(low=2.8, high=3.0, shape=(1,)),  # Maximum lookahead for most natural sound
    }

    # Short reverb with increased early reflections
    distributions_FDN = {
        "b": Normal(mean=1/4, std=0.01, shape=(6,2)),  # Increased input gain for stronger early reflections
        "c": Normal(mean=0.9, std=0.05, shape=(2,6)),  # Standard output gain
        "U": Normal(mean=0, std=(1/(6**0.5)), shape=(6,6), transformation=lambda x: unilossless(x)),  # Standard density
        "gamma": NormalRamp(mean_low=0.65, mean_high=0.04, std_low=0.2, std_high=0.02, log=True, shape=(49, 1), 
                           transformation=lambda x: T60togamma(torch.clamp(x, min=0, max=T60_max), sample_rate=sample_rate)),  # Short decay times
    }

    # EQ for the reverb - warm character
    distributions_PEQ_FDN = {
        "peak1_gain": Normal(mean=1.0, std=0.5, shape=(1,)),  # Slight boost in mids
        "peak1_freq": Normal(mean=700, std=100, shape=(1,), transformation=lambda x: torch.clamp(x, min=200, max=2500)),  # Mid frequency
        "peak1_Q": Normal(mean=0.8, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.51, max=20)),  # Broad Q for natural sound
    
        "peak2_gain": Normal(mean=0.0, std=0.5, shape=(1,)),  # Neutral upper mids
        "peak2_freq": Normal(mean=1500, std=200, shape=(1,), transformation=lambda x: torch.clamp(x, min=600, max=7000)),  # Upper mid frequency
        "peak2_Q": Normal(mean=0.8, std=0.1, shape=(1,), transformation=lambda x: torch.clamp(x, min=0.6, max=20)),  # Broad Q
    
        "lowshelf_gain": Normal(mean=-1.0, std=0.5, shape=(1,)),  # Slight cut in lows for clearer reverb
        "lowshelf_freq": Normal(mean=200, std=30, shape=(1,), transformation=lambda x: torch.clamp(x, min=30, max=450)),  # Low frequency
    
        "highshelf_gain": Normal(mean=-2.0, std=0.5, shape=(1,)),  # Cut highs for warmer reverb
        "highshelf_freq": Normal(mean=3500, std=300, shape=(1,), transformation=lambda x: torch.clamp(x, min=1500, max=16000)),  # High frequency
    }

    distributions_pan = {
        "pan_param": Uniform(low=0.47, high=0.53, shape=(1,), transformation=lambda x: torch.clamp(x, 0, 1)),  # Nearly centered for bass
    }

    distributions_RMSnorm = {
        "RMSnorm": Normal(mean=-25, std=3, shape=(1,), transformation=lambda x: torch.clamp(x, -40, -10)),  # Natural level for country bass
    }

    return {
        "PEQ": distributions_PEQ,
        "CompExp": distributions_CompExp,
        "FDN": distributions_FDN,
        "PEQ_FDN": distributions_PEQ_FDN,
        "pan": distributions_pan,
        "RMSnorm": distributions_RMSnorm,
    }