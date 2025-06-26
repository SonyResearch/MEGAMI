
import torch

from fx_model.processors.diffvox_functional import (
    lowpass_biquad_parallel,
    highpass_biquad_parallel,
    equalizer_biquad_parallel,
    lowshelf_biquad_parallel,
    highshelf_biquad_parallel,
)

from fx_model.processors.transformations import Identity, UniLossLess, MinMax, SmoothingCoef, WrappedPositive

EPS= 1e-8

def prepare_PEQ_FDN_parameters(
    sample_rate: int,
    device: str = "cpu",
    ):

    Q_shelf = torch.tensor(0.707, device=device)  # Default Q for lowshelf

    dict_non_optimizable = {
        "Q_shelf": Q_shelf,  # shape (1,)
        "sample_rate": sample_rate,  # shape (1,)
    }

    dict_transformations = {
        "peak1_gain": Identity(),
        "peak1_freq": MinMax(200.0, 2500.0),
        "peak1_Q": MinMax(0.1, 3.0),
        "peak2_gain": Identity(),
        "peak2_freq": MinMax(600.0, 7000.0),
        "peak2_Q": MinMax(0.1, 3.0),
        "lowshelf_gain": Identity(),
        "lowshelf_freq": MinMax(30.0, 450.0),
        "highshelf_gain": Identity(),
        "highshelf_freq": MinMax(1500.0, 16000.0),
    }

    peak1_gain=torch.tensor(0.0, device=device)  # Initialize peak1_gain to 0
    peak1_gain=dict_transformations["peak1_gain"].right_inverse(peak1_gain)  # Apply transformation

    peak1_freq=torch.tensor(800.0, device=device)  # Initialize peak1_freq to 800 Hz
    peak1_freq=dict_transformations["peak1_freq"].right_inverse(peak1_freq)  # Apply transformation

    peak1_Q=torch.tensor(0.707, device=device)  # Initialize peak1_Q to 1.0
    peak1_Q=dict_transformations["peak1_Q"].right_inverse(peak1_Q)  # Apply transformation

    peak2_gain=torch.tensor(0.0, device=device)  # Initialize peak2_gain to 0
    peak2_gain=dict_transformations["peak2_gain"].right_inverse(peak2_gain)  # Apply transformation

    peak2_freq=torch.tensor(4000.0, device=device)  # Initialize peak2_freq to 4000 Hz
    peak2_freq=dict_transformations["peak2_freq"].right_inverse(peak2_freq)  # Apply transformation

    peak2_Q=torch.tensor(0.707, device=device)  # Initialize peak2_Q to 1.0
    peak2_Q=dict_transformations["peak2_Q"].right_inverse(peak2_Q)  # Apply transformation

    lowshelf_gain=torch.tensor(0.0, device=device)  # Initialize lowshelf_gain to 0
    lowshelf_gain=dict_transformations["lowshelf_gain"].right_inverse(lowshelf_gain)  # Apply transformation

    lowshelf_freq=torch.tensor(115.0, device=device)  # Initialize lowshelf_freq to 115 Hz
    lowshelf_freq=dict_transformations["lowshelf_freq"].right_inverse(lowshelf_freq)  # Apply transformation

    highshelf_gain=torch.tensor(0.0, device=device)  # Initialize highshelf_gain to 0
    highshelf_gain=dict_transformations["highshelf_gain"].right_inverse(highshelf_gain)  # Apply transformation

    highshelf_freq=torch.tensor(8000.0, device=device)  # Initialize highshelf_freq to 6000 Hz
    highshelf_freq=dict_transformations["highshelf_freq"].right_inverse(highshelf_freq)  # Apply transformation


    dict_optimizable = {
        "peak1_gain": peak1_gain.to(device),  # shape (1,)
        "peak1_freq": peak1_freq.to(device),  # shape (1,)
        "peak1_Q": peak1_Q.to(device),  # shape (1,)
        "peak2_gain": peak2_gain.to(device),  # shape (1,)      
        "peak2_freq": peak2_freq.to(device),  # shape (1,)
        "peak2_Q": peak2_Q.to(device),  # shape (1,)
        "lowshelf_gain": lowshelf_gain.to(device),  # shape (1,)
        "lowshelf_freq": lowshelf_freq.to(device),  # shape (1,)
        "highshelf_gain": highshelf_gain.to(device),  # shape (1,)  
        "highshelf_freq": highshelf_freq.to(device),  # shape (1,)
    }

    return dict_optimizable, dict_non_optimizable, dict_transformations





def prepare_PEQ_parameters(
    sample_rate: int,
    device: str = "cpu",
    ):

    Q_shelf = torch.tensor(0.707, device=device)  # Default Q for lowshelf

    dict_non_optimizable = {
        "Q_shelf": Q_shelf,  # shape (1,)
        "sample_rate": sample_rate,  # shape (1,)
    }

    dict_transformations = {
        "peak1_gain": Identity(),
        "peak1_freq": MinMax(33.0, 5400.0),
        "peak1_Q": MinMax(0.2, 20.0),
        "peak2_gain": Identity(),
        "peak2_freq": MinMax(200.0, 17500.0),
        "peak2_Q": MinMax(0.2, 20.0),
        "lowshelf_gain": Identity(),
        "lowshelf_freq": MinMax(30.0, 200.0),
        "highshelf_gain": Identity(),
        "highshelf_freq": MinMax(750.0, 8300.0),
        "lowpass_freq": MinMax(200.0, 18000.0),
        "lowpass_Q": MinMax(0.5, 10.0),
        "highpass_freq": MinMax(16.0, 5300.0),
        "highpass_Q": MinMax(0.5, 10.0),
    }

    peak1_gain=torch.tensor(0.0, device=device)  # Initialize peak1_gain to 0
    peak1_gain=dict_transformations["peak1_gain"].right_inverse(peak1_gain)  # Apply transformation

    peak1_freq=torch.tensor(800.0, device=device)  # Initialize peak1_freq to 800 Hz
    peak1_freq=dict_transformations["peak1_freq"].right_inverse(peak1_freq)  # Apply transformation

    peak1_Q=torch.tensor(0.707, device=device)  # Initialize peak1_Q to 1.0
    peak1_Q=dict_transformations["peak1_Q"].right_inverse(peak1_Q)  # Apply transformation

    peak2_gain=torch.tensor(0.0, device=device)  # Initialize peak2_gain to 0
    peak2_gain=dict_transformations["peak2_gain"].right_inverse(peak2_gain)  # Apply transformation

    peak2_freq=torch.tensor(4000.0, device=device)  # Initialize peak2_freq to 4000 Hz
    peak2_freq=dict_transformations["peak2_freq"].right_inverse(peak2_freq)  # Apply transformation

    peak2_Q=torch.tensor(0.707, device=device)  # Initialize peak2_Q to 1.0
    peak2_Q=dict_transformations["peak2_Q"].right_inverse(peak2_Q)  # Apply transformation

    lowshelf_gain=torch.tensor(0.0, device=device)  # Initialize lowshelf_gain to 0
    lowshelf_gain=dict_transformations["lowshelf_gain"].right_inverse(lowshelf_gain)  # Apply transformation

    lowshelf_freq=torch.tensor(115.0, device=device)  # Initialize lowshelf_freq to 115 Hz
    lowshelf_freq=dict_transformations["lowshelf_freq"].right_inverse(lowshelf_freq)  # Apply transformation

    highshelf_gain=torch.tensor(0.0, device=device)  # Initialize highshelf_gain to 0
    highshelf_gain=dict_transformations["highshelf_gain"].right_inverse(highshelf_gain)  # Apply transformation

    highshelf_freq=torch.tensor(6000.0, device=device)  # Initialize highshelf_freq to 6000 Hz
    highshelf_freq=dict_transformations["highshelf_freq"].right_inverse(highshelf_freq)  # Apply transformation

    lowpass_freq=torch.tensor(17500.0, device=device)  # Initialize lowpass_freq to 17500 Hz
    lowpass_freq=dict_transformations["lowpass_freq"].right_inverse(lowpass_freq)  # Apply transformation

    lowpass_Q=torch.tensor(0.707, device=device)  # Initialize lowpass_Q to 1.0
    lowpass_Q=dict_transformations["lowpass_Q"].right_inverse(lowpass_Q)  # Apply

    highpass_freq=torch.tensor(200.0, device=device)  # Initialize highpass_freq to 200 Hz
    highpass_freq=dict_transformations["highpass_freq"].right_inverse(highpass_freq)  # Apply transformation

    highpass_Q=torch.tensor(0.707, device=device)  # Initialize highpass_Q to 1.0
    highpass_Q=dict_transformations["highpass_Q"].right_inverse(highpass_Q)  # Apply transformation

    dict_optimizable = {
        "peak1_gain": peak1_gain.to(device),  # shape (1,)
        "peak1_freq": peak1_freq.to(device),  # shape (1,)
        "peak1_Q": peak1_Q.to(device),  # shape (1,)
        "peak2_gain": peak2_gain.to(device),  # shape (1,)      
        "peak2_freq": peak2_freq.to(device),  # shape (1,)
        "peak2_Q": peak2_Q.to(device),  # shape (1,)
        "lowshelf_gain": lowshelf_gain.to(device),  # shape (1,)
        "lowshelf_freq": lowshelf_freq.to(device),  # shape (1,)
        "highshelf_gain": highshelf_gain.to(device),  # shape (1,)  
        "highshelf_freq": highshelf_freq.to(device),  # shape (1,)
        "lowpass_freq": lowpass_freq.to(device),  # shape (1,)
        "lowpass_Q": lowpass_Q.to(device),  # shape (1,)
        "highpass_freq": highpass_freq.to(device),  # shape (1,)
        "highpass_Q": highpass_Q.to(device),  # shape (1,)
    }

    return dict_optimizable, dict_non_optimizable, dict_transformations




def peq_FDN_functional(
    x: torch.Tensor,
    peak1_gain: torch.Tensor,
    peak1_freq: torch.Tensor,
    peak1_Q: torch.Tensor,
    peak2_gain: torch.Tensor,
    peak2_freq: torch.Tensor,
    peak2_Q: torch.Tensor,
    lowshelf_gain: torch.Tensor,
    lowshelf_freq: torch.Tensor,
    highshelf_gain: torch.Tensor,
    highshelf_freq: torch.Tensor,
    Q_shelf: torch.Tensor,
    sample_rate: float,
):
    """Compressor/Expander with gating functionality.
 
    Args:
        x (torch.Tensor): Input audio tensor with shape (bs, chs, seq_len)
        sample_rate (float): Audio sample rate.
        peak1_freq (torch.Tensor): Frequency for the first peak filter.
        peak2_freq (torch.Tensor): Frequency for the second peak filter.
        lowshelf_freq (torch.Tensor): Frequency for the low shelf filter.
        highshelf_freq (torch.Tensor): Frequency for the high shelf filter.
        lowpass_freq (torch.Tensor): Frequency for the low pass filter.
        highpass_freq (torch.Tensor): Frequency for the high pass filter.
 
    Returns:
        torch.Tensor: Processed audio signal.
    """
    #    bs, chs, seq_len = x.size()

    Q_shelf=Q_shelf.view(-1, 1)  # Ensure Q_shelf is a column vector
    Q_shelf=Q_shelf.repeat(x.shape[0],1)

    peak1_freq=peak1_freq.view(-1, 1)  # Ensure peak1_freq is a column vector
    peak1_gain=peak1_gain.view(-1, 1)  # Ensure peak1_gain is a column vector
    peak1_Q=peak1_Q.view(-1, 1)  # Ensure peak

    peak2_freq=peak2_freq.view(-1, 1)  # Ensure peak2_freq is a column vector
    peak2_gain=peak2_gain.view(-1, 1)  # Ensure peak
    peak2_Q=peak2_Q.view(-1, 1)  # Ensure peak2_Q is a column vector

    lowshelf_freq=lowshelf_freq.view(-1, 1)  # Ensure lowshelf_freq is a column vector
    lowshelf_gain=lowshelf_gain.view(-1, 1)  # Ensure lows

    highshelf_freq=highshelf_freq.view(-1, 1)  # Ensure highshelf_freq is a column vector
    highshelf_gain=highshelf_gain.view(-1, 1)  # Ensure

    #output=torch.zeros_like(x)

    x= equalizer_biquad_parallel(
            x,
            sample_rate=sample_rate,
            center_freq=peak1_freq,
            Q=peak1_Q,
            gain=peak1_gain
        )
    
    x= equalizer_biquad_parallel(
            x,
            sample_rate=sample_rate,
            center_freq=peak2_freq,
            Q=peak2_Q,
            gain=peak2_gain
        )
    
    x= lowshelf_biquad_parallel(
            x,
            sample_rate=sample_rate,
            cutoff_freq=lowshelf_freq,
            gain=lowshelf_gain,
            Q=Q_shelf
        )
    
    x= highshelf_biquad_parallel(
            x,
            sample_rate=sample_rate,
            cutoff_freq=highshelf_freq,
            gain=highshelf_gain,
            Q=Q_shelf
        )
    

    return x


def peq_functional(
    x: torch.Tensor,
    peak1_gain: torch.Tensor,
    peak1_freq: torch.Tensor,
    peak1_Q: torch.Tensor,
    peak2_gain: torch.Tensor,
    peak2_freq: torch.Tensor,
    peak2_Q: torch.Tensor,
    lowshelf_gain: torch.Tensor,
    lowshelf_freq: torch.Tensor,
    highshelf_gain: torch.Tensor,
    highshelf_freq: torch.Tensor,
    lowpass_freq: torch.Tensor,
    lowpass_Q: torch.Tensor,
    highpass_freq: torch.Tensor,
    highpass_Q: torch.Tensor,
    Q_shelf: torch.Tensor,
    sample_rate: float,
):
    """Compressor/Expander with gating functionality.
 
    Args:
        x (torch.Tensor): Input audio tensor with shape (bs, chs, seq_len)
        sample_rate (float): Audio sample rate.
        peak1_freq (torch.Tensor): Frequency for the first peak filter.
        peak2_freq (torch.Tensor): Frequency for the second peak filter.
        lowshelf_freq (torch.Tensor): Frequency for the low shelf filter.
        highshelf_freq (torch.Tensor): Frequency for the high shelf filter.
        lowpass_freq (torch.Tensor): Frequency for the low pass filter.
        highpass_freq (torch.Tensor): Frequency for the high pass filter.
 
    Returns:
        torch.Tensor: Processed audio signal.
    """
    #    bs, chs, seq_len = x.size()

    Q_shelf=Q_shelf.view(-1, 1)  # Ensure Q_shelf is a column vector
    Q_shelf=Q_shelf.repeat(x.shape[0],1)

    peak1_freq=peak1_freq.view(-1, 1)  # Ensure peak1_freq is a column vector
    peak1_gain=peak1_gain.view(-1, 1)  # Ensure peak1_gain is a column vector
    peak1_Q=peak1_Q.view(-1, 1)  # Ensure peak

    peak2_freq=peak2_freq.view(-1, 1)  # Ensure peak2_freq is a column vector
    peak2_gain=peak2_gain.view(-1, 1)  # Ensure peak
    peak2_Q=peak2_Q.view(-1, 1)  # Ensure peak2_Q is a column vector

    lowshelf_freq=lowshelf_freq.view(-1, 1)  # Ensure lowshelf_freq is a column vector
    lowshelf_gain=lowshelf_gain.view(-1, 1)  # Ensure lows

    highshelf_freq=highshelf_freq.view(-1, 1)  # Ensure highshelf_freq is a column vector
    highshelf_gain=highshelf_gain.view(-1, 1)  # Ensure

    lowpass_freq=lowpass_freq.view(-1, 1)  # Ensure lowpass_freq is a column vector
    lowpass_Q=lowpass_Q.view(-1, 1)  # Ensure low

    highpass_freq=highpass_freq.view(-1, 1)  # Ensure highpass_freq is a column vector
    highpass_Q=highpass_Q.view(-1, 1)  # Ensure high

    x= equalizer_biquad_parallel(
            x,
            sample_rate=sample_rate,
            center_freq=peak1_freq,
            Q=peak1_Q,
            gain=peak1_gain
        )
    
    x= equalizer_biquad_parallel(
            x,
            sample_rate=sample_rate,
            center_freq=peak2_freq,
            Q=peak2_Q,
            gain=peak2_gain
        )
    
    x= lowshelf_biquad_parallel(
            x,
            sample_rate=sample_rate,
            cutoff_freq=lowshelf_freq,
            gain=lowshelf_gain,
            Q=Q_shelf
        )
    
    x= highshelf_biquad_parallel(
            x,
            sample_rate=sample_rate,
            cutoff_freq=highshelf_freq,
            gain=highshelf_gain,
            Q=Q_shelf
        )
    
    x= lowpass_biquad_parallel(
            x,
            sample_rate=sample_rate,
            cutoff_freq=lowpass_freq,
            Q=lowpass_Q
        )
    
    
    x= highpass_biquad_parallel(
            x,
            sample_rate=sample_rate,
            cutoff_freq=highpass_freq,
            Q=highpass_Q
        )

    return x