import soundfile as sf

def read_wav_segment(file_path, start=None, end=None, dtype="float32"):
    """
    Reads a specific segment from a .wav file efficiently.

    Args:
        file_path (str): Path to the .wav file.
        start (int): Start frame index.
        end (int): End frame index.

    Returns:
        numpy.ndarray: Audio data for the specified segment.
        int: Sample rate of the audio file.
    """
    # Open the .wav file
    if start is None or end is None:
        data, samplerate = sf.read(file_path, dtype=dtype)
    else:
        with sf.SoundFile(file_path) as audio_file:
            # Read only the required frames
            audio_file.seek(start)
            data = audio_file.read(frames=end-start, dtype=dtype)
            samplerate = audio_file.samplerate

    return data, samplerate

def get_audio_length(file_path):
    """
    Retrieves the length of an audio file in seconds and frames.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        float: Length of the audio file in seconds.
        int: Total number of frames in the audio file.
        int: Sample rate of the audio file.
    """
    with sf.SoundFile(file_path) as audio_file:
        total_frames = len(audio_file)  # Total number of frames
        samplerate = audio_file.samplerate  # Sample rate
        duration = total_frames / samplerate  # Duration in seconds

    return duration, total_frames, samplerate

def taxonomy2track(input_class, num_instr=8):

    assert num_instr==8, "num_instr should be 8 for this function, the rest is not implemented yet"

    if input_class is None:
        return 'unknown'  
    if num_instr == 8:
        mapping = {0000: 'other', 1100: 'drums', 1200: 'drums', 1300: 'other', 2000: 'bass', 3000: 'guitar', 4100: 'piano', 4200: 'piano', 4300: 'piano', 4400: 'other', 4500: 'other', 4600: 'other', 4700: 'other', 4900: 'other', 5000: 'brass', 6100: 'strings', 6210: 'brass', 6220: 'brass', 8100: 'guitar', 8200: 'brass', 9000: 'vocals'}
    else:
        raise NotImplementedError()
    
    code_length = len(str(input_class))
    if code_length < 4:
        #pad zeros to the right to make it 4 digits
        input_class = int(str(input_class) + "0" * (4 - code_length))
 
    class_str = str(input_class)
    for i in range(len(class_str), 0, -1):
        general_class = int(class_str[:i] + "0" * (len(class_str) - i))
        if general_class in mapping:
            return mapping[general_class]  
       
    try:
        raise ValueError(f"No mapping found for input class {input_class} with num_instr {num_instr}")
    except ValueError as e:
        print(f"Error: {e}")
        return "other"  # Return a default value if no mapping is found

import torch
def efficient_roll(x, shift, dims=-1):
    """
    Efficiently roll tensor elements along a dimension without creating a full copy.
    
    Args:
        x: Input tensor
        shift: Number of places to roll (negative for left roll)
        dim: Dimension along which to roll
    
    Returns:
        Rolled tensor view where possible, minimal copy where necessary
    """
    if shift == 0:
        return x
    
    # Get the size of the dimension
    dim_size = x.size(dims)
    
    # Handle shift larger than dimension size
    shift = shift % dim_size
    if shift < 0:
        shift += dim_size
    
    # Create indices for the roll
    indices = torch.cat([torch.arange(dim_size-shift, dim_size), 
                         torch.arange(0, dim_size-shift)])
    
    # Use index_select for the roll
    return torch.index_select(x, dims, indices)


def apply_RMS_normalization(x, RMS_norm=-25, device=None):
        if device is None:
            device = x.device

        RMS= torch.tensor(RMS_norm, device=device).view(1, 1, 1).repeat(x.shape[0],1,1)  # Use fixed RMS for evaluation

        x_RMS=20*torch.log10(torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True).mean(dim=-2, keepdim=True)))

        gain= RMS - x_RMS
        gain_linear = 10 ** (gain / 20 + 1e-6)  # Convert dB gain to linear scale, adding a small value to avoid division by zero
        x=x* gain_linear.view(-1, 1, 1)

        return x


def synthesize_sinc_train(num_samples, sample_rate=44100, min_cutoff_ratio=0.1, max_cutoff_ratio=0.99, min_gain=0.2, max_gain=1.0, mean_pulse_rate=3):
    """
    Synthesize a sinc train signal with random parameters.
    
    Args:
        duration (float): Duration of the signal in seconds.
        sample_rate (int): Sample rate in Hz.
        min_cutoff_ratio (float): Minimum cutoff frequency ratio.
        max_cutoff_ratio (float): Maximum cutoff frequency ratio.
        min_gain (float): Minimum gain for the sinc function.
        max_gain (float): Maximum gain for the sinc function.
    
    Returns:
        torch.Tensor: Synthesized sinc train signal.
    """

    duration = num_samples / sample_rate  # Calculate duration from number of samples and sample rate

    time = torch.linspace(0, duration, num_samples)

    # Nyquist frequency is half the sample rate
    nyquist_freq = sample_rate / 2  # 22050 Hz

    # Cutoff ratio range (how close to Nyquist)
    # Function to create a sinc pulse with specified cutoff frequency
    def sinc_pulse(t, center, cutoff_freq, gain):
        # For a lowpass filter, the sinc function is scaled by 2*cutoff_freq
        normalized_t = 2 * cutoff_freq * (t - center)
        
        # Handle the special case where normalized_t is close to zero
        zero_indices = torch.abs(normalized_t) < 1e-10
        result = torch.zeros_like(normalized_t)
        result[zero_indices] = 2 * cutoff_freq  # The peak value
        
        # Calculate sinc for non-zero values
        non_zero = ~zero_indices
        result[non_zero] = 2 * cutoff_freq * torch.sin(torch.pi * normalized_t[non_zero]) / (torch.pi * normalized_t[non_zero])
    
        # flip polarity with a probability of 50%
        if torch.rand(1).item() < 0.5:
            result = -result
        
        # Apply gain
        return gain * result

    # Generate random pulse locations
    total_pulses = int(mean_pulse_rate * duration)
    pulse_times = torch.sort(torch.rand(total_pulses) * duration)[0]

    # Generate random gains for each pulse
    pulse_gains = min_gain + (max_gain - min_gain) * torch.rand(total_pulses)

    # Generate random cutoff ratios for each pulse
    pulse_cutoff_ratios = min_cutoff_ratio + (max_cutoff_ratio - min_cutoff_ratio) * torch.rand(total_pulses)
    pulse_cutoff_freqs = pulse_cutoff_ratios * nyquist_freq

    # Create the signal
    signal = torch.zeros(num_samples)
    for i, pulse_time in enumerate(pulse_times):
        cutoff_freq = pulse_cutoff_freqs[i]
        pulse = sinc_pulse(time, pulse_time, cutoff_freq, pulse_gains[i])
        signal += pulse

    # Normalize the signal to avoid clipping
    if torch.max(torch.abs(signal)) > 0:
        signal = signal / torch.max(torch.abs(signal))

    return signal.unsqueeze(0)  # Ensure the output is a 1D tensor

from utils.dsp_features import compute_log_rms_gated_v2, compute_crest_factor, compute_stereo_width, compute_stereo_imbalance, compute_log_spread

def apply_RMS_normalization(x, RMS_norm=-25, device=None, use_gate=False, stereo=False, threshold_dB=-70.0):
        if device is None:
            device = x.device

        RMS= torch.tensor(RMS_norm, device=device).view(1, 1, 1).repeat(x.shape[0],1,1)  # Use fixed RMS for evaluation

        #x_RMS_ref=20*torch.log10(torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True).mean(dim=-2, keepdim=True)))

        if use_gate:
            x_RMS = compute_log_rms_gated_v2(x).unsqueeze(-1)
        else:
            x_RMS=20*torch.log10(torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True).mean(dim=-2, keepdim=True)))

        if stereo:
            x_RMS = x_RMS.mean(dim=(-2), keepdim=True)
        
        #print("ref RMS", x_RMS_ref.shape, x_RMS.shape)

        gain= RMS - x_RMS

        #replace gain > 70dB with 0dB

        gain = torch.where(gain > -threshold_dB, torch.zeros_like(gain), gain)

        
        #if x_RMS < threshold_dB:
        #    gain=0
        #else:

        gain_linear = 10 ** (gain / 20 + 1e-6)  # Convert dB gain to linear scale, adding a small value to avoid division by zero
        #print("x.shape", x.shape, "gain_linear.shape", gain_linear.shape)
        x=x* gain_linear

        #print("xshape", x.shape, "gain_linear", gain_linear.shape)

        return x