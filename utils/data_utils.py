import soundfile as sf
import torchaudio

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

#import loudness
import numpy as np

def apply_loud_normalization(x, lufs=-23, sample_rate=44100,device=None):
    """
    x shaPe: (batch_size, channels, time)
    """

    in_shape= x.shape
    if x.ndim != 3:
        x=x.view(-1, in_shape[-2], in_shape[-1])  # Ensure x is 3D

    B, C, T = x.shape

    if device is None:
        device = x.device

    x_out = torch.zeros_like(x)
    #for b in range(B):
    #    x_i=x[b].cpu().numpy().T
    #    lufs_in=loudness.integrated_loudness(x_i, sample_rate)

    #    delta_loudness= lufs - lufs_in
    #    gain=np.power(10, delta_loudness / 20)  # Convert dB to linear gain

    #    x_out[b] = torch.tensor(x_i.T * gain, device=device)

    x=x.view(B* C,1, T)  # Ensure x is 3D

    loudness=torchaudio.functional.loudness(x+1e-5, sample_rate=sample_rate)
    delta_loudness = lufs - loudness
    gain= torch.pow(10, delta_loudness / 20)  # Convert dB to linear gain
    if gain.isnan().any():
        print("NaN detected in gain, setting to -30 dB")
        gain = torch.nan_to_num(gain, nan=-30.0)

    x_out = x * gain.view(B * C, 1, 1)  # Apply gain to each channel

    
    x_out = x_out.view(in_shape)

    return x_out



from utils.feature_extractors.dsp_features import compute_log_rms_gated_max, compute_crest_factor, compute_stereo_width, compute_stereo_imbalance, compute_log_spread

def apply_RMS_normalization(x, RMS_norm=-25, device=None, use_gate=False):
        if device is None:
            device = x.device

        RMS= torch.tensor(RMS_norm, device=device).view(1, 1, 1).repeat(x.shape[0],1,1)  # Use fixed RMS for evaluation

        x_RMS_ref=20*torch.log10(torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True).mean(dim=-2, keepdim=True)))
        if use_gate:
            x_RMS = compute_log_rms_gated_max(x).unsqueeze(-1)
        else:
            x_RMS=20*torch.log10(torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True).mean(dim=-2, keepdim=True)))
        

        gain= RMS - x_RMS
        gain_linear = 10 ** (gain / 20 + 1e-6)  # Convert dB gain to linear scale, adding a small value to avoid division by zero
        x=x* gain_linear

        return x


import pyloudnorm as pyln   

def loudness_normalize(audio, target_loudness=-23.0, sample_rate=44100):
    """
    Normalize the loudness of the audio to a target level.
    """

    pylnmeter = pyln.Meter(sample_rate)  # Create a meter for 44100 Hz sampling rate

    audio= np.array(audio, dtype=np.float32).T
    loudness = pylnmeter.integrated_loudness(audio)


    # loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(audio, loudness, -14.0)

    return  torch.tensor(loudness_normalized_audio.T, dtype=torch.float32)
