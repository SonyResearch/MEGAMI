import soundfile as sf

def read_wav_segment(file_path, start, end, dtype="float32"):
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