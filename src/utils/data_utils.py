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