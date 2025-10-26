import numpy as np
import librosa

def compute_spectrogram(file_path, set_limit=False, limit_seconds=10):
    """Compute the spectrogram of an audio file."""
    y, sr = librosa.load(file_path, sr=None)
    if set_limit:
        y = y[:int(sr*limit_seconds)]
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db, sr, y

def smooth_curve(data, window_size=30):
    """Smooth a curve using a moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    
def compute_crest_factor(y, sr, frame_length=2048, hop_length=512):
    # compute crest factor by windowing
    crest_factor_total = []
    for i in range(0, len(y), hop_length):
        frame = y[i:i+frame_length]
        peak = np.max(np.abs(frame))
        rms = np.sqrt(np.mean(frame**2))
        cur_crest_factor = peak / (rms + 1e-6)
        crest_factor_total.append(cur_crest_factor)
    crest_factor = np.asarray(crest_factor_total)
    return crest_factor

def compute_audio_features(y, sr, smooth_window_size=1, average_time=True):
    # features to compute: RMS energy, crest factor, dynamic spread, spectral centroid, spectral contrast, spectral flatness, spectral balance, and spectral bandwidth
    y_mono = y.mean(axis=0)
    spectral_centroid = smooth_curve(librosa.feature.spectral_centroid(y=y_mono, sr=sr)[0], window_size=smooth_window_size)
    spectral_contrast = smooth_curve(np.mean(librosa.feature.spectral_contrast(y=y_mono, sr=sr), axis=0), window_size=smooth_window_size)
    spectral_flatness = smooth_curve(librosa.feature.spectral_flatness(y=y_mono)[0], window_size=smooth_window_size)
    rms_energy = smooth_curve(librosa.feature.rms(y=y_mono)[0], window_size=smooth_window_size)
    # Crest Factor: Peak-to-RMS ratio
    crest_factor = smooth_curve(compute_crest_factor(y_mono, sr, frame_length=2048, hop_length=512), window_size=smooth_window_size)
    # Dynamic Spread
    dynamic_spread = np.std(rms_energy)
    # Spectral Balance (Low, Mid, High Frequency Energy Ratios)
    spec = np.abs(librosa.stft(y_mono))**2
    freqs = librosa.fft_frequencies(sr=sr)
    low_energy = np.sum(spec[freqs < 200])
    mid_energy = np.sum(spec[(freqs >= 200) & (freqs < 2000)])
    high_energy = np.sum(spec[freqs >= 2000])
    total_energy = low_energy + mid_energy + high_energy
    spectral_balance = np.array([low_energy / total_energy, mid_energy / total_energy, high_energy / total_energy])
    spectral_bandwidth = smooth_curve(librosa.feature.spectral_bandwidth(y=y_mono, sr=sr)[0], window_size=smooth_window_size)
    # Stereo Width + Mid-Side Ratio
    y_mid = (y[:len(y)//2] + y[len(y)//2:]) / 2
    y_side = (y[:len(y)//2] - y[len(y)//2:]) / 2
    stereo_width = np.mean(np.abs(y_side)) / (np.mean(np.abs(y_mid)) + 1e-6)
    mid_side_ratio = np.mean(np.abs(y_mid)) / (np.mean(np.abs(y_side)) + 1e-6)
    # average across time
    if average_time:
        spectral_centroid = np.mean(spectral_centroid)
        spectral_flatness = np.mean(spectral_flatness)
        spectral_contrast = np.mean(spectral_contrast)
        spectral_flatness = np.mean(spectral_flatness)
        rms_energy = np.mean(rms_energy)
        crest_factor = np.mean(crest_factor)
        spectral_bandwidth = np.mean(spectral_bandwidth)

    # return as a dictionary
    return {'spectral_centroid': spectral_centroid,
            'spectral_contrast': spectral_contrast,
            'spectral_flatness': spectral_flatness,
            'rms_energy': rms_energy,
            'crest_factor': crest_factor,
            'dynamic_spread': dynamic_spread,
            'spectral_balance': spectral_balance,
            'spectral_bandwidth': spectral_bandwidth,
            'stereo_width': stereo_width,
            'mid_side_ratio': mid_side_ratio}
    

