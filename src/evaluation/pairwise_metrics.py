
import torchaudio
import os
from importlib import import_module
import yaml
import torch
import warnings
import pyloudnorm as pyln
import sklearn
import scipy
import librosa
import numpy as np

# Utils - general-purpose functions
def amp_to_db(x):
    return 20*np.log10(x + 1e-30)
def db_to_amp(x):
    return 10**(x/20)

def get_mape(x, y):

    try:
        return sklearn.metrics.mean_absolute_percentage_error(x, y)
    except Exception as e:
        # If an exception occurs, return np.nan and print an error message
        print(f"Error calculating mean absolute percentage error: {e}")
        return np.nan

def running_mean_std(x, N):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        cumsum2 = np.cumsum(np.insert(x**2, 0, 0)) 
        mean = (cumsum[N:] - cumsum[:-N]) / float(N)
        std = np.sqrt(((cumsum2[N:] - cumsum2[:-N]) / N) - (mean * mean)) 
    return mean, std
def get_running_stats(x, features, N=20):
    """
    Returns running mean and standard deviation from array x. This, based on the previous N frames
    Args:
        x: multi-dimensional array containing features
        features: index of features to be taken into account
    Returns:
        mean and std arrays
    """
    
    mean = []
    std = []
    x = x.copy()
    for i in range(len(features)):
        mean_, std_ = running_mean_std(x[:,i], N)
        mean.append(mean_)
        std.append(std_)
    mean = np.asarray(mean)
    std = np.asarray(std)
    
    return mean, std
def compute_stft(samples, hop_length, fft_size, stft_window):
    """
    Compute the STFT of `samples` applying a Hann window of size `FFT_SIZE`, shifted for each frame by `hop_length`.
    Args:
        samples: num samples x channels
        hop_length: window shift in samples
        fft_size: FFT size which is also the window size
        stft_window: STFT analysis window
    Returns:
        stft: frames x channels x freqbins
    """
    n_channels = samples.shape[1]
    n_frames = 1+int((samples.shape[0] - fft_size)/hop_length)
    stft = np.empty((n_frames, n_channels, fft_size//2+1), dtype=np.complex64)
    # convert into f_contiguous (such that [:,n] slicing is c_contiguous)
    samples = np.asfortranarray(samples)
    for n in range(n_channels):
        # compute STFT (output has size `n_frames x N_BINS`)
        stft[:, n, :] = librosa.stft(samples[:, n],
                                     n_fft=fft_size,
                                     hop_length=hop_length,
                                     window=stft_window,
                                     center=False).transpose()
    return stft
# Functions to compute Fx-related low-level features
# Loudness
def get_lufs_peak_frames(x, sr, frame_size, hop_size, min_lufs=-70.0):
    """
    Computes lufs and peak loudness in a frame-wise manner
    Args:
        x: audio
        sr: sampling rate
        frame_size: frame size, ideally larger than 400ms (LUFS)
        hop_size: frame hop
    Returns:
        loudness_ and peak_ arrays
    """ 

    x = x.copy()
    x_frames = librosa.util.frame(x.T, frame_length=frame_size, hop_length=hop_size)

    peak_ = []
    loudness_ = []
    for i in range(x_frames.shape[-1]):
        x_ = x_frames[:,:,i]
        peak = np.max(np.abs(x_.T))
        peak_db = 20.0 * np.log10(peak)
        peak_.append(peak_db)
        meter = pyln.Meter(sr, block_size=0.4) # create BS.1770 meter
        loudness = meter.integrated_loudness(x_.T)
        loudness_.append(loudness)
    peak_ = np.asarray(peak_)
    loudness_ = np.asarray(loudness_)
    
    peak_ = np.expand_dims(peak_, -1)
    loudness_ = np.expand_dims(loudness_, -1)

    #replace -Inf with min_lufs, and any value below min_lufs with min_lufs
    loudness_[loudness_ == -np.inf] = min_lufs
    loudness_[loudness_ < min_lufs] = min_lufs

    peak_[peak_ == -np.inf] = -70.0
    peak_[peak_ < -70.0] = -70.0
    
    return loudness_, peak_
    
def compute_loudness_features(audio_out, audio_tar, sr,  frame_size=17640, hop_size=8820):
    """
    Computes lufs and peak loudness mape error using a running mean
    Args:
        audio_out: automix audio (output of models)
        audio_tar: target audio
        sr: sampling rate
        frame_size: frame size, ideally larger than 400ms (LUFS)
        hop_size: frame hop
    Returns:
        loudness_ dictionary
    """ 

    audio_out = audio_out.copy()
    audio_tar = audio_tar.copy()

    loudness_ = {key:[] for key in ['lufs_loudness', 'peak_loudness', ]}



    loudness_tar, peak_tar = get_lufs_peak_frames(audio_tar, sr, frame_size, hop_size)
    loudness_out, peak_out = get_lufs_peak_frames(audio_out, sr, frame_size, hop_size)
    
    eps = 1e-10
    N = 40 # Considers previous 40 frames
    
    #print("loudness_tar", loudness_tar)
    #print("loudness_out", loudness_out)

    #print("peak_tar", peak_tar)
    #print("peak_out", peak_out)

    if peak_tar.shape[0] > N:
        
        mean_peak_tar, std_peak_tar = get_running_stats(peak_tar+eps, [0], N=N)
        mean_peak_out, std_peak_out = get_running_stats(peak_out+eps, [0], N=N)
        mean_lufs_tar, std_lufs_tar = get_running_stats(loudness_tar+eps, [0], N=N)
        mean_lufs_out, std_lufs_out = get_running_stats(loudness_out+eps, [0], N=N)
        
    else:
        
        mean_peak_tar = np.expand_dims(np.asarray([np.mean(peak_tar+eps)]), 0)
        mean_peak_out = np.expand_dims(np.asarray([np.mean(peak_out+eps)]), 0)
        mean_lufs_tar = np.expand_dims(np.asarray([np.mean(loudness_tar+eps)]), 0)
        mean_lufs_out = np.expand_dims(np.asarray([np.mean(loudness_out+eps)]), 0)

    #print(f'mean_peak_tar: {mean_peak_tar}, mean_peak_out: {mean_peak_out}')
    #print(f'mean_lufs_tar: {mean_lufs_tar}, mean_lufs_out: {mean_lufs_out}')

    mape_mean_peak = sklearn.metrics.mean_absolute_percentage_error(mean_peak_tar[0], mean_peak_out[0])
    mape_mean_lufs = sklearn.metrics.mean_absolute_percentage_error(mean_lufs_tar[0], mean_lufs_out[0])
    mape_mean_peak = get_mape(mean_peak_tar[0], mean_peak_out[0])
    mape_mean_lufs = get_mape(mean_lufs_tar[0], mean_lufs_out[0])

    loudness_['peak_loudness'] = mape_mean_peak
    loudness_['lufs_loudness'] = mape_mean_lufs
    
    loudness_['mean_mape_loudness'] = np.mean([loudness_['peak_loudness'],
                                      loudness_['lufs_loudness'],
                                      ])
    return loudness_
# Spectral
def compute_spectral_features_no(audio_out, audio_tar, sr, fft_size=4096, hop_length=1024, channels=2):
    """
    Computes spectral features' mape error using a running mean
    Args:
        audio_out: automix audio (output of models)
        audio_tar: target audio
        sr: sampling rate
        fft_size: fft_size size
        hop_length: fft hop size
        channels: channels to compute
    Returns:
        spectral_ dictionary
    """ 

    audio_out = audio_out.copy()
    audio_tar = audio_tar.copy()

    audio_out_ = pyln.normalize.peak(audio_out, -1.0)
    audio_tar_ = pyln.normalize.peak(audio_tar, -1.0)

    spec_out_ = compute_stft(audio_out_,
                         hop_length,
                         fft_size,
                         np.sqrt(np.hanning(fft_size+1)[:-1]))
    spec_out_ = np.transpose(spec_out_, axes=[1, -1, 0])
    spec_out_ = np.abs(spec_out_)
    
    spec_tar_ = compute_stft(audio_tar_,
                             hop_length,
                             fft_size,
                             np.sqrt(np.hanning(fft_size+1)[:-1]))
    spec_tar_ = np.transpose(spec_tar_, axes=[1, -1, 0])
    spec_tar_ = np.abs(spec_tar_)

    spectral_ = {key:[] for key in ['centroid',
                                    'bandwidth',
                                    'contrast',
                                    # 'contrast_lows',
                                    # 'contrast_mids',
                                    # 'contrast_highs',
                                    'rolloff',
                                    'flatness',
                                    'mean_mape_spectral',
                                   ]}
        
    centroid_mean_ = []
    centroid_std_ = []
    bandwidth_mean_ = []
    bandwidth_std_ = []
    contrast_l_mean_ = []
    contrast_l_std_ = []
    contrast_m_mean_ = []
    contrast_m_std_ = []
    contrast_h_mean_ = []
    contrast_h_std_ = []
    rolloff_mean_ = []
    rolloff_std_ = []
    flatness_mean_ = []
    for ch in range(channels):
        tar = spec_tar_[ch]
        out = spec_out_[ch]
        tar_sc = librosa.feature.spectral_centroid(y=None, sr=sr, S=tar,
                             n_fft=fft_size, hop_length=hop_length)
        out_sc = librosa.feature.spectral_centroid(y=None, sr=sr, S=out,
                             n_fft=fft_size, hop_length=hop_length)
        tar_bw = librosa.feature.spectral_bandwidth(y=None, sr=sr, S=tar,
                                                    n_fft=fft_size, hop_length=hop_length, 
                                                    centroid=tar_sc, norm=True, p=2)
        out_bw = librosa.feature.spectral_bandwidth(y=None, sr=sr, S=out,
                                                    n_fft=fft_size, hop_length=hop_length, 
                                                    centroid=out_sc, norm=True, p=2)
        
        # l = 0-250, m = 1-2-3 = 250 - 2000, h = 2000 - SR/2
        tar_ct = librosa.feature.spectral_contrast(y=None, sr=sr, S=tar,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   fmin=250.0, n_bands=4, quantile=0.02, linear=False)
        out_ct = librosa.feature.spectral_contrast(y=None, sr=sr, S=out,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   fmin=250.0, n_bands=4, quantile=0.02, linear=False)
        tar_ro = librosa.feature.spectral_rolloff(y=None, sr=sr, S=tar,
                                                  n_fft=fft_size, hop_length=hop_length, 
                                                  roll_percent=0.85)
        out_ro = librosa.feature.spectral_rolloff(y=None, sr=sr, S=out,
                                                  n_fft=fft_size, hop_length=hop_length, 
                                                  roll_percent=0.85)
        
        tar_ft = librosa.feature.spectral_flatness(y=None, S=tar,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   amin=1e-10, power=2.0)
        out_ft = librosa.feature.spectral_flatness(y=None, S=out,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   amin=1e-10, power=2.0)

        # Flatness is usually computed in dB
        tar_ft = amp_to_db(tar_ft)
        out_ft = amp_to_db(out_ft)
        # projection to avoid mape errors
        tar_ft = (-1*tar_ft) + 1.0
        out_ft = (-1*out_ft) + 1.0

        eps = 1e-0
        N = 40
        mean_sc_tar, std_sc_tar = get_running_stats(tar_sc.T+eps, [0], N=N)
        mean_sc_out, std_sc_out = get_running_stats(out_sc.T+eps, [0], N=N)
        
        assert np.isnan(mean_sc_tar).any() == False, f'NAN values mean_sc_tar'
        assert np.isnan(mean_sc_out).any() == False, f'NAN values mean_sc_out'
        
        mean_bw_tar, std_bw_tar = get_running_stats(tar_bw.T+eps, [0], N=N)
        mean_bw_out, std_bw_out = get_running_stats(out_bw.T+eps, [0], N=N)
        
        assert np.isnan(mean_bw_tar).any() == False, f'NAN values tar mean bw'
        assert np.isnan(mean_bw_out).any() == False, f'NAN values out mean bw'
        
        mean_ct_tar, std_ct_tar = get_running_stats(tar_ct.T, list(range(tar_ct.shape[0])), N=N)
        mean_ct_out, std_ct_out = get_running_stats(out_ct.T, list(range(out_ct.shape[0])), N=N)
        
        assert np.isnan(mean_ct_tar).any() == False, f'NAN values tar mean ct'
        assert np.isnan(mean_ct_out).any() == False, f'NAN values out mean ct'
        
        mean_ro_tar, std_ro_tar = get_running_stats(tar_ro.T+eps, [0], N=N)
        mean_ro_out, std_ro_out = get_running_stats(out_ro.T+eps, [0], N=N)
        
        assert np.isnan(mean_ro_tar).any() == False, f'NAN values tar mean ro'
        assert np.isnan(mean_ro_out).any() == False, f'NAN values out mean ro'

        mean_ft_tar, std_ft_tar = get_running_stats(tar_ft.T, [0], N=40) # If flatness mean error is too large, increase N. e.g. 100, 1000
        mean_ft_out, std_ft_out = get_running_stats(out_ft.T, [0], N=40)
        mean_ft_tar, std_ft_tar = get_running_stats(tar_ft.T, [0], N=N) 
        mean_ft_out, std_ft_out = get_running_stats(out_ft.T, [0], N=N)

        mape_mean_sc = sklearn.metrics.mean_absolute_percentage_error(mean_sc_tar[0], mean_sc_out[0])
        mape_mean_sc = get_mape(mean_sc_tar[0], mean_sc_out[0])

        mape_mean_bw = sklearn.metrics.mean_absolute_percentage_error(mean_bw_tar[0], mean_bw_out[0])
        mape_mean_bw = get_mape(mean_bw_tar[0], mean_bw_out[0])

        mape_mean_ct_l = sklearn.metrics.mean_absolute_percentage_error(mean_ct_tar[0], mean_ct_out[0])
        mape_mean_ct_l = get_mape(mean_ct_tar[0], mean_ct_out[0])

        mape_mean_ct_m = sklearn.metrics.mean_absolute_percentage_error(np.mean(mean_ct_tar[1:4], axis=0),
                                                                        np.mean(mean_ct_out[1:4], axis=0))
        mape_mean_ct_m = get_mape(np.mean(mean_ct_tar[1:4], axis=0), np.mean(mean_ct_out[1:4], axis=0))

        mape_mean_ct_h = sklearn.metrics.mean_absolute_percentage_error(mean_ct_tar[-1], mean_ct_out[-1])
        mape_mean_ct_h = get_mape(mean_ct_tar[-1], mean_ct_out[-1])

        mape_mean_ro = sklearn.metrics.mean_absolute_percentage_error(mean_ro_tar[0], mean_ro_out[0])
        mape_mean_ro = get_mape(mean_ro_tar[0], mean_ro_out[0])

        mape_mean_ft = sklearn.metrics.mean_absolute_percentage_error(mean_ft_tar[0], mean_ft_out[0])
        mape_mean_ft = get_mape(mean_ft_tar[0], mean_ft_out[0])

        centroid_mean_.append(mape_mean_sc)
        bandwidth_mean_.append(mape_mean_bw)
        contrast_l_mean_.append(mape_mean_ct_l)
        contrast_m_mean_.append(mape_mean_ct_m)
        contrast_h_mean_.append(mape_mean_ct_h)
        rolloff_mean_.append(mape_mean_ro)
        flatness_mean_.append(mape_mean_ft)
    spectral_['centroid'] = np.mean(centroid_mean_)
    spectral_['bandwidth'] = np.mean(bandwidth_mean_)
    contrast_l = np.mean(contrast_l_mean_)
    contrast_m = np.mean(contrast_m_mean_)
    contrast_h = np.mean(contrast_h_mean_)
    spectral_['contrast'] = np.mean([contrast_l, 
                                     contrast_m, 
                                     contrast_h])
    spectral_['rolloff'] = np.mean(rolloff_mean_)
    spectral_['flatness'] = np.mean(flatness_mean_)
    # spectral_['mean_mape_spectral'] = np.mean([np.mean(centroid_mean_),
    #                                   np.mean(bandwidth_mean_),
    #                                   np.mean(contrast_l_mean_),
    #                                   np.mean(contrast_m_mean_),
    #                                   np.mean(contrast_h_mean_),
    #                                   np.mean(rolloff_mean_),
    #                                   np.mean(flatness_mean_),
    #                                 ])
    spectral_['mean_mape_spectral'] = np.mean([spectral_['centroid'],
                                      spectral_['bandwidth'],
                                      spectral_['contrast'],
                                      spectral_['rolloff'],
                                      spectral_['flatness'],
                                     ])

    return spectral_
# PANNING 
def get_SPS(x, n_fft=2048, hop_length=1024, smooth=False, frames=False):
    
    """
    Computes Stereo Panning Spectrum (SPS) and similarity measure (phi)
    
    See: 
    Tzanetakis, George, Randy Jones, and Kirk McNally. "Stereo Panning Features for Classifying Recording Production Style." ISMIR. 2007.
    
    Args:
        x: input audio array
        n_fft: fft size
        hop_length: fft hop
        smooth: Applies smoothing filter to SPS
        frames: SPS is calculated in a frame-wise manner
    Returns:
        SPS_mean, phi_mean mean arrays
        SPS, phi arrays
    """ 
    
    
    x = np.copy(x)
    eps = 1e-20
        
    audio_D = compute_stft(x,
                 hop_length,
                 n_fft,
                 np.sqrt(np.hanning(n_fft+1)[:-1]))
    
    audio_D_l = np.abs(audio_D[:, 0, :] + eps)
    audio_D_r = np.abs(audio_D[:, 1, :] + eps)
    
    phi = 2 * (np.abs(audio_D_l*np.conj(audio_D_r)))/(np.abs(audio_D_l)**2+np.abs(audio_D_r)**2)
    
    phi_l = np.abs(audio_D_l*np.conj(audio_D_r))/(np.abs(audio_D_l)**2)
    phi_r = np.abs(audio_D_r*np.conj(audio_D_l))/(np.abs(audio_D_r)**2)
    delta = phi_l - phi_r
    delta_ = np.sign(delta)
    SPS = (1-phi)*delta_
    
    phi_mean = np.mean(phi, axis=0)
    if smooth:
        phi_mean = scipy.signal.savgol_filter(phi_mean, 501, 1, mode='mirror')
    
    SPS_mean = np.mean(SPS, axis=0)
    if smooth:
        SPS_mean = scipy.signal.savgol_filter(SPS_mean, 501, 1, mode='mirror')
        
    return SPS_mean, phi_mean, SPS, phi
def get_panning_rms_frame(sps_frame, freqs=[0,22050], sr=44100, n_fft=2048):
    """
    Computes Stereo Panning Spectrum RMS energy within a specifc frequency band.
    
    Args:
        sps_frame: sps frame
        freqs: frequency band
        sr: sampling rate
        n_fft: fft size
    Returns:
        p_rms SPS rms energy
    """ 
    
    
    idx1 = freqs[0]
    idx2 = freqs[1]
    f1 = int(np.floor(idx1*n_fft/sr))
    f2 = int(np.floor(idx2*n_fft/sr))
    
    p_rms = np.sqrt((1/(f2-f1)) * np.sum(sps_frame[f1:f2]**2))
    
    return p_rms
def get_panning_rms(sps, freqs=[[0, 22050]], sr=44100, n_fft=2048):
    """
    Computes Stereo Panning Spectrum RMS energy within a specifc frequency band.
    
    Args:
        sps: sps
        freqs: frequency band
        sr: sampling rate
        n_fft: fft size
    Returns:
        p_rms SPS rms energy array
    """ 
    
    p_rms = []
    for frame in sps:
        p_rms_ = []
        for f in freqs:
            rms = get_panning_rms_frame(frame, freqs=f, sr=sr, n_fft=n_fft)
            p_rms_.append(rms)
        p_rms.append(p_rms_)
    
    return np.asarray(p_rms)
def compute_panning_features(audio_out, audio_tar, sr, fft_size=4096, hop_length=1024):
    """
    Computes panning features' mape error using a running mean
    Args:
        audio_out: automix audio (output of models)
        audio_tar: target audio
        sr: sampling rate
        fft_size: fft_size size
        hop_length: fft hop size
    Returns:
        panning_ dictionary
    """ 
    audio_out = audio_out.copy()
    audio_tar = audio_tar.copy()

    audio_out_ = pyln.normalize.peak(audio_out, -1.0)
    audio_tar_ = pyln.normalize.peak(audio_tar, -1.0)
    
    panning_ = {}
                               
    freqs=[[0, sr//2], [0, 250], [250, 2500], [2500, sr//2]]  
    
    _, _, sps_frames_tar, _ = get_SPS(audio_tar_, n_fft=fft_size,
                                  hop_length=hop_length,
                                  smooth=True, frames=True)
    
    _, _, sps_frames_out, _ = get_SPS(audio_out_, n_fft=fft_size,
                                      hop_length=hop_length,
                                      smooth=True, frames=True)
    p_rms_tar = get_panning_rms(sps_frames_tar,
                    freqs=freqs,
                    sr=sr,
                    n_fft=fft_size)
    
    p_rms_out = get_panning_rms(sps_frames_out,
                    freqs=freqs,
                    sr=sr,
                    n_fft=fft_size)
    
    # to avoid num instability, deletes frames with zero rms from target
    if np.min(p_rms_tar) == 0.0:
        id_zeros = np.where(p_rms_tar.T[0] == 0)
        p_rms_tar_ = []
        p_rms_out_ = []
        for i in range(len(freqs)):
            temp_tar = np.delete(p_rms_tar.T[i], id_zeros)
            temp_out = np.delete(p_rms_out.T[i], id_zeros)
            p_rms_tar_.append(temp_tar)
            p_rms_out_.append(temp_out)
        p_rms_tar_ = np.asarray(p_rms_tar_)
        p_rms_tar = p_rms_tar_.T
        p_rms_out_ = np.asarray(p_rms_out_)
        p_rms_out = p_rms_out_.T
    
    N = 40 
    
    mean_tar, std_tar = get_running_stats(p_rms_tar, freqs, N=N)
    mean_out, std_out = get_running_stats(p_rms_out, freqs, N=N)

    panning_['panning_rms_total'] = sklearn.metrics.mean_absolute_percentage_error(mean_tar[0], mean_out[0])
    panning_['panning_rms_lows'] = sklearn.metrics.mean_absolute_percentage_error(mean_tar[1], mean_out[1])
    panning_['panning_rms_mids'] = sklearn.metrics.mean_absolute_percentage_error(mean_tar[2], mean_out[2])
    panning_['panning_rms_highs'] = sklearn.metrics.mean_absolute_percentage_error(mean_tar[3], mean_out[3])
    panning_['panning_rms_total'] = get_mape(mean_tar[0], mean_out[0])
    panning_['panning_rms_lows'] = get_mape(mean_tar[1], mean_out[1])
    panning_['panning_rms_mids'] = get_mape(mean_tar[2], mean_out[2])
    panning_['panning_rms_highs'] = get_mape(mean_tar[3], mean_out[3])

    panning_['mean_mape_panning'] = np.mean([panning_['panning_rms_total'],
                                      panning_['panning_rms_lows'],
                                      panning_['panning_rms_mids'],
                                      panning_['panning_rms_highs'],
                                     ])
    
    return panning_
# DYNAMIC
def get_rms_dynamic_crest(x, frame_length, hop_length):
    """
    computes rms level, dynamic spread and crest factor
    
    See: Ma, Zheng, et al. "Intelligent multitrack dynamic range compression." Journal of the Audio Engineering Society 
    
    Args:
        x: input audio array
        frame_length: frame size
        hop_length: frame hop
    Returns:
        rms, dynamic_spread, crest arrays
    """ 
    
    rms = []
    dynamic_spread = []
    crest = []
    for ch in range(x.shape[-1]):
        frames = librosa.util.frame(x[:, ch], frame_length=frame_length, hop_length=hop_length)
        rms_ = []
        dynamic_spread_ = []
        crest_ = []
        for i in frames.T:
            x_rms = amp_to_db(np.sqrt(np.sum(i**2)/frame_length))   
            x_d = np.sum(amp_to_db(np.abs(i)) - x_rms)/frame_length
            x_c = amp_to_db(np.max(np.abs(i)))/x_rms
            
            rms_.append(x_rms)
            dynamic_spread_.append(x_d)
            crest_.append(x_c)
        rms.append(rms_)
        dynamic_spread.append(dynamic_spread_)
        crest.append(crest_)
        
    rms = np.asarray(rms)
    dynamic_spread = np.asarray(dynamic_spread)
    crest = np.asarray(crest)  
    
    rms = np.mean(rms, axis=0)
    dynamic_spread = np.mean(dynamic_spread, axis=0)
    crest = np.mean(crest, axis=0)
    
    rms = np.expand_dims(rms, axis=0)
    dynamic_spread = np.expand_dims(dynamic_spread, axis=0)
    crest = np.expand_dims(crest, axis=0)
    
    return rms, dynamic_spread, crest
def lowpassFiltering(x, f0, sr):
    """
    low pass filters
    Args:
        x: input audio array
        f0:cut-off frequency
        sr: sampling rate
    Returns:
        filtered audio array
    """ 
    
    b1, a1 = scipy.signal.butter(4, f0/(sr/2),'lowpass')
    x_f = []
    for ch in range(x.shape[-1]):
        x_f_ = scipy.signal.filtfilt(b1, a1, x[:, ch]).copy(order='F')
        x_f.append(x_f_)
    return np.asarray(x_f).T  
def compute_dynamic_features(audio_out, audio_tar, sr, fft_size=4096, hop_length=1024):
    """
    Computes dynamic features' mape error using a running mean
    Args:
        audio_out: automix audio (output of models)
        audio_tar: target audio
        sr: sampling rate
        fft_size: fft_size size
        hop_length: fft hop size
    Returns:
        spectral_ dictionary
    """ 

    audio_out = audio_out.copy()
    audio_tar = audio_tar.copy()

    audio_out_ = pyln.normalize.peak(audio_out, -1.0)
    audio_tar_ = pyln.normalize.peak(audio_tar, -1.0)
    
    dynamic_ = {}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
    
        rms_tar, dyn_tar, crest_tar = get_rms_dynamic_crest(audio_tar_, fft_size, hop_length)
        rms_out, dyn_out, crest_out = get_rms_dynamic_crest(audio_out_, fft_size, hop_length)
        
    N = 40
    
    eps = 1e-10
    
    rms_tar = (-1*rms_tar) + 1.0
    rms_out = (-1*rms_out) + 1.0
    dyn_tar = (-1*dyn_tar) + 1.0
    dyn_out = (-1*dyn_out) + 1.0
   
    mean_rms_tar, std_rms_tar = get_running_stats(rms_tar.T, [0], N=N)
    mean_rms_out, std_rms_out = get_running_stats(rms_out.T, [0], N=N)
    
    mean_dyn_tar, std_dyn_tar = get_running_stats(dyn_tar.T, [0], N=N)
    mean_dyn_out, std_dyn_out = get_running_stats(dyn_out.T, [0], N=N)
    
    mean_crest_tar, std_crest_tar = get_running_stats(crest_tar.T, [0], N=N)
    mean_crest_out, std_crest_out = get_running_stats(crest_out.T, [0], N=N)

    dynamic_['rms_level'] = sklearn.metrics.mean_absolute_percentage_error(mean_rms_tar, mean_rms_out)
    dynamic_['dynamic_spread'] = sklearn.metrics.mean_absolute_percentage_error(mean_dyn_tar, mean_dyn_out)
    dynamic_['crest_factor'] = sklearn.metrics.mean_absolute_percentage_error(mean_crest_tar, mean_crest_out)
    dynamic_['rms_level'] = get_mape(mean_rms_tar, mean_rms_out)
    dynamic_['dynamic_spread'] = get_mape(mean_dyn_tar, mean_dyn_out)
    dynamic_['crest_factor'] = get_mape(mean_crest_tar, mean_crest_out)

    dynamic_['mean_mape_dynamic'] = np.mean([dynamic_['rms_level'],
                                      dynamic_['dynamic_spread'],
                                      dynamic_['crest_factor'],
                                     ])
    
    return dynamic_
def get_features(target, output, sr=44100,
                 frame_size=17640, frame_hop=8820,
                 fft_size=4096, fft_hop=1024):
    
    """
    Computes all features' mape error using a running mean
    Args:
        output: automix audio (output of models)
        target: target audio
        sr: sampling rate
        frame_size: frame size for loudness computations, ideally larger than 400ms (LUFS) 
        hop_size: frame hop
        fft_size: fft_size size
        hop_length: fft hop size
    Returns:
        features dictionary
    """ 
    
    # Finds the starting and ending silences in target_mix and trims both mixes

    features = {}

    x = target.astype(np.float64)
    y = output.astype(np.float64)

    x = target.T
    y = output.T

    x, idx = librosa.effects.trim(x.T, top_db=45, frame_length=4096, hop_length=1024)
    x = x.T
    y = y[idx[0]:idx[1],:]
    assert x.shape == y.shape

    # Compute Loudness features

    # print("Computing Loudness features...")

    loudness_features = compute_loudness_features(y, x, sr, frame_size, frame_hop)
    for k, i in loudness_features.items():
        features[k] = i

    # Compute spectral features

    # print("Computing Spectral features...")

    n_channels = x.shape[1]
    spectral_features = compute_spectral_features(y, x, sr, fft_size, fft_hop, n_channels)
    for k, i in spectral_features.items():
        features[k] = i

    # Computes panning features

    # print("Computing Panning features...")

    panning_features = compute_panning_features(y, x, sr, fft_size, fft_hop)  

    for k, i in panning_features.items():
        features[k] = i

    # Computes dynamic features

    # print("Computing Dynamic features...")

    dynamic_features = compute_dynamic_features(y, x, sr, fft_size, fft_hop)
    for k, i in dynamic_features.items():
        features[k] = i
        
    return features

class PairwiseMetric:
    """
    Base class for pairwise metrics.
    
    This class should be subclassed to implement specific pairwise metrics.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the PairwiseMetric instance.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def compute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")


def load_fx_encoder(model_args, device):
    """
    Load the FX Encoder model.
    
    Args:
        model_args: Arguments for the FX Encoder model.
        device: Device to load the model on (CPU or GPU).
        
    Returns:
        a function that extracts features from audio.
    """
    assert model_args is not None, "model_args must be provided for fx_encoder type"

    ckpt_path=model_args.ckpt_path

    #from utils.feature_extractors.fx_encoder import load_effects_encoder
    from utils.feature_extractors.networks import Effects_Encoder

    def reload_weights(model, ckpt_path, device):
        checkpoint = torch.load(ckpt_path, map_location=device)
    
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)


    with open(os.path.join('.','utils','feature_extractors', 'networks', 'configs.yaml'), 'r') as f:
        configs = yaml.full_load(f)

    cfg_enc = configs['Effects_Encoder']['default']

    effects_encoder = Effects_Encoder(cfg_enc)
    reload_weights(effects_encoder, ckpt_path, device)
    effects_encoder.to(device)
    effects_encoder.eval()

    return lambda x: effects_encoder(x)

def load_AFxRep(model_args, device, sample_rate=44100):

    assert model_args is not None, "model_args must be provided for AFxRep type"

    ckpt_path=model_args.ckpt_path

    config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    encoder_configs = config["model"]["init_args"]["encoder"]

    module_path, class_name = encoder_configs["class_path"].rsplit(".", 1)
    module_path = module_path.replace("lcap", "utils.st_ito")

    module = import_module(module_path)

    model = getattr(module, class_name)(**encoder_configs["init_args"])

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # load state dicts
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("encoder"):
            state_dict[k.replace("encoder.", "", 1)] = v

    model.load_state_dict(state_dict)

    model.eval()

    model.to(device)

    def wrapper_fn(x, sample_rate):

        x=x.to(device)

        #x=torch.transpose(x,-1,-2)

        if sample_rate != 48000:
            x=torchaudio.functional.resample(x, sample_rate, 48000)

        bs= x.shape[0]
        #peak normalization. I do it because this is what ST-ITO get_param_embeds does. Not sure if it is good that this representation is invariant to gain
        for batch_idx in range(bs):
            x[batch_idx, ...] /= x[batch_idx, ...].abs().max().clamp(1e-8)

        mid_embeddings, side_embeddings = model(x)

        # check for nan
        if torch.isnan(mid_embeddings).any():
            print("Warning: NaNs found in mid_embeddings")
            mid_embeddings = torch.nan_to_num(mid_embeddings)
        elif torch.isnan(side_embeddings).any():
            print("Warning: NaNs found in side_embeddings")
            side_embeddings = torch.nan_to_num(side_embeddings)

        mid_embeddings = torch.nn.functional.normalize(mid_embeddings, p=2, dim=-1)
        side_embeddings = torch.nn.functional.normalize(side_embeddings, p=2, dim=-1)
        
        embeddings_all= torch.cat([mid_embeddings, side_embeddings], dim=-1)

        return embeddings_all

    feat_extractor = lambda x: wrapper_fn(x, sample_rate=sample_rate)

    return feat_extractor



class PairwiseFeatures(PairwiseMetric):
    """
    Class for computing the pairwise spectral metric.
    
    This class inherits from PairwiseMetric and implements the compute method
    to calculate the pairwise spectral metric.
    """
    def __init__(self,
        type=None,
        sample_rate=44100,
                  *args, **kwargs):
        """
        Initialize the PairwiseSpectral instance.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.type = type
        self.sample_rate = sample_rate

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.type == "fx_encoder":
            self.model_args= kwargs.get("fx_encoder_args", None)

            assert self.model_args is not None, "model_args must be provided for fx_encoder type"

            self.distance_type=self.model_args.distance_type

            self.feat_extractor = load_fx_encoder(self.model_args, self.device)


            #self.feat_extractor = load_effects_encoder(ckpt_path=ckpt_path).to(self.device)
        
        elif self.type== "AFxRep-mid" or self.type== "AFxRep-side" or self.type== "AFxRep":

            self.model_args= kwargs.get("AFxRep_args", None)

            assert self.model_args is not None, "model_args must be provided for AFxRep type"

            self.distance_type=self.model_args.distance_type

            feat_extractor = load_AFxRep(self.model_args, self.device)

            if self.type == "AFxRep-mid":
                def feat_extractor_mid(x):

                    features= feat_extractor(x, sample_rate)

                    #divide by 2 to get mid and side features

                    feat_mid, feat_side = features.chunk(2, dim=-1)

                    return feat_mid

                self.feat_extractor = feat_extractor_mid
            
            elif self.type == "AFxRep-side":
                def feat_extractor_side(x):

                    features= feat_extractor(x, sample_rate)

                    #divide by 2 to get mid and side features

                    feat_mid, feat_side = features.chunk(2, dim=-1)

                    return feat_side

                self.feat_extractor = feat_extractor_side
            else:
                self.feat_extractor = feat_extractor


        super().__init__(*args, **kwargs)
    
    def compute_feature_distance(self, y, y_hat, sample_rate, type):    

        y=torch.tensor(y).permute(1,0).unsqueeze(0).to(self.device)
        y_hat=torch.tensor(y_hat).permute(1,0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_y= self.feat_extractor(y)
            feat_y_hat= self.feat_extractor(y_hat)

        if self.distance_type == "cosine":
            cos_dist= 1- torch.cosine_similarity(feat_y_hat, feat_y, dim=1)

            print("cos_dist", cos_dist.shape, cos_dist, cos_dist.mean().item())

            return {"distance": cos_dist.mean().item()}



    def compute(self, dict_y, dict_y_hat, dict_x,   *args, **kwargs):
        """
        Compute the pairwise spectral metric.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The computed pairwise spectral metric.
        """



        dict_features={}

        for key in dict_y.keys():
            y= dict_y[key]
            y_hat= dict_y_hat[key]

            assert y.shape == y_hat.shape, f"Shape mismatch for key {key}: {y.shape} vs {y_hat.shape}"

            c, d=y.shape
            #assert b==1, f"Expected batch size of 1, got {b} for key {key}"

            assert c==2, f"Expected 2 channels, got {c} for key {key}"

            y=y.T
            y_hat=y_hat.T


            if self.type == "spectral":
                from evaluation.automix_evaluation import compute_spectral_features 
                dict_features_out= compute_spectral_features(y_hat, y ,self.sample_rate)
                dict_features[key] = dict_features_out['mean_mape_spectral']
            elif self.type=="panning":
                from evaluation.automix_evaluation import compute_panning_features 
                dict_features_out = compute_panning_features(y_hat, y, self.sample_rate)
                dict_features[key] = dict_features_out['mean_mape_panning']
            elif self.type=="loudness":
                from evaluation.automix_evaluation import compute_loudness_features 
                dict_features_out = compute_loudness_features(y_hat, y, self.sample_rate)
                dict_features[key] = dict_features_out['mean_mape_loudness']
            elif self.type=="dynamic":
                from evaluation.automix_evaluation import compute_dynamic_features 
                dict_features_out = compute_dynamic_features(y_hat, y, self.sample_rate)
                dict_features[key] = dict_features_out['mean_mape_dynamic']
            elif self.type=="fx_encoder":
                dict_features_out = self.compute_feature_distance(y, y_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="AFxRep":
                dict_features_out = self.compute_feature_distance(y, y_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            

        # Compute the mean of the features across all keys

        mean_features = sum(dict_features.values()) / len(dict_features)

        return  mean_features, dict_features

class PairwiseLDR(PairwiseMetric):
    """
    Class for computing the pairwise LDR metric.
    
    This class inherits from PairwiseMetric and implements the compute method
    to calculate the pairwise LDR metric.
    """
    def __init__(self, mode=None, *args, **kwargs):
        """
        Initialize the PairwiseLDR instance.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

        assert mode is not None, "Mode must be specified for PairwiseLDR"

        if mode == "mldr-lr":
            from evaluation.ldr import MLDRLoss
            self.metric= MLDRLoss(
                sr=44100,
                s_taus=[50, 100],
                l_taus=[1000, 2000],
            ).cuda()
        elif mode == "mldr-ms":
            from evaluation.ldr import MLDRLoss
            self.metric= MLDRLoss(
                sr=44100,
                s_taus=[50, 100],
                l_taus=[1000, 2000],
                mid_side=True
            ).cuda()

    def compute(self, dict_y, dict_y_hat, dict_x,   *args, **kwargs):
        """
        Compute the pairwise spectral metric.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The computed pairwise spectral metric.
        """

        dict_metrics={}

        for key in dict_y.keys():
            y= dict_y[key]
            y_hat= dict_y_hat[key]

            assert y.shape == y_hat.shape, f"Shape mismatch for key {key}: {y.shape} vs {y_hat.shape}"

            c, d=y.shape
            #assert b==1, f"Expected batch size of 1, got {b} for key {key}"

            assert c==2, f"Expected 2 channels, got {c} for key {key}"

            y=y.T
            y_hat=y_hat.T
            
            y=torch.from_numpy(y).cuda().unsqueeze(0)
            y_hat=torch.from_numpy(y_hat).cuda().unsqueeze(0)

            metric=self.metric(y_hat, y)

            dict_metrics[key] = metric.item()

        mean_features = sum(dict_metrics.values()) / len(dict_metrics)

        return  mean_features, dict_metrics


class PairwiseAuraloss(PairwiseMetric):
    """
    Class for computing the pairwise MSS metric.
    
    This class inherits from PairwiseMetric and implements the compute method
    to calculate the pairwise MSS metric.
    """
    def __init__(self, mode=None, *args, **kwargs):
        """
        Initialize the PairwiseMSS instance.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

        assert mode is not None, "Mode must be specified for PairwiseMSS"

        if mode == "mss-lr":
            from auraloss.freq import MultiResolutionSTFTLoss
            self.metric=MultiResolutionSTFTLoss(
                [128, 512, 2048],
                [32, 128, 512],
                [128, 512, 2048],
                sample_rate=44100,
                perceptual_weighting=True,
            ).cuda()
        elif mode == "mss-ms":
            from auraloss.freq import  SumAndDifferenceSTFTLoss
            self.metric=SumAndDifferenceSTFTLoss(
            [128, 512, 2048],
            [32, 128, 512],
            [128, 512, 2048],
            sample_rate=44100,
            perceptual_weighting=True,
            ).cuda()

    def compute(self, dict_y, dict_y_hat, dict_x,   *args, **kwargs):
        """
        Compute the pairwise spectral metric.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The computed pairwise spectral metric.
        """

        dict_metrics={}

        for key in dict_y.keys():
            y= dict_y[key]
            y_hat= dict_y_hat[key]

            assert y.shape == y_hat.shape, f"Shape mismatch for key {key}: {y.shape} vs {y_hat.shape}"

            c, d=y.shape
            #assert b==1, f"Expected batch size of 1, got {b} for key {key}"

            assert c==2, f"Expected 2 channels, got {c} for key {key}"

            #y=y.T
            #y_hat=y_hat.T
            
            y=torch.from_numpy(y).cuda().unsqueeze(0)
            y_hat=torch.from_numpy(y_hat).cuda().unsqueeze(0)

            metric=self.metric(y_hat, y)

            dict_metrics[key] = metric.item()

        mean_features = sum(dict_metrics.values()) / len(dict_metrics)

        return  mean_features, dict_metrics

def metric_factory(metric_name, sample_rate, *args, **kwargs):
    """
    Factory function to create a metric function based on the metric name.
    
    Args:
        metric_name (str): The name of the metric to create.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        
    Returns:
        An instance of a class that implements the metric function.
    """
    if metric_name == "pairwise-spectral":
        return PairwiseFeatures(*args, **kwargs, type="spectral", sample_rate=sample_rate)
    elif metric_name == "pairwise-panning":
        return PairwiseFeatures(*args, **kwargs, type="panning", sample_rate=sample_rate)
    elif metric_name == "pairwise-loudness":
        return PairwiseFeatures(*args, **kwargs, type="loudness", sample_rate=sample_rate)
    elif metric_name == "pairwise-dynamic":
        return PairwiseFeatures(*args, **kwargs, type="dynamic", sample_rate=sample_rate)
    elif metric_name == "pairwise-mss-lr":
        return PairwiseAuraloss(mode="mss-lr",*args, **kwargs)
    elif metric_name == "pairwise-mss-ms":
        return PairwiseAuraloss(mode="mss-ms",*args, **kwargs)
    elif metric_name == "pairwise-mldr-lr":
        return PairwiseLDR(mode="mldr-lr",*args, **kwargs)
    elif metric_name == "pairwise-mldr-ms":
        return PairwiseLDR(mode="mldr-ms",*args, **kwargs)
    elif metric_name == "pairwise-fx_encoder":
        return PairwiseFeatures(*args, **kwargs, type="fx_encoder", sample_rate=sample_rate, model_args=kwargs.get('fx_encoder_args', None))
    elif metric_name == "pairwise-AFxRep":
        return PairwiseFeatures(*args, **kwargs, type="AFxRep", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None))
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

# Example usage:
#metric_instance = metric_factory("pairwise-spectral")
#```
