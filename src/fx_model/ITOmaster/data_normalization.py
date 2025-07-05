"""
    Implementation of the 'audio effects chain normalization'
"""
import numpy as np
import scipy
import soundfile as sf
import pyloudnorm as pyln

from glob import glob
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from utils_data_normalization import amp_to_db, compute_stft, get_SPS, get_mean_peak, \
                                        get_eq_matching, get_comp_matching
from normalization_imager import normalize_imager, lr_to_ms, print_balance


'''
    Audio Effects Chain Normalization
    process: normalizes input stems according to given precomputed features
'''
class Audio_Effects_Normalizer:
    def __init__(self, precomputed_feature_path=None, \
                    STEMS=['drums', 'bass', 'other', 'vocals'], \
                    EFFECTS=['eq', 'compression', 'imager', 'loudness'], \
                    audio_extension='wav'):
        self.STEMS = STEMS # Stems to be normalized
        self.EFFECTS = EFFECTS # Effects to be normalized, order matters
        self.audio_extension = audio_extension
        self.precomputed_feature_path = precomputed_feature_path

        # Audio settings
        self.SR = 44100
        self.SUBTYPE = 'PCM_16'

        # General Settings
        self.FFT_SIZE = 2**16
        self.HOP_LENGTH = self.FFT_SIZE//4

        # Loudness
        self.NTAPS = 1001
        self.LUFS = -30
        self.MIN_DB = -40 # Min amplitude to apply EQ matching

        # Compressor
        self.COMP_USE_EXPANDER = False
        self.COMP_PEAK_NORM = -10.0
        self.COMP_TRUE_PEAK = False
        self.COMP_PERCENTILE = 75 # features_mean (v1) was done with 25
        self.COMP_MIN_TH = -40
        self.COMP_MAX_RATIO = 20
        comp_settings = {key:{} for key in self.STEMS}
        for key in comp_settings:
            if key=='vocals':
                comp_settings[key]['attack'] = 7.5
                comp_settings[key]['release'] = 400.0
                comp_settings[key]['ratio'] = 4
                comp_settings[key]['n_mels'] = 128
            elif key=='drums':
                comp_settings[key]['attack'] = 10.0
                comp_settings[key]['release'] = 180.0
                comp_settings[key]['ratio'] = 6
                comp_settings[key]['n_mels'] = 128
            elif key=='bass':
                comp_settings[key]['attack'] = 10.0
                comp_settings[key]['release'] = 500.0
                comp_settings[key]['ratio'] = 5
                comp_settings[key]['n_mels'] = 16
            elif key=='other' or key=='mixture':
                comp_settings[key]['attack'] = 15.0
                comp_settings[key]['release'] = 666.0
                comp_settings[key]['ratio'] = 4
                comp_settings[key]['n_mels'] = 128
        self.comp_settings = comp_settings

        if precomputed_feature_path!=None and os.path.isfile(precomputed_feature_path):
            # Load Pre-computed Audio Effects Features
            features_mean = np.load(precomputed_feature_path, allow_pickle='TRUE')[()]
            self.features_mean = self.smooth_feature(features_mean)


    # compute audio effects' mean feature values
    def compute_mean(self, base_dir_path, save_feat=True, single_file=False):

        audio_path_dict = {}
        for cur_stem in self.STEMS:
            # if single_file=True, base_dir_path = the target file path
            audio_path_dict[cur_stem] = [base_dir_path] if single_file else glob(os.path.join(base_dir_path, "**", f"{cur_stem}.{self.audio_extension}"), recursive=True)

        features_dict = {}
        features_mean = {}
        for effect in self.EFFECTS:
            features_dict[effect] = {key:[] for key in self.STEMS}
            features_mean[effect] = {key:[] for key in self.STEMS}

        stems_names = self.STEMS.copy()
        for effect in self.EFFECTS:
            print(f'{effect} ...')
            j=0
            for key in self.STEMS:
                print(f'{key} ...')
                i = []
                for i_, p_ in enumerate(audio_path_dict[key]):
                    i.append(i_)  
                i = np.asarray(i) + j
                j += len(i)

                features_ = []
                for cur_i, cur_audio_path in enumerate(audio_path_dict[key]):
                    print(f'getting {effect} features for {key}- stem {cur_i} of {len(audio_path_dict[key])-1} {cur_audio_path}') 
                    features_.append(self.get_norm_feature(cur_audio_path, cur_i, effect, key))
                
                features_dict[effect][key] = features_
                
                print(effect, key, len(features_dict[effect][key]))
                s = np.asarray(features_dict[effect][key])
                s = np.mean(s, axis=0)
                features_mean[effect][key] = s
                
                if effect == 'eq':
                    assert len(s)==1+self.FFT_SIZE//2, len(s)
                elif effect == 'compression':
                    assert len(s)==2, len(s)
                elif effect == 'panning':
                    assert len(s)==1+self.FFT_SIZE//2, len(s)
                elif effect == 'loudness':
                    assert len(s)==1, len(s)
                
                if effect == 'eq':
                    if key in ['other', 'vocals', 'mixture']:
                        f = 401
                    else:
                        f = 151
                    features_mean[effect][key] = scipy.signal.savgol_filter(features_mean[effect][key],
                                                                            f, 1, mode='mirror')
                elif effect == 'panning':
                    features_mean[effect][key] = scipy.signal.savgol_filter(features_mean[effect][key],
                                                                            501, 1, mode='mirror')
        if save_feat:
            np.save(self.precomputed_feature_path, features_mean)
        self.features_mean = self.smooth_feature(features_mean)
        print('---feature mean computation completed---')

        return self.features_mean


    def get_norm_feature(self, path, i, effect, stem):
        
        if isinstance(path, str): 
            audio, fs = sf.read(path)
            assert(fs == self.SR)
        else:
            audio = path
            fs = self.SR
        all_zeros = not np.any(audio)

        if all_zeros == False:

            audio = np.pad(audio, ((self.FFT_SIZE, self.FFT_SIZE), (0, 0)), mode='constant')

            max_db = amp_to_db(np.max(np.abs(audio)))

            if max_db > self.MIN_DB:

                if effect == 'loudness':
                    meter = pyln.Meter(self.SR) 
                    loudness = meter.integrated_loudness(audio)
                    return [loudness]
                
                elif effect == 'eq':
                    audio = lufs_normalize(audio, self.SR, self.LUFS, log=False) 
                    audio_spec = compute_stft(audio,
                                    self.HOP_LENGTH,
                                    self.FFT_SIZE,
                                    np.sqrt(np.hanning(self.FFT_SIZE+1)[:-1]))
                    audio_spec = np.abs(audio_spec)
                    audio_spec_avg = np.mean(audio_spec, axis=(0,1))
                    return audio_spec_avg
                
                elif effect == 'panning':
                    phi = get_SPS(audio,
                                n_fft=self.FFT_SIZE,
                                hop_length=self.HOP_LENGTH,
                                smooth=False,
                                frames=False)
                    return(phi[1])
                
                elif effect == 'compression':
                    x = pyln.normalize.peak(audio, self.COMP_PEAK_NORM)
                    peak_std = get_mean_peak(x,
                                            sr=self.SR,
                                            true_peak=self.COMP_TRUE_PEAK,
                                            percentile=self.COMP_PERCENTILE,
                                            n_mels=self.comp_settings[stem]['n_mels'])

                    if peak_std is not None:
                        return peak_std
                    else:
                        return None
                
                elif effect == 'imager':
                    mid, side = lr_to_ms(audio[:,0], audio[:,1])
                    return print_balance(mid, side, verbose=False)
                    
            else:
                print(f'{path} is silence...')
                return None
            
        else:
                
            print(f'{path} is only zeros...')
            return None


    # normalize current audio input with the order of designed audio FX
    def normalize_audio(self, audio, src):
        assert src in self.STEMS

        normalized_audio = audio
        for cur_effect in self.EFFECTS:
            normalized_audio = self.normalize_audio_per_effect(normalized_audio, src=src, effect=cur_effect)

        return normalized_audio


    # normalize current audio input with current targeted audio FX
    def normalize_audio_per_effect(self, audio, src, effect):
        audio = audio.astype(dtype=np.float32)
        audio_track = np.pad(audio, ((self.FFT_SIZE, self.FFT_SIZE), (0, 0)), mode='constant')
        
        assert len(audio_track.shape) == 2  # Always expects two dimensions
        
        if audio_track.shape[1] == 1:    # Converts mono to stereo with repeated channels
            audio_track = np.repeat(audio_track, 2, axis=-1)
            
        output_audio = audio_track.copy()
        
        max_db = amp_to_db(np.max(np.abs(output_audio)))
        if max_db > self.MIN_DB:
        
            if effect == 'eq':
                # normalize each channel
                for ch in range(audio_track.shape[1]):
                    audio_eq_matched = get_eq_matching(output_audio[:, ch],
                                                        self.features_mean[effect][src],
                                                        sr=self.SR,
                                                        n_fft=self.FFT_SIZE,
                                                        hop_length=self.HOP_LENGTH,
                                                        min_db=self.MIN_DB,
                                                        ntaps=self.NTAPS,
                                                        lufs=self.LUFS)
                    np.copyto(output_audio[:,ch], audio_eq_matched)

            elif effect == 'compression':
                assert(len(self.features_mean[effect][src])==2)
                # normalize each channel
                for ch in range(audio_track.shape[1]):
                    try:
                        audio_comp_matched = get_comp_matching(output_audio[:, ch],
                                                                self.features_mean[effect][src][0], 
                                                                self.features_mean[effect][src][1],
                                                                self.comp_settings[src]['ratio'],
                                                                self.comp_settings[src]['attack'],
                                                                self.comp_settings[src]['release'],
                                                                sr=self.SR,
                                                                min_db=self.MIN_DB,
                                                                min_th=self.COMP_MIN_TH, 
                                                                comp_peak_norm=self.COMP_PEAK_NORM,
                                                                max_ratio=self.COMP_MAX_RATIO,
                                                                n_mels=self.comp_settings[src]['n_mels'],
                                                                true_peak=self.COMP_TRUE_PEAK,
                                                                percentile=self.COMP_PERCENTILE, 
                                                                expander=self.COMP_USE_EXPANDER)

                        np.copyto(output_audio[:,ch], audio_comp_matched[:, 0])
                    except:
                        break

            elif effect == 'loudness':
                output_audio = lufs_normalize(output_audio, self.SR, self.features_mean[effect][src], log=False)
                
            elif effect == 'imager':
                # threshold of applying Haas effects
                mono_threshold = 0.99 if src=='bass' else 0.975
                audio_imager_matched = normalize_imager(output_audio, \
                                                        target_side_mid_bal=self.features_mean[effect][src][0], \
                                                        mono_threshold=mono_threshold, \
                                                        sr=self.SR)

                np.copyto(output_audio, audio_imager_matched)
        
        output_audio = output_audio[self.FFT_SIZE:self.FFT_SIZE+audio.shape[0]]

        return output_audio


    def smooth_feature(self, feature_dict_):
        
        for effect in self.EFFECTS:
            for key in self.STEMS:
                if effect == 'eq':
                    if key in ['other', 'vocals', 'mixture']:
                        f = 401
                    else:
                        f = 151
                    feature_dict_[effect][key] = scipy.signal.savgol_filter(feature_dict_[effect][key],
                                                                            f, 1, mode='mirror')
                elif effect == 'panning':
                    feature_dict_[effect][key] = scipy.signal.savgol_filter(feature_dict_[effect][key],
                                                                            501, 1, mode='mirror')
        return feature_dict_


    # compute "normalization" based on a single sample
    def feature_matching(self, src_aud_path, ref_aud_path):
        # compute mean features from reference audio
        mean_feature = self.compute_mean(ref_aud_path, save_feat=False, single_file=True)
        print(mean_feature)

        src_aud, sr = sf.read(src_aud_path)
        normalized_audio = self.normalize_audio(src_aud, 'mixture')

        return normalized_audio


def lufs_normalize(x, sr, lufs=-14., log=False):
    # measure the loudness first 
    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(x+1e-10)
    if log:
        print("original loudness: ", loudness," max value: ", np.max(np.abs(x)))

    loudness_normalized_audio = pyln.normalize.loudness(x, loudness, lufs)
    
    maxabs_amp = np.maximum(1.0, 1e-6 + np.max(np.abs(loudness_normalized_audio)))
    loudness_normalized_audio /= maxabs_amp
    
    loudness = meter.integrated_loudness(loudness_normalized_audio)
    if log:
        print("new loudness: ", loudness," max value: ", np.max(np.abs(loudness_normalized_audio)))

    return loudness_normalized_audio
