
import pyloudnorm as pyln
import soundfile as sf
import torch
import numpy as np
import os
import glob
import yaml
from pathlib import Path
import random

from utils.data_utils import read_wav_segment, get_audio_length
import torch.distributed as dist

import pickle

from tqdm import tqdm

 
def find_time_offset(x: torch.Tensor, y: torch.Tensor):
    x = x.double()
    y = y.double()
    N = x.size(-1)
    M = y.size(-1)

    X = torch.fft.rfft(x, n=N + M - 1)
    Y = torch.fft.rfft(y, n=N + M - 1)
    corr = torch.fft.irfft(X.conj() * Y)
    shifts = torch.argmax(corr, dim=-1)

    return torch.where(shifts >= N, shifts - N - M + 1, shifts)

def process_id_list_lead_vocal(
    song_id_list
):
    wet_subdir = "multi"
    dry_subdir = "dry"

    dict_filt = lambda d, f: {k: v for k, v in d.items() if f(k, v)}


    results = []

    for song_id in song_id_list:
        #print(song_id)
        song_id=Path(song_id)
        match_yaml = song_id / "correspondence_pp.yaml"


        if match_yaml.exists():
            with open(match_yaml) as f:
                try:
                    hierachy = yaml.safe_load(f)
                except:
                    print(f"skip {song_id} as the yaml file format is not correct")
                    continue

                one2one = dict_filt(hierachy["matched"], lambda _, v: len(v) == 1)
                lead_vocal_only = dict_filt(
                    one2one,
                    lambda k, _: (
                        "lv" in k.lower()
                        or "ld" in k.lower()
                        or "lead_vocal" in k.lower()
                        or "9210" in k.lower()
                        or "9220" in k.lower()
                    )
                    and not (
                        "synth" in k.lower()
                        or "gtr" in k.lower()
                        or "guitar" in k.lower()
                        or "piano" in k.lower()
                        or "strings" in k.lower()
                        or "eg" in k.lower()
                        or "ag" in k.lower()
                        or "bv" in k.lower()
                    ),
                )
    
                #print("lead_vocal_only", len(lead_vocal_only))  

                something=False
                for k, v in lead_vocal_only.items():
                    wet_file = song_id / wet_subdir / k.split("/")[-1]
                    dry_file = song_id / dry_subdir / v[0].split("/")[-1]

                    results.append((dry_file, wet_file))
                    something=True
                if not something:
                    print(f"skip {song_id} as no lead vocal found")
                    continue
        else:
            print(f"skip {song_id} as no yaml file found")
            continue

    print("results", len(results))
    return results


def check_side_energy( x_dry, dry_file, side_energy_threshold=-30):
            if x_dry.size(0) > 1:
                left = x_dry[0]
                right = x_dry[1]
                left = left / left.max()
                right = right / right.max()
                side = (left - right) * 0.707
                side_energy = 20*torch.log10(side.abs().max()).item()
            else:
                side_energy = -torch.inf
            
            if side_energy > side_energy_threshold:
                print(f"Skip {dry_file} because of high side energy"+str(side_energy))
                return False
            else:
                return True


def load_audio( file, start=None, end=None, stereo=True):

            x, fs=read_wav_segment(file, start, end)
            if stereo:
                if len(x.shape)==1:
                    #print( "dry not stereo , doubling channels", x_dry.shape)
                    x=x[:,np.newaxis]
                    x= np.concatenate((x, x), axis=-1)
                elif len(x.shape)==2 and x.shape[-1]==1:
                    #print( "dry not stereo , doubling channels", x_dry.shape)
                    x = np.concatenate((x, x), axis=-1)

            x=torch.from_numpy(x).permute(1,0)

            return x, fs


class TencyMastering_Vocals_Test(torch.utils.data.Dataset):

    def __init__(self,
        fs=44100,
        segment_length=131072,
        base_dir=None,
        mode="dry-wet",
        normalize_params=None,
        align=False,
        align_mode=None,
        stereo=True,
        num_examples=4,
        RMS_threshold_dB=-40,
        seed=42
        ):

        super().__init__()
        print(num_examples, "num_examples")

        np.random.seed(seed)
        random.seed(seed)



        id_list=[]
        id_list=glob.glob(os.path.join(base_dir, "dry","*"))
    
        assert len(id_list)>0, "No files found in the dataset"

        self.id_list=id_list

        self.segment_length=segment_length
        self.fs=fs

        self.align=align
        if self.align:
            assert align_mode in ["cross_correlation", "pickle"], "align_mode must be either 'cross_correlation' or 'pickle'"
            self.align_mode=align_mode

        self.normalize_mode=normalize_params.normalize_mode

        self.stereo=stereo

        if self.normalize_mode=="loudness_dry":
            meter = pyln.Meter(fs)
            def normaliser_fn(x):
                x=x.numpy().T
                x=pyln.normalize.loudness(x, meter.integrated_loudness(x), normalize_params.loudness_dry)
                x=torch.from_numpy(x.T).float()
                return x

            self.normaliser = normaliser_fn

        elif self.normalize_mode=="rms_dry":
            RMS_target=normalize_params.rms_dry #-16 dB
            def normaliser_fn(x):
                RMS=20*torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1)))
                scaler=10**((RMS_target-RMS)/20)
                x=x*scaler.view(-1,1)
                return x

            self.normaliser = normaliser_fn
            
        elif self.normalize_mode is None:
            self.normaliser = lambda x: x
        

        self.RMS_threshold_dB=RMS_threshold_dB

        self.get_RMS=lambda x: 20 * torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1)))

        self.test_samples = []

        counter=0

        #for dry_file, wet_file in tqdm(self.pair_list):
        for id in tqdm(self.id_list):
            #if counter >= num_examples and num_examples!= -1:
            #    break
            print("mode", mode)
            if mode=="dry-wet":
                dry_file=id / "dry" / "vocals.wav"

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)    

                out= load_audio(dry_file, stereo=self.stereo)

                if out is None:
                    continue
                else:
                    x_dry_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
                x_long= x_dry_long

            
            elif mode=="dryfxnorm-wet":
                
                dry_file=id / "dry" / "vocals_normalized.wav"

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)

                out= load_audio(dry_file, stereo=self.stereo)

                if out is None:
                    continue
                else:
                    x_dry_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
                x_long= x_dry_long

                
            elif mode=="wetfxnorm-wet":

                dry_file=id / "wet" / "vocals_normalized.wav"

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)    

                out= load_audio(dry_file, stereo=self.stereo)

                if out is None:
                    continue
                else:
                    x_dry_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
                x_long= x_dry_long

                
            wet_file=id / "wet" / "vocals.wav"

            assert wet_file.exists(), "wet file does not exist: {}".format(wet_file)

            out= load_audio(wet_file, stereo=self.stereo)

            if out is None:
                    continue
            else:
                    y_wet_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
            y_long=y_wet_long
            
            assert self.align==False, "align is not supported yet for this dataset"


            #pad x_dry if it is not a multiple of segment_length
            max_length = max(x_long.size(-1), y_wet_long.size(-1))

            #define max_length as the next multiple of segment_length (padding)
            max_length = (max_length + self.segment_length - 1) // self.segment_length * self.segment_length

            #print("x_dry_long", x_dry_long.shape, "y_wet_long", y_wet_long.shape, "max_length", max_length)
            x_long = torch.nn.functional.pad(x_long, (0, max_length - x_long.size(-1)), mode='constant', value=0)
            y_long = torch.nn.functional.pad(y_long, (0, max_length - y_long.size(-1)), mode='constant', value=0)

            #print("x_dry_long", x_dry_long.shape, "y_wet_long", y_wet_long.shape)

            assert x_long.size(-1)% self.segment_length == 0, "x_dry_long must be a multiple of segment_length, got {}".format(x_long.size(-1))
            assert y_long.size(-1)% self.segment_length == 0, "y_wet_long must be a multiple of segment_length, got {}".format(y_wet_long.size(-1))

            #assert now the two have the same length
            assert x_long.shape==y_wet_long.shape, "x_dry and y_wet must have the same shape, got {} and {}".format(x_long.shape, y_wet_long.shape)

            #divide into non-overlapping segments of segment_length

            for i in range(0, x_long.size(-1), self.segment_length):
                x_dry = x_long[:, i:i + self.segment_length]
                y_wet = y_wet_long[:, i:i + self.segment_length]

                RMS_dB=self.get_RMS(y_wet.mean(0))
                if RMS_dB<self.RMS_threshold_dB:
                    continue

                if self.normalize_mode is not None:
                    if "dry" in self.normalize_mode:
                        #potentially slow
                        x_dry=self.normaliser(x_dry)
    
                        if torch.isnan(x_dry).any():
                            continue
                    else:
                        pass

                #print("x_dry", x_dry.shape, "y_wet", y_wet.shape)
                self.test_samples.append(( y_wet, x_dry)) 
                counter+=1

        #permute the test samples
        random.shuffle(self.test_samples)

        if num_examples != -1:
            self.test_samples = self.test_samples[:num_examples]

        print("test_samples", len(self.test_samples), "num_examples", num_examples)

    def __getitem__(self, idx):
        return self.test_samples[idx]

    def __len__(self):
        return len(self.test_samples)

class TencyMastering_Vocals(torch.utils.data.IterableDataset):

    def __init__(self,
        fs=44100,
        segment_length=131072,
        base_dir=None,
        mode="dry-wet",
        normalize_params=None,
        align=False,
        align_mode=None,
        stereo=True,
        num_examples=4,
        RMS_threshold_dB=-40,
        seed=42
        ):

        super().__init__()
        print(num_examples, "num_examples")

        np.random.seed(seed)
        random.seed(seed)



        id_list=[]
        id_list=glob.glob(os.path.join(base_dir, "dry","*"))
    
        assert len(id_list)>0, "No files found in the dataset"

        self.id_list=id_list

        self.segment_length=segment_length
        self.fs=fs

        self.align=align
        if self.align:
            assert align_mode in ["cross_correlation", "pickle"], "align_mode must be either 'cross_correlation' or 'pickle'"
            self.align_mode=align_mode

        self.normalize_mode=normalize_params.normalize_mode

        self.stereo=stereo

        if self.normalize_mode=="loudness_dry":
            meter = pyln.Meter(fs)
            def normaliser_fn(x):
                x=x.numpy().T
                x=pyln.normalize.loudness(x, meter.integrated_loudness(x), normalize_params.loudness_dry)
                x=torch.from_numpy(x.T).float()
                return x

            self.normaliser = normaliser_fn

        elif self.normalize_mode=="rms_dry":
            RMS_target=normalize_params.rms_dry #-16 dB
            def normaliser_fn(x):
                RMS=20*torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1)))
                scaler=10**((RMS_target-RMS)/20)
                x=x*scaler.view(-1,1)
                return x

            self.normaliser = normaliser_fn
            
        elif self.normalize_mode is None:
            self.normaliser = lambda x: x
        

        self.RMS_threshold_dB=RMS_threshold_dB

        self.get_RMS=lambda x: 20 * torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1)))

        self.mode=mode


    def __iter__(self):

        while True:

            #for id in tqdm(self.id_list):

            num=np.random.randint(0,len(self.id_list)-1)

            id=self.id_list[num]

            #if counter >= num_examples and num_examples!= -1:
            #    break
            #print("mode", mode)

            if self.mode=="dry-wet":
                dry_file=id / "dry" / "vocals.wav"

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)    

            
            elif self.mode=="dryfxnorm-wet":
                
                dry_file=id / "dry" / "vocals_normalized.wav"

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)

                
            elif self.mode=="wetfxnorm-wet":

                dry_file=id / "wet" / "vocals_normalized.wav"

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)    


            wet_file=id / "wet" / "vocals.wav"
            assert wet_file.exists(), "wet file does not exist: {}".format(wet_file)


            dry_duration, dry_total_frames, dry_samplerate=get_audio_length(dry_file)
            wet_duration, wet_total_frames, wet_samplerate=get_audio_length(wet_file)

            assert dry_total_frames== wet_total_frames, "dry and wet files must have the same number of frames, got {} and {}".format(dry_total_frames, wet_total_frames)

            total_frames=dry_total_frames

            start=np.random.randint(0,total_frames-self.segment_length)
            end=start+ self.segment_length

            out=load_audio(dry_file, start, end, stereo=self.stereo)

            if out is None:
                continue
            else:
                x_dry, fs=out
                assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                

            out= load_audio(wet_file, start, end, stereo=self.stereo)

            if out is None:
                continue
            else:
                y_wet, fs=out
                assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
            
            assert self.align==False, "align is not supported yet for this dataset"

            RMS_dB=self.get_RMS(y_wet.mean(0))
            if RMS_dB<self.RMS_threshold_dB:
                continue

            if self.normalize_mode is not None:

                    if "dry" in self.normalize_mode:
                        #potentially slow
                        x_dry=self.normaliser(x_dry)
                        
                        #detect NaNs here

                        if torch.isnan(x_dry).any():
                            continue

                    else:
                        pass
                

            yield  y_wet, x_dry

            #except Exception as e:
            #    print(e)
            #    continue

    def __getitem__(self, idx):
        return self.test_samples[idx]

    def __len__(self):
        return len(self.test_samples)


