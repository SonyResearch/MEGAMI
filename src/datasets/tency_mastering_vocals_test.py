
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
        skip=None,
        base_dir=None,
        processed_dir=None,
        mode="dry-wet",
        subdirs=None,
        skip_list=None,
        normalize_params=None,
        align=False,
        align_mode=None,
        stereo=True,
        num_examples=4,
        RMS_threshold_dB=-40,
        side_energy_threshold=-10,
        seed=42
        ):

        super().__init__()
        print(num_examples, "num_examples")

        np.random.seed(seed)
        random.seed(seed)

        base_path=os.path.join(base_dir)

        id_list=[]
        for subdir in subdirs:
            path=os.path.join(base_path,subdir)
            id_list=glob.glob(os.path.join(path,"*"))

        def filter_ids(path):
            id_track=path.split("/")[-1]
            partition=path.split("/")[-2]
            id_name=partition+"/"+id_track
            if id_name in skip_list:
                return False
            else:
                return True

        id_list=[x for x in id_list if filter_ids(x)]
        assert len(id_list)>0, "No files found in the dataset"

        ids_processed_dir=glob.glob(os.path.join(processed_dir, "dry","*"))

        print("ids_processed_dir", len(ids_processed_dir))

        #take only the ids that are in the processed_dir
        id_list=[x for x in id_list if os.path.join(processed_dir, "dry", x.split("/")[-1]) in ids_processed_dir]

        print("id_list", len(id_list))

        self.pair_list=process_id_list_lead_vocal(id_list)

        print("pair_list", len(self.pair_list))

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

        self.side_energy_threshold=side_energy_threshold

        self.test_samples = []

        counter=0

        for dry_file, wet_file in tqdm(self.pair_list):
            #if counter >= num_examples and num_examples!= -1:
            #    break

            if mode=="dry-wet":

                out= load_audio(dry_file, stereo=self.stereo)

                if out is None:
                    continue
                else:
                    x_dry_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
                x_long= x_dry_long
            
            elif mode=="dryfxnorm-wet":

                id=dry_file.parent.parent.name
                print("id", id)

                dryfxnorm_file=os.path.join(processed_dir, "dry", id, "vocals_normalized.wav")

                out= load_audio(dryfxnorm_file, stereo=self.stereo)

                if out is None:
                    continue
                else:
                    x_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
            elif mode=="wetfxnorm-wet":

                id=dry_file.parent.parent.name
                print("id", id)

                wetfxnorm_file=os.path.join(processed_dir, "wet", id, "vocals_normalized.wav")

                out= load_audio(wetfxnorm_file, stereo=self.stereo)
                if out is None:
                    continue
                else:
                    x_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)

                
            
            out= load_audio(wet_file, stereo=self.stereo)
            if out is None:
                continue
            else:
                y_wet_long, fs=out
                assert fs==self.fs, "wrong sampling rate: {}".format(fs)


            #total_frames=min(dry_total_frames, wet_total_frames)

            #sample "start" uniformly in the range [0, total_frames-seg_len]
            #start=np.random.randint(0,total_frames-self.segment_length)

            #start=total_frames//2 #fixed
            #end=start+ self.segment_length

            if self.align and self.align_mode=="pickle":
                    #aligning using a pickle file with a dictionary, as done for GRAFx

                    #dry_file is path/sond_id/dry/track.wav
                    #pickle_file is path/song_id/alignment.pickle

                    alignment_file=dry_file.parent.parent / "alignment.pickle"
                    alignment_data = pickle.load(open(alignment_file, "rb"))

                    dry_align=alignment_data.get("dry_alignment",0)
                    #multi_align=alignment_data.get("multi_alignment",0)
                    x_long=torch.roll(x_long, shifts=int(dry_align), dims=-1)
            
            elif self.align and self.align_mode=="cross_correlation":
                    #do that on GPU if it is slow
                    shifts = find_time_offset(x_long.mean(0), y_wet_long.mean(0), margin=3000).item()
                    #TODO: ensure shifts is small, otherwise skip
                    x_dry_long = torch.roll(x_long, shifts=int(shifts), dims=1)
                
            #pad x_dry if it is not a multiple of segment_length
            max_length = max(x_long.size(-1), y_wet_long.size(-1))

            #define max_length as the next multiple of segment_length (padding)
            max_length = (max_length + self.segment_length - 1) // self.segment_length * self.segment_length

            #print("x_dry_long", x_dry_long.shape, "y_wet_long", y_wet_long.shape, "max_length", max_length)
            x_long = torch.nn.functional.pad(x_long, (0, max_length - x_long.size(-1)), mode='constant', value=0)
            y_wet_long = torch.nn.functional.pad(y_wet_long, (0, max_length - y_wet_long.size(-1)), mode='constant', value=0)

            #print("x_dry_long", x_dry_long.shape, "y_wet_long", y_wet_long.shape)

            assert x_long.size(-1)% self.segment_length == 0, "x_dry_long must be a multiple of segment_length, got {}".format(x_long.size(-1))
            assert y_wet_long.size(-1)% self.segment_length == 0, "y_wet_long must be a multiple of segment_length, got {}".format(y_wet_long.size(-1))

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

class TencyMastering_Vocals_Test_v1(torch.utils.data.Dataset):

    def __init__(self,
        fs=44100,
        segment_length=131072,
        skip=None,
        base_dir=None,
        subdirs=None,
        skip_list=None,
        normalize_params=None,
        align=False,
        align_mode=None,
        stereo=True,
        num_examples=4,
        RMS_threshold_dB=-40,
        side_energy_threshold=-10,
        seed=42
        ):

        super().__init__()
        #np.random.seed(seed)

        base_path=os.path.join(base_dir)

        id_list=[]
        for subdir in subdirs:
            path=os.path.join(base_path,subdir)
            id_list=glob.glob(os.path.join(path,"*"))

        def filter_ids(path):
            id_track=path.split("/")[-1]
            partition=path.split("/")[-2]
            id_name=partition+"/"+id_track
            if id_name in skip_list:
                return False
            else:
                return True

        id_list=[x for x in id_list if filter_ids(x)]
        assert len(id_list)>0, "No files found in the dataset"

        self.pair_list=process_id_list_lead_vocal(id_list)
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

        self.side_energy_threshold=side_energy_threshold

        self.test_samples = []

        counter=0

        for dry_file, wet_file in self.pair_list:
            if counter >= num_examples and num_examples!= -1:
                break

            dry_duration, dry_total_frames, dry_samplerate=get_audio_length(dry_file)
            wet_duration, wet_total_frames, wet_samplerate=get_audio_length(wet_file)

            total_frames=min(dry_total_frames, wet_total_frames)

            #sample "start" uniformly in the range [0, total_frames-seg_len]
            #start=np.random.randint(0,total_frames-self.segment_length)

            start=total_frames//2 #fixed
            end=start+ self.segment_length

            if self.align and self.align_mode=="pickle":
                    #aligning using a pickle file with a dictionary, as done for GRAFx

                    #dry_file is path/sond_id/dry/track.wav
                    #pickle_file is path/song_id/alignment.pickle

                    alignment_file=dry_file.parent.parent / "alignment.pickle"
                    alignment_data = pickle.load(open(alignment_file, "rb"))

                    dry_align=alignment_data.get("dry_alignment",0)
                    #multi_align=alignment_data.get("multi_alignment",0)
                    print("dry_align", dry_align)

                    dry_start = start - dry_align
                    dry_end= dry_start + self.segment_length
            else:
                dry_start = start
                dry_end= dry_start + self.segment_length


            out= load_audio(dry_file, dry_start, dry_end, stereo=self.stereo)
            if out is None:
                continue
            else:
                x_dry, fs=out

            assert fs==self.fs, "wrong sampling rate: {}".format(fs)
            
            if not check_side_energy(x_dry, dry_file, side_energy_threshold=self.side_energy_threshold):
                continue

            out= load_audio(wet_file, start, end, stereo=self.stereo)
            if out is None:
                continue    
            else:
                y_wet, fs=out

            assert fs==self.fs, "wrong sampling rate: {}".format(fs)

            RMS_dB=self.get_RMS(y_wet.mean(0))
            if RMS_dB<self.RMS_threshold_dB:
                continue

            if self.align:
                if self.align_mode=="cross_correlation":
                    #do that on GPU if it is slow
                    shifts = find_time_offset(x_dry.mean(0), y_wet.mean(0), margin=3000).item()
                    #TODO: ensure shifts is small, otherwise skip
                    x_dry = torch.roll(x_dry, shifts=int(shifts), dims=1)


            if self.normalize_mode is not None:
                if "dry" in self.normalize_mode:
                    #potentially slow
                    x_dry=self.normaliser(x_dry)

                    if torch.isnan(x_dry).any():
                        continue
                else:
                    pass

            self.test_samples.append(( y_wet, x_dry))
            counter+=1

    def __getitem__(self, idx):
        return self.test_samples[idx]

    def __len__(self):
        return len(self.test_samples)
