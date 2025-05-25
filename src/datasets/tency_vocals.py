
import pyloudnorm as pyln
import soundfile as sf
import torch
import numpy as np
import os
import glob
import yaml
from pathlib import Path

from utils.data_utils import read_wav_segment, get_audio_length
import torch.distributed as dist



 
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

                for k, v in lead_vocal_only.items():
                    wet_file = song_id / wet_subdir / k.split("/")[-1]
                    dry_file = song_id / dry_subdir / v[0].split("/")[-1]
    
                    results.append((dry_file, wet_file))
        else:
            continue

    print("results", len(results))
    return results



class TencyMastering_Vocals(torch.utils.data.IterableDataset):
    def __init__(self,
        fs=44100,
        segment_length=131072,
        skip=None,
        base_dir=None,
        subdirs=None,
        skip_list=None,
        normalize_params=None,
        align=False,
        stereo=True,
        RMS_threshold_dB=-40,
        seed=42,
        side_energy_threshold=10
        ):


        super().__init__()
        self.seed = seed
        #print("data loader rank", rank)

        #np.random.seed(seed+rank)

        base_path=os.path.join(base_dir)

        id_list=[]
        for subdir in subdirs:
            path=os.path.join(base_path,subdir)
            id_list.extend(glob.glob(os.path.join(path,"*")))

        print("id_list", len(id_list))
        def filter_ids(path):
            id_track=path.split("/")[-1]
            partition=path.split("/")[-2]
            id_name=partition+"/"+id_track
            if id_name in skip_list:
                return False
            else:
                return True

        id_list=[x for x in id_list if filter_ids(x)]
        print("id_list after filtering", len(id_list))
        assert len(id_list)>0, "No files found in the dataset"

        self.pair_list=process_id_list_lead_vocal(id_list)
        print("pair_list", len(self.pair_list))
        self.segment_length=segment_length
        self.fs=fs

        self.align=align
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

        #raise ValueError("stop here")

    def __iter__(self):
        while True:
            try:
                #bignum=np.random.randint(0,10000-1)

                num=np.random.randint(0,len(self.pair_list)-1)
                dry_file, wet_file=self.pair_list[num]

                dry_duration, dry_total_frames, dry_samplerate=get_audio_length(dry_file)
                wet_duration, wet_total_frames, wet_samplerate=get_audio_length(wet_file)

                total_frames=min(dry_total_frames, wet_total_frames)

                #sample "start" uniformly in the range [0, total_frames-seg_len]
                start=np.random.randint(0,total_frames-self.segment_length)
                end=start+ self.segment_length

                x_dry, fs=read_wav_segment(dry_file, start, end)
                assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                if self.stereo:
                    if len(x_dry.shape)==1:
                        #print( "dry not stereo , doubling channels", x_dry.shape)
                        x_dry=x_dry[:,np.newaxis]
                        x_dry = np.concatenate((x_dry, x_dry), axis=-1)
                    elif len(x_dry.shape)==2 and x_dry.shape[-1]==1:
                        #print( "dry not stereo , doubling channels", x_dry.shape)
                        x_dry = np.concatenate((x_dry, x_dry), axis=-1)

                x_dry=torch.from_numpy(x_dry).permute(1,0)
    
                if x_dry.size(0) > 1:
                    left = x_dry[0]
                    right = x_dry[1]
                    left = left / left.max()
                    right = right / right.max()
                    side = (left - right) * 0.707
                    side_energy = 20*torch.log10(side.abs().max()).item()
                else:
                    side_energy = -torch.inf
                
                if side_energy > self.side_energy_threshold:
                    print(f"Skip {dry_file} because of high side energy"+str(side_energy))
                    continue


                y_wet, fs=read_wav_segment(wet_file, start, end)
                assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                if self.stereo:
                    assert y_wet.shape[-1]==2, "not stereo"
                y_wet=torch.from_numpy(y_wet).permute(1,0)

                RMS_dB=self.get_RMS(y_wet.mean(0))
                if RMS_dB<self.RMS_threshold_dB:
                    #print(r"skip wet file because of low RMS (silence) {}".format(RMS_dB))
                    continue


                if self.align:
                    #do that on GPU if it is slow
                    shifts = find_time_offset(x_dry.mean(0), y_wet.mean(0), margin=3000).item()
                    #TODO: ensure shifts is small, otherwise skip
                    x_dry = torch.roll(x_dry, shifts=int(shifts), dims=1)


                if self.normalize_mode is not None:
                    if "dry" in self.normalize_mode:
                        #potentially slow
                        x_dry=self.normaliser(x_dry)
                    else:
                        pass
                
                #print("x_dry", 20*torch.log10(x_dry.std()), "y_wet", 20*torch.log10(y_wet.std()))

                yield  y_wet, x_dry

            except Exception as e:
                print(e)
                continue

class TencyMastering_Vocals_Test(torch.utils.data.Dataset):

    def __init__(self,
        fs=44100,
        segment_length=131072,
        skip=None,
        base_dir=None,
        subdirs=None,
        skip_list=None,
        normalize_params=None,
        align=False,
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
            if counter >= num_examples:
                break

            dry_duration, dry_total_frames, dry_samplerate=get_audio_length(dry_file)
            wet_duration, wet_total_frames, wet_samplerate=get_audio_length(wet_file)

            total_frames=min(dry_total_frames, wet_total_frames)

            #sample "start" uniformly in the range [0, total_frames-seg_len]
            #start=np.random.randint(0,total_frames-self.segment_length)

            start=total_frames//2 #fixed
            end=start+ self.segment_length


            x_dry, fs=read_wav_segment(dry_file, start, end)
            assert fs==self.fs, "wrong sampling rate: {}".format(fs)
            if self.stereo:
                if self.stereo:
                    if len(x_dry.shape)==1:
                        print( "dry not stereo , doubling channels", dry_file.shape)
                        x_dry=x_dry[:,np.newaxis]
                        x_dry = np.concatenate((x_dry, x_dry), axis=-1)
                    elif len(x_dry.shape)==2 and x_dry.shape[-1]==1:
                        print( "dry not stereo , doubling channels", dry_file.shape)
                        x_dry = np.concatenate((x_dry, x_dry), axis=-1)

            x_dry=torch.from_numpy(x_dry).permute(1,0)

            if x_dry.size(0) > 1:
                left = x_dry[0]
                right = x_dry[1]
                left = left / left.max()
                right = right / right.max()
                side = (left - right) * 0.707
                side_energy = 20*torch.log10(side.abs().max()).item()
            else:
                side_energy = -torch.inf
            
            if side_energy > self.side_energy_threshold:
                print(f"Skip {dry_file}")
                continue

            y_wet, fs=read_wav_segment(wet_file, start, end)
            assert fs==self.fs, "wrong sampling rate: {}".format(fs)
            if self.stereo:
                assert y_wet.shape[-1]==2, "not stereo"
            y_wet=torch.from_numpy(y_wet).permute(1,0)

            if self.get_RMS(y_wet.mean(0))<self.RMS_threshold_dB:
                #print("skip wet file because of low RMS (silence)")
                continue

            if self.align:
                #do that on GPU if it is slow
                shifts = find_time_offset(x_dry.mean(0), y_wet.mean(0), margin=3000).item()
                x_dry = torch.roll(x_dry, shifts=int(shifts), dims=1)
            

            if self.normalize_mode is not None:
                if "dry" in self.normalize_mode:
                    #potentially slow
                    x_dry=self.normaliser(x_dry)
                else:
                    pass

            self.test_samples.append(( y_wet, x_dry))
            counter+=1

    def __getitem__(self, idx):
        return self.test_samples[idx]

    def __len__(self):
        return len(self.test_samples)
