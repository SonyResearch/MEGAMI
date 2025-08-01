
import pandas as pd
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


class TencyMastering_Test(torch.utils.data.Dataset):

    def __init__(self,
        fs=44100,
        segment_length=131072,
        num_tracks=-1,
        mode="dry-wet",
        normalize_params=None,
        stereo=True,
        num_examples=4,
        RMS_threshold_dB=-40,
        x_as_mono=False,
        seed=42,
        tracks=["all",],
        path_csv=None,
        clusters=[0,1]
        ):

        super().__init__()
        print(num_examples, "num_examples")

        np.random.seed(seed)
        random.seed(seed)



        self.segment_length=segment_length
        self.fs=fs


        self.normalize_mode=normalize_params.normalize_mode

        self.stereo=stereo

        self.num_tracks=num_tracks

        if self.normalize_mode=="loudness_dry":
            meter = pyln.Meter(fs)
            def normaliser_fn(x):
                x=x.numpy().T
                ln=meter.integrated_loudness(x)
                #replace -Infinity with -50 dB
                if ln == -np.inf:
                    ln = -50.0
                x=pyln.normalize.loudness(x, ln, normalize_params.loudness_dry)
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
        

        self.x_as_mono=x_as_mono

        self.RMS_threshold_dB=RMS_threshold_dB

        self.get_RMS=lambda x: 20 * torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1)))

        self.test_samples = []

        counter=0

        self.tracks=tracks
        self.path_csv=path_csv
        assert self.path_csv is not None, "path_csv must be provided"

        self.df=pd.read_csv(self.path_csv)
        self.df= self.df[self.df['cluster'].isin(clusters)]

        if self.num_tracks != -1:
            self.df = self.df.sample(n=self.num_tracks, random_state=seed)


        print("mode", mode)
        #for dry_file, wet_file in tqdm(self.pair_list):
        for row in tqdm(self.df.iterrows()):
            #if counter >= num_examples and num_examples!= -1:
            #    break
            if mode=="dry-wet" or mode=="dry-wet-train":

                path=row[1]["path"] 
                cluster=row[1]["cluster"]
                
                dry_path=os.path.join(path, "dry")
                files= glob.glob(os.path.join(dry_path, "*.wav"))
                file_taxonomy = [os.path.basename(f)[:2] for f in files]

                #get indexes with file_taxonomy = "92" . could be more than one
                indexes = [i for i, x in enumerate(file_taxonomy) if x == "92"]

                #take a random index
                try:
                    index= random.choice(indexes)
                except:
                    raise ValueError("No files found with taxonomy '92' in {}".format(dry_path))

                file=files[index]
                    
                dry_file=Path(file)

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)    

                out= load_audio(str(dry_file), stereo=self.stereo)

                if out is None:
                    raise ValueError("Could not load dry audio file: {}".format(dry_file))
                    continue
                else:
                    x_dry_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
                #replace NaNs with zeros
                x_dry_long = torch.nan_to_num(x_dry_long, nan=0.0)


                x_long=x_dry_long+torch.randn_like(x_dry_long)*0.0001 #add a small noise to avoid NaNs in the future

            
            elif mode=="dryfxnorm-wet" or mode=="dryfxnorm-wet-train":
                raise NotImplementedError("dryfxnorm-wet is not implemented yet")
                
                #dry_file=id / "dry" / "vocals_normalized.wav"
                print("loading dryfxnorm-wet", id_i)
                dry_file=os.path.join(id_i, "vocals_normalized.wav") 
                dry_file=Path(dry_file)

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)

                out= load_audio(str(dry_file), stereo=self.stereo)

                if out is None:
                    raise ValueError("Could not load dry audio file: {}".format(dry_file))
                    continue
                else:
                    x_dry_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
                x_long= x_dry_long

            elif mode=="dryfxnormdr-wet":
                raise NotImplementedError("dryfxnormdr-wet is not implemented yet")
                
                #dry_file=id / "dry" / "vocals_normalized.wav"
                dry_file=os.path.join(id_i, "vocals_normalized_dr.wav") 
                dry_file=Path(dry_file)

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)

                out= load_audio(str(dry_file), stereo=self.stereo)

                if out is None:
                    raise ValueError("Could not load dry audio file: {}".format(dry_file))
                    continue
                else:
                    x_dry_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
                x_long= x_dry_long

                
            elif mode=="wetfxnorm-wet" or mode=="wetfxnorm-wet-train":
                raise NotImplementedError("wetfxnorm-wet is not implemented yet")

                print("loading wetfxnorm-wet", id_i)
                #dry_file=id / "wet" / "vocals_normalized.wav"
                dry_file=os.path.join(id_i, "vocals_normalized.wav").replace("dry", "wet")
                dry_file=Path(dry_file)

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)    

                out= load_audio(str(dry_file), stereo=self.stereo)

                if out is None:
                    raise ValueError("Could not load dry audio file: {}".format(dry_file))
                    continue
                else:
                    x_dry_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
                x_long= x_dry_long

            elif mode=="wetfxnormdr-wet":
                raise NotImplementedError("wetfxnormdr-wet is not implemented yet")

                #dry_file=id / "wet" / "vocals_normalized.wav"
                dry_file=os.path.join(id_i, "vocals_normalized_dr.wav").replace("dry", "wet")
                dry_file=Path(dry_file)

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)    

                out= load_audio(str(dry_file), stereo=self.stereo)

                if out is None:
                    raise ValueError("Could not load dry audio file: {}".format(dry_file))
                    continue
                else:
                    x_dry_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
                x_long= x_dry_long

                
            #wet_file=id / "wet" / "vocals.wav"
            #wet_file=os.path.join(id_i,"vocals.wav").replace("dry", "wet")
            #wet_file=Path(wet_file)

            #assert wet_file.exists(), "wet file does not exist: {}".format(wet_file)

            #out= load_audio(str(wet_file), stereo=self.stereo)

            #if out is None:
            #        raise ValueError("Could not load wet audio file: {}".format(wet_file))
            #        continue
            #else:
            #        y_wet_long, fs=out
            #        assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
            #y_long=y_wet_long
            
            #assert self.align==False, "align is not supported yet for this dataset"


            #pad x_dry if it is not a multiple of segment_length
            #max_length = max(x_long.size(-1), y_wet_long.size(-1))

            #define max_length as the next multiple of segment_length (padding)
            max_length = x_long.size(-1)
            max_length = (max_length + self.segment_length - 1) // self.segment_length * self.segment_length

            #print("x_dry_long", x_dry_long.shape, "y_wet_long", y_wet_long.shape, "max_length", max_length)
            x_long = torch.nn.functional.pad(x_long, (0, max_length - x_long.size(-1)), mode='constant', value=0)
            #y_long = torch.nn.functional.pad(y_long, (0, max_length - y_long.size(-1)), mode='constant', value=0)


            #print("x_dry_long", x_dry_long.shape, "y_wet_long", y_wet_long.shape)

            assert x_long.size(-1)% self.segment_length == 0, "x_dry_long must be a multiple of segment_length, got {}".format(x_long.size(-1))
            #assert y_long.size(-1)% self.segment_length == 0, "y_wet_long must be a multiple of segment_length, got {}".format(y_wet_long.size(-1))

            #assert now the two have the same length
            #assert x_long.shape==y_long.shape, "x_dry and y_wet must have the same shape, got {} and {}".format(x_long.shape, y_wet_long.shape)

            #divide into non-overlapping segments of segment_length

            for i in range(0, x_long.size(-1), self.segment_length):
                x_dry = x_long[:, i:i + self.segment_length]
                #y_wet = y_long[:, i:i + self.segment_length]


                RMS_dB=self.get_RMS(x_dry.mean(0))
                if RMS_dB<self.RMS_threshold_dB:
                    continue

                if self.normalize_mode is not None:
                    if "dry" in self.normalize_mode:
                        #potentially slow
                        x_dry=self.normaliser(x_dry)  # add a small value to avoid division by zero
    
                        if torch.isnan(x_dry).any():
                            raise ValueError("NaN values found in x_dry after normalization")
                            continue
                    else:
                        pass

                if self.x_as_mono:
                    x_dry = x_dry.mean(dim=0, keepdim=True)
                    #how many NaNs are there?
                    x_dry= torch.cat((x_dry, x_dry), dim=0)
                
                if torch.isnan(x_dry).any():
                    raise ValueError("NaN values found in x_dry"+str(dry_file))
                    continue

                #print("x_dry", x_dry.shape, "y_wet", y_wet.shape)
                self.test_samples.append(( x_dry, cluster)) 
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

class TencyMastering(torch.utils.data.IterableDataset):

    def __init__(self,
        fs=44100,
        segment_length=131072,
        mode="dry-wet",
        normalize_params=None,
        stereo=True,
        num_examples=4,
        RMS_threshold_dB=-40,
        seed=42,
        x_as_mono=False,
        effect_randomizer=None,
        tracks=["all",],
        path_csv=None,
        clusters=[0,1]
        ):

        super().__init__()
        print(num_examples, "num_examples")

        #np.random.seed(seed)
        #random.seed(seed)

    

        self.segment_length=segment_length
        self.fs=fs


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

        self.effect_randomizer=effect_randomizer

        self.x_as_mono=x_as_mono

        self.tracks=tracks

        self.path_csv=path_csv

        self.df=pd.read_csv(self.path_csv)
        self.df= self.df[self.df['cluster'].isin(clusters)]

    def __iter__(self):

        while True:
            #print("Iterating over TencyMastering_Vocals dataset")

            #for id in tqdm(self.id_list):

            num=np.random.randint(0,len(self.df)-1)

            row=self.df.iloc[num]
            
            if self.mode=="dry-wet":
                path=row["path"]
                cluster=row["cluster"]
    
                dry_path=os.path.join(path, "dry")
                files= glob.glob(os.path.join(dry_path, "*.wav"))
                if len(files)==0:
                    print("row", row)
                    print("dry_path", dry_path)

                file_taxonomy = [os.path.basename(f)[:2] for f in files]
    
                #get indexes with file_taxonomy = "92" . could be more than one
                indexes = [i for i, x in enumerate(file_taxonomy) if x == "92"]
    
                #take a random index
                try:
                    index= random.choice(indexes)
                except:
                    print("taxonomy not found in", dry_path)
                    print("files", files)
                    raise ValueError("No files found with taxonomy '92' in {}".format(dry_path))
    
                file=files[index]
                        
                dry_file=Path(file)
    
                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)


            
            elif self.mode=="dryfxnorm-wet":
                raise NotImplementedError("dryfxnorm-wet is not implemented yet")
                
                dry_file=os.path.join(id, "vocals_normalized.wav")
                dry_file=Path(dry_file)

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)

            elif self.mode=="dryfxnormdr-wet":
                raise NotImplementedError("dryfxnormdr-wet is not implemented yet")
                
                dry_file=os.path.join(id, "vocals_normalized_dr.wav")
                dry_file=Path(dry_file)

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)

                
            elif self.mode=="wetfxnorm-wet":
                raise NotImplementedError("wetfxnorm-wet is not implemented yet")

                dry_file=os.path.join(id, "vocals_normalized.wav").replace("dry", "wet")
                dry_file=Path(dry_file)

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)    

            elif self.mode=="wetfxnormdr-wet":
                raise NotImplementedError("wetfxnormdr-wet is not implemented yet")

                dry_file=os.path.join(id, "vocals_normalized_dr.wav").replace("dry", "wet")
                dry_file=Path(dry_file)

                assert dry_file.exists(), "dry file does not exist: {}".format(dry_file)    


            #wet_file=os.path.join(id,"vocals.wav").replace("dry", "wet")
            #wet_file=Path(wet_file)

            #assert wet_file.exists(), "wet file does not exist: {}".format(wet_file)


            dry_duration, dry_total_frames, dry_samplerate=get_audio_length(str(dry_file))
            #wet_duration, wet_total_frames, wet_samplerate=get_audio_length(str(wet_file))

            #assert dry_total_frames== wet_total_frames, "dry and wet files must have the same number of frames, got {} and {}".format(dry_total_frames, wet_total_frames)

            total_frames=dry_total_frames

            start=np.random.randint(0,total_frames-self.segment_length)
            end=start+ self.segment_length


            out=load_audio(str(dry_file), start, end, stereo=self.stereo)

            if out is None:
                continue
            else:
                x_dry, fs=out
                assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                

            #out= load_audio(str(wet_file), start, end, stereo=self.stereo)

            #if out is None:
            #    continue
            #else:
            #    y_wet, fs=out
            #    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
            
            RMS_dB=self.get_RMS(x_dry.mean(0))
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
                
            
            if self.x_as_mono:
                x_dry = x_dry.mean(dim=0, keepdim=True)
                x_dry= torch.cat((x_dry, x_dry), dim=0)

            

            yield  x_dry, cluster


