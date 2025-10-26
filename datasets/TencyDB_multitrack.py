
import torch.nn.functional as F
import pandas as pd
import soundfile as sf
import torch
import numpy as np
import os
import glob
import yaml
from pathlib import Path
import random

from utils.data_utils import read_wav_segment, get_audio_length

import pickle

from tqdm import tqdm

 

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



class TencyDB_Multitrack_Dataset(torch.utils.data.IterableDataset):

    def __init__(self,
        fs=44100,
        segment_length=131072,
        normalize_params=None,
        stereo=True,
        mode="dry-wet",
        num_examples=4,
        RMS_threshold_dB=-40,
        seed=42,
        effect_randomizer=None,
        tracks="all",
        tencymastering_path=None,
        tency_path=None,
        path_csv=None,
        random_order=False,
        random_polarity=False,
        only_dry=False, #if True, only dry files are used, if False, both dry and wet files are used
        ):

        super().__init__()
        print(num_examples, "num_examples")

        self.random_order=random_order

        self.random_polarity=random_polarity

        self.only_dry=only_dry

        self.mode=mode
    
        self.segment_length=segment_length
        self.segment_length= int(segment_length*48000/44100)+1 #most of the data is samples at 48kHz, so we convert the segment length to 48kHz samples. will be resampled and cut later
        self.fs=fs

        self.normalize_mode=normalize_params.normalize_mode

        self.stereo=stereo


        if self.normalize_mode=="rms_dry":
            RMS_target=normalize_params.rms_dry #-16 dB
            def normaliser_fn(x):
                RMS=20*torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1).mean(dim=-1)))
                scaler=10**((RMS_target-RMS)/20)
                x=x*scaler.view(-1,1)
                return x, scaler.view(-1,1)

            self.normaliser = normaliser_fn
            
        elif self.normalize_mode is None:
            self.normaliser = lambda x: x
        

        self.RMS_threshold_dB=RMS_threshold_dB

        self.get_RMS=lambda x: 20 * torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1)))


        self.effect_randomizer=effect_randomizer

        self.tracks=tracks
        assert self.tracks=="all", "only all model is implemented for now"

        self.tencymastering_path=tencymastering_path
        self.tency_path=tency_path
        self.path_csv=path_csv
        assert self.path_csv is not None, "path_csv must be provided"

        self.df=pd.read_csv(self.path_csv)

    def __iter__(self):

        
        while True:
            num=np.random.randint(0,len(self.df)-1)

            row=self.df.iloc[num]

            subdir=row["subdir"]

            path=row["path"]

            if subdir=="TencyMastering":
                wet_path=os.path.join(self.tencymastering_path, path, "multi")
                files= glob.glob(os.path.join(wet_path, "*.wav"))
            else:
                wet_path=os.path.join(self.tency_path, subdir, path, "multi")
                files= glob.glob(os.path.join(wet_path,  "*.flac"))
                

            if len(files)==0:
                print("no dry files found in", wet_path, subdir, path)
                continue


            if self.tracks == "all":
                wet_files=[]
                for i in range(len(files)):
                    wet_files.append(files[i])
            else:
                raise NotImplementedError("all mode is only implemented for now")

            
            assert len(wet_files) > 0, "no wet files found"

            total_frames= 999999999999

            wet_frames=[]
            for i in range(len(wet_files)):
                wet_duration, wet_total_frames, wet_samplerate=get_audio_length(str(wet_files[i]))
                total_frames= min(total_frames, wet_total_frames)
                wet_frames.append(wet_total_frames)

            start=np.random.randint(0,total_frames-self.segment_length)
            end=start+ self.segment_length

            selected_tracks_wet=[]

            for i,  wet_file in enumerate(wet_files):
                
                try:
                    out=load_audio(str(wet_file), start, end, stereo=self.stereo)
                except:
                    continue

                if out is None:
                    raise Exception(f"Error loading wet file {wet_file} at segment {start}-{end}")
                
                x_wet, fs=out
                
                assert x_wet.shape[-1]==self.segment_length, "x_wet_long must have the same length as segment_length, got {}".format(x_wet.shape[-1])
                assert x_wet.shape[0]==2, "x_wet_long must have 2 channels, got {}".format(x_wet.shape[0])
    
                if self.random_polarity:
                    if np.random.rand() > 0.5:
                        x_wet = -x_wet

                RMS_dB_wet=self.get_RMS(x_wet.mean(0))
                if RMS_dB_wet<self.RMS_threshold_dB:
                    continue
            
                selected_tracks_wet.append(x_wet)

            assert len(selected_tracks_wet) > 0, "no wet tracks found after filtering by RMS threshold"

            selected_tracks_wet=torch.stack(selected_tracks_wet, dim=0)

            if self.random_order:
                #randomly shuffle the tracks
                indices = torch.randperm(selected_tracks_wet.shape[0])
                selected_tracks_wet=selected_tracks_wet[indices]
            else:
                raise NotImplementedError("random_order must be used")


            yield  selected_tracks_wet, path, fs



