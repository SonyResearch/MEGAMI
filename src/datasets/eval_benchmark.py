
import torch.nn.functional as F
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

from utils.common_audioeffects import AugmentationChain, ConvolutionalReverb, Compressor, Equaliser, Panner, Haas, Gain
import pickle

from tqdm import tqdm

from utils.data_utils import taxonomy2track


def trackname2taxonomy(tracks):
    """
    Convert track names to taxonomy codes.
    """
    taxonomy = {
        "vocals": "92",
        "bass": "2",
        "drums": "11",
    }
    
    return [taxonomy[track] for track in tracks if track in taxonomy]
 
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


class Eval_Benchmark(torch.utils.data.Dataset):

    def __init__(self,
        fs=44100,
        segment_length=525312,
        num_tracks=-1,
        num_examples=-1,
        num_segments_per_track=-1,
        mode="dry-wet", #dry-wet, dry-only, dry-mixture, dry-wet-mixture 
        format="all_tracks", #all_tracks, 4instr
        only_dry=False, #if True, only dry files are used, if False, both dry and wet files are used
        random_order_examples=False, #if True, the order of the samples is randomized
        random_order_tracks=False, #if True, the order of the tracks is randomized
        path=None,
        RMS_threshold_dB=-60,
        ):

        super().__init__()

        self.segment_length=segment_length
        self.fs=fs

        self.mode=mode
        self.format=format

        self.random_order_examples=random_order_examples
        self.random_order_tracks=random_order_tracks

        self.num_examples=num_examples
        self.num_tracks=num_tracks


        self.test_samples = []

        self.path=path

        self.RMS_threshold_dB=RMS_threshold_dB
        self.get_RMS=lambda x: 20 * torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1)))

        counter=0

        #glob directories that are children of the path

        self.song_dirs= sorted(glob.glob(os.path.join(self.path, "*")))

        if self.mode=="dry-wet" or self.mode=="dry-wet-mixture":
            #only TM tracks are available
            self.song_dirs = [d for d in self.song_dirs if "TM" in os.path.basename(d)]


        for song_dir in tqdm(self.song_dirs):

            song_id= os.path.basename(song_dir)

            #glob all the subdirectories corresponding to segments:

            segment_subdirs = sorted(glob.glob(os.path.join(song_dir, "*")))

            for i, segment_subdir in enumerate(segment_subdirs):

                if i>=num_segments_per_track and num_segments_per_track!=-1:
                    break


                segment_id= os.path.basename(segment_subdir)

                if self.format == "4instr":
                    dry_path= os.path.join(segment_subdir, "dry_4instr")
                elif self.format == "all_tracks":
                    dry_path= os.path.join(segment_subdir, "dry_multi")

                dry_files= glob.glob(os.path.join(dry_path, "*.wav"))

                if self.format == "4instr":
                    assert len(dry_files) ==4 , "No dry files found in {}".format(dry_path)
                elif self.format == "all_tracks":
                    assert len(dry_files) > 0, "No dry files found in {}".format(dry_path)

                x_dry_tracks=[]

                if "wet" in self.mode:
                    if self.format == "4instr":
                        wet_path= os.path.join(segment_subdir, "wet_4instr")
                    elif self.format == "all_tracks":
                        wet_path= os.path.join(segment_subdir, "multi")

                    files_wet= glob.glob(os.path.join(wet_path, "*.wav"))

                    if self.format == "4instr":
                        assert len(files_wet) == 4, "No wet files found in {}".format(wet_path)
                    elif self.format == "all_tracks":
                        assert len(files_wet) > 0, "No wet files found in {}".format(wet_path)

                    x_wet_tracks=[]


                for f in dry_files:
                    out=load_audio(str(f), stereo=True)
                    if out is None:
                        raise ValueError("Could not load dry audio file: {}".format(f))
                    x_dry, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)

                    assert x_dry.shape[-1]== self.segment_length, "x_dry must be of length segment_length, got {}".format(x_dry.shape[-1])
                    
                    #stereo to mono
                    x_dry = x_dry.mean(dim=0, keepdim=True)

                    if self.format == "all_tracks":
                        #discard tracks with no enouugh energy
                        RMS_dB=self.get_RMS(x_dry.mean(0))
                        if RMS_dB<self.RMS_threshold_dB:
                            continue

                    x_dry_tracks.append(x_dry)

                    if "wet" in self.mode:
                        wet_file = os.path.join(wet_path, os.path.basename(f))
                        out=load_audio(str(wet_file), stereo=True)
                        if out is None:
                            raise ValueError("Could not load wet audio file: {}".format(wet_file))
                        x_wet, fs=out
                        assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                        assert x_wet.shape[-1]== self.segment_length, "x_wet must be of length segment_length, got {}".format(x_wet.shape[-1])

                        x_wet_tracks.append(x_wet)

                assert len(x_dry_tracks) > 0, "No dry tracks found in {}".format(dry_path)

                if "wet" in self.mode:
                    assert len(x_dry_tracks)==len(x_wet_tracks) 

                x_dry_all=torch.stack(x_dry_tracks, dim=0)
                print("x_dry_all", x_dry_all.shape)

                if "wet" in self.mode:
                    x_wet_all=torch.stack(x_wet_tracks, dim=0)
                    print("x_wet_all", x_wet_all.shape)



                if self.random_order_tracks:
                    #randomly shuffle the tracks
                    indices = torch.randperm(x_dry_all.shape[0])
                    x_dry_all = x_dry_all[indices]
                    if "wet" in self.mode:
                        x_wet_all=x_wet_all[indices]

                if "mixture" in self.mode:
                    #load the wet mixture
                    mix_path= os.path.join(segment_subdir,"mix", "mix.wav")
                    assert os.path.exists(mix_path), "No mixture file found in {}".format(mix_path)

                    x_mix, fs=load_audio(mix_path, stereo=True)
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                    assert x_mix.shape[-1]== self.segment_length, "x_mix must be of length segment_length, got {}".format(x_mix.shape[-1])


                if self.mode=="dry-wet":
                    self.test_samples.append(( x_dry_all, x_wet_all, None ,song_id, segment_id, segment_subdir))
                elif self.mode=="dry-only":
                    self.test_samples.append(( x_dry_all, None, None, song_id, segment_id, segment_subdir))
                elif self.mode=="dry-mixture":
                    self.test_samples.append(( x_dry_all, None, x_mix, song_id, segment_id, segment_subdir))
                elif self.mode=="dry-wet-mixture":
                    self.test_samples.append(( x_dry_all, x_wet_all, x_mix, song_id, segment_id, segment_subdir))


            counter+=1

            if self.num_tracks != -1 and counter >= self.num_tracks:
                break

        if self.random_order_examples:
            random.shuffle(self.test_samples)

        if num_examples != -1:
            self.test_samples = self.test_samples[:num_examples]

    def __getitem__(self, idx):
        return self.test_samples[idx]

    def __len__(self):
        return len(self.test_samples)



