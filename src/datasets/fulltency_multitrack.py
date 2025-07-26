
import torch.nn.functional as F
import pandas as pd
#import pyloudnorm as pyln
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

from utils.data_utils import taxonomy2track

from utils.common_audioeffects import AugmentationChain, ConvolutionalReverb, Compressor, Equaliser, Panner, Haas, Gain

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
        tracks="all",
        path_csv=None,
        only_dry=False, #if True, only dry files are used, if False, both dry and wet files are used
        random_order=False, #if True, the order of the samples is randomized
        ):

        super().__init__()
        print(num_examples, "num_examples")

        np.random.seed(seed)
        random.seed(seed)


        self.segment_length=segment_length
        self.fs=fs

        self.only_dry=only_dry

        self.random_order=random_order


        self.normalize_mode=normalize_params.normalize_mode

        self.stereo=stereo

        self.num_tracks=num_tracks

        if self.normalize_mode=="loudness_dry":
            raise NotImplementedError
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
                RMS=20*torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1).mean(dim=-1)))
                scaler=10**((RMS_target-RMS)/20)
                x=x*scaler.view(-1,1)
                return x, scaler.view(-1,1)

            self.normaliser = normaliser_fn
            
        elif self.normalize_mode is None:
            self.normaliser = lambda x: x
        

        self.x_as_mono=x_as_mono

        self.RMS_threshold_dB=RMS_threshold_dB

        self.get_RMS=lambda x: 20 * torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1)))

        self.test_samples = []

        counter=0

        self.tracks=tracks
        assert self.tracks=="all", "only all model is implemented for now"

        self.path_csv=path_csv
        assert self.path_csv is not None, "path_csv must be provided"

        self.df=pd.read_csv(self.path_csv)

        if self.num_tracks != -1:
            self.df = self.df.sample(n=self.num_tracks, random_state=seed)
        else:
            self.df = self.df.sample(frac=1, random_state=seed)

        num_skips=0

        for row in tqdm(self.df.iterrows()):

            skip_iteration = False

            path=row[1]["path"] 

            dry_path=os.path.join(path, "dry_multi")
            wet_path=os.path.join(path, "multi")

            files= glob.glob(os.path.join(dry_path, "*.wav"))
            if len(files)==0:
                print("no dry files found in", dry_path)
                num_skips+=1
                continue
            files_taxonomy = [os.path.basename(f)[:4] for f in files]


            if self.tracks == "all":
                #filter files by tracks
                dry_files=[]
                wet_files=[]
                taxonomies=[]
                for i, file_taxonomy in enumerate(files_taxonomy):
                    dry_files.append(files[i])
                    taxonomies.append(file_taxonomy)
                    wet_file = os.path.join(wet_path, files[i].split("/")[-1])
                    assert os.path.exists(wet_file), "wet file does not exist: {}".format(wet_file)
                    wet_files.append(wet_file)
            else:
                raise NotImplementedError("all mode is only implemented for now")
                target_taxonomies= trackname2taxonomy(self.tracks)
                dry_files=[]
                for track_id in target_taxonomies:
                    #check if the track_id is in the files_taxonomy list. It should start with the track_id (first of the 4 characters)
                    found_i=False
                    for i, file_taxonomy in enumerate(files_taxonomy):
                        if file_taxonomy.startswith(str(track_id)):
                            found_i=True
                            dry_files.append(files[i])
                            #if there is more than one, we ignore the rest, it is OK as the dataset is synthetic
                            break
                    if not found_i:
                        skip_iteration = True
                        print("track not found in", dry_path, "for track", track_id)
                        print("tracks", self.tracks)
                        print("files_taxonomy", files_taxonomy)

            if skip_iteration:
                num_skips+=1
                continue

            assert len(dry_files) > 0, "no dry files found in {}".format(dry_path)
            assert len(dry_files) == len(taxonomies), "number of dry files and taxonomies must match, got {} and {}".format(len(dry_files), len(taxonomies))
            assert len(wet_files) == len(dry_files), "number of wet files and dry files must match, got {} and {}".format(len(wet_files), len(dry_files))
        
            for i, (dry_file, wet_file) in enumerate(zip(dry_files, wet_files)):

                try:
                    out=load_audio(str(dry_file), stereo=self.stereo)
                except:
                    continue
                if out is None:
                    skip_iteration = True
                    print("Could not load dry audio file: {}".format(dry_file))
                    continue

                x_dry_long, fs=out
                assert fs==self.fs, "wrong sampling rate: {}".format(fs)

                x_dry_long = x_dry_long.mean(0, keepdim=True)

                if not self.only_dry:
                    try:
                        out=load_audio(str(wet_file), stereo=self.stereo)
                    except:
                        continue
                    if out is None:
                        skip_iteration = True
                        print("Could not load wet audio file: {}".format(wet_file))
                        continue
                    
                    x_wet_long, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
                if i==0:
                    if not self.only_dry:
                        if x_dry_long.shape[-1]<x_wet_long.shape[-1]:
                            x_wet_long= x_wet_long[..., :x_dry_long.shape[-1]]
                        elif x_dry_long.shape[-1]>x_wet_long.shape[-1]:
                            x_dry_long = x_dry_long[..., :x_wet_long.shape[-1]]

                        x_all_wet=torch.zeros((len(dry_files),2, x_dry_long.shape[-1]), dtype=torch.float32)
                        x_all_wet[i]=x_wet_long

                    x_all=torch.zeros((len(dry_files),2, x_dry_long.shape[-1]), dtype=torch.float32)
                    x_all[i]=x_dry_long

                    x_sum=x_dry_long

                else:
                    if x_dry_long.shape[-1]<x_all.shape[-1]:
                        padding = x_all.shape[-1] - x_dry_long.shape[-1]
                        x_dry_long = torch.nn.functional.pad(x_dry_long, (0, padding))
                    elif x_dry_long.shape[-1]>x_all.shape[-1]:
                        #pas
                        x_dry_long = x_dry_long[..., :x_all.shape[-1]]

                    x_all[i] = x_dry_long
                    x_sum+=x_dry_long
                    
                    if not self.only_dry:
                        if x_wet_long.shape[-1]<x_all_wet.shape[-1]:
                            padding = x_all_wet.shape[-1] - x_wet_long.shape[-1]
                            x_wet_long = torch.nn.functional.pad(x_wet_long, (0, padding))
                        elif x_wet_long.shape[-1]>x_all_wet.shape[-1]:
                            #pas
                            x_wet_long = x_wet_long[..., :x_all_wet.shape[-1]]
                        x_all_wet[i] = x_wet_long

                    

            if not self.only_dry:
                alignment_file=Path(dry_files[0]).parent.parent / "alignment.pickle"
                alignment_data = pickle.load(open(alignment_file, "rb"))
                dry_align=alignment_data.get("dry_alignment",0)
                x_all=torch.roll(x_all, shifts=int(dry_align), dims=-1)
                x_sum= torch.roll(x_sum, shifts=int(dry_align), dims=-1)

            
            max_length = x_sum.size(-1)
            max_length = (max_length + self.segment_length - 1) // self.segment_length * self.segment_length

            x_all = torch.nn.functional.pad(x_all, (0, max_length - x_all.size(-1)), mode='constant', value=0)
            if not self.only_dry:
                x_all_wet = torch.nn.functional.pad(x_all_wet, (0, max_length - x_all_wet.size(-1)), mode='constant', value=0)
            x_sum = torch.nn.functional.pad(x_sum, (0, max_length - x_sum.size(-1)), mode='constant', value=0)

            assert x_all.size(-1)% self.segment_length == 0, "x_dry_long must be a multiple of segment_length, got {}".format(x_all.size(-1))
            if not self.only_dry:
                assert x_all_wet.size(-1)% self.segment_length == 0, "x_wet_long must be a multiple of segment_length, got {}".format(x_all_wet.size(-1))
            assert x_sum.size(-1)% self.segment_length == 0, "x_dry_long must be a multiple of segment_length, got {}".format(x_sum.size(-1))
            

            for i in range(0, x_sum.size(-1), self.segment_length):
                x_all_i= x_all[..., i:i+self.segment_length]
                if not self.only_dry:
                    x_all_wet_i= x_all_wet[..., i:i+self.segment_length]
                x_sum_i= x_sum[..., i:i+self.segment_length]

                skip_segment=False
                selected_tracks = torch.ones((x_all_i.shape[0],), dtype=torch.bool)

                for i in range(x_all_i.shape[0]):
                    RMS_dB=self.get_RMS(x_all_i[i].mean(0))
                    if RMS_dB<self.RMS_threshold_dB:
                        #remove track if it is below the threshold
                        selected_tracks[i]=False
                
                if selected_tracks.sum() == 0:
                    skip_segment=True
                if skip_segment:
                    continue

                #now remove the tracks that are not selected
                x_all_i = x_all_i[selected_tracks]
                if not self.only_dry:
                    x_all_wet_i = x_all_wet_i[selected_tracks]
                taxonomies_selected = [taxonomies[i] for i in range(len(taxonomies)) if selected_tracks[i]]
                        
                if self.normalize_mode is not None:

                    if "dry" in self.normalize_mode:
                        #potentially slow
                        x_sum_i, scaler=self.normaliser(x_sum_i)
                        
                        x_all_i*=scaler.unsqueeze(0)

                        assert not torch.isnan(x_all).any(), "NaN values found in x_all after normalization"

                    else:
                        pass

                if self.random_order:
                    #randomly shuffle the tracks
                    indices = torch.randperm(x_all_i.shape[0])
                    x_all_i=x_all_i[indices]
                    if not self.only_dry:
                        x_all_wet_i=x_all_wet_i[indices]
                    taxonomies_selected = [taxonomies_selected[i] for i in indices]

                if self.only_dry:
                    self.test_samples.append(( x_all_i, None,   taxonomies_selected, path)) 
                else:
                    self.test_samples.append(( x_all_i,  x_all_wet_i, taxonomies_selected, path)) 
                counter+=1

            if skip_iteration:
                num_skips+=1
                continue

        random.shuffle(self.test_samples)

        if num_examples != -1:
            self.test_samples = self.test_samples[:num_examples]

        print("test_samples", len(self.test_samples), "num_examples", num_examples, "num_skips", num_skips)

    def __getitem__(self, idx):
        return self.test_samples[idx]

    def __len__(self):
        return len(self.test_samples)

class TencyMastering(torch.utils.data.IterableDataset):

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
        apply_augmentation=False,
        RIR_path_csv="/data5/eloi/ImpulseResponses/rir_files_train.csv",
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


        self.apply_augmentation=apply_augmentation

        if self.apply_augmentation:
            df= pd.read_csv(RIR_path_csv)
            RIR_files = df['rir_file'].tolist()
            print(RIR_files[:10])
    
            acceoted_sampling_rates = [44100]
    
            dataset_rir=pd.DataFrame(columns=['impulse_response'])
            dataset_rir=pd.DataFrame(columns=['impulse_response'])
            for i, file in enumerate(RIR_files):
                #load the RIR file
                x,fs=sf.read(file)
                n_samples=x.shape[0]
                if len(x.shape) == 1:
                    x = x[:, None]
                my_tuple=(int(n_samples), x)
                #add my_tuple to the dataset_rir DataFrame (column 'impulse_response')
                dataset_rir.loc[i, 'impulse_response'] = my_tuple

            self.augment_chain= AugmentationChain([
                        (ConvolutionalReverb(impulse_responses=dataset_rir, sample_rates=acceoted_sampling_rates), 0.5),
                        (Haas(sample_rates=acceoted_sampling_rates), 0.5),
                        (Gain(), 0.8),
                        (Panner(sample_rates=acceoted_sampling_rates), 0.5),
                        (Compressor(sample_rates=acceoted_sampling_rates), 0.5),
                        (Equaliser(n_channels=2,sample_rates=acceoted_sampling_rates), 0.5),
                        ],
                        shuffle=True, apply_to='target')

    def __iter__(self):

        
        while True:
            skip_iteration = False
            #print("Iterating over TencyMastering_Vocals dataset")

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

            files_taxonomy = [os.path.basename(f)[:4] for f in files]


            if self.tracks == "all":
                wet_files=[]
                taxonomies=[]
                for i, file_taxonomy in enumerate(files_taxonomy):
                    wet_files.append(files[i])
                    taxonomies.append(file_taxonomy)
            else:
                raise NotImplementedError("all mode is only implemented for now")
                        

            if skip_iteration:
                continue
            
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
                    skip_iteration = True
                    print("Could not load wet audio file: {}".format(wet_file))
                    continue
                
                x_wet, fs=out
                #assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                
                assert x_wet.shape[-1]==self.segment_length, "x_wet_long must have the same length as segment_length, got {}".format(x_wet.shape[-1])
                assert x_wet.shape[0]==2, "x_wet_long must have 2 channels, got {}".format(x_wet.shape[0])
    
                if self.random_polarity:
                    if np.random.rand() > 0.5:
                        x_wet = -x_wet

                RMS_dB_wet=self.get_RMS(x_wet.mean(0))
                if RMS_dB_wet<self.RMS_threshold_dB:
                    continue
            
                selected_tracks_wet.append(x_wet)

            if len(selected_tracks_wet) == 0:
                continue

            selected_tracks_wet=torch.stack(selected_tracks_wet, dim=0)

            if self.random_order:
                #randomly shuffle the tracks
                indices = torch.randperm(selected_tracks_wet.shape[0])
                selected_tracks_wet=selected_tracks_wet[indices]
            else:
                raise NotImplementedError("random_order must be used")

            if self.apply_augmentation:
                #apply augmentation
                augmented_tracks = []
                for i in range(selected_tracks_wet.shape[0]):
                    track_i= selected_tracks_wet[i]
                    _, track_augmented=self.augment_chain(track_i.cpu().numpy().T, track_i.cpu().numpy().T)
                    track_augmented = torch.from_numpy(track_augmented.T).float().to(track_i.device)
                    assert track_augmented.shape == track_i.shape, "augment_target shape must match x_wet shape, got {} and {}".format(augment_target.shape, x_wet.shape)
                    augmented_tracks.append(track_augmented)
                
                augmented_tracks = torch.stack(augmented_tracks, dim=0)

                yield selected_tracks_wet, augmented_tracks, path, fs
            
            else:
            
                yield  selected_tracks_wet, None, path, fs



