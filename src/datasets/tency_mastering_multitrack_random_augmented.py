

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

import pickle

from tqdm import tqdm

from utils.data_utils import taxonomy2track, efficient_roll
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
        path_csv=None,
        random_polarity=False,
        random_shift=1024, #shift in samples to apply to the audio, if random_shift is None, no shift is applied
        only_dry=False, #if True, only dry files are used, if False, both dry and wet files are used
        RIR_path_csv="/data5/eloi/ImpulseResponses/rir_files_train.csv",
        sinc_train_params=None,
        apply_effects_on_dry=False, #if True, the dry signal will be augmented with effects
        ):

        super().__init__()
        print(num_examples, "num_examples")

        #np.random.seed(seed)
        #random.seed(seed)
        self.apply_effects_on_dry = apply_effects_on_dry    

        #self.random_num_tracks=random_num_tracks

        self.random_polarity=random_polarity
        self.random_shift=random_shift

        self.only_dry=only_dry

        #torch.manual_seed(seed)

        self.mode=mode
    
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
                #RMS=20*torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1)))
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

        self.path_csv=path_csv
        assert self.path_csv is not None, "path_csv must be provided"

        import pandas as pd
        self.df=pd.read_csv(self.path_csv)

        #self.df= self.df[self.df['cluster'].isin(clusters)]
        self.audio_length_cache = {}
        self.path_files_cache = {}


        df= pd.read_csv(RIR_path_csv)
        #list of RIR files
        RIR_files = df['rir_file'].tolist()
        print(RIR_files[:10])

        acceoted_sampling_rates = [44100]

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


        if sinc_train_params is not None:
            self.sinc_train_probability=sinc_train_params.probability
            self.probability_mono=sinc_train_params.probability_mono
            from utils.data_utils import synthesize_sinc_train
            
            self.synthesize_sinc_train_fn = lambda num_samples:  synthesize_sinc_train(num_samples, 
                                                                                   sample_rate=self.fs, 
                                                                                   min_cutoff_ratio=sinc_train_params.min_cutoff_ratio, 
                                                                                   max_cutoff_ratio=sinc_train_params.max_cutoff_ratio, 
                                                                                   min_gain=sinc_train_params.min_gain, 
                                                                                   max_gain=sinc_train_params.max_gain, 
                                                                                   mean_pulse_rate=sinc_train_params.mean_pulse_rate)
        else:
            self.sinc_train_probability=0.0
            self.synthesize_sinc_train_fn = None
                                                                                
        

    def __iter__(self):

        
        while True:
            skip_iteration = False
            #print("Iterating over TencyMastering_Vocals dataset")

            #for id in tqdm(self.id_list):
            if self.sinc_train_probability > 0 and np.random.rand() < self.sinc_train_probability:
                print("Generating sinc train sample")
                
                if np.random.rand() < self.probability_mono:
                    #mono sinc train
                    x_dry=self.synthesize_sinc_train_fn(self.segment_length)
                    #expand to stereo
                    x_dry = x_dry.repeat(2, 1)
                    x_wet= x_dry.clone()  # For sinc train, wet is the same as dry
                else:
                    x_dry_left = self.synthesize_sinc_train_fn(self.segment_length)
                    x_dry_right = self.synthesize_sinc_train_fn(self.segment_length)
                    x_dry = torch.cat((x_dry_left, x_dry_right), dim=0)
                    x_wet= x_dry.clone()  # For sinc train, wet is the same as dry

                _, augment_target=self.augment_chain(x_wet.cpu().numpy().T, x_wet.cpu().numpy().T)
                augment_target = torch.from_numpy(augment_target.T).float().to(x_wet.device)

                assert augment_target.shape == x_wet.shape, "augment_target shape must match x_wet shape, got {} and {}".format(augment_target.shape, x_wet.shape)
                assert x_dry.shape == augment_target.shape, "x_dry shape must match augment_target shape, got {} and {}".format(x_dry.shape, augment_target.shape)

                if self.apply_effects_on_dry:
                    _, x_dry = self.augment_chain(x_dry.cpu().numpy().T, x_dry.cpu().numpy().T)
                    x_dry = torch.from_numpy(x_dry.T).float().to(x_wet.device)
                
                assert x_dry.shape == augment_target.shape, "x_dry shape must match augment_target shape, got {} and {}".format(x_dry.shape, augment_target.shape)

                x_dry= x_dry.mean(0, keepdim=True)  # Convert stereo to mono for dry

                yield  x_dry, augment_target



            num=np.random.randint(0,len(self.df)-1)

            row=self.df.iloc[num]
            
            path=row["path"]
    
            dry_path=os.path.join(path, "dry_multi")
            wet_path=os.path.join(path, "multi")

            if dry_path in self.path_files_cache.keys():
                files= self.path_files_cache[dry_path]
            else:
                files= glob.glob(os.path.join(dry_path, "*.wav"))
                self.path_files_cache[dry_path] = glob.glob(os.path.join(dry_path, "*.wav"))

            if len(files)==0:
                print("no dry files found in", dry_path)
                continue

            files_taxonomy = [os.path.basename(f)[:4] for f in files]


            if self.tracks == "all":
                dry_files=[]
                wet_files=[]
                taxonomies=[]
                for i, file_taxonomy in enumerate(files_taxonomy):
                    dry_files.append(files[i])
                    taxonomies.append(file_taxonomy)
                    if not self.only_dry:
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
                continue
            
            assert len(dry_files) > 0, "no dry files found in {}".format(dry_path)
            assert len(dry_files) == len(taxonomies), "number of dry files and taxonomies must match, got {} and {}".format(len(dry_files), len(taxonomies))
            if not self.only_dry:
                assert len(wet_files) == len(dry_files), "number of wet files and dry files must match, got {} and {}".format(len(wet_files), len(dry_files))
            

            total_frames= 999999999999
            dry_frames=[]

            for i in range(len(dry_files)):
                if str(dry_files[i]) in self.audio_length_cache.keys():
                    #print("dry file FUND in cache", dry_files[i])
                    dry_duration, dry_total_frames, dry_samplerate = self.audio_length_cache[str(dry_files[i])]
                else:
                    #print("dry file not in cache", dry_files[i])
                    dry_duration, dry_total_frames, dry_samplerate=get_audio_length(str(dry_files[i]))
                    self.audio_length_cache[str(dry_files[i])] = (dry_duration, dry_total_frames, dry_samplerate)

                total_frames= min(total_frames, dry_total_frames)
                dry_frames.append(dry_total_frames)

            if not self.only_dry:
                wet_frames=[]
                for i in range(len(wet_files)):
                    if str(wet_files[i]) in self.audio_length_cache.keys():
                        #print("wet file in cache", wet_files[i])
                        wet_duration, wet_total_frames, wet_samplerate = self.audio_length_cache[str(wet_files[i])]
                    else:
                        #print("wet file not in cache", wet_files[i])
                        wet_duration, wet_total_frames, wet_samplerate=get_audio_length(str(wet_files[i]))
                        self.audio_length_cache[str(wet_files[i])] = (wet_duration, wet_total_frames, wet_samplerate)
                    total_frames= min(total_frames, wet_total_frames)
                    wet_frames.append(wet_total_frames)

                alignment_file=Path(dry_files[0]).parent.parent / "alignment.pickle"
                alignment_data = pickle.load(open(alignment_file, "rb"))
                dry_align=alignment_data.get("dry_alignment",0)
                wet_align=alignment_data.get("multi_alignment",0)

                # Calculate the valid range for the start index considering alignments
                max_dry_offset = max(0, dry_align)
                max_wet_offset = max(0, wet_align)
                min_dry_offset = min(0, dry_align)
                min_wet_offset = min(0, wet_align)
    
                # Ensure we have enough samples at the beginning and end
                valid_start_min = max(0, -min_dry_offset, -min_wet_offset)
                valid_end_max = min(total_frames, total_frames - max_dry_offset, total_frames - max_wet_offset)
    
                # Ensure we have enough space for the segment
                valid_range = valid_end_max - valid_start_min - self.segment_length
                if valid_range <= 0:
                    # Handle the case where the segment can't fit with alignments
                    # Either reduce segment length or use padding
                    print(f"Cannot fit segment of length {self.segment_length} with alignments {dry_align}, {wet_align}")
                    continue
    
                # Choose a random start within the valid range
                start = valid_start_min + np.random.randint(0, valid_range)
                end = start + self.segment_length
    

                #print("valid_start_min", valid_start_min, "valid range", valid_range,"start", start, "end", end, "total_frames", total_frames, "segment_length", self.segment_length, "dry_align", dry_align, "wet_align", wet_align)
    
                # Apply alignments
                start_dry = start + dry_align
                end_dry = start_dry + self.segment_length
    
                start_wet = start + wet_align
                end_wet = start_wet + self.segment_length

            else:
                start_dry=np.random.randint(0,total_frames-self.segment_length)
                end_dry=start_dry+ self.segment_length


            order = np.random.permutation(len(dry_files))
            dry_files = [dry_files[i] for i in order]
            if not self.only_dry:
                wet_files = [wet_files[i] for i in order]
            taxonomies = [taxonomies[i] for i in order]

            found=False

            if self.only_dry:
                wet_files = [None] * len(dry_files)  # Placeholder for wet files if only dry is used

            for i, (dry_file, wet_file) in enumerate(zip(dry_files, wet_files)):

                #print("dry_file", dry_file, "wet_file", wet_file, i, "of", len(dry_files), "start", start, "end", end)
                out=load_audio(str(dry_file), start_dry, end_dry, stereo=self.stereo)
                if out is None:
                    skip_iteration = True
                    print("Could not load dry audio file: {}".format(dry_file))
                    continue
                
                x_dry, fs=out
                assert fs==self.fs, "wrong sampling rate: {}".format(fs)

                assert x_dry.shape[-1]==self.segment_length, "x_dry must have the same length as segment_length, got {}".format(x_dry.shape[-1])
                assert x_dry.shape[0]==2, "x_dry must have 2 channels, got {}".format(x_dry.shape[0])

                #convert stereo to mono for dry
                x_dry=x_dry.mean(0, keepdim=True) 

                #x_all[i]=x_dry

                if not self.only_dry:
                    if self.random_shift is not None:
                        if self.random_shift > 0:
                            shift = np.random.randint(-self.random_shift, self.random_shift)
                            start_wet = start_wet + shift
                            end_wet = start_wet + self.segment_length
                            if start_wet < 0:
                                start_wet = max(0, start_wet)
                                end_wet= start_wet + self.segment_length
                            if end_wet > wet_frames[i]:
                                end_wet = min(wet_frames[i], end_wet)
                                start_wet = end_wet - self.segment_length
    
                    out=load_audio(str(wet_file), start_wet, end_wet, stereo=self.stereo)
                    if out is None:
                        skip_iteration = True
                        print("Could not load wet audio file: {}".format(wet_file))
                        continue
                    
                    x_wet, fs=out
                    assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                    
                    assert x_wet.shape[-1]==self.segment_length, "x_wet_long must have the same length as segment_length, got {}".format(x_wet.shape[-1])
                    assert x_wet.shape[0]==2, "x_wet_long must have 2 channels, got {}".format(x_wet.shape[0])
    
                    if self.random_polarity:
                        if np.random.rand() > 0.5:
                            x_wet = -x_wet

                    RMS_dB_wet=self.get_RMS(x_wet.mean(0))
                    if RMS_dB_wet<self.RMS_threshold_dB:
                        continue
                #x_all_wet[i]=x_wet

                #x_sum+=x_dry
                #x_sum_wet+=x_wet

                RMS_dB=self.get_RMS(x_dry.mean(0))
                if RMS_dB<self.RMS_threshold_dB:
                    continue

            
                if self.normalize_mode is not None:
                    if "dry" in self.normalize_mode:
                        #potentially slow
                        x_dry, scaler=self.normaliser(x_dry)
                    else:
                        pass
            
                found=True
                taxonomy_found= taxonomies[i]
                break

            if not found:
                print("No valid dry file found in", dry_path)
                continue

            if self.only_dry:
                yield x_dry
            else:

                _, augment_target=self.augment_chain(x_wet.cpu().numpy().T, x_wet.cpu().numpy().T)
                augment_target = torch.from_numpy(augment_target.T).float().to(x_wet.device)
                assert augment_target.shape == x_wet.shape, "augment_target shape must match x_wet shape, got {} and {}".format(augment_target.shape, x_wet.shape)

                #print("yielding dry file", x_dry.shape, "wet file", augment_target.shape, x_dry.dtype, augment_target.dtype)

                yield  x_dry, augment_target





class SincTrainTest(torch.utils.data.Dataset):

    def __init__(self,
        fs=44100,
        segment_length=131072,
        mode="dry-wet",
        num_examples=4,
        seed=42,
        sinc_train_params=None,
        RIR_path_csv="/data5/eloi/ImpulseResponses/rir_files_train.csv",
        ):

        super().__init__()
        print(num_examples, "num_examples")

        np.random.seed(seed)
        random.seed(seed)


        self.segment_length=segment_length
        self.fs=fs


        if sinc_train_params is not None:
            self.probability_mono=sinc_train_params.probability_mono
            from utils.data_utils import synthesize_sinc_train
            
            self.synthesize_sinc_train_fn = lambda num_samples:  synthesize_sinc_train(num_samples, 
                                                                                   sample_rate=self.fs, 
                                                                                   min_cutoff_ratio=sinc_train_params.min_cutoff_ratio, 
                                                                                   max_cutoff_ratio=sinc_train_params.max_cutoff_ratio, 
                                                                                   min_gain=sinc_train_params.min_gain, 
                                                                                   max_gain=sinc_train_params.max_gain, 
                                                                                   mean_pulse_rate=sinc_train_params.mean_pulse_rate)
        else:
            raise ValueError("sinc_train_params must be provided")
                                                                                
 
        df= pd.read_csv(RIR_path_csv)
        #list of RIR files
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

       


        counter=0

        self.num_examples=num_examples
        self.test_samples=[]

        for i in range(self.num_examples):
            if self.probability_mono > 0 and np.random.rand() < self.probability_mono:
                #mono sinc train
                x_dry=self.synthesize_sinc_train_fn(self.segment_length)
                #expand to stereo
                x_dry = x_dry.repeat(2, 1)
                x_wet= x_dry.clone()


            else:
                x_dry_left = self.synthesize_sinc_train_fn(self.segment_length)
                x_dry_right = self.synthesize_sinc_train_fn(self.segment_length)
                x_dry = torch.cat((x_dry_left, x_dry_right), dim=0)
                x_wet= x_dry.clone()
            

            _, augment_target=self.augment_chain(x_wet.cpu().numpy().T, x_wet.cpu().numpy().T)

            augment_target = torch.from_numpy(augment_target.T).float().to(x_wet.device)
            assert augment_target.shape == x_wet.shape, "augment_target shape must match x_wet shape, got {} and {}".format(augment_target.shape, x_wet.shape)

            assert x_dry.shape == augment_target.shape, "x_dry shape must match augment_target shape, got {} and {}".format(x_dry.shape, augment_target.shape)

            assert x_dry.shape[0] == 2, "x_dry must have 2 channels, got {}".format(x_dry.shape[0])


            self.test_samples.append(( x_dry,  augment_target)) 

        random.shuffle(self.test_samples)

        if num_examples != -1:
            self.test_samples = self.test_samples[:num_examples]


    def __getitem__(self, idx):
        return self.test_samples[idx]

    def __len__(self):
        return len(self.test_samples)
