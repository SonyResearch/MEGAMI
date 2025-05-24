
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
    wet_filename = "vocals.flac"
    fxnorm_filename = "vocals_normalized.flac"


    results = []

    for song_id in song_id_list:
        #print(song_id)
        song_id=Path(song_id)

        file_wet=song_id / wet_filename
        file_normalized=song_id / fxnorm_filename

        if os.path.exists(file_wet) and os.path.exists(file_normalized):

            results.append((file_wet, file_normalized))

        else:
            continue

    print("results", len(results))
    return results



class Tency1_FxNorm_Vocals(torch.utils.data.IterableDataset):
    def __init__(self,
        fs=44100,
        segment_length=131072,
        skip=None,
        base_dir=None,
        subdirs=None,
        skip_list=None,
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

        #print("id_list", len(id_list))
        def filter_ids(path):
            id_track=path.split("/")[-1]
            partition=path.split("/")[-2]
            id_name=partition+"/"+id_track
            if id_name in skip_list:
                return False
            else:
                return True

        #id_list=[x for x in id_list if filter_ids(x)]
        #print("id_list after filtering", len(id_list))
        assert len(id_list)>0, "No files found in the dataset"

        self.pair_list=process_id_list_lead_vocal(id_list)
        #print("pair_list", len(self.pair_list))
        self.segment_length=segment_length
        self.fs=fs


        self.stereo=stereo


        
        self.RMS_threshold_dB=RMS_threshold_dB

        self.get_RMS=lambda x: 20 * torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1)))
        
        self.side_energy_threshold=side_energy_threshold

        #raise ValueError("stop here")

    def __iter__(self):
        while True:
            try:
                #bignum=np.random.randint(0,10000-1)

                num=np.random.randint(0,len(self.pair_list)-1)
                wet_file, norm_file=self.pair_list[num]

                norm_duration, norm_total_frames, norm_samplerate=get_audio_length(norm_file)
                wet_duration, wet_total_frames, wet_samplerate=get_audio_length(wet_file)

                total_frames=min(norm_total_frames, wet_total_frames)

                #sample "start" uniformly in the range [0, total_frames-seg_len]
                start=np.random.randint(0,total_frames-self.segment_length)
                end=start+ self.segment_length

                x_norm, fs=read_wav_segment(norm_file, start, end)
                assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                if self.stereo:
                    if len(x_norm.shape)==1:
                        x_norm=x_norm[:,np.newaxis]
                        x_norm = np.concatenate((x_norm, x_norm), axis=-1)
                    elif len(x_norm.shape)==2 and x_norm.shape[-1]==1:
                        x_norm = np.concatenate((x_norm, x_norm), axis=-1)

                x_norm=torch.from_numpy(x_norm).permute(1,0)
    
                #if x_norm.size(0) > 1:
                #    left = x_norm[0]
                #    right = x_norm[1]
                #    left = left / left.max()
                #    right = right / right.max()
                #    side = (left - right) * 0.707
                #    side_energy = 20*torch.log10(side.abs().max()).item()
                #else:
                #    side_energy = -torch.inf
                
                #if side_energy > self.side_energy_threshold:
                #    print(f"Skip {norm_file} because of high side energy"+str(side_energy))
                #    continue


                y_wet, fs=read_wav_segment(wet_file, start, end)
                assert fs==self.fs, "wrong sampling rate: {}".format(fs)
                if self.stereo:
                    assert y_wet.shape[-1]==2, "not stereo"
                y_wet=torch.from_numpy(y_wet).permute(1,0)

                RMS_dB=self.get_RMS(y_wet.mean(0))
                if RMS_dB<self.RMS_threshold_dB:
                    #print(r"skip wet file because of low RMS (silence) {}".format(RMS_dB))
                    continue


                

                yield  y_wet, x_norm

            except Exception as e:
                print(e)
                continue

class Tency1_FxNorm_Vocals_Test(torch.utils.data.Dataset):

    def __init__(self,
        fs=44100,
        segment_length=131072,
        skip=None,
        base_dir=None,
        subdirs=None,
        skip_list=None,
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

        #id_list=[x for x in id_list if filter_ids(x)]
        assert len(id_list)>0, "No files found in the dataset"

        self.pair_list=process_id_list_lead_vocal(id_list)

        self.segment_length=segment_length
        self.fs=fs


        self.stereo=stereo

        
        self.RMS_threshold_dB=RMS_threshold_dB

        self.get_RMS=lambda x: 20 * torch.log10(torch.sqrt(torch.mean(x ** 2, dim=-1)))

        self.side_energy_threshold=side_energy_threshold

        self.test_samples = []

        counter=0

        for wet_file, norm_file in self.pair_list:
            if counter >= num_examples:
                break

            norm_duration, norm_total_frames, norm_samplerate=get_audio_length(norm_file)
            wet_duration, wet_total_frames, wet_samplerate=get_audio_length(wet_file)

            total_frames=min(norm_total_frames, wet_total_frames)

            #sample "start" uniformly in the range [0, total_frames-seg_len]
            #start=np.random.randint(0,total_frames-self.segment_length)

            start=total_frames//2 #fixed
            end=start+ self.segment_length


            x_norm, fs=read_wav_segment(norm_file, start, end)
            assert fs==self.fs, "wrong sampling rate: {}".format(fs)
            if self.stereo:
                if self.stereo:
                    if len(x_norm.shape)==1:
                        print( "norm not stereo , doubling channels", norm_file.shape)
                        x_norm=x_norm[:,np.newaxis]
                        x_norm = np.concatenate((x_norm, x_norm), axis=-1)
                    elif len(x_norm.shape)==2 and x_norm.shape[-1]==1:
                        print( "norm not stereo , doubling channels", norm_file.shape)
                        x_norm = np.concatenate((x_norm, x_norm), axis=-1)

            x_norm=torch.from_numpy(x_norm).permute(1,0)

            #if x_norm.size(0) > 1:
            #    left = x_norm[0]
            #    right = x_norm[1]
            #    left = left / left.max()
            #    right = right / right.max()
            #    side = (left - right) * 0.707
            #    side_energy = 20*torch.log10(side.abs().max()).item()
            #else:
            #    side_energy = -torch.inf
            
            #if side_energy > self.side_energy_threshold:
            #    print(f"Skip {norm_file}" +str(side_energy))
            #    continue
            #else:
            #    print(f"not skipping {norm_file} because of side energy"+str(side_energy))


            y_wet, fs=read_wav_segment(wet_file, start, end)
            assert fs==self.fs, "wrong sampling rate: {}".format(fs)
            if self.stereo:
                assert y_wet.shape[-1]==2, "not stereo"
            y_wet=torch.from_numpy(y_wet).permute(1,0)

            if self.get_RMS(y_wet.mean(0))<self.RMS_threshold_dB:
                #print("skip wet file because of low RMS (silence)")
                continue


            self.test_samples.append(( y_wet, x_norm))
            counter+=1

    def __getitem__(self, idx):
        return self.test_samples[idx]

    def __len__(self):
        return len(self.test_samples)
