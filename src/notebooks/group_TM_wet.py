
import os
import yaml
import einops
import torch
import torchaudio
import soundfile as sf
from glob import glob
import tqdm


def group_classes(input_class, num_instr):
    if input_class is None:
        return 'unknown'  
    if num_instr == 1:
        mapping = {10000: 'mixture', 11000: 'mixture', 12000: 'mixture', 13000: 'mixture', 14000: 'mixture', 15000: 'mixture', 16000: 'mixture', 18000: 'mixture', 19000: 'mixture'}
    elif num_instr == 2:
        mapping = {10000: 'accompaniment', 11000: 'accompaniment', 12000: 'accompaniment', 13000: 'accompaniment', 14000: 'accompaniment', 14700: 'accompaniment', 14900: 'accompaniment', 15000: 'accompaniment', 16000: 'accompaniment', 19000: 'vocals'}
    elif num_instr == 4:
        mapping = {10000: 'other', 11100: 'drums', 11200: 'drums', 11300: 'other', 12000: 'bass', 13000: 'other', 14000: 'other', 14700: 'other', 14900: 'other', 15000: 'other', 16000: 'other', 18000: 'other', 19000: 'vocals'}
    elif num_instr == 6:
        mapping = {10000: 'other', 11100: 'drums', 11200: 'drums', 11300: 'other', 12000: 'bass', 13000: 'guitar', 14100: 'piano', 14200: 'piano', 14300: 'piano', 14400: 'other', 14500: 'other', 14600: 'other', 14700: 'other', 14900: 'other', 15000: 'other', 16000: 'other', 18100: 'guitar', 18200: 'other', 19000: 'vocals'}
    elif num_instr == 8:
        mapping = {10000: 'other', 11100: 'drums', 11200: 'drums', 11300: 'other', 12000: 'bass', 13000: 'guitar', 14100: 'piano', 14200: 'piano', 14300: 'piano', 14400: 'other', 14500: 'other', 14600: 'other', 14700: 'other', 14900: 'other', 15000: 'brass', 16100: 'strings', 16210: 'brass', 16220: 'brass', 18100: 'guitar', 18200: 'brass', 19000: 'vocals'}
    elif num_instr == 10:
        mapping = {10000: 'other', 11100: 'drums', 11200: 'drums', 11300: 'other', 12000: 'bass', 13000: 'guitar', 14100: 'piano', 14200: 'keyboard', 14300: 'keyboard', 14400: 'other', 14500: 'other', 14600: 'other', 14700: 'other', 14900: 'other', 15000: 'brass', 16100: 'strings', 16210: 'brass', 16220: 'woodwind', 18100: 'guitar', 18200: 'woodwind', 19000: 'vocals'}
    else:
        raise NotImplementedError()
 
    class_str = str(input_class)
    for i in range(len(class_str), 0, -1):
        general_class = int(class_str[:i] + "0" * (len(class_str) - i))
        print(f"Checking general class {general_class} for input class {input_class}")
        if general_class in mapping:
            return mapping[general_class]  
       
    raise ValueError(f"No mapping found for input class {input_class} with num_instr {num_instr}")



path_TM="/data2/eloi/TencyMastering"
subdirs=["part1","part2", "part3", "part4", "part4_test", "part4_validation"]



dir_out="wet_4instr"
dir_dry="multi"

do_selection=True

for subdir in subdirs:
    path=os.path.join(path_TM, subdir)

    #Iterate over each subdirectory
    ids = glob(os.path.join(path,  "*"))

    for id in tqdm.tqdm(ids):
        try:
            #if os.path.basename(id) in skip_ids:
            #    print(f"Skipping {id} because it is in the skip list")
            #    continue
            path_in= os.path.join(id, dir_dry)
            path_out= os.path.join(id, dir_out)
            #create the output directory if it does not exist
            if not os.path.exists(path_out):
                os.makedirs(path_out)
            #check if path_out is empty, if not, skip it
            #if os.listdir(path_out) and do_selection:
            #    print(f"Skipping {id} because {path_out} is not empty")
            #    continue
            #else:
            #    print(f"Processing {id}")
    
    
            if not os.path.exists(path_in):
                raise FileNotFoundError(f"Input directory {path_in} does not exist")
    
            #glob the wav files in the path_dry directory
            wav_files = glob(os.path.join(path_in, "*.wav"))
    
            tracks= {}
    
            #iterate over each wav file
            for i, wav_file in enumerate(wav_files):
                #take the first 4 characters of the filename as the taxonomy id
    
                taxonomy=os.path.basename(wav_file)[:4]
                #prepend a '1' to the taxonomy id 
                taxonomy = "1" + taxonomy
                inst_class= group_classes(int(taxonomy), 4)
    
                print(f"Processing {wav_file} with class {inst_class}, taxonomy {taxonomy}")
    
                if i==0:
                    x, fs = sf.read(wav_file)
                    x=torch.tensor(x).float()
    
                    tracks["vocals"]=torch.zeros_like(x)
                    tracks["bass"]=torch.zeros_like(x)
                    tracks["drums"]=torch.zeros_like(x)
                    tracks["other"]=torch.zeros_like(x)
    
                    tracks[inst_class] += x
                else:
                    x, fs = sf.read(wav_file)
                    x=torch.tensor(x).float()

                    print("x.shape", x.shape, "tracks[inst_class].shape", tracks[inst_class].shape)
    
                    #cut the track to the same length as the first track
                    if x.shape[0] > tracks[inst_class].shape[0]:
                        x = x[:tracks[inst_class].shape[0]]
                    elif x.shape[0] < tracks[inst_class].shape[0]:
                        #shape is [N, C] for stereo, we need to pad the second dimension
                        x=x.permute(1,0)
                        #shape is [C, N], we need to pad the second dimension
                        x = torch.nn.functional.pad(x, (0, tracks[inst_class].shape[0] - x.shape[-1]))
                        x= x.permute(1,0)
    
                    print("x.shape", x.shape, "tracks[inst_class].shape", tracks[inst_class].shape)
                    tracks[inst_class] += x
    
    
            for inst_class, track in tracks.items():
                print(f"Writing {inst_class} track with shape {track.shape} to {os.path.join(path_out, f'{inst_class}.wav')}")
                sf.write(os.path.join(path_out, f"{inst_class}.wav"), track.numpy(), fs)

        except Exception as e:
            print(f"Error processing {id}: {e}")
            continue
