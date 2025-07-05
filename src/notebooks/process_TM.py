import os
import yaml
import einops
import torch
import torchaudio
import soundfile as sf
from glob import glob
import tqdm


path_TM="/data5/eloi/TencyMastering"
subdirs=["part1","part2", "part3", "part4", "part4_test", "part4_validation"]

dir_out="dry_multi"
correspondence_file = "correspondence.yaml"
path_dry="dry"

#mix_filenames= ["nbv-bv-ld-nomasterfx_dry.wav", "nbv-bv-ld-NoMasterFx_dry.wav", "nbv-ld-nomasterfx_dry.wav", "nbv-ld-NoMasterFx_dry.wav","nbv-bv-nomasterfx_dry.wav", "nbv-nomasterfx_dry.wav"]

#skip_ids = ["6699", "33787", "65486", "66235", "67570", "67808", "12212", "67437", "67476", "67658", "67719", "10356"]

do_selection=True

for subdir in subdirs:
    path=os.path.join(path_TM, subdir)

    #Iterate over each subdirectory
    ids = glob(os.path.join(path,  "*"))

    for id in tqdm.tqdm(ids):
        #if os.path.basename(id) in skip_ids:
        #    print(f"Skipping {id} because it is in the skip list")
        #    continue
        try: 

            correspondence_file_path = os.path.join(id, correspondence_file)
    
            if not os.path.exists(correspondence_file_path):
                print(f"Skipping {id} because the correspondence file does not exist")
                continue
    
    
            if not os.path.exists(os.path.join(id, dir_out)):
                os.makedirs(os.path.join(id, dir_out))
    
            # Load the .yaml correspondence file
            correspondence = yaml.safe_load(open(correspondence_file_path, 'r'))
    
            routing=correspondence["routing"][0]["routing_hierarchy"]
    
            multi_tracks=correspondence["routing"][0]["routing_hierarchy"].keys()
            multi_tracks = list(multi_tracks)
    
            for track in multi_tracks:
                routing_track = routing[track]
                if "children" not in routing_track:
                    print(f"Skipping {id} because the track {track} has no children")
                    continue
    
                path_dry_track = os.path.join(id, path_dry)
    
                for i, child in enumerate(routing_track["children"]):
                    #check if the child is a file
                    if not os.path.isfile(os.path.join(path_dry_track, child+".wav")):
                        print(f"Skipping {id} because the child {child} is not a file")
                        continue
                    else:
                        # Load the audio file
                        audio, sr = torchaudio.load(os.path.join(path_dry_track, child+".wav"))
                        audio = audio

                        if audio.shape[0] ==1:
                            audio= audio.repeat(2, 1)
    
                        #get gain
                        gain = routing_track["children"][child]["decomposition_gain"]
                        gain = torch.tensor(gain )
    
                        if i==0:
                            mix = audio * gain
                        else:
                            mix += audio * gain
                # Save the mix                     
                mix = mix.cpu()
                mix = einops.rearrange(mix, 'c t -> t c')
                print(f"Saving mix for {id} in {os.path.join(id, dir_out, track+'.wav')}")
    
                sf.write(os.path.join(id, dir_out, track+'.wav'), mix.numpy(), sr)


        except Exception as e:
                print(f"Error processing {id}: {e}")
                continue
            
            



