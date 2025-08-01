import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


file_dry=os.path.join( "audio_examples","vocals.wav")


import soundfile as sf

dry, sr = sf.read(file_dry)

dry=torch.from_numpy(dry.T).float().unsqueeze(0)

dry=dry.mean(dim=1, keepdim=True)

start_t=15*sr
segment_length = 524288
dry_segment = dry[...,start_t:start_t + segment_length]


from distribution_presets.clusters_vocals import get_distributions_Cluster0 
from fx_pipeline import EffectRandomizer

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

distributions_dict=get_distributions_Cluster0(sample_rate=sr)

effect_randomizer=EffectRandomizer(sample_rate=sr, distributions_dict=distributions_dict, device=device)

batch_size=8
inputs = dry_segment.repeat(batch_size, 1, 1).to(device) #simulate a batch of 8 identical inputs


res=effect_randomizer.forward(inputs)

res_list=res.cpu().unbind(dim=0)


for i, r in enumerate(res_list):
    output_file = f"output_{i}.wav"
    sf.write(output_file, r.cpu().numpy().T, sr)



