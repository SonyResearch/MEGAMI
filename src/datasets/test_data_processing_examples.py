
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.collators import collate_multitrack_paired
from fx_model.apply_effects_multitrack_utils import multitrack_batched_processing
import os
from datasets.tency_mastering_multitrack_paired import TencyMastering_Test
import omegaconf


def load_paired_multitrack_dataset():
    normalize_params=omegaconf.OmegaConf.create(
        {
        "normalize_mode": "rms_dry",
        "rms_dry": -25.0
        }
    )

    dataset_val= TencyMastering_Test(
        mode= "dry-wet",
        segment_length= 525312,
        fs= 44100,
        stereo= True,
        tracks= "all",
        num_tracks= 2,
        path_csv= "/data5/eloi/TencyMastering/PANNs_country_pop/val_split.csv",
        normalize_params=normalize_params,
        num_examples= -1, #use all examples
        RMS_threshold_dB= -40.0,
        seed= 42
    )

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from fx_model.fxnormaug_v3_noiseless import FxNormAug
    randomizer=FxNormAug(sample_rate=44100, mode="train", device=device )
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    import torch
    batch_size=32
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=1, collate_fn=lambda x: x)


    outputs_dry = {}
    outputs_wet = {}
    for i,  data in enumerate(val_loader):
        print(i)
        data2 = data.copy()
    
        collated_data = collate_multitrack_paired(data2)
    
        x=collated_data['x'].to(device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        y=collated_data['y'].to(device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        taxonomy=collated_data['taxonomies']  # taxonomy is a list of lists of taxonomies, each list is a track, each taxonomy is a string of 2 digits
        masks=collated_data['masks'].to(device)  # masks is a tensor of shape [B, N] where B is the batch size and N is the number of tracks, it is used to mask the tracks that are not present in the batch
        paths=collated_data['paths']  # paths is a list of paths to the audio files, each path is a string
    
    
    
        func = lambda x, taxonomy: randomizer.forward(x, taxonomy=taxonomy) 
    
        x_dry_norm= multitrack_batched_processing(x.clone(), taxonomy=taxonomy, function=func, class_dependent=False, masks=masks)
    
        y_wet_norm= multitrack_batched_processing(y.clone(), taxonomy=taxonomy, function=func, class_dependent=False, masks=masks)
    
    
        if x_dry_norm.shape[2] != 2:
            x_dry_norm= x_dry_norm.repeat(1,1, 2, 1)
    
        if y_wet_norm.shape[2] != 2:
            y_wet_norm= y_wet_norm.repeat(1,1, 2, 1)


