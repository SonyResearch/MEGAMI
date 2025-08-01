#load generator model
import hydra
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import utils.training_utils as tr_utils


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


#Loading EffectsDiT
config_file_rel="../conf"
config_path="/home/eloi/projects/project_mfm_eloi/src/conf"
config_name="conf_S9_tencymastering_multitrack_paired_stylefxenc2048AF_contentCLAP_CLAPadaptor.yaml"
model_dir="/data5/eloi/checkpoints/S9"
ckpt="1C_tencymastering_vocals-110000.pt"

#config_name="conf_S7_tencymastering_multitrack_paired_stylefxenc2048AF_contentCLAP.yaml"
#model_dir="/data5/eloi/checkpoints/S7"
#ckpt="1C_tencymastering_vocals-50000.pt"

overrides = [
    f"model_dir={model_dir}",
    f"tester.checkpoint={ckpt}",
    "tester.cfg_scale=1.0",
    "tester.sampling_params.T=30",
]

with initialize(version_base=None, config_path=config_file_rel):
    args = compose(config_name=config_name, overrides=overrides)

if not os.path.exists(args.model_dir):
    raise Exception(f"Model directory {args.model_dir} does not exist")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

diff_params=hydra.utils.instantiate(args.diff_params)

network=hydra.utils.instantiate(args.network)
network=network.to(device)
state_dict = torch.load(os.path.join(args.model_dir,args.tester.checkpoint), map_location=device, weights_only=False)

tr_utils.load_state_dict(state_dict, ema=network)

sampler = hydra.utils.instantiate(args.tester.sampler, network, diff_params, args, )


#Loading blackbox mapper

#config_name="conf_M2_mapper_blackbox_predictive_fxenc2048AFv3CLAP_paired.yaml"
#model_dir="/data5/eloi/checkpoints/M2_mapper_blackbox_TM"
#ckpt="mapper_blackbox_TCN-44000.pt"

config_name="conf_MF3wet_mapper_blackbox_predictive_fxenc2048AFv3CLAP_paired.yaml"
model_dir="/data5/eloi/checkpoints/MF3wet"
ckpt="mapper_blackbox_TCN-85000.pt"

overrides = [
    f"model_dir={model_dir}",
    f"tester.checkpoint={ckpt}",
]

with initialize(version_base=None, config_path=config_file_rel):
    args = compose(config_name=config_name, overrides=overrides)

if not os.path.exists(args.model_dir):
    raise Exception(f"Model directory {args.model_dir} does not exist")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mapper=hydra.utils.instantiate(args.network)
mapper=mapper.to(device)
state_dict = torch.load(os.path.join(args.model_dir,args.tester.checkpoint), map_location=device, weights_only=False)

tr_utils.load_state_dict(state_dict, network=mapper)


import torch
import os



import torch
from utils.collators import collate_multitrack_paired
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from datasets.tency_mastering_multitrack_paired import TencyMastering_Test
import omegaconf

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
  num_tracks= 2,
  tracks="all",
  path_csv= "/data5/eloi/TencyMastering/PANNs_country_pop/train_split.csv",
  normalize_params=normalize_params,
  num_examples= -1, #use all examples
  RMS_threshold_dB= -40.0,
  seed= 42
)

batch_size = 1
val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, num_workers=1, collate_fn=lambda x: x) 

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs=[]  # List to store input audio tensors
outputs_ref=[]


import math
import omegaconf
import torch

Fxencoder_kwargs=omegaconf.OmegaConf.create(
{
        "ckpt_path": "/home/eloi/projects/project_mfm_eloi/src/utils/feature_extractors/ckpt/fxenc_plusplus_default.pt"
}
)

from evaluation.feature_extractors import load_fx_encoder_plusplus_2048
feat_extractor = load_fx_encoder_plusplus_2048(Fxencoder_kwargs, device)

from utils.AF_features_embedding_v2 import AF_fourier_embedding
AFembedding= AF_fourier_embedding(device=device)

def FxEnc(x):
    """
    x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
    """
    z=feat_extractor(x)
    z= torch.nn.functional.normalize(z, dim=-1, p=2)  # normalize to unit variance
    z= z* math.sqrt(z.shape[-1])  # rescale to keep the same scale

    z_af,_=AFembedding.encode(x)
    z_af=z_af* math.sqrt(z_af.shape[-1])  # rescale to keep the same scale

    z_all= torch.cat([z, z_af], dim=-1)

    #now L2 normalize

    norm_z= z_all/ math.sqrt(z_all.shape[-1])  # normalize by dividing by sqrt(dim) to keep the same scale

    return norm_z

def embedding_post_processing(z):
    """
    L2 normalize each of the features in z
    """
    z_fxenc=z[..., :2048]  # assuming the FxEncoder features are the first 2048 dimensions
    z_af=z[..., 2048:]  # assuming the AF features are the last 2048 dimensions

    z_fxenc=torch.nn.functional.normalize(z_fxenc, dim=-1, p=2)  # normalize to unit variance
    z_af=torch.nn.functional.normalize(z_af, dim=-1, p=2)

    z_fxenc=z_fxenc * math.sqrt(z_fxenc.shape[-1])  # rescale to keep the same scale
    z_af=z_af * math.sqrt(z_af.shape[-1])  # rescale to

    z_all= torch.cat([z_fxenc, z_af], dim=-1)

    return z_all/ math.sqrt(z_all.shape[-1])  # normalize by dividing by sqrt(dim) to keep the same scale


def get_log_rms_from_z(z):

    z = z * math.sqrt(z.shape[-1])  # rescale to keep the same scale
    AF=z[...,2048:]  # assuming the AF features are the last 2048 dimensions
    AF=AF/ math.sqrt(AF.shape[-1])  # normalize to unit variance

    features= AFembedding.decode(AF)
    log_rms=features[0]

    return log_rms


def generate_Fx(x, input_type="dry",num_samples=1, T=30):
    B, N, C, L = x.shape  # B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio

    masks= torch.ones((B, N), dtype=torch.bool, device=device)  # Create masks for all tracks, assuming all tracks are present

    shape=sampler.diff_params.default_shape
    shape= [num_samples, N,*shape[2:]]  # B is the batch size, we want to sample B samples

    sampler.T=T

    with torch.no_grad():
        is_wet= "wet" in input_type
        cond, x_preprocessed=sampler.diff_params.transform_forward(x,  is_condition=True, is_test=True, masks=masks, is_wet=is_wet)
        cond=cond.expand(shape[0], -1, -1,-1)  # Expand the condition to match the batch size
        preds, noise_init = sampler.predict_conditional(shape, cond=cond.contiguous(), cfg_scale=args.tester.cfg_scale, device=device,  masks=masks)

    return preds

def apply_rms(y_hat, z_pred):
    """
    Apply RMS normalization to the generated audio y_hat based on the predicted features z_pred.
    """
    pred_logrms=get_log_rms_from_z(z_pred)  # get the log RMS from the generated features
    rms_y=10**(pred_logrms.unsqueeze(-1)/20)  # Convert log RMS to linear scale
    y_final= y_hat * (rms_y / torch.sqrt(torch.mean(y_hat**2, dim=(-1), keepdim=True) + 1e-6))
    return y_final

from utils.data_utils import apply_RMS_normalization

def apply_effects(x, z_pred):

    x_norm= x[0].mean(dim=1, keepdim=True)  # Normalize the input audio by its mean across the tracks
    x_norm=apply_RMS_normalization(x_norm,-25.0, device=device)

    with torch.no_grad():
        y_hat=mapper(x_norm, z_pred)
    
    y_final=apply_rms(y_hat, z_pred)  # Apply RMS normalization to the generated audio

    return y_final
        

import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE   
from utils.log import make_PCA_figure
for i in range(len(dataset_val)):
    datav=dataset_val[i],
    collated_data= collate_multitrack_paired(datav)
    x=collated_data['x'].to(device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
    y=collated_data['y'].to(device)  # y is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio

    z_ref=FxEnc(y[0])  # z_y is a tensor of shape [B, N, D] where D is the dimension of the features (2048 + 2048 = 4096)

    taxonomy=collated_data['taxonomies']  # taxonomy is a list of lists of taxonomies, each list is a track, each taxonomy is a string of 2 digits

    mixture=y.sum(dim=1, keepdim=False)


    from IPython.display import Audio, display
    print("reference")
    #Audio(mixture[0].cpu().clamp(-1,1), rate=44100, normalize=False)  # Play the first track of the input audio
    #save audio file
    sf.write(f"examples/ref_{i}.wav", mixture[0].cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')

    y_final=apply_effects(x, z_ref)  # Apply the effects to the input audio

    y_hat_mixture=y_final.sum(dim=0, keepdim=False)
    a=Audio(y_hat_mixture.cpu().clamp(-1,1), rate=44100, normalize=False)  # Play the first track of the generated audio
    sf.write(f"examples/ref_blackbox_{i}.wav", y_hat_mixture.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')  # Save the generated audio
    #display(a)

    preds=generate_Fx(x, input_type="dry", num_samples=5) 
    z_pred=embedding_post_processing(preds)  # post-process the generated features

    print("predictions")
    for j in range(z_pred.shape[0]):
        y_final=apply_effects(x, z_pred[j])  # Apply the effects to the input audio
    
        y_hat_mixture=y_final.sum(dim=0, keepdim=False)
    
        #a=Audio(y_hat_mixture.cpu().clamp(-1,1), rate=44100, normalize=False)  # Play the first track of the generated audio
        #display(a)  # Display the audio player in the notebook
        sf.write(f"examples/pred_blackbox_{i}_{j}.wav", y_hat_mixture.cpu().clamp(-1,1).numpy().T, 44100, subtype='PCM_16')


    preds=generate_Fx(x, input_type="dry", num_samples=300, T=30) 
    preds=embedding_post_processing(preds)  # post-process the generated features

    # Step 1: Optional PCA preprocessing (recommended for high dimensions)
    # If you want to add PCA preprocessing:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100)
    combined_data_flat = torch.cat([preds.view(preds.shape[0],-1), z_ref.view(-1).unsqueeze(0)], dim=0).cpu().numpy()
    combined_data = pca.fit_transform(combined_data_flat)

    tsne =TSNE(n_components=2, perplexity=30)
    #combined_data = torch.cat([preds.view(preds.shape[0],-1), z_ref.view(-1).unsqueeze(0)], dim=0).cpu().numpy()
    combined_result = tsne.fit_transform(combined_data)

    data_dict = {
       "predicted": combined_result[:-1],
       "reference": combined_result[-1:]
         } 

    fig = make_PCA_figure(data_dict, title="TSNE")
    fig.savefig(f"examples/tsne_blackbox_{i}.png")
