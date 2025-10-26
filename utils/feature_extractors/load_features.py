import os
import math
import sys
import torchaudio
from importlib import import_module
import torch
import yaml
import torch.nn.functional as F

def load_fx_encoder_plusplus(model_args, device, *args, **kwargs):
    from utils.feature_extractors.fx_encoder_plus_plus import load_model 

    assert model_args is not None, "model_args must be provided for fx_encoder type"

    ckpt_path=model_args.ckpt_path

    model=load_model(
        model_path=ckpt_path,
        device=device,
    )

    def effects_encoder_fn(x):
        assert x.ndim == 3, f"Input tensor x must be 2D, got {x.ndim}D"
        assert x.shape[1] == 2, f"Input tensor x must have 2 channels, got {x.shape[1]} channels"

        emb=model.get_fx_embedding(x)
        #l2 normalize the embeddings
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        return emb
        

    return lambda x: effects_encoder_fn(x)

def load_fx_encoder_plusplus_2048(model_args, device, *args, **kwargs):
    from utils.feature_extractors.fx_encoder_plus_plus import load_model 

    assert model_args is not None, "model_args must be provided for fx_encoder type"

    ckpt_path=model_args.ckpt_path

    model=load_model(
        model_path=ckpt_path,
        device=device,
    )

    def effects_encoder_fn(x):
        assert x.ndim == 3, f"Input tensor x must be 2D, got {x.ndim}D"
        assert x.shape[1] == 2, f"Input tensor x must have 2 channels, got {x.shape[1]} channels"

        emb=model.fx_encoder(x)
        emb=emb["embedding"]  # Extract the embedding from the dictionary

        return emb
        

    return lambda x: effects_encoder_fn(x)

def add_isotropic_noise(z, sigma=0.1):
        """
        z: [..., D] normalized embeddings (e.g., from CLAP or a regressor)
        sigma: scale of noise to inject
        Returns: z with orthogonal Gaussian noise added
        """
        n=torch.randn_like(z)  # isotropic noise
        z_noisy = F.normalize(z + sigma * n, dim=-1)
        return z_noisy




def load_CLAP(model_args, device, *args, **kwargs):

    #original_path = sys.path.copy()
    from utils.laion_clap.hook import CLAP_Module
    model= CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    #sys.path = original_path


    print("checkpoint",model_args.ckpt_path)
    #print current sys.path
    print("sys.path", sys.path)
    model.load_ckpt(model_args.ckpt_path)
    model.to(device)

    normalize = model_args.normalize

    if model_args.use_adaptor:
        if model_args.adaptor_type == "MLP_CLAP_regressor":
            from networks.MLP_CLAP_regressor import MLP_CLAP_regressor
            adaptor=MLP_CLAP_regressor()
            ckpt=torch.load(model_args.adaptor_checkpoint, map_location=device, weights_only=False)
            adaptor.load_state_dict(ckpt["network"], strict=True)
            adaptor.to(device)

        


    def clap_fn(x, type=None):
        B, C, T = x.shape
        if C > 1:
            x= x.mean(dim=1, keepdim=True)  # Convert to mono if stereo

        with torch.no_grad():
            x=torchaudio.functional.resample(x, orig_freq=44100, new_freq=48000)
            x= x.squeeze(1)  # Remove channel dimension for CLAP
            emb=model.get_audio_embedding_from_data(x,use_tensor=True)

            if type is not None:
                if type == "wet":
                    #print("wet mode")
                    if model_args.use_adaptor:
                        emb= adaptor(emb)  # Apply the adaptor if specified
    
            if model_args.add_noise:
                emb= torch.nn.functional.normalize(emb, p=2, dim=-1)  # Normalize before adding noise
                emb = add_isotropic_noise(emb, sigma=model_args.noise_sigma)

            # Normalize the embeddings
            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

            return emb

    return lambda x, type: clap_fn(x, type=type)


             
def load_fx_encoder(model_args, device, *args, **kwargs):
    """
    Load the FX Encoder model.
    
    Args:
        model_args: Arguments for the FX Encoder model.
        device: Device to load the model on (CPU or GPU).
        
    Returns:
        a function that extracts features from audio.
    """
    assert model_args is not None, "model_args must be provided for fx_encoder type"

    ckpt_path=model_args.ckpt_path

    #from utils.feature_extractors.fx_encoder import load_effects_encoder
    from utils.feature_extractors.networks import Effects_Encoder

    def reload_weights(model, ckpt_path, device):
        checkpoint = torch.load(ckpt_path, map_location=device)
    
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)


    try:
        with open(os.path.join('.','utils','feature_extractors', 'networks', 'configs.yaml'), 'r') as f:
            configs = yaml.full_load(f)
    except:
        with open(model_args.config_file, 'r') as f:
            configs = yaml.full_load(f)

    cfg_enc = configs['Effects_Encoder']['default']

    effects_encoder = Effects_Encoder(cfg_enc)
    reload_weights(effects_encoder, ckpt_path, device)
    effects_encoder.to(device)
    effects_encoder.eval()

    def effects_encoder_fn(x):
        emb=effects_encoder(x)
        #l2 normalize the embeddings
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb
        

    return lambda x, *args: effects_encoder_fn(x)

def load_AFxRep(model_args, device, sample_rate=44100, peak_scaling=True, *args, **kwargs):

    assert model_args is not None, "model_args must be provided for AFxRep type"

    ckpt_path=model_args.ckpt_path

    config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    encoder_configs = config["model"]["init_args"]["encoder"]

    module_path, class_name = encoder_configs["class_path"].rsplit(".", 1)
    module_path = module_path.replace("lcap", "utils.st_ito")

    module = import_module(module_path)

    model = getattr(module, class_name)(**encoder_configs["init_args"])

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # load state dicts
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("encoder"):
            state_dict[k.replace("encoder.", "", 1)] = v

    model.load_state_dict(state_dict)

    model.eval()

    model.to(device)

    def wrapper_fn(x, sample_rate):

        x=x.to(device)

        #x=torch.transpose(x,-1,-2)

        if sample_rate != 48000:
            x=torchaudio.functional.resample(x, sample_rate, 48000)

        bs= x.shape[0]
        #peak normalization. I do it because this is what ST-ITO get_param_embeds does. Not sure if it is good that this representation is invariant to gain
        if peak_scaling:
            x_max=[]
            for batch_idx in range(bs):
                #x[batch_idx, ...] /= x[batch_idx, ...].abs().max().clamp(1e-8)
                x_max.append( x[batch_idx, ...].abs().max().clamp(1e-8) )
            
            if x.ndim == 3:
                x_max=torch.stack(x_max, dim=0).view(bs, 1, 1)
            elif x.ndim == 2:
                x_max=torch.stack(x_max, dim=0).view(bs, 1)
    
            x=x/ x_max

        mid_embeddings, side_embeddings = model(x)

        # check for nan
        if torch.isnan(mid_embeddings).any():
            print("Warning: NaNs found in mid_embeddings")
            mid_embeddings = torch.nan_to_num(mid_embeddings)
        elif torch.isnan(side_embeddings).any():
            print("Warning: NaNs found in side_embeddings")
            side_embeddings = torch.nan_to_num(side_embeddings)

        mid_embeddings = torch.nn.functional.normalize(mid_embeddings, p=2, dim=-1)
        side_embeddings = torch.nn.functional.normalize(side_embeddings, p=2, dim=-1)
        
        embeddings_all= torch.cat([mid_embeddings, side_embeddings], dim=-1)

        return embeddings_all

    feat_extractor = lambda x, *args: wrapper_fn(x, sample_rate=sample_rate)

    return feat_extractor


