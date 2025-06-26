import os
import sys
import torchaudio
from importlib import import_module
import torch
import yaml

def load_fx_encoder_plusplus(model_args, device):
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

def load_CLAP(model_args, device):

    #original_path = sys.path.copy()
    from utils.laion_clap.hook import CLAP_Module
    model= CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    #sys.path = original_path


    model.load_ckpt(model_args.ckpt_path)
    model.to(device)

    normalize = model_args.normalize

    def clap_fn(x):
        B, C, T = x.shape
        if C > 1:
            x= x.mean(dim=1, keepdim=True)  # Convert to mono if stereo

        with torch.no_grad():
            x=torchaudio.functional.resample(x, orig_freq=44100, new_freq=48000)
            x= x.squeeze(1)  # Remove channel dimension for CLAP
            emb=model.get_audio_embedding_from_data(x,use_tensor=True)

            # Normalize the embeddings
            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

            return emb

    return lambda x: clap_fn(x)


def load_MERT(model_args, device):

    from transformers import Wav2Vec2FeatureExtractor
    from transformers import AutoModel
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    model.to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)


    layer=model_args.layer
    average=model_args.average
    normalize=model_args.normalize


    def mert_fn(x):
        B, C, T = x.shape
        #first convert to mono
        if C > 1:
            x = x.mean(dim=1, keepdim=True)

        with torch.no_grad():
            #now resample
            x=torchaudio.functional.resample(x, orig_freq=44100, new_freq=processor.sampling_rate)
    
            inputs=processor(x.squeeze(1), sampling_rate=processor.sampling_rate, return_tensors="pt")
    
            input_values = inputs.input_values.to(device)[0]
            attention_mask=torch.ones_like(input_values).to(device)

             
            outputs = model(input_values, attention_mask, output_hidden_states=True)
            all_layer_hidden_states = torch.stack(outputs.hidden_states) #shape [L, B, T, D] where L is the number of layers, B is the batch size, T is the sequence length, and D is the hidden size

            if layer is not None:
                layer_hidden_states = all_layer_hidden_states[layer] #shape [B, T, D]
            else:
                layer_hidden_states = all_layer_hidden_states[-1] 

            if average:
                layer_hidden_states= layer_hidden_states.mean(dim=1) #shape [B, D]
             
            if normalize:
                layer_hidden_states = torch.nn.functional.normalize(layer_hidden_states, p=2, dim=-1)
            return layer_hidden_states
            
    return lambda x: mert_fn(x)         
            
             
             
def load_fx_encoder(model_args, device):
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
        

    return lambda x: effects_encoder_fn(x)

def load_AFxRep(model_args, device, sample_rate=44100, peak_scaling=True):

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

    feat_extractor = lambda x: wrapper_fn(x, sample_rate=sample_rate)

    return feat_extractor


