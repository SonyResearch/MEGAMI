import os
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

def load_CLAP_standard(model_args, device, *args, **kwargs):

    #original_path = sys.path.copy()
    from utils.laion_clap.hook import CLAP_Module
    model= CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    #sys.path = original_path


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

        
    def add_isotropic_noise(z, sigma=0.1):
        """
        z: [..., D] normalized embeddings (e.g., from CLAP or a regressor)
        sigma: scale of noise to inject
        Returns: z with orthogonal Gaussian noise added
        """
        n=torch.randn_like(z)  # isotropic noise
        z_noisy = F.normalize(z + sigma * n, dim=-1)
        return z_noisy

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

    return lambda x: clap_fn(x)

def load_CLAP(model_args, device, *args, **kwargs):

    #original_path = sys.path.copy()
    from utils.laion_clap.hook import CLAP_Module
    model= CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    #sys.path = original_path


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

        
    def add_isotropic_noise(z, sigma=0.1):
        """
        z: [..., D] normalized embeddings (e.g., from CLAP or a regressor)
        sigma: scale of noise to inject
        Returns: z with orthogonal Gaussian noise added
        """
        n=torch.randn_like(z)  # isotropic noise
        z_noisy = F.normalize(z + sigma * n, dim=-1)
        return z_noisy

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


def load_MERT(model_args, device, *args, **kwargs):

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
            
    return lambda x, *args: mert_fn(x)         
            
             
             
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


def load_bark_spectrum(model_args,  *args, **kwargs):

    from utils.ITOMaster_loss import compute_barkspectrum

    def wrapper_fn(x):

        
        B, C, T = x.shape

        bark= compute_barkspectrum(x)

        #print("Bark spectrum shape:", bark.shape)

        return bark.view(B, -1)  # Flatten to shape [1, D]


    
    feat_extractor = lambda x, *args: wrapper_fn(x)

    return feat_extractor


import pyloudnorm as pyln
import librosa
from evaluation.automix_evaluation import compute_stft, amp_to_db, get_running_stats
import numpy as np  

def load_spectral_features(model_args, device, *args, **kwargs):

    fft_size= 4096
    hop_length= 1024
    sr=44100
    def wrapper_fn(x, normalization_dict=None):
        B, C, T = x.shape
        assert B==1 # Only support batch size of 1 for spectral features
        x=x[0].T  # Transpose to shape [T, C] for processing
        x= x.cpu().numpy()
        x = pyln.normalize.peak(x, -1.0)
        spec = compute_stft( x, hop_length, fft_size, np.sqrt(np.hanning(fft_size+1)[:-1]))
        spec = np.transpose(spec, axes=[1, -1, 0])
        spec = np.abs(spec)

        ct_mean=[]
        bw_mean=[]
        sc_mean=[]
        ro_mean=[]
        ft_mean=[]

        for c in range(C):
            s=spec[c]
            sc = librosa.feature.spectral_centroid(y=None, sr=sr, S=s,
                             n_fft=fft_size, hop_length=hop_length)

            bw= librosa.feature.spectral_bandwidth(y=None, sr=sr, S=s,
                             n_fft=fft_size, hop_length=hop_length, centroid=sc, norm=True, p=2)
        
            ct = librosa.feature.spectral_contrast(y=None, sr=sr, S=s,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   fmin=250.0, n_bands=4, quantile=0.02, linear=False)
            ro = librosa.feature.spectral_rolloff(y=None, sr=sr, S=s,
                                                  n_fft=fft_size, hop_length=hop_length, 
                                                  roll_percent=0.85)
        
            ft = librosa.feature.spectral_flatness(y=None, S=s,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   amin=1e-10, power=2.0)

            ft = amp_to_db(ft)
            ft= (-1 * ft) + 1.0
        
            eps = 1e-0
            N = 40
            mean_sc, std_sc = get_running_stats(sc.T+eps, [0], N=N)
            mean_bw, std_bw = get_running_stats(bw.T+eps, [0], N=N)
            mean_ct, std_ct = get_running_stats(ct.T, list(range(ct.shape[0])), N=N)
            mean_ro, std_ro = get_running_stats(ro.T+eps, [0], N=N)
            mean_ft, std_ft = get_running_stats(ft.T+eps, [0], N=N)

            ct_mean.append(mean_ct)
            bw_mean.append(mean_bw)
            sc_mean.append(mean_sc)
            ro_mean.append(mean_ro)
            ft_mean.append(mean_ft)
        
        ct_mean = np.mean(ct_mean)
        bw_mean = np.mean(bw_mean)
        sc_mean = np.mean(sc_mean)
        ro_mean = np.mean(ro_mean)
        ft_mean = np.mean(ft_mean)


        arrays_to_concat = []
        for arr in [ct_mean, bw_mean, sc_mean, ro_mean, ft_mean]:
            if np.ndim(arr) == 0:  # If it's a scalar
                arrays_to_concat.append(np.array([arr]))  # Convert to 1D array
            else:
                arrays_to_concat.append(arr)

        features = np.concatenate(arrays_to_concat, axis=0)
        #features = np.concatenate([ct_mean, bw_mean, sc_mean, ro_mean, ft_mean], axis=0)

        features = torch.tensor(features, device=device).unsqueeze(0)  # Add batch dimension


        if normalization_dict is not None:
            features = (features - normalization_dict['shift']) / normalization_dict['scale']

        return features
    
    return wrapper_fn

from evaluation.automix_evaluation import get_rms_dynamic_crest
def load_dynamic_features(model_args, device, *args, **kwargs):
    """
    Load the dynamic features extractor.
    
    Args:
        model_args: Arguments for the dynamic features extractor.
        device: Device to load the model on (CPU or GPU).
        
    Returns:
        a function that extracts dynamic features from audio.
    """

    fft_size = 4096
    hop_length = 1024
    sr = 44100

    def wrapper_fn(x ):
        # Assuming x is a tensor of shape [B, C, T]
        B, C, T = x.shape
        assert B == 1, "Only batch size of 1 is supported for dynamic features"
        
        x = x[0].T  # Transpose to shape [T, C] for processing
        x = x.cpu().numpy()  # Convert to numpy array

        x= pyln.normalize.peak(x, -1.0)  # Normalize to -1 dB peak

        rms, dyn, crest= get_rms_dynamic_crest(x,  fft_size, hop_length)

        rms=(-1* rms) + 1.0
        dyn=(-1* dyn) + 1.0

        mean_rms, std_rms = get_running_stats(rms.T, [0], N=40)
        mean_dyn, std_dyn = get_running_stats(dyn.T, [0], N=40)
        mean_crest, std_crest = get_running_stats(crest.T, [0], N=40)


        mean_rms = np.mean(mean_rms, axis=-1)  # Mean across time frames
        mean_dyn = np.mean(mean_dyn, axis=-1)  # Mean across time frames
        mean_crest = np.mean(mean_crest, axis=-1)  # Mean across time frames

        features = np.concatenate([mean_rms, mean_dyn,  mean_crest], axis=0)

        features = torch.tensor(features, device=device).unsqueeze(0)  # Add batch dimension


        return features
    
    return wrapper_fn

from evaluation.automix_evaluation import get_SPS, get_panning_rms

def load_panning_features(model_args, device, *args, **kwargs):
    """
    Load the panning features extractor.
    
    Args:
        model_args: Arguments for the panning features extractor.
        device: Device to load the model on (CPU or GPU).

    Returns:
        a function that extracts panning features from audio.
    """

    fft_size = 4096
    hop_length = 1024
    sr = 44100

    def wrapper_fn(x):
        # Assuming x is a tensor of shape [B, C, T]
        B, C, T = x.shape
        assert B == 1, "Only batch size of 1 is supported for panning features"
        
        x = x[0].T  # Transpose to shape [T, C] for processing
        x = x.cpu().numpy()  # Convert to numpy array

        x=pyln.normalize.peak(x, -1.0)  # Normalize to -1 dB peak

        freqs=[[0, sr//2], [0, 250], [250, 2500], [2500, sr//2]]  

        _, _, sps_frames, _ = get_SPS(x, n_fft=fft_size,
                                  hop_length=hop_length,
                                  smooth=True, frames=True)
    
    
        p_rms = get_panning_rms(sps_frames,
                    freqs=freqs,
                    sr=sr,
                    n_fft=fft_size)


        # to avoid num instability, deletes frames with zero rms from target
        if np.min(p_rms) == 0.0:
            id_zeros = np.where(p_rms.T[0] == 0)
            p_rms_out_ = []
            for i in range(len(freqs)):
                temp_out = np.delete(p_rms.T[i], id_zeros)
                p_rms_out_.append(temp_out)
            p_rms_out_ = np.asarray(p_rms)
            p_rms = p_rms_out_.T
        
        N = 40 

        mean, std = get_running_stats(p_rms, freqs, N=N)
        mean= np.mean(mean, axis=-1)  # Mean across time frames

        features= mean

        #features = np.concatenate(mean, axis=0)

        features = torch.tensor(features, device=device).unsqueeze(0)



        return features
    
    return wrapper_fn
    