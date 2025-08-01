import os
import yaml
import torch
import sys
sys.path.append('.')
from networks import Effects_Encoder


def load_effects_encoder(ckpt_path, device='cuda'):
    # load network configurations
    with open(os.path.join('.', 'networks', 'configs.yaml'), 'r') as f:
        configs = yaml.full_load(f)
    cfg_enc = configs['Effects_Encoder']['default']

    effects_encoder = Effects_Encoder(cfg_enc)
    reload_weights(effects_encoder, ckpt_path, device)
    effects_encoder.to(device)
    effects_encoder.eval()

    return effects_encoder


def reload_weights(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
