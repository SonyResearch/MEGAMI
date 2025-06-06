import torch
import torchaudio
from importlib import import_module
import yaml
import os
import einops
import numpy as np

import utils.training_utils as utils
from diff_params.shared import SDE

import torch.distributed as dist
from diff_params.edm_LDM import EDM_LDM


class EDM_Style(EDM_LDM):
    """
        This implements the time-frequency domain diffusion
        Definition of the diffusion parameterization, following ( Karras et al., "Elucidating...", 2022). 
        This includes only the utilities needed for training, not for sampling.
    """

    def __init__(self,
        FXenc_args,
        sample_rate,
        *args,
        **kwargs
        ):

        super().__init__(*args, **kwargs) 

        if FXenc_args.type=="AFxRep":

            ckpt_path=FXenc_args.ckpt_path
            config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
        
            with open(config_path) as f:
                config = yaml.safe_load(f)
        
            encoder_configs = config["model"]["init_args"]["encoder"]
        
            module_path, class_name = encoder_configs["class_path"].rsplit(".", 1)
            module_path = module_path.replace("lcap", "utils.st_ito")

            print("module path", module_path, "class name", class_name)
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
        
            model.to(self.device)


            def fxencode_fn(x):
                if sample_rate != 48000:
                    x=torchaudio.functional.resample(x, sample_rate, 48000)

                bs= x.shape[0]
                #peak normalization. I do it because this is what ST-ITO get_param_embeds does. Not sure if it is good that this representation is invariant to gain
                for batch_idx in range(bs):
                    x[batch_idx, ...] /= x[batch_idx, ...].abs().max().clamp(1e-8)

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

                embed=torch.cat([mid_embeddings, side_embeddings], dim=-1)

                embed=embed.view(embed.shape[0], 64, -1)

                return embed
            
            def reshape_fn(embed):

                embed=embed.view(embed.shape[0], -1)

                #embed_mid, embed_side = torch.chunk(embed, 2, dim=1)

                return embed

            
            self.FXenc=fxencode_fn
            self.FXenc_compiled=torch.compile(fxencode_fn)
            self.FXenc_reshape=reshape_fn

        else:
            raise NotImplementedError("Only AFxRep is implemented for now")



    def loss_fn(self, net, sample=None, context=None, *args, **kwargs):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """

        y=sample

        t = self.sample_time_training(y.shape[0]).to(y.device)



        with torch.no_grad():
            y=self.style_encode(y, compile=True)

            if context is not None:
                context=self.transform_forward(context, compile=True, is_condition=True)
                if self.cfg_dropout_prob > 0.0:
                    #context=self.transform_forward(context)
                    null_embed = torch.zeros_like(context, device=context.device)
                    #dropout context with probability cfg_dropout_prob
                    mask = torch.rand(context.shape[0], device=context.device) < self.cfg_dropout_prob
                    context = torch.where(mask.view(-1,1,1), null_embed, context)
    

        input, target, cnoise = self.prepare_train_preconditioning(y, t )


        if len(cnoise.shape)==1:
            cnoise=cnoise.unsqueeze(-1)
        if input.ndim==2:
            input=input.unsqueeze(1)

        context=einops.rearrange(context, "b c t -> b t c") if context is not None else None

        estimate = net(input, cnoise, cross_attn_cond=context)
        
        if target.ndim==2 and estimate.ndim==3:
            estimate=estimate.squeeze(1)

        error=torch.square(torch.abs(estimate-target))


        return error, self._std(t)



    def denoiser(self, xn , net, t, cond=None,cfg_scale=1.0, **kwargs):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (CQT shape?) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        sigma = self._std(t).unsqueeze(-1)
        sigma = sigma.view(*sigma.size(), *(1,)*(xn.ndim - sigma.ndim))

        cskip = self.cskip(sigma)
        cout = self.cout(sigma)
        cin = self.cin(sigma)
        cnoise = self.cnoise(sigma.squeeze())

        #check if cnoise is a scalar, if so, repeat it
        if len(cnoise.shape) == 0:
            cnoise = cnoise.repeat(xn.shape[0],).unsqueeze(-1)
        else:
            cnoise = cnoise.view(xn.shape[0],).unsqueeze(-1)


        #print(xn.shape, cskip.shape, cout.shape, cin.shape, cnoise.shape)
        x_in=cin*xn

        cond=einops.rearrange(cond, "b c t -> b t c") if cond is not None else None

        if cfg_scale != 1.0:
            net_out=net(x_in, cnoise.to(torch.float32), cross_attn_cond=cond)  #this will crash because of broadcasting problems, debug later!
        else:
            null_embed = self.get_null_embed(cond)

            inputs_cond= torch.cat([cond, null_embed], dim=0)

            x_in_cat= torch.cat([x_in, x_in], dim=0)

            cnoise= torch.cat([cnoise, cnoise], dim=0)

            net_out_batch=net(x_in_cat, cnoise.to(torch.float32), cross_attn_cond=inputs_cond )

            cond_output, uncond_output = torch.chunk(net_out_batch, 2, dim=0)

            net_out = uncond_output + (cond_output - uncond_output) * cfg_scale

        x_hat=cskip*xn + cout*net_out   

        return x_hat

    def style_encode(self, x, compile=False):
        """
        Encode the input audio using the style encoder
        Args:
            x (Tensor): shape: (B,T) Audio to encode
        """

        if compile:
            x=self.FXenc_compiled(x)
        else:
            x=self.FXenc(x)
        
        print("x shape after FXenc", x.shape)

        #assert x.ndim==2

        #x=x.view(x.shape[0], x.shape[1], 1)

        return x

        
    def transform_inverse(self, z):

        return self.FXenc_reshape(z)

        
    def normalize(self, x):
        return x

    def denormalize(self, x):
        return x
 