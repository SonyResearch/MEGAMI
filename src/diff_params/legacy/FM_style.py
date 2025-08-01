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
from diff_params.FM_LDM import FM_LDM




class FM_Style(FM_LDM):
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

        print("Initializing EDM_Style with args:", args, kwargs)
        super().__init__(*args, **kwargs) 

        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
        elif FXenc_args.type=="fx_encoder_++":
            raise NotImplementedError("fx_encoder_++ is not implemented yet")
        else:
            raise NotImplementedError("Only AFxRep is implemented for now")

    def prepare_train_preconditioning(self, x, t, n=None, *args, **kwargs):

        with torch.no_grad():
            if n is None:
                z = self.sample_prior(shape=x.shape).to(x.device)
            else:
                z = n
    
            #linear interpolation
            #print(t.shape, x.shape, z.shape)
    
            t=t.view(t.shape[0],1,1) if x.ndim==3 else t.view(t.shape[0],1)
            #x=x/self.sigma_data
            # Eloi implementation
            x_t= t*z+(1.-t)*x
            #CFM objective
            target= z-x

        return x_t, target, t

    def _mean(self, x, t):
        return x

    def _std(self, t):
        return t

    def _ode_integrand(self, x, t, v):
        return  v

    def Tweedie2v(self, tweedie, xt, t, *args, **kwargs):
        #raise NotImplementedError
        return (tweedie - self._mean(xt, t)) / self._std(t)**2
        #return (tweedie - self._mean(xt, t)) / self._std(t)**2

    def v2Tweedie(self, vt, xt, t, *args, **kwargs):
        if xt.dim() == 3:
            x0= xt.detach().clone()- vt *t.view(-1,1,1)
        elif xt.dim() == 2:
            x0= xt.detach().clone()- vt *t.view(-1,1)
        else:
            raise ValueError("Invalid dimension of xt: "+str(xt.dim()))
        return x0

    def loss_fn(self, net,  sample=None, context=None, *args, **kwargs):
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
            y_style=self.style_encode(y, compile=True)

            if context is not None:
                z, x=self.transform_forward(context, compile=True, is_condition=True)
                if self.cfg_dropout_prob > 0.0:
                    #context=self.transform_forward(context)
                    null_embed = torch.zeros_like(z, device=z.device)
                    #dropout context with probability cfg_dropout_prob
                    mask = torch.rand(z.shape[0], device=z.device) < self.cfg_dropout_prob
                    z = torch.where(mask.view(-1,1,1), null_embed, z)
    

        input, target, cnoise = self.prepare_train_preconditioning(y_style, t )


        if len(cnoise.shape)==1:
            cnoise=cnoise.unsqueeze(-1)
        elif cnoise.ndim==3:
            cnoise=cnoise.squeeze(-1)

        if input.ndim==2:
            input=input.unsqueeze(1)

        z=einops.rearrange(z, "b c t -> b t c") if z is not None else None

        print("input shape", input.shape, "cnoise shape", cnoise.shape, "z shape", z.shape)
        estimate = net(input, cnoise, cross_attn_cond=z)
        
        if target.ndim==2 and estimate.ndim==3:
            estimate=estimate.squeeze(1)

        error=torch.square(torch.abs(estimate-target))


        return error, self._std(t), x, y

    def model_call(self, x, net, t, cond=None,  cfg_scale=1.0, *args, **kwargs):

        if len(t.shape)==0:
            t=t.unsqueeze(-1).unsqueeze(-1)
        elif len(t.shape)==1:
            t=t.unsqueeze(-1)
        elif len(t.shape)==2:
            pass
        else:
            raise ValueError("Invalid shape of t: "+str(t.shape))

        cond=einops.rearrange(cond, "b c t -> b t c") if cond is not None else None

        cnoise=t
        if cnoise.shape[0] != x.shape[0]:
            cnoise= cnoise.repeat(x.shape[0], 1)

        #v=net(x.unsqueeze(1), t).squeeze(1)
        #print("x shape", x.shape, "cnoise shape", cnoise.shape, "cond shape", cond.shape)
        if cfg_scale == 1.0:
            v=net(x, cnoise.to(torch.float32), cross_attn_cond=cond)  #this will crash because of broadcasting problems, debug later!
        else:
            null_embed = self.get_null_embed(cond)

            inputs_cond= torch.cat([cond, null_embed], dim=0)

            x_in_cat= torch.cat([x, x], dim=0)

            cnoise= torch.cat([cnoise, cnoise], dim=0)

            net_out_batch=net(x_in_cat, cnoise.to(torch.float32), cross_attn_cond=inputs_cond )

            cond_output, uncond_output = torch.chunk(net_out_batch, 2, dim=0)

            v = uncond_output + (cond_output - uncond_output) * cfg_scale
        
        return v


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
        
        x=x/self.sigma_data
        #print("x shape after FXenc", x.shape)

        #assert x.ndim==2

        #x=x.view(x.shape[0], x.shape[1], 1)

        return x

        
    def transform_inverse(self, z):

        z=z*self.sigma_data
        return self.FXenc_reshape(z)

        
    def normalize(self, x):
        return x

    def denormalize(self, x):
        return x
 