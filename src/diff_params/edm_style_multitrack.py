import torch
import time
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

from fx_model.apply_effects_multitrack_utils import multitrack_batched_processing



class EDM_Style_Multitrack(EDM_LDM):
    """
        This implements the time-frequency domain diffusion
        Definition of the diffusion parameterization, following ( Karras et al., "Elucidating...", 2022). 
        This includes only the utilities needed for training, not for sampling.
    """

    def __init__(self,
        FXenc_args,
        sample_rate,
        context_signal="dry",
        *args,
        **kwargs
        ):

        print("Initializing EDM_Style with args:", args, kwargs)
        super().__init__(*args, **kwargs) 

        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=device

        self.context_signal=context_signal

        if FXenc_args.type=="AFxRep":

            AFxRep_args= kwargs.get("AFxRep_args", None)
            from evaluation.feature_extractors import load_AFxRep
            AFxRep_encoder= load_AFxRep(AFxRep_args, device=self.device)
        

            def fxencode_fn(x):

                z=AFxRep_encoder(x)

                z=z.view(z.shape[0], 64, -1)

                return z

            
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


    def loss_fn(self, net,  sample=None, context=None, clusters=None, taxonomy=None, masks=None, compile=False,*args, **kwargs):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """

        start=time.time()
        y=sample

        t = self.sample_time_training(y.shape[0]).to(y.device)

        if self.context_signal == "wet":
            context = y
        else:
            assert context is not None, "Context must be provided if context_signal is not 'wet'"

        with torch.no_grad():
            
            y_style=self.style_encode(y, compile=compile, masks=masks, taxonomy=taxonomy)

            #print("Style encoding took", time.time()-start, "seconds")


            if context is not None:
                z, x=self.transform_forward(context, compile=compile, is_condition=True, clusters=clusters, masks=masks, taxonomy=taxonomy, is_wet=(self.context_signal == "wet"))
                #print("Transform forward took", time.time()-start, "seconds")
                if self.cfg_dropout_prob > 0.0:
                    #context=self.transform_forward(context)
                    null_embed = torch.zeros_like(z, device=z.device)
                    #dropout context with probability cfg_dropout_prob
                    mask = torch.rand(z.shape[0], device=z.device) < self.cfg_dropout_prob
                    z = torch.where(mask.view(-1,1,1,1), null_embed, z)

        input, target, cnoise = self.prepare_train_preconditioning(y_style, t )


        if len(cnoise.shape)==1:
            cnoise=cnoise.unsqueeze(-1)
        if input.ndim==2:
            input=input.unsqueeze(1)

        #print("input shape", input.shape, "cnoise shape", cnoise.shape, "z shape", z.shape)
        estimate = net(input, cnoise, cross_attn_cond=z, taxonomy=taxonomy, mask=masks, cross_attn_cond_mask=masks) 
        #print("net forward took", time.time()-start, "seconds")
        
        if target.ndim==2 and estimate.ndim==3:
            estimate=estimate.squeeze(1)

        error=torch.square(torch.abs(estimate-target))

        # do not propagate the error of the padded tracks
        error= error*masks.view(masks.shape[0], masks.shape[1], 1, 1) 

        # compensate for the facct that we will later reduce to the mean, but in practice we are discarding some tracks because of the masks
        # the compensating factor should be the total number of elements in the masks, divided by the number of elements in the masks that are not zero
        compensating_scalar= torch.numel(masks)/ torch.sum(masks, dim=(0,1), keepdim=False).clamp(min=1.0)
        error= error * compensating_scalar.view(-1, 1, 1, 1)

        return error, self._std(t), x, y



    def denoiser(self, xn , net, t, cond=None,cfg_scale=1.0, masks=None, taxonomy=None, **kwargs):
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


        x_in=cin*xn

        #cond=einops.rearrange(cond, "b n c t -> b n t c") if cond is not None else None

        #print("x_in shape", x_in.shape, "cond", cond.shape)
        if cfg_scale == 1.0:
            net_out=net(x_in, cnoise.to(torch.float32), cross_attn_cond=cond, mask=masks, taxonomy=taxonomy,  cross_attn_cond_mask=masks)  #this will crash because of broadcasting problems, debug later!
        else:
            null_embed = self.get_null_embed(cond)

            inputs_cond= torch.cat([cond, null_embed], dim=0)

            x_in_cat= torch.cat([x_in, x_in], dim=0)

            cnoise= torch.cat([cnoise, cnoise], dim=0)

            masks_in= torch.cat([masks, masks], dim=0) if masks is not None else None

            #taxonomy is a list of lists, we need to double the list to match the batch size
            taxonomy_in=taxonomy + taxonomy

            net_out_batch=net(x_in_cat, cnoise.to(torch.float32), cross_attn_cond=inputs_cond , mask=masks_in, taxonomy=taxonomy_in,  cross_attn_cond_mask=masks_in)  #this will crash because of broadcasting problems, debug later!0

            cond_output, uncond_output = torch.chunk(net_out_batch, 2, dim=0)

            net_out = uncond_output + (cond_output - uncond_output) * cfg_scale

        x_hat=cskip*xn + cout*net_out   

        x_hat=x_hat* masks.view(masks.shape[0], masks.shape[1], 1, 1) 



        return x_hat

    def style_encode(self, x, compile=False, masks=None, taxonomy=None):
        """
        Encode the input audio using the style encoder
        Args:
            x (Tensor): shape: (B,N, C, T) Audio to encode
            masks (Tensor): shape: (B, N) Mask indicating which tracks are present in the batch
        """
        def apply_fxenc(x_masked, taxonommy=None):
            if compile:
                x_emb=self.FXenc_compiled(x_masked)
            else:
                x_emb=self.FXenc(x_masked)

            return x_emb

        assert taxonomy is not None, "taxonomy must be provided for style encoding"
        assert masks is not None, "masks must be provided for style encoding"

        output_emb=multitrack_batched_processing( x, taxonomy=taxonomy ,function=apply_fxenc, class_dependent=False, masks=masks)

        #assert masks is not None

        #original_shape=x.shape

        #x=einops.rearrange(x, "b n c t -> (b n) c t")  #flatten the batch and number of tracks
        
        #mask_reshaped=einops.rearrange(masks, "b n -> (b n)") if masks is not None else None

        #x_masked=x[mask_reshaped]


        #if compile:
        #    x_emb=self.FXenc_compiled(x_masked)
        #else:
        #    x_emb=self.FXenc(x_masked)

        #output_shape = (original_shape[0], original_shape[1], x_emb.shape[-2],x_emb.shape[-1])
        #output_emb = torch.zeros(output_shape, dtype=x_emb.dtype, device=x_emb.device)

        # Create a counter to keep track of where we are in x_emb
        #emb_idx = 0

        #for b in range(original_shape[0]):
        #    for n in range(original_shape[1]):
        #        if masks[b, n]:
        #            output_emb[b, n] = x_emb[emb_idx]
        #            emb_idx += 1

        return output_emb

        
    def transform_inverse(self, z):
        #shape is (B, N, C, T)
        B, N, C, T = z.shape
        # Reshape z to (B*N, C, T)
        z_reshaped = einops.rearrange(z, "b n c t -> (b n) c t")
        z_reshaped= self.FXenc_reshape(z_reshaped)

        # Reshape back to (B, N, C)

        z_out = einops.rearrange(z_reshaped, "(b n) c -> b n c", b=B, n=N)

        return z_out

        
    def normalize(self, x):
        return x

    def denormalize(self, x):
        return x
 
    def preprocessor(self, x, is_test=False, taxonomy=None):
            """
                x: tensor of shape (BxN, C, T) where B is the batch size, N is the number of tracks, C is the number of channels and T is the number of time steps
                taxonomy: list of lists of strings, where each string is the taxonomy of the track with length BxN. It may be useful if we want to apply different augmentations depending on the taxonomy of the track.
            """
            if self.apply_fxnormaug:
                if is_test:
                    x=self.fxnormaug_inference.forward(x, taxonomy=taxonomy)
                else:
                    x=self.fxnormaug_train.forward(x, taxonomy=taxonomy)
            return x

    def transform_forward(self, x, y=None, compile=False, is_condition=False, is_test=False, clusters=None, masks=None, taxonomy=None, is_wet=False):
        #TODO: Apply forward transform here
        #fake stereo
        if self.AE_type=="oracle":
            raise NotImplementedError("Oracle AE is not implemented for EDM_Style_Multitrack")
      

        assert masks is not None


        def prerprocess_and_encode(x_masked, taxonomy=None):
            if is_condition:
                x_masked=self.preprocessor(x_masked, is_test=is_test, taxonomy=taxonomy)
    
            if compile:
                with torch.no_grad():
                    x_emb=self.AE_encode_compiled(x_masked, type="wet" if is_wet else "dry")
            else:
                with torch.no_grad():
                    x_emb=self.AE_encode(x_masked, type="wet" if is_wet else "dry")

            return x_emb, x_masked

        z, x_out=multitrack_batched_processing(
            x, taxonomy=taxonomy, function=prerprocess_and_encode, class_dependent=False, masks=masks, number_outputs=2
        )
        #original_shape=x.shape

        #x_reshaped=einops.rearrange(x, "b n c t -> (b n) c t")  #flatten the batch and number of tracks

        #mask_reshaped=einops.rearrange(masks, "b n -> (b n)") if masks is not None else None

        #reshape the list "taxonomy" in the same way as masks
        #taxonomy_reshaped=[]
        #if taxonomy is not None:
        #    for b in range(original_shape[0]):
        #        for n in range(original_shape[1]):
        #            if masks[b, n]:
        #                taxonomy_reshaped.append(taxonomy[b][n])

        #x_masked=x_reshaped[mask_reshaped]

        #if is_condition:
        #    x_masked=self.preprocessor(x_masked, is_test=is_test, taxonomy=taxonomy_reshaped)

        #if compile:
        #    with torch.no_grad():
        #        x_emb=self.AE_encode_compiled(x_masked)
        #else:
        #    with torch.no_grad():
        #        x_emb=self.AE_encode(x_masked)


        #output_shape = (original_shape[0], original_shape[1], x_emb.shape[-2],x_emb.shape[-1])
        #z = torch.zeros(output_shape, dtype=x_emb.dtype, device=x_emb.device)
        #x_out=torch.zeros_like(x, dtype=x_emb.dtype, device=x_emb.device)

        # Create a counter to keep track of where we are in x_emb
        #emb_idx = 0


        #for b in range(original_shape[0]):
        #    for n in range(original_shape[1]):
        #        if masks[b, n]:
        #            z[b, n] = x_emb[emb_idx]
        #            x_out[b, n] = x_masked[emb_idx]
        #            emb_idx += 1


        return z, x_out