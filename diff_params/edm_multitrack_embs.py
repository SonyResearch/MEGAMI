import torch
import torch.nn.functional as F
import sys
import math
import time
import os
import einops
import numpy as np

from utils.multitrack_utils import multitrack_batched_processing
from utils.data_utils import apply_RMS_normalization

class EDM_Multitrack_Embeddings:
    """
        This implements the time-frequency domain diffusion
        Definition of the diffusion parameterization, following ( Karras et al., "Elucidating...", 2022). 
        This includes only the utilities needed for training, not for sampling.
    """

    def __init__(self,
        type,
        sde_hp,
        default_shape,
        cfg_dropout_prob=0.2,
        sample_rate=44100,
        context_signal="dry",
        content_encoder_type="CLAP",
        style_encoder_type="fx_encoder2048+AFv6",
        *args,
        **kwargs
        ):

        self.type = type
        self.sde_hp = sde_hp

        self.sigma_data = self.sde_hp.sigma_data  # depends on the training data!! precalculated variance of the dataset
        self.sigma_min = self.sde_hp.sigma_min
        self.sigma_max = self.sde_hp.sigma_max
        self.rho = self.sde_hp.rho

        self.default_shape = torch.Size(default_shape)

        try:
            self.max_t = self.sde_hp.max_sigma
        except Exception as e:
            print(e)
            print("max_sigma not defined, please add it. It should be the highest sigma value seen during training")

        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=device

        self.cfg_dropout_prob = cfg_dropout_prob

        self.context_signal=context_signal

        self.sample_rate=sample_rate

        self.prepare_content_encoder(content_encoder_type, sample_rate, *args, **kwargs)
        self.prepare_style_encoder(style_encoder_type, *args, **kwargs)

    def prepare_content_encoder(self, type, sample_rate, *args, **kwargs):

        if type=="CLAP":
            CLAP_args= kwargs.get("CLAP_args", None)
            assert CLAP_args is not None, "CLAP_args must be provided for CLAP AE"

            # Save original path
            from utils.feature_extractors.load_features import load_CLAP
            CLAP_encoder= load_CLAP(CLAP_args, device=self.device)

            def encode_fn(x, *args, **kwargs):
                x=x.to(self.device)

                type= kwargs.get("type", None)
                z=CLAP_encoder(x, type) #shape (B, C)

                z=z.view(z.shape[0], 64, -1) #shape (B, 64, N)

                z=z.permute(0, 2, 1) #shape (B, N, 64)

                return z
            
            self.content_encode_fn=encode_fn

        else:
            raise NotImplementedError(f"AE type {AE_type} not implemented")


    def prepare_style_encoder(self, type, *args, **kwargs):

        if type=="FxEncoder++_DynamicFeatures":

            Fxencoder_plusplus_args=kwargs.get("fx_encoder_plusplus_args", None)

            from utils.feature_extractors.load_features import load_fx_encoder_plusplus_2048
            feat_extractor = load_fx_encoder_plusplus_2048(Fxencoder_plusplus_args, self.device)

            from utils.feature_extractors.AF_features_embedding import AF_fourier_embedding
            AFembedding= AF_fourier_embedding(device=self.device)

            def fxencode_fn(x):
                """
                x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
                """
                z=feat_extractor(x)
                z=torch.nn.functional.normalize(z, dim=-1, p=2)
                z=z*math.sqrt(z.shape[-1])  # rescale to keep the same scale

                z_af,_=AFembedding.encode(x)
                z_af=z_af* math.sqrt(z_af.shape[-1])  # rescale to keep the same scale


                z_all= torch.cat([z, z_af], dim=-1)

                #now L2 normalize

                norm_z= z_all/ math.sqrt(z_all.shape[-1])  # normalize by dividing by sqrt(dim) to keep the same scale

                norm_z=norm_z.view(norm_z.shape[0], 64, -1)  # Reshape to [B, 64, L//64] where L//64 is the number of frames

                return norm_z

            def reshape_fn(embed):
                """
                embed: tensor of shape [B, 64, L//64] where B is the batch size
                """
                embed=embed.view(embed.shape[0], -1)

                return embed


            self.style_encode_fn=fxencode_fn
            self.style_reshape=reshape_fn       

        else:
            raise NotImplementedError(f"FX encoder type {type} not implemented")

    def sample_time_training(self, N):
        """
        For training, getting t according to a similar criteria as sampling. Simpler and safer to what Karras et al. did
        Args:
            N (int): batch size
        """
        a = torch.rand(N)
        t = (self.sigma_max**(1/self.rho) +a *(self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho)))**self.rho

        return t

    def sample_prior(self, shape=None, t=None, dtype=None):
        """
        Just sample some gaussian noise, nothing more
        Args:
            shape (tuple): shape of the noise to sample, something like (B,T)
        """
        assert shape is not None
        if t is not None:
            n = torch.randn(shape).to(t.device) * t
        else:
            n = torch.randn(shape)
        return n

    def cskip(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        
        """
        return self.sigma_data**2 *(sigma**2+self.sigma_data**2)**-1

    def cout(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return sigma*self.sigma_data* (self.sigma_data**2+sigma**2)**(-0.5)

    def cin(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (self.sigma_data**2+sigma**2)**(-0.5)

    def cnoise(self, sigma):
        """
        preconditioning of the noise embedding
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (1/4)*torch.log(sigma)

    def lambda_w(self, sigma):
        """
        Score matching loss weighting
        """
        return (sigma*self.sigma_data)**(-2) * (self.sigma_data**2+sigma**2)

    def prepare_train_preconditioning(self, x, t, n=None, *args, **kwargs):

        mu, sigma = self._mean(x, t), self._std(t).unsqueeze(-1)
        sigma = sigma.view(*sigma.size(), *(1,)*(x.ndim - sigma.ndim))
        if n is None:
            n=self.sample_prior(shape=x.shape).to(x.device)
        x_perturbed = mu + sigma *n
        #self.sample_prior(x.shape).to(x.device)

        cskip = self.cskip(sigma)
        cout = self.cout(sigma)
        cin = self.cin(sigma)
        cnoise = self.cnoise(sigma.squeeze())

        #check if cnoise is a scalar, if so, repeat it
        if len(cnoise.shape) == 0:
            cnoise = cnoise.repeat(x.shape[0],)
        else:
            cnoise = cnoise.view(x.shape[0],)

        target = 1/cout * (x - cskip * x_perturbed)

        return cin * x_perturbed, target, cnoise

    def loss_fn(self, net,  sample=None,  sample_aug=None, context=None, clusters=None, taxonomy=None, masks=None, *args, **kwargs):
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
            if sample_aug is not None:
                context = sample_aug
            else:
                context = y.clone()  # use the wet signal as context

        else:
            assert context is not None, "Context must be provided if context_signal is not 'wet'"
        
        a=time.time

        with torch.no_grad():
            
            y_style=self.style_encode(y,  masks=masks, taxonomy=taxonomy)

            if context is not None:
                z, x=self.transform_forward(context,  is_condition=True, clusters=clusters, masks=masks, taxonomy=taxonomy, is_wet=(self.context_signal == "wet"))
                if self.cfg_dropout_prob > 0.0:
                    null_embed = torch.zeros_like(z, device=z.device)
                    #dropout context with probability cfg_dropout_prob
                    mask = torch.rand(z.shape[0], device=z.device) < self.cfg_dropout_prob
                    z = torch.where(mask.view(-1,1,1,1), null_embed, z)


            input, target, cnoise = self.prepare_train_preconditioning(y_style, t )


            if len(cnoise.shape)==1:
                cnoise=cnoise.unsqueeze(-1)
            if input.ndim==2:
                input=input.unsqueeze(1)

        estimate = net(input, cnoise, cross_attn_cond=z, taxonomy=taxonomy, mask=masks, cross_attn_cond_mask=masks) 
        
        if target.ndim==2 and estimate.ndim==3:
            estimate=estimate.squeeze(1)

        error=torch.square(torch.abs(estimate-target))

        # do not propagate the error of the padded tracks
        error= error*masks.view(masks.shape[0], masks.shape[1], 1, 1) 

        compensating_scalar= torch.numel(masks)/ torch.sum(masks, dim=(0,1), keepdim=False).clamp(min=1.0)
        error= error * compensating_scalar.view(-1, 1, 1, 1)

        return error, self._std(t), x, y


    def get_null_embed(self, context):
        null_embed = torch.zeros_like(context, device=context.device)
        return null_embed


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

        if cfg_scale == 1.0:
            net_out=net(x_in, cnoise.to(torch.float32), cross_attn_cond=cond, mask=masks, taxonomy=taxonomy,  cross_attn_cond_mask=masks)  #this will crash because of broadcasting problems, debug later!
        else:
            null_embed = self.get_null_embed(cond)

            inputs_cond= torch.cat([cond, null_embed], dim=0)

            x_in_cat= torch.cat([x_in, x_in], dim=0)

            cnoise= torch.cat([cnoise, cnoise], dim=0)

            masks_in= torch.cat([masks, masks], dim=0) if masks is not None else None


            net_out_batch=net(x_in_cat, cnoise.to(torch.float32), cross_attn_cond=inputs_cond , mask=masks_in,  cross_attn_cond_mask=masks_in)  #this will crash because of broadcasting problems, debug later!0

            cond_output, uncond_output = torch.chunk(net_out_batch, 2, dim=0)

            net_out = uncond_output + (cond_output - uncond_output) * cfg_scale


        x_hat=cskip*xn + cout*net_out   
        x_hat=x_hat* masks.view(masks.shape[0], masks.shape[1], 1, 1) 

        return x_hat

    def style_encode(self, x,  masks=None, taxonomy=None, use_adaptor=False):
        """
        Encode the input audio using the style encoder
        Args:
            x (Tensor): shape: (B,N, C, T) Audio to encode
            masks (Tensor): shape: (B, N) Mask indicating which tracks are present in the batch
        """
        def apply_fxenc(x_masked, taxonommy=None):
            x_emb=self.style_encode_fn(x_masked)

            return x_emb

        assert masks is not None, "masks must be provided for style encoding"

        output_emb=multitrack_batched_processing( x, taxonomy=taxonomy ,function=apply_fxenc, class_dependent=False, masks=masks)

        return output_emb

    def _mean(self, x, t):
        return x
    
    def _std(self, t):
        return t
    
    def _ode_integrand(self, x, t, score):
        return -t * score
    
    def transform_inverse(self, z):
        #shape is (B, N, C, T)
        B, N, C, T = z.shape
        # Reshape z to (B*N, C, T)
        z_reshaped = einops.rearrange(z, "b n c t -> (b n) c t")
        z_reshaped= self.style_reshape(z_reshaped)

        # Reshape back to (B, N, C)

        z_out = einops.rearrange(z_reshaped, "(b n) c -> b n c", b=B, n=N)

        return z_out
        
    def preprocessor(self, x, is_test=False, taxonomy=None):
            """
                x: tensor of shape (BxN, C, T) where B is the batch size, N is the number of tracks, C is the number of channels and T is the number of time steps
                taxonomy: list of lists of strings, where each string is the taxonomy of the track with length BxN. It may be useful if we want to apply different augmentations depending on the taxonomy of the track.
            """
            if x.shape[1] == 2:
                        x = torch.mean(x, dim=1, keepdim=True).expand(-1, 2, -1)  # convert to stereo if it is mono
            elif x.shape[1] == 1:  # if context is mono, we expand it to stereo
                        x = x.expand(-1, 2, -1)
                    
            if not is_test:
                #random flip
                if np.random.rand() > 0.5:
                    x = -x

            #rms normalize context to -25 dB
            x= apply_RMS_normalization(x, -25, device=self.device)

            return x

    def Tweedie2score(self, tweedie, xt, t, *args, **kwargs):
        return (tweedie - self._mean(xt, t)) / self._std(t)**2

    def score2Tweedie(self, score, xt, t, *args, **kwargs):
        return self._std(t)**2 * score + self._mean(xt, t)

    def transform_forward(self, x, y=None,  is_condition=False, is_test=False, clusters=None, masks=None, taxonomy=None, is_wet=False):

        assert masks is not None

        #convert y to mono and rms normalize itdd

        def prerprocess_and_encode(x_masked, taxonomy=None):
            if is_condition:

                x_masked=self.preprocessor(x_masked, is_test=is_test, taxonomy=taxonomy)
    
            with torch.no_grad():
                x_emb=self.content_encode_fn(x_masked, type="wet" if is_wet else "dry")

            return x_emb, x_masked

        z, x_out=multitrack_batched_processing(
            x, taxonomy=taxonomy, function=prerprocess_and_encode, class_dependent=False, masks=masks, number_outputs=2
        )
        return z, x_out
