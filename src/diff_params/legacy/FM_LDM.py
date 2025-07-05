import torch
import hydra
import einops
import numpy as np

import utils.training_utils as utils
from diff_params.shared import SDE

import torch.distributed as dist
import sys

class FM_LDM(SDE):
    """
        This implements the time-frequency domain diffusion
        Definition of the diffusion parameterization, following ( Karras et al., "Elucidating...", 2022). 
        This includes only the utilities needed for training, not for sampling.
    """

    def __init__(self,
        type,
        AE_type,
        sde_hp,
        cfg_dropout_prob,
        default_shape,
        apply_fxnormaug=False,
        fxnormaug_train=None,
        fxnormaug_inference=None,
        **kwargs
        ):

        super().__init__(type, sde_hp)

        self.sigma_data = self.sde_hp.sigma_data #depends on the training data!! precalculated variance of the dataset
        self.t_min = self.sde_hp.t_min
        self.t_max = self.sde_hp.t_max

        self.default_shape = torch.Size(default_shape)

        self.apply_fxnormaug = apply_fxnormaug
        self.fxnormaug_train = fxnormaug_train
        self.fxnormaug_inference = fxnormaug_inference

        try:
            self.max_t= self.sde_hp.max_sigma
        except Exception as e:
            print(e)
            print("max_sigma not defined, please add it. It should be the highest sigma value seen during training")


        try:
            rank=dist.get_rank()
            self.device = torch.device(f"cuda:{rank}")
        except:
            self.device = torch.device("cuda:0")

        self.cfg_dropout_prob = cfg_dropout_prob

        if AE_type=="SAO_VAE":
            from stable_audio_tools import get_pretrained_model
            VAE, VAE_model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
            self.AE=VAE.to(self.device)

            def encode_fn(x, mono=False):
                x = x.to(self.device)
                if mono:
                    x=torch.stack([x, x], dim=1)
                z=self.AE.pretransform.encode(x)
                z=einops.rearrange(z, "b t c -> b c t")
                return z
            
            def decode_fn(z, mono=False):
                z=einops.rearrange(z, "b c t -> b t c")
                x=self.AE.pretransform.decode(z)
                #invert fake stereo
                if mono:
                    x=x[:,0,:]
                return x

            self.AE_encode=encode_fn
            self.AE_encode_compiled=torch.compile(encode_fn)
            self.AE_decode=decode_fn

        elif AE_type=="MERT":
            MERT_args= kwargs.get("MERT_args", None)
            assert MERT_args is not None, "MERT_args must be provided for MERT AE"

            from evaluation.feature_extractors import load_MERT

            MERT_encoder= load_MERT(MERT_args, device=self.device)

            def encode_fn(x):
                x=x.to(self.device)
                z=MERT_encoder(x) #shape (B, C)

                #print("MERT z shape", z.shape)

                z=z.view(z.shape[0], 64, -1) #shape (B, 64, N)

                #print("MERT z shape after reshape", z.shape)
                z=z.permute(0, 2, 1) #shape (B, N, 64)

                return z.contiguous()
            
            self.AE_encode=encode_fn
            self.AE_encode_compiled=torch.compile(encode_fn)

            self.AE_decode=lambda x: x

        elif AE_type=="CLAP":
            CLAP_args= kwargs.get("CLAP_args", None)
            assert CLAP_args is not None, "CLAP_args must be provided for CLAP AE"

            # Save original path
            original_path = sys.path.copy()
            print("path", sys.path)
   
            from evaluation.feature_extractors import load_CLAP
            CLAP_encoder= load_CLAP(CLAP_args, device=self.device)

            sys.path = original_path
            print("path", sys.path)

            def encode_fn(x):
                x=x.to(self.device)
                z=CLAP_encoder(x) #shape (B, C)

                z=z.view(z.shape[0], 64, -1) #shape (B, 64, N)

                z=z.permute(0, 2, 1) #shape (B, N, 64)

                return z
            

            self.AE_encode=encode_fn
            self.AE_encode_compiled=torch.compile(encode_fn)

            self.AE_decode=lambda x: x

        elif AE_type=="Music2Latent4":
            from music2latent4 import Inferencer

            self.AE = Inferencer(device=self.device)
            #self.AE=self.AE.to(self.device)
            def latent2seq(latent):
                """
                Convert the latent representation to a sequence of latent vectors.
                """
                # Reshape the latent representation to match the expected input shape
                latent = latent.view(latent.size(0),-1, latent.size(-1))
                return latent

            def seq2latent(latent_sequence):
                """
                Convert the sequence of latent vectors back to the original latent representation.
                """
                # Reshape the latent sequence to match the expected output shape
                latent = latent_sequence.view(latent_sequence.size(0), -1,8, latent_sequence.size(-1))
                return latent


            def encode_fn(x, mono=False):
                assert x.ndim==3

                if not mono:
                    assert x.shape[1]==2
                else:
                    raise NotImplementedError("Mono not implemented yet")
                

                z = self.AE.encode(x)
                z=latent2seq(z)
                return z
            
            
            def decode_fn(z, mono=False):
                assert mono==False
                z=seq2latent(z)
                x = self.AE.decode(z)
                return x

            self.AE_encode=encode_fn
            self.AE_encode_compiled=torch.compile(encode_fn)

            self.AE_decode=decode_fn


        else:
            raise NotImplementedError("Only SAO VAE is implemented for now")


    def sample_time_training(self, N):
        """
        For training, getting t according to a similar criteria as sampling. Simpler and safer to what Karras et al. did
        Args:
            N (int): batch size
        """
        t = torch.rand(N)* (self.t_max - self.t_min) + self.t_min

        return t


    def sample_prior(self, shape=None, t=None, dtype=None):
        """
        Just sample some gaussian noise, nothing more
        Args:
            shape (tuple): shape of the noise to sample, something like (B,T)
        """
        assert shape is not None
        #print(shape)
        if t is not None:
            n = torch.randn(shape, dtype=dtype).to(t.device) * t
        else:
            n = torch.randn(shape, dtype=dtype)
        return n



    def cskip(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        
        """
        raise NotImplementedError("cskip is not implemented yet, please use cskip_eloi instead")

    def cout(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        raise NotImplementedError("cout is not implemented yet, please use cout_eloi instead")

    def cin(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        raise NotImplementedError("cin is not implemented yet, please use cin_eloi instead")

    def cnoise(self, t):
        """
        preconditioning of the noise embedding
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return t

    def lambda_w(self, sigma):
        """
        Score matching loss weighting
        """
        raise NotImplementedError("lambda_w is not implemented yet, please use lambda_w_eloi instead")
        #return (sigma*self.sigma_data)**(-2) * (self.sigma_data**2+sigma**2)
        
    def Tweedie2score(self, tweedie, xt, t, *args, **kwargs):
        return (tweedie - self._mean(xt, t)) / self._std(t)**2

    def score2Tweedie(self, score, xt, t, *args, **kwargs):
        return self._std(t)**2 * score + self._mean(xt, t)

    def _mean(self, x, t):
        return x
    
    def _std(self, t):
        return t
    
    def _ode_integrand(self, x, t, score):
        raise NotImplementedError("This is not implemented yet, please use _ode_integrand_eloi instead")
        return -t * score
    
    def add_white_noise(self, x, t, n=None):
        sigma=t
        if n is None:
            n=self.sample_prior(shape=x.shape, dtype=x.dtype).to(x.device)
        x_perturbed = x + sigma *n
        return x_perturbed

    def prepare_train_preconditioning(self, x, t,n=None, *args, **kwargs):
        raise NotImplementedError("This is not implemented yet, please use prepare_train_preconditioning_eloi instead")

    def get_null_embed(self, context):
        null_embed = torch.zeros_like(context, device=context.device)
        return null_embed

    def loss_fn(self, net, sample=None, context=None, *args, **kwargs):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        raise NotImplementedError("This is not implemented yet, please use loss_fn_eloi instead")


    def add_noise(self,x, is_test=False):
        SNR_mean= self.context_preproc.SNR_mean
        SNR_std= self.context_preproc.SNR_std   
        if is_test and not self.randomize_parameters_at_test:
            SNR_std=0.0
        SNR= torch.randn(x.shape[0], device=x.device) * SNR_std + SNR_mean
        x= utils.add_pink_noise(x, SNR)
        return x

    def preprocessor(self, x, is_test=False):
            if self.apply_fxnormaug:
                if is_test and not self.randomize_parameters_at_test:
                    x=self.fxnormaug_inference.forward(x)
                else:
                    x=self.fxnormaug_train.forward(x)
            return x
    
    def apply_RMS_randomization(self, x, is_test=False):
        if is_test and not self.randomize_parameters_at_test:
            self.RMS=torch.zeros(x.shape[0], device=x.device) + self.RMS_mean
        else:
            self.RMS=torch.randn(x.shape[0], device=x.device) * self.RMS_std + self.RMS_mean

        self.RMS=self.RMS.unsqueeze(-1).unsqueeze(-1)

        x_RMS=20*torch.log10(torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True).mean(dim=-2, keepdim=True)))

        gain= self.RMS - x_RMS
        gain_linear = 10 ** (gain / 20)
        #print("gain_linear", gain_linear.shape, gain_linear.device, x.shape, x.device)
        x=x* gain_linear.view(-1, 1, 1)

        return x
        
    def transform_forward(self, x, compile=False, is_condition=False, is_test=False):
        #TODO: Apply forward transform here
        #fake stereo

        if is_condition:
            x=self.preprocessor(x, is_test=is_test)
        if compile:
            with torch.no_grad():
                z=self.AE_encode_compiled(x)
        else:
            with torch.no_grad():
                z=self.AE_encode(x)

        z=einops.rearrange(z, "b t c -> b c t")

        return z, x
    
    def transform_inverse(self, z):

        z=einops.rearrange(z, "b c t -> b t c")
        x=self.AE_decode(z)

        return x
        
    def normalize(self, x):
        return x

    def denormalize(self, x):
        return x
 