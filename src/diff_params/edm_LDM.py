import torch
import einops
import numpy as np

import utils.training_utils as utils
from diff_params.shared import SDE

import torch.distributed as dist


class EDM_LDM(SDE):
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
        context_preproc=None,
        ):

        super().__init__(type, sde_hp)

        self.sigma_data = self.sde_hp.sigma_data #depends on the training data!! precalculated variance of the dataset
        self.sigma_min = self.sde_hp.sigma_min
        self.sigma_max = self.sde_hp.sigma_max
        self.rho = self.sde_hp.rho

        self.default_shape = torch.Size(default_shape)

        self.context_preproc = context_preproc

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
        
    def Tweedie2score(self, tweedie, xt, t, *args, **kwargs):
        return (tweedie - self._mean(xt, t)) / self._std(t)**2

    def score2Tweedie(self, score, xt, t, *args, **kwargs):
        return self._std(t)**2 * score + self._mean(xt, t)

    def _mean(self, x, t):
        return x
    
    def _std(self, t):
        return t
    
    def _ode_integrand(self, x, t, score):
        return -t * score
    
    def _corrector_(self, x, score, gamma, t):
        w=torch.randn_like(x)
        #annealed langevin dynamics
        step_size=0.5*(gamma*t)**2 

        return x + step_size*score + torch.sqrt(2 * step_size) * w


    def prepare_train_preconditioning(self, x, t,n=None, *args, **kwargs):

        mu, sigma = self._mean(x, t), self._std(t).unsqueeze(-1)
        sigma = sigma.view(*sigma.size(), *(1,)*(x.ndim - sigma.ndim))
        if n is None:
            n=self.sample_prior(shape=x.shape, dtype=x.dtype).to(x.device)

        #print("mu device", mu.device, "sigma device", sigma.device, "n device", n.device, "rank", dist.get_rank())

        x_perturbed = mu + sigma *n

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
            y=self.transform_forward(y, compile=True)
            print("y", y.std())

            if context is not None:


                context=self.transform_forward(context, compile=True, is_condition=True, is_test=False)

                if self.cfg_dropout_prob > 0.0:
                    #context=self.transform_forward(context)
                    null_embed = torch.zeros_like(context, device=context.device)
                    #dropout context with probability cfg_dropout_prob
                    mask = torch.rand(context.shape[0], device=context.device) < self.cfg_dropout_prob
                    context = torch.where(mask.view(-1,1,1), null_embed, context)
    

        #print("y shape", y.shape, "y stdev", y.std(), "context shape", context.shape, "t shape", t.shape)
        #x=self.transform_forward(context)
        #x=self.flatten(x)

        #print("y shape", y.shape, "t shape", t.shape, "context shape", context.shape)


        input, target, cnoise = self.prepare_train_preconditioning(y, t )


        if len(cnoise.shape)==1:
            cnoise=cnoise.unsqueeze(-1)
        if input.ndim==2:
            input=input.unsqueeze(1)

        estimate = net(input, cnoise, input_concat_cond=context)
        #estimate = net(input, cnoise, input_concat_cond=None)
        
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


        #x_in=self.unflatten(x_in)
        net_out=net(x_in, cnoise.to(torch.float32), input_concat_cond=cond, cfg_scale=cfg_scale)  #this will crash because of broadcasting problems, debug later!
        #net_out=self.flatten(net_out).to(xn.dtype)

        x_hat=cskip*xn + cout*net_out   

        #x_hat has CQT shape

        return x_hat


        
    def transform_forward(self, x, compile=False, is_condition=False, is_test=False):
        #TODO: Apply forward transform here
        #fake stereo

        if is_condition:
            if self.context_preproc is not None:
                    if self.context_preproc.add_noise:
                        if self.context_preproc.noise_type=="pink":
                            #add pink noise to context
                            print("Adding pink noise to context")
                            SNR_mean= self.context_preproc.SNR_mean
                            SNR_std= self.context_preproc.SNR_std   
                            if is_test:
                                SNR_std=0.0
                            SNR= torch.randn(x.shape[0], device=x.device) * SNR_std + SNR_mean
                            x= utils.add_pink_noise(x, SNR)
                        else:
                            raise NotImplementedError("Only pink noise is implemented for now")

        if compile:
            z=self.AE_encode_compiled(x)
        else:
            z=self.AE_encode(x)
        z=einops.rearrange(z, "b t c -> b c t")
        return z
    
    def transform_inverse(self, z):

        z=einops.rearrange(z, "b c t -> b t c")
        x=self.AE_decode(z)

        return x
        
    def normalize(self, x):
        return x

    def denormalize(self, x):
        return x
 