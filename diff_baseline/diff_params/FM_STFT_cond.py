
import torch
import einops
import numpy as np
import omegaconf

import utils.training_utils as utils
from diff_params.shared import SDE
import torch.distributed as dist

#from utils.cqt_nsgt_pytorch import CQT_nsgt

def get_window(window_type, window_length):
	if window_type == 'sqrthann':
		return torch.sqrt(torch.hann_window(window_length, periodic=True))
	elif window_type == 'hann':
		return torch.hann_window(window_length, periodic=True)
	else:
		raise NotImplementedError(f"Window type {window_type} not implemented!")

class FM_STFT(SDE):
    """
        This implements the time-frequency domain diffusion
        Definition of the diffusion parameterization, following ( Karras et al., "Elucidating...", 2022). 
        This includes only the utilities needed for training, not for sampling.
    """

    def __init__(self,
        type,
        sde_hp,
        stft,
        cfg_dropout_prob, 
        default_shape,
        sigma_T
        ):

        super().__init__(type, sde_hp)

        self.default_shape = torch.Size(default_shape)

        self.sde_hp = sde_hp
        self.sigma_max= sde_hp.sigma_max
        self.sigma_min= sde_hp.sigma_min

        self.sigma_T=sigma_T
        self.sigma_data = self.sde_hp.sigma_data  # depends on the training 


        self.cfg_dropout_prob = cfg_dropout_prob

        try:
            rank=dist.get_rank()
            self.device = torch.device(f"cuda:{rank}")
        except:
            self.device = torch.device("cuda:0")

        self.stft_kwargs = omegaconf.OmegaConf.create({
            "n_fft": stft.n_fft,
            "hop_length": stft.hop_length,
            "onesided": stft.onesided,
            "center": stft.center
        })

        #self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        window = get_window("hann", stft.n_fft)


        self.N_pad_spec = stft.N_pad_spec

        def stft_func( sig, window=None):
            #sig shape is (B, N, C, T) where B is batch size, N is number of samples, C is number of channels and T is the time dimension

            B, N, C, T = sig.shape

            window = window.to(sig.device)
            C=sig.shape[1]
            sig=einops.rearrange(sig, "b n c t -> (b n c) t")   
            spec= torch.stft(sig, **{**self.stft_kwargs, "window": window}, return_complex=True)
            spec=einops.rearrange(spec, "(b n c) f t -> b n c f t", b=B, n=N)

            N_pad = self.N_pad_spec
            if spec.shape[-1] % N_pad != 0:
                num_pad= N_pad - spec.shape[-1] % N_pad
                spec= torch.nn.functional.pad(spec, (0, num_pad, 0, 0), mode="constant", value=0)
            spec = spec.type(torch.complex64)

            if stft.compression_alpha != 1 or stft.compression_beta != 1:
                e=stft.compression_alpha
                beta=stft.compression_beta

                spec = spec.abs()**e * torch.exp(1j * spec.angle())
                spec = spec * beta

            return spec


        def istft( spec, length=None, window=None):

            B, N, C, F, T = spec.shape

            if stft.compression_alpha != 1 or stft.compression_beta != 1:
                spec = spec / stft.compression_beta
                e = stft.compression_alpha
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())

            window = window.to(spec.device)
            c=spec.shape[1]

            spec=einops.rearrange(spec, "b n c f t -> (b n c) f t")

            sig= torch.istft(spec, **{**self.stft_kwargs, "window": window}, length=length)
            sig=einops.rearrange(sig, "(b n c) t -> b n c t", b=B, n=N, c=C)
            return sig[..., :length]

        self.stft_fwd=lambda x: stft_func(x, window=window)

        self.stft_bwd=lambda x: istft(x, length=stft.audio_len, window=window)



    def sample_time_training(self, N):
        """
        For training, getting t according to a similar criteria as sampling. Simpler and safer to what Karras et al. did
        Args:
            N (int): batch size
        """
        t = torch.rand(N) * (1- 1e-5) + 1e-5

        return t

    def sample_prior(self, shape=None, t=None, dtype=None):
        """
        Just sample some gaussian noise, nothing more
        Args:
            shape (tuple): shape of the noise to sample, something like (B,T)
        """
        assert shape is not None
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
        raise NotImplementedError("This method should be implemented in the subclass")
        return -t * score
    
    def _corrector_(self, x, score, gamma, t):
        w=torch.randn_like(x)
        #annealed langevin dynamics
        step_size=0.5*(gamma*t)**2 

        return x + step_size*score + torch.sqrt(2 * step_size) * w


    def prepare_train_preconditioning(self, x, t,n=None, *args, **kwargs):
        #weight=self.lambda_w(sigma)
        #Eloi: Is calling the denoiser here a good idea? Maybe it would be better to apply directly the preconditioning as in the paper, even though Karras et al seem to do it this way in their code
        #Jm: I don't mind that preconditioning form actually, makes the loss also more normalized and easier to compare in between runs/SDEs i.m.o.

        mu, sigma = self._mean(x, t), self._std(t).unsqueeze(-1)
        sigma = sigma.view(*sigma.size(), *(1,)*(x.ndim - sigma.ndim))
        if n is None:
            n=self.sample_prior(shape=x.shape, dtype=x.dtype).to(x.device)

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

    def loss_fn(self, net, sample=None, context=None, *args, **kwargs):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        y=sample #shape (B,N,C,T)

        t = self.sample_time_training(y.shape[0]).to(y.device)



        with torch.no_grad():

            Y=self.transform_forward(y)/self.sigma_data

            x=context
            if x.shape[2]==1:
                #expand to 2 channels
                x=x.repeat(1, 1, 2, 1)

            X=self.transform_forward(x)/ self.sigma_data

            cond=X.clone()

            n=torch.randn_like(Y, dtype=Y.dtype).to(Y.device) * self.sigma_T
            X= X + n

            t=t.view(-1, 1,1,1,1)

            input= t*X + (1-t)*Y

            target= X - Y


        estimate = net(input, t.squeeze(-1), input_concat_cond=cond)
        
        error=torch.square(torch.abs(estimate-target))

        #compute mean over all dimensions except the first one

        error = error.mean(dim=tuple(range(1, error.ndim))).unsqueeze(-1)

        return error, t

    def model_call(self, x, net, t, cond=None,  cfg_scale=1.0, *args, **kwargs):

        if len(t.shape)==0:
            t=t.unsqueeze(-1).unsqueeze(-1)
        elif len(t.shape)==1:
            t=t.unsqueeze(-1)
        elif len(t.shape)==2:
            pass
        else:
            raise ValueError("Invalid shape of t: "+str(t.shape))

        #cond=einops.rearrange(cond, "b c t -> b t c") if cond is not None else None

        cnoise=t
        if cnoise.shape[0] != x.shape[0]:
            cnoise= cnoise.repeat(x.shape[0], 1)

        v=net(x, cnoise.to(torch.float32), input_concat_cond=cond)  #this will crash because of broadcasting problems, debug later!
        
        return v


    def flatten(self, X):

        self._X_shape=X.shape
        X_flat=X.flatten(start_dim=2)

        #transform complex to real
        #X_concat=torch.view_as_real(X_concat)
        #X_concat=X_concat.view(X_concat.shape[0], X_concat.shape[1], -1)


        return X_flat
    
    def unflatten(self, X):

        #real to complex

        #X_concat=X_concat.view(X_concat.shape[0], X_concat.shape[1], -1, 2)
        #X_concat=torch.view_as_complex(X_concat)

        X=X.view(self._X_shape)

        return X

        
    def transform_forward(self, x, *args, **kwargs):
        #TODO: Apply forward transform here
        X=self.stft_fwd(x)

        return X
    
    def transform_inverse(self, X, *args, **kwargs):

        x=self.stft_bwd(X)

        return x
        
    def normalize(self, x):
        return x

    def denormalize(self, x):
        return x
 
