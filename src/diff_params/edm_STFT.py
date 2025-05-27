
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

class EDM_STFT(SDE):
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
        default_shape
        ):

        super().__init__(type, sde_hp)

        self.default_shape = torch.Size(default_shape)

        self.sigma_data = self.sde_hp.sigma_data #depends on the training data!! precalculated variance of the dataset
        self.sigma_min = self.sde_hp.sigma_min
        self.sigma_max = self.sde_hp.sigma_max
        self.rho = self.sde_hp.rho

        try:
            self.max_t= self.sde_hp.max_sigma
        except Exception as e:
            print(e)
            print("max_sigma not defined, please add it. It should be the highest sigma value seen during training")
        

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
            window = window.to(sig.device)
            C=sig.shape[1]
            sig=einops.rearrange(sig, "b c t -> (b c) t")   
            spec= torch.stft(sig, **{**self.stft_kwargs, "window": window}, return_complex=True)
            # spec= torch.stft(sig, **{**vars(self.stft_kwargs), "window": window}, return_complex=True)
            spec=einops.rearrange(spec, "(b c) f t -> b c f t", c=C)
            #pad in the time axis if the resulting spec is not a multiple of 16
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

            if stft.compression_alpha != 1 or stft.compression_beta != 1:
                spec = spec / stft.compression_beta
                e = stft.compression_alpha
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())

            window = window.to(spec.device)
            c=spec.shape[1]
            spec=einops.rearrange(spec, "b c f t -> (b c) f t")
            sig= torch.istft(spec, **{**self.stft_kwargs, "window": window}, length=length)
            # sig= torch.istft(spec, **{**vars(self.stft_kwargs), "window": window}, length=length)
            sig=einops.rearrange(sig, "(b c) t -> b c t", c=c)
            return sig[..., :length]

        self.stft_fwd=lambda x: stft_func(x, window=window)

        self.stft_bwd=lambda x: istft(x, length=stft.audio_len, window=window)



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
        y=sample

        t = self.sample_time_training(y.shape[0]).to(y.device)

        #print("sample std", x_CQT.std())
        with torch.no_grad():
            Y=self.transform_forward(y)

            x=context
            X=self.transform_forward(x, is_condition=True)
            if self.cfg_dropout_prob > 0.0:
                    #context=self.transform_forward(context)
                    null_embed = torch.zeros_like(X, device=X.device)
                    #dropout context with probability cfg_dropout_prob
                    mask = torch.rand(X.shape[0], device=X.device) < self.cfg_dropout_prob
                    X = torch.where(mask.view(-1,1,1,1), null_embed, X)
    
    
        
        print("Y std", Y.std())

        input, target, cnoise = self.prepare_train_preconditioning(Y, t)

        #print("x_in", input.std(), input.shape, input.dtype)
        #print("target", target.std(), target.shape, target.dtype)
        #print("sample std", x_CQT.std())

        if len(cnoise.shape)==1:
            cnoise=cnoise.unsqueeze(-1)
        if input.ndim==2:
            input=input.unsqueeze(1)

        estimate = net(input, cnoise.squeeze(-1), input_concat_cond=X)
        
        if target.ndim==2 and estimate.ndim==3:
            estimate=estimate.squeeze(1)

        error=torch.square(torch.abs(estimate-target))
        

        return error, self._std(t)



    def denoiser(self, xn , net, t, cond=None, *args, **kwargs):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (CQT shape?) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """

        dtype_in=xn.dtype
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


        net_out=net(x_in.to(dtype_in), cnoise.squeeze(-1).to(torch.float32), input_concat_cond=cond)  #this will crash because of broadcasting problems, debug later!

        net_out=net_out.to(xn.dtype)

        x_hat=cskip*xn + cout*net_out   

        #x_hat has CQT shape

        return x_hat

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


        
    def transform_forward(self, x, is_condition=False):
        #TODO: Apply forward transform here
        X=self.stft_fwd(x)
        if is_condition:
            X=X.abs()

        #X_list is a list of tensors with shape (B, C, time, freq), where time and freq are different for each tensor

        #I want to map it to a tensor with shape (B, C, C2), the transform should be invertible

        return X
    
    def transform_inverse(self, X):

        x=self.stft_bwd(X)


        return x
        
    def normalize(self, x):
        return x

    def denormalize(self, x):
        return x
 