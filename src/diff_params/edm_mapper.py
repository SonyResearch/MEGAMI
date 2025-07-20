import torch
import numpy as np

from diff_params.shared import SDE

import torch.distributed as dist


class EDM(SDE):
    """
        Definition of the diffusion parameterization, following ( Karras et al., "Elucidating...", 2022). 
        This includes only the utilities needed for training, not for sampling.
    """

    def __init__(self,
        type,
        sde_hp,
        cfg_dropout_prob, 
        default_shape,
        **kwargs
        ):

        super().__init__(type, sde_hp)

        self.sigma_data = self.sde_hp.sigma_data #depends on the training data!! precalculated variance of the dataset
        self.sigma_min = self.sde_hp.sigma_min
        self.sigma_max = self.sde_hp.sigma_max
        self.rho = self.sde_hp.rho

        self.default_shape = torch.Size(default_shape)

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
        self.embedding = None

        from utils.ITOMaster_loss import MultiScale_Spectral_Loss_MidSide_DDSP

        multi_scale_spectral_midside = MultiScale_Spectral_Loss_MidSide_DDSP(mode='midside', eps=1e-6, device=self.device, reduce=False)
        multi_scale_spectral_ori = MultiScale_Spectral_Loss_MidSide_DDSP(mode='ori', eps=1e-6, device=self.device, reduce=False)

        def MSS_loss_fn(x_pred, y):
            """
            Loss function, which is the mean squared error between the denoised latent and the clean latent
            Args:
                x_pred (Tensor): shape: (B,T) Intermediate noisy latent to denoise
                y (Tensor): shape: (B,T) Clean latent
            """

            loss_midside = multi_scale_spectral_midside(x_pred, y)
            loss_ori = multi_scale_spectral_ori(x_pred, y)

            loss_dictionary = {
                "loss_mss_midside": loss_midside,
                "loss_mss_lr": loss_ori,
            }

            return loss_dictionary

        self.losses = {"loss_mss_midside": [], "loss_mss_lr": []}
        self.loss_weights = {"loss_mss_midside": 0.25, "loss_mss_lr": 0.25}
        self.MSS_loss_fn = MSS_loss_fn  

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





    def denoiser(self, xn , net, t, cond=None, *args, **kwargs):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (B,1,T) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        #print("xn",xn.shape)
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
        
        #TODO implement CFG with respect to the embedding

        x_hat=cskip * xn + cout * net((cin * xn).to(torch.float32),self.embedding, time_cond=cnoise.to(torch.float32), input_concat_cond=cond).to(xn.dtype)  #this will crash because of broadcasting problems, debug later!

        return x_hat
        
    def denormalize(self, x):
        return x
    
    def normalize(self, x):
        return x
        
    def save_embedding(self,net, x, z,  *args, **kwargs):
        if net.use_CLAP:
            with torch.no_grad():
                z= net.merge_CLAP_embeddings(x,z)

        self.embedding = z

        
    def prepare_train_preconditioning(self, x, t,n=None, *args, **kwargs):

        mu, sigma = self._mean(x, t), self._std(t).unsqueeze(-1)
        sigma = sigma.view(*sigma.size(), *(1,)*(x.ndim - sigma.ndim))
        if n is None:
            n=self.sample_prior(shape=x.shape).to(x.device)
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

        return cin * x_perturbed, target, cnoise, cskip, cout, cin

    def loss_fn(self, net, sample=None, x=None, embedding=None, *args, **kwargs):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        y=sample

        if self.cfg_dropout_prob > 0.0:
                #context=self.transform_forward(context)
                null_embed = torch.zeros_like(embedding, device=embedding.device)
                #dropout context with probability cfg_dropout_prob
                mask = torch.rand(embedding.shape[0], device=embedding.device) < self.cfg_dropout_prob
                embedding = torch.where(mask.view(-1,1), null_embed, embedding)
    

        t = self.sample_time_training(y.shape[0]).to(y.device)


        input, target, cnoise, cskip, cout, cin = self.prepare_train_preconditioning(y, t)



        #print("input",input.shape,"cnoise", cnoise.shape)

        if len(cnoise.shape) == 1:
            #dirty patch
            cnoise = cnoise.unsqueeze(-1)

        if input.ndim==2:
            input=input.unsqueeze(1)

        if net.use_CLAP:
            with torch.no_grad():
                embedding= net.merge_CLAP_embeddings(x,embedding)

        #print("input.shape", input.shape, "embedding.shape", embedding.shape, "cnoise.shape", cnoise.shape)
        estimate = net(input, embedding, time_cond=cnoise, input_concat_cond=x)
        
        if target.ndim==2 and estimate.ndim==3:
            estimate=estimate.squeeze(1)

        error = estimate - target

        x_hat= cskip * (input/cin) + cout * estimate

        mss_loss = self.MSS_loss_fn(x_hat, y)


        loss_dict= {
            "l2": (error**2).mean(-1).mean(-1).unsqueeze(-1),
            "MSS_loss_ms": mss_loss["loss_mss_midside"].mean(-1)* self.loss_weights["loss_mss_midside"],
            "MSS_loss_lr": mss_loss["loss_mss_lr"].mean(-1)* self.loss_weights["loss_mss_lr"],
        }

        return loss_dict, self._std(t)
    
    def transform_forward(self,x,*args, **kwargs):
        """
        Transform the input x to the forward diffusion process
        Args:
            x (Tensor): shape: (B,T) Input to transform
        """
        return x

    def transform_inverse(self, x, *args, **kwargs):
        """
        Transform the input x to the inverse diffusion process
        Args:
            x (Tensor): shape: (B,T) Input to transform
        """
        return x