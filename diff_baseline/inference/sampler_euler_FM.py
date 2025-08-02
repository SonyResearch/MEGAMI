from tqdm import tqdm
import torch
from inference.sampler import Sampler


class SamplerEulerFM(Sampler):

    def __init__(self, model, diff_params, args):
        super().__init__(model, diff_params, args)

        # stochasticity parameters
        self.Schurn = self.args.tester.sampling_params.Schurn
        self.Snoise = self.args.tester.sampling_params.Snoise
        self.Stmin = self.args.tester.sampling_params.Stmin
        self.Stmax = self.args.tester.sampling_params.Stmax

        # order of the sampler
        self.order = self.args.tester.sampling_params.order
        self.cond=None
        self.cfg_scale = 1.0

    def predict_conditional(
            self,
            shape,  # observations (lowpssed signal) Tensor with shape ??
            cond=None,
            cfg_scale=1.0,
            device=None,  # device
            apply_inverse_transform=True  # whether to apply inverse transform
    ):
        self.cond = cond
        assert self.cond is not None, "Conditional input is None"
        if self.cond.shape[2]==1:
            #fake stereo channel, repeat it
            self.cond = self.cond.repeat(1, 1, 2, 1, 1)


        self.cfg_scale = cfg_scale

        return self.predict(shape, device, apply_inverse_transform=apply_inverse_transform)

    def predict_unconditional(
            self,
            shape,  # observations (lowpssed signal) Tensor with shape ??
            device
    ):
        self.y = None
        self.degradation = None

        return self.predict(shape, device)

    def get_gamma(self, t):
        """
        Get the parameter gamma that defines the stochasticity of the sampler
        Args
            t (Tensor): shape: (N_steps, ) Tensor of timesteps, from which we will compute gamma
        """
        N = t.shape[0]
        gamma = torch.zeros(t.shape).to(t.device)

        # If desired, only apply stochasticity between a certain range of noises Stmin is 0 by default and Stmax is a huge number by default. (Unless these parameters are specified, this does nothing)
        indexes = torch.logical_and(t > self.Stmin, t < self.Stmax)

        # We use Schurn=5 as the default in our experiments
        gamma[indexes] = gamma[indexes] + torch.min(torch.Tensor([self.Schurn / N, 2 ** (1 / 2) - 1]))

        return gamma

    def get_Tweedie_estimate(self, x, t_i):

        if x.ndim==2:
            x_=x.unsqueeze(1)
        elif x.ndim==3:
            pass

        if self.cond is not None:
            x_hat = self.diff_params.denoiser(x, self.model, t_i, cond=self.cond, cfg_scale=self.cfg_scale)
        else:
            x_hat = self.diff_params.denoiser(x, self.model, t_i)

        return x_hat

    def Tweedie2score(self, tweedie, xt, t):
        return self.diff_params.Tweedie2score(tweedie, xt, t)

    def score2Tweedie(self, score, xt, t):
        return self.diff_params.score2Tweedie(score, xt, t)

    def stochastic_timestep(self, x, t, gamma, Snoise=1):
        t_hat = t + gamma * t  # if gamma_sig[i]==0 this is a deterministic step, make sure it doed not crash
        t_hat=torch.clamp(t_hat, 0, self.diff_params.max_t)
        epsilon = torch.randn(x.shape).to(x.device) * Snoise  # sample Gaussiannoise, Snoise is 1 by default
        if t_hat <= t:
            x_hat = x
            #print(f"t_hat<=t, gamma {gamma}")
        else:
            #print(t_hat, t)
            x_hat = x + ((t_hat ** 2 - t ** 2) ** (1 / 2)) * epsilon  # Perturb data
        return x_hat, t_hat

    def step(self, x_i, t_i, t_iplus1):

        with torch.no_grad():

            #x_den = self.get_Tweedie_estimate(x_hat, t_hat)
            v=self.diff_params.model_call(x_i,self.model, t_i,  cfg_scale=self.cfg_scale, cond=self.cond/self.diff_params.sigma_data)
            #x_den= self.diff_params.v2Tweedie(v, x_i, t_i )

            #score = self.Tweedie2score(x_den, x_hat, t_hat)
            #ode_integrand = self.diff_params._ode_integrand(x_i, t_i, v)
            ode_integrand=v

            dt = t_iplus1 - t_i

            x_iplus1 = x_i + dt * ode_integrand

            return x_iplus1
    
    def get_domain_shape(self, shape, device):

        x=torch.zeros(shape, dtype=torch.float32).to(device)
        X=self.diff_params.transform_forward(x)

        return X.shape, X.dtype


    def predict(
            self,
            shape,  # observations (lowpssed signal) Tensor with shape ??
            device,  # lambda function
            dtype=torch.float32,  # data type
            apply_inverse_transform=True  # whether to apply inverse transform
    ):

        # get the noise schedule
        t = self.create_schedule().to(device).to(torch.float32)


        #shape_example, dtype = self.get_domain_shape(shape, device)
        #print("shape_example", shape_example, dtype)

        # sample prior
        #x = self.diff_params.sample_prior(t=t[0], shape=shape, dtype=dtype)
        print(self.diff_params.sigma_data, self.diff_params.sigma_T)
        x=self.cond/self.diff_params.sigma_data + torch.randn_like(self.cond)*self.diff_params.sigma_T

        x_init = x.clone()
        print("x_init", x_init.shape, "x", x.shape, x.std(), x_init.std())

        # parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic

        for i in tqdm(range(0, self.T-1, 1)):
            self.step_counter = i
            x = self.step(x, t[i], t[i + 1])

        if apply_inverse_transform:
            x=x* self.diff_params.sigma_data 
            x_init=x_init*self.diff_params.sigma_data

            with torch.no_grad():
                x_den_wave=self.diff_params.transform_inverse(x.detach())
                x_init_wave=self.diff_params.transform_inverse(x_init.detach())

            return x_den_wave.detach(), x_init_wave.detach()
        else:
            return x.detach(), None

    def create_schedule(self, sigma_min=None, sigma_max=None, rho=None, T=None):
        """
        EDM schedule by default
        """
        if T is None:
            T=self.T

        t = torch.linspace(self.diff_params.sigma_max, self.diff_params.sigma_min, T+1)
        return t


