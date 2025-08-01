
import torch

class Distribution:
    """
    Base class for distributions.
    """

    def __init__(self, shape, transformation=None):
        self.shape = shape
        self.transformation = transformation

    def sample(self, n_samples: int):
        """
        Sample from the distribution.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class Uniform(Distribution):
    """
    Uniform distribution class.
    Samples uniformly from a specified range.
    """

    def __init__(self, low, high, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low = low
        self.high = high

    def sample(self, n_samples: int):
        """
        Sample from the uniform distribution.
        """
        shape= (n_samples, *self.shape)

        sample= torch.rand(shape) * (self.high - self.low) + self.low

        if self.transformation is not None:
            sample = self.transformation(sample)

        return sample

class LogUniform(Distribution):
    """
    Uniform distribution class.
    Samples uniformly from a specified range.
    """

    def __init__(self, low, high, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low = torch.tensor(low)
        self.high = torch.tensor(high)

    def sample(self, n_samples: int):
        """
        Sample from the uniform distribution.
        """
        shape= (n_samples, *self.shape)

        sample= torch.rand(shape) * (torch.log10(self.high) - torch.log10(self.low)) + torch.log10(self.low)
        sample = torch.pow(10, sample)

        if self.transformation is not None:
            sample = self.transformation(sample)

        return sample

class UniformRamp(Distribution):
    """
    Normal distribution class.
    Samples from a normal distribution with specified mean and standard deviation.
    """

    def __init__(self, low_low, low_high, high_low, high_high, log=False, *args, **kwargs):
        super().__init__(*args,**kwargs )

        if not log:
            #linear ramp
            arange=torch.arange(self.shape[0], dtype=torch.float32)
            self.low = arange * (low_high - low_low) / (self.shape[0] - 1) + low_low
            self.high = arange * (high_high - high_low) / (self.shape[0] - 1) + high_low

        else:
            arange = torch.arange(self.shape[0], dtype=torch.float32)
            log_factor = torch.log10(torch.tensor(low_high / low_low))
            self.low = low_low * torch.pow(10, (arange * log_factor) / (self.shape[0] - 1))
            log_factor = torch.log10(torch.tensor(high_high / high_low))
            self.high = high_low * torch.pow(10, (arange * log_factor) / (self.shape[0] - 1))

        self.low = self.low.view(-1, *((1,) * (len(self.shape) - 1)))
        self.high = self.high.view(-1, *((1,) * (len(self.shape)- 1)))


    def sample(self, n_samples: int):
        """
        Sample from the normal distribution.
        """
        shape = (n_samples, *self.shape)
        sample= torch.rand(shape) * (self.high.unsqueeze(0) - self.low.unsqueeze(0)) + self.low.unsqueeze(0)
        #print("T60", sample)
        if self.transformation is not None:
            sample = self.transformation(sample)
        return sample
class NormalRamp(Distribution):
    """
    Normal distribution class.
    Samples from a normal distribution with specified mean and standard deviation.
    """

    def __init__(self, mean_low, mean_high, std_low, std_high, log=False, *args, **kwargs):
        super().__init__(*args,**kwargs )

        if not log:
            #linear ramp
            arange=torch.arange(self.shape[0], dtype=torch.float32)
            self.mean = arange * (mean_high - mean_low) / (self.shape[0] - 1) + mean_low
            self.std = arange * (std_high - std_low) / (self.shape[0] - 1) + std_low

        else:
            arange = torch.arange(self.shape[0], dtype=torch.float32)
            log_factor = torch.log10(torch.tensor(mean_high / (mean_low + 1e-4)))  # Adding a small value to avoid log(0)
            self.mean = mean_low * torch.pow(10, (arange * log_factor) / (self.shape[0] - 1))
            log_factor = torch.log10(torch.tensor(std_high /( std_low + 1e-4)))  # Adding a small value to avoid log(0)
            self.std = std_low * torch.pow(10, (arange * log_factor) / (self.shape[0] - 1))

        self.mean = self.mean.view(-1, *((1,) * (len(self.shape) - 1)))
        self.std = self.std.view(-1, *((1,) * (len(self.shape)- 1)))


    def sample(self, n_samples: int):
        """
        Sample from the normal distribution.
        """
        shape = (n_samples, *self.shape)
        sample= torch.randn(shape) * self.std.unsqueeze(0) + self.mean.unsqueeze(0)
        if self.transformation is not None:
            sample = self.transformation(sample)
        return sample

class Normal(Distribution):
    """
    Normal distribution class.
    Samples from a normal distribution with specified mean and standard deviation.
    """

    def __init__(self, mean, std, *args, **kwargs):
        super().__init__(*args,**kwargs )
        self.mean = mean
        self.std = std

    def sample(self, n_samples: int):
        """
        Sample from the normal distribution.
        """
        shape = (n_samples, *self.shape)
        sample= torch.randn(shape) * self.std + self.mean
        if self.transformation is not None:
            sample = self.transformation(sample)
        return sample

def sample_from_distribution_dict(distribution_dict, n_samples: int, device="cpu"):
    """
    Sample from a dictionary of distributions.
    """
    sampled_params = {}
    for key, distribution in distribution_dict.items():
        sampled_params[key] = distribution.sample(n_samples).to(device)
    return sampled_params


def unilossless(x):

    B, N,N= x.shape

    out= torch.zeros_like(x)
    for i in range(B):
        out[i]=torch.linalg.matrix_exp(x[i].triu(1) - x[i].triu(1).T)
    
    return out
