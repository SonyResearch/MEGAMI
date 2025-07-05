
import time
from fx_model.processors.compexp import prepare_compexp_parameters, compexp_functional
import math

from fx_model.processors.fdn import prepare_FDN_parameters, fdn_functional
from fx_model.processors.peq import prepare_PEQ_parameters, peq_functional, prepare_PEQ_FDN_parameters, peq_FDN_functional
import torch
import torch.nn as nn

from fx_model.processors.transformations import  SmoothingCoef, MinMax

from utils.distributions import Uniform, Normal, sample_from_distribution_dict


def panning( x, pan_param):
        angle = pan_param.view(-1,1) * torch.pi * 0.5
        amp = torch.concat([angle.cos(), angle.sin()],dim=1).view(-1,2, 1) * math.sqrt(2)  # Normalize to avoid gain increase
        return x * amp

def RMS_normalization(x, rms_norm):
    """
    Normalize the RMS of the input tensor to a target value.
    x: shape (batch_size, num_channels, num_samples)
    rms_norm: target RMS value in dB
    """
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
    target_rms = 10 ** (rms_norm / 20)  # Convert dB to linear scale
    return x * (target_rms / (rms + 1e-6))  # Avoid division by zero



class MyModel(nn.Module):

    def __init__(self, 
        sample_rate=44100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ):
        super(MyModel, self).__init__()

        self.sample_rate = sample_rate

        params_PEQ_optimizable, params_PEQ_non_optimizable, dict_transformations = prepare_PEQ_parameters(sample_rate, device=device)
        self.PEQ_params = nn.ParameterDict({
            key: torch.nn.Parameter(value) for key, value in params_PEQ_optimizable.items()
        })
        self.params_PEQ_non_optimizable = params_PEQ_non_optimizable
        self.PEQ_transformations = dict_transformations


        params_PEQ_FDN_optimizable, params_PEQ_FDN_non_optimizable, dict_transformations = prepare_PEQ_FDN_parameters(sample_rate, device=device)
        self.PEQ_FDN_params = nn.ParameterDict({
            key: torch.nn.Parameter(value) for key, value in params_PEQ_FDN_optimizable.items()
        })
        self.params_PEQ_FDN_non_optimizable = params_PEQ_FDN_non_optimizable
        self.PEQ_FDN_transformations = dict_transformations


        params_CompExp_optimizable, params_CompExp_non_optimizable, dict_transformations = prepare_compexp_parameters(sample_rate, device=device)
        self.CompExp_params = nn.ParameterDict({
            key: torch.nn.Parameter(value) for key, value in params_CompExp_optimizable.items()
        })

        self.params_CompExp_non_optimizable = params_CompExp_non_optimizable
        self.CompExp_transformations = dict_transformations

        
        params_FDN_optimizable, params_FDN_non_optimizable, dict_transformations = prepare_FDN_parameters(sample_rate, device=device,  ir_duration = 6.0)
        self.FDN_params = nn.ParameterDict({
            key: torch.nn.Parameter(value, requires_grad=True) for key, value in params_FDN_optimizable.items()
        })

        self.params_FDN_non_optimizable = params_FDN_non_optimizable
        self.FDN_transformations = dict_transformations


        pan=torch.tensor(0.5, device=device)  # Initialize panning parameter to 0.5 (centered)
        self.pan_param_transformation = SmoothingCoef()

        self.pan_param= torch.nn.Parameter(self.pan_param_transformation.right_inverse(pan), requires_grad=True)

        #self.RMS_norm= nn.Parameter(torch.tensor(-18.0), requires_grad=True) #in DB
        #self.RMS_norm_transformation = MinMax(-30.0, 0.0)  # Normalize RMS to a range between -30 dB and 0 dB


    def forward(self, x):
        """
        Process the data using GRAFx-specific algorithms.
        x: shape (batch_size, num_channels, num_samples)
        """
        #x= RMS_normalization(x, self.RMS_norm)

        start= time.time()


        batched_params_PEQ={}
        for key, param in self.PEQ_params.items():
            param_transformed= self.PEQ_transformations[key](param)
            batched_params_PEQ[key] = param_transformed.unsqueeze(0).repeat(x.size(0), *([1] * (param_transformed.dim() )))
        
        x= peq_functional(x, **batched_params_PEQ, **self.params_PEQ_non_optimizable)

        batched_params_CompExp={}
        for key, param in self.CompExp_params.items():
            param_transformed= self.CompExp_transformations[key](param)
            batched_params_CompExp[key] = param_transformed.unsqueeze(0).repeat(x.size(0), *([1] * (param_transformed.dim() )))
        
        x= compexp_functional(x, **batched_params_CompExp, **self.params_CompExp_non_optimizable)

        batched_params_FDN={}
        for key, param in self.FDN_params.items():
            param_transformed= self.FDN_transformations[key](param)
            batched_params_FDN[key] = param_transformed.unsqueeze(0).repeat(x.size(0), *([1] * (param_transformed.dim() )))
        
        batched_params_PEQ_FDN={}
        for key, param in self.PEQ_FDN_params.items():
            param_transformed= self.PEQ_FDN_transformations[key](param)
            batched_params_PEQ_FDN[key] = param_transformed.unsqueeze(0).repeat(x.size(0), *([1] * (param_transformed.dim() )))
        
        def eq_fn(h):
            B, C1, C2, T = h.shape
            h= h.view(B, C1 * C2, T)
            h=peq_FDN_functional(h, **batched_params_PEQ_FDN, **self.params_PEQ_FDN_non_optimizable)
            return h.view(B, C1, C2, T)

        x_to_fdn=x.repeat(1,2,1)
        x_fdn=fdn_functional(x_to_fdn, **batched_params_FDN, **self.params_FDN_non_optimizable, eq=eq_fn)


        pan_param_transformed= self.pan_param_transformation(self.pan_param)
        x_pan=panning( x, pan_param_transformed)


        x = x_fdn + x_pan  # Add the panned signal to the original signal

        return x

class EffectRandomizer:
    """
    Randomizes the parameters of the model.
    """
    def __init__(self,
                sample_rate=44100,
                distributions_dict=None,
                device="cuda" if torch.cuda.is_available() else "cpu",
                ):

        self.device = device

        assert distributions_dict is not None, "distributions_PEQ must be provided"
        self.distributions_PEQ = distributions_dict.get("PEQ", None)
        assert self.distributions_PEQ is not None, "distributions_PEQ must be provided"

        self.distributions_CompExp = distributions_dict.get("CompExp", None)
        assert self.distributions_CompExp is not None, "distributions_CompExp must be provided"

        self.distributions_FDN = distributions_dict.get("FDN", None)
        assert self.distributions_FDN is not None, "distributions_FDN must be provided"

        self.distributions_PEQ_FDN = distributions_dict.get("PEQ_FDN", None)
        assert self.distributions_PEQ_FDN is not None, "distributions_PEQ_FDN must be provided"

        self.distributions_pan = distributions_dict.get("pan", None)
        assert self.distributions_pan is not None, "distributions_pan must be provided"

        self.distributions_RMS_norm = distributions_dict.get("RMSnorm", None)
        assert self.distributions_RMS_norm is not None, "distributions_RMS_norm must be provided"

        _, self.params_PEQ_non_optimizable, _ = prepare_PEQ_parameters(sample_rate, device=device)


        _, self.params_PEQ_FDN_non_optimizable, _ = prepare_PEQ_FDN_parameters(sample_rate, device=device)


        _, self.params_CompExp_non_optimizable, _ = prepare_compexp_parameters(sample_rate, device=device)

        
        _, self.params_FDN_non_optimizable, _ = prepare_FDN_parameters(sample_rate, device=device, ir_duration=6.0)

    def apply_RMS_normalization(self, x, RMS=None):

        #if random:
        #    RMS=self.distribution_RMS.sample(x.shape[0]).view(-1, 1, 1).to(x.device)  # Ensure RMS is broadcastable
        #else:
        #RMS= torch.tensor(RMS, device=x.device).view(1, 1, 1).repeat(x.shape[0],1,1)  # Use fixed RMS for evaluation
        RMS= RMS.view(-1, 1, 1).to(x.device)  # Ensure RMS is broadcastable

        x_RMS=20*torch.log10(torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True).mean(dim=-2, keepdim=True)))

        gain= RMS - x_RMS
        gain_linear = 10 ** (gain / 20)
        x=x* gain_linear.view(-1, 1, 1)

        return x

    def forward(self, x):
        """
        Process the data using GRAFx-specific algorithms.
        x: shape (batch_size, num_channels, num_samples)
        """
        #x= x * self.pre_gain
        a=time.time()

        B= x.size(0)

        #convert to mono if stereo
        if x.size(1) == 2:
            x= x.mean(dim=1, keepdim=True)  # Average the two channels to create a mono signal

        params_PEQ = sample_from_distribution_dict(self.distributions_PEQ, B, device=self.device)
        #print("Time to sample PEQ parameters:", time.time() - a)
        x= peq_functional(x, **params_PEQ, **self.params_PEQ_non_optimizable)
        #print("Time to apply PEQ:", time.time() - a)

        params_CompExp = sample_from_distribution_dict(self.distributions_CompExp, B, device=self.device)
        #print("Time to sample CompExp parameters:", time.time() - a)
        x= compexp_functional(x, **params_CompExp, **self.params_CompExp_non_optimizable)
        #print("Time to apply CompExp:", time.time() - a)

        params_FDN= sample_from_distribution_dict(self.distributions_FDN, B, device=self.device)
        #print("Time to sample FDN parameters:", time.time() - a)
        
        params_PEQ_FDN = sample_from_distribution_dict(self.distributions_PEQ_FDN, B, device=self.device)
        #print("Time to sample PEQ_FDN parameters:", time.time() - a)    
        
        def eq_fn(h):
            B, C1, C2, T = h.shape
            h= h.view(B, C1 * C2, T)
            h=peq_FDN_functional(h, **params_PEQ_FDN, **self.params_PEQ_FDN_non_optimizable)
            return h.view(B, C1, C2, T)

        x_to_fdn=x.repeat(1,2,1)
        x_fdn=fdn_functional(x_to_fdn, **params_FDN, **self.params_FDN_non_optimizable, eq=eq_fn)

        #print("Time to apply FDN:", time.time() - a)

        params_pan = sample_from_distribution_dict(self.distributions_pan, B, device=self.device)
        #print("Time to sample pan parameters:", time.time() - a)
        pan_param= params_pan["pan_param"]
        x_pan=panning( x, pan_param)
        #print("Time to apply panning:", time.time() - a)

        #x=x_pan
        x = x_fdn + x_pan  # Add the panned signal to the original signal

        RMS_norm = sample_from_distribution_dict(self.distributions_RMS_norm, B, device=self.device)
        #print("Time to sample RMS normalization parameters:", time.time() - a)


        x = self.apply_RMS_normalization(x, RMS=RMS_norm["RMSnorm"])
        
        #print("Time to apply RMS normalization:", time.time() - a)

        return x