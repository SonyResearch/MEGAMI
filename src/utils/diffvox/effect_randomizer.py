
from functools import partial
import json
import numpy as np  
import hydra
import torch
from utils.diffvox.modules.utils import  get_chunks, vec2statedict


import logging

# Set the logging level for the Numba CUDA driver to WARNING
logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)


class DiffvoxRandomizer:

    def __init__(self,
        strategy=None,  # Strategy can be 'random', 'sequential', etc.
        std_range=3,
        fx=None,
        npz_path=None,
        info_path=None,
        seed=None):

        """
        Initializes the DiffvoxRandomizer with an optional seed for reproducibility.
        :param seed: Optional seed for random number generation.
        """

        print(fx)
        self.fx=fx

        gauss_data = np.load(npz_path)

        self.mean_vec = torch.tensor(gauss_data["mean"])
        self.cov = torch.tensor(gauss_data["cov"])

        self.stds = torch.sqrt(torch.diag(self.cov))  # Standard deviations are the square root of the diagonal elements of the covariance matrix


        with open("/home/eloi/projects/diffvox/presets/internal/info.json") as f:
            info = json.load(f)
    
        param_keys = info["params_keys"]
        original_shapes = list(
            map(lambda lst: lst if len(lst) else [1], info["params_original_shapes"])
        )
    
        *vec2dict_args, dimensions_not_need = get_chunks(param_keys, original_shapes)
        vec2dict_args = [param_keys, original_shapes] + vec2dict_args
        vec2dict = partial(
            vec2statedict,
            **dict(
                zip(
                    [
                        "keys",
                        "original_shapes",
                        "selected_chunks",
                        "position",
                        "U_matrix_shape",
                    ],
                    vec2dict_args,
                )
            ),
        )
    
        ndim_dict = {k: v.ndim for k, v in self.fx.state_dict().items()}
        self.to_fx_state_dict = lambda x: {
            k: v[0] if ndim_dict[k] == 0 else v for k, v in vec2dict(x).items()
        }


    def apply_random_effect(self, x, std_range=3):
        """
        Applies a random effect to the model parameters.
        :param num_samples: Number of random samples to generate.
        :return: A dictionary containing the modified state dictionary.
        """

        print(f"Applying random effect with std_range: {std_range}")

        unbatch = False
        if len(x.shape) == 2:
            unbatch = True
            x = x.unsqueeze(0)

        self.fx.to(x.device)

        mean= self.mean_vec.to(x.device)


        # Step 2: Define the range for each dimension
        lower_bound = mean - std_range * self.stds.to(x.device)
        upper_bound = mean + std_range * self.stds.to(x.device)
    
        # Step 3: Sample uniformly within the range for each dimension
        num_dimensions = mean.shape[0]

        num_samples = x.shape[0]  # Assuming x is a batch of inputs, use its batch size as num_samples  

        uniform_samples = torch.empty(num_samples, num_dimensions).to(x.device)
        for i in range(num_dimensions):
            uniform_samples[:, i] = torch.empty(num_samples).uniform_(lower_bound[i], upper_bound[i]).to(x.device)


        output=torch.empty_like(x)

        if x.shape[1] == 2:
            x = x.mean(dim=1, keepdim=True)

        for i in range(num_samples):
                state_dict= self.to_fx_state_dict(uniform_samples[i])
                self.fx.load_state_dict(state_dict, strict=False)
                #output[i]=torch.func.functional_call(self.fx, state_dict, x[i])
                output[i] = self.fx(x[i].unsqueeze(0))


        assert output.shape[0] == x.shape[0], f"Output shape {output.shape} does not match input shape {x.shape}"
        assert output.shape[-1] == x.shape[-1], f"Output shape {output.shape} does not match input shape {x.shape}"

        if unbatch:
            output = output.squeeze(0)
        
        return output
