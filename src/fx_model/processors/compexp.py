import torch
import torch.nn.functional as F
import torchcomp

EPS= 1e-8


from fx_model.processors.diffvox_functional import compressor_expander as compressor_expander_diffvox_func

from fx_model.processors.transformations import Identity, UniLossLess, MinMax, SmoothingCoef, WrappedPositive


def prepare_compexp_parameters(
        sample_rate: int,
        max_lookahead: float = 15.0,
        device="cpu",
    ):

    sinc_length = int(sample_rate * (max_lookahead + 1) * 0.001) + 1
    left_pad_size = int(sample_rate * 0.001)

    pad_size = (left_pad_size, sinc_length - left_pad_size - 1)
    arange=torch.arange(sinc_length) - left_pad_size

    dict_non_optimizable = {
        "pad_size": pad_size,  # padding size for lookahead
        "arange": arange,  # range for sinc filter
        "sample_rate": sample_rate,  # sample rate for sinc filter
    }


    dict_transformations= {
        "comp_ratio": MinMax(1.0, 20.0),
        "exp_ratio": SmoothingCoef(),
        "at_coef": SmoothingCoef(),
        "rt_coef": SmoothingCoef(),
        "avg_coef": SmoothingCoef(),
        "comp_th": Identity(),
        "exp_th": Identity(),
        "make_up": Identity(),
        "lookahead":  WrappedPositive(max_lookahead), 
    }

    #same intitialization as DiffVox

    comp_ratio = torch.tensor([2.0])
    comp_ratio = dict_transformations["comp_ratio"].right_inverse(comp_ratio)

    exp_ratio = torch.tensor([0.5])  # 
    exp_ratio = dict_transformations["exp_ratio"].right_inverse(exp_ratio)

    at_ms= torch.tensor([50.0])  #
    at_coef=torchcomp.ms2coef(at_ms, sample_rate)
    at_coef = dict_transformations["at_coef"].right_inverse(at_coef)

    rt_ms= torch.tensor([50.0])  #
    rt_coef=torchcomp.ms2coef(rt_ms, sample_rate)
    rt_coef = dict_transformations["rt_coef"].right_inverse(rt_coef)

    avg_coef = torch.tensor([0.3])  # average coefficient
    avg_coef = dict_transformations["avg_coef"].right_inverse(avg_coef)

    comp_th = torch.tensor([-18.0])  # 
    exp_th= torch.tensor([-54.0])  # 

    make_up = torch.tensor([0.0])  # make-up gain dB

    lookahead= torch.ones(1) / sample_rate * 1000 # lookahead in ms
    lookahead = dict_transformations["lookahead"].right_inverse(lookahead)

    dict_optimizable = {
        "comp_ratio": comp_ratio.to(device),  # shape (1,)
        "exp_ratio": exp_ratio.to(device),  # shape (1,)
        "at_coef": at_coef.to(device),  # shape (1,)
        "rt_coef": rt_coef.to(device),  # shape (1,)
        "avg_coef": avg_coef.to(device),  # shape (1,)
        "comp_th": comp_th.to(device),  # shape (1,)
        "exp_th": exp_th.to(device),  # shape (1,)
        "make_up": make_up.to(device),  # shape (1,)
        "lookahead": lookahead.to(device),  # shape (1,)
    }

    return dict_optimizable, dict_non_optimizable, dict_transformations



        
 
def compexp_functional(
    x: torch.Tensor,
    comp_ratio: torch.Tensor,
    exp_ratio: torch.Tensor,
    at_coef: torch.Tensor,
    rt_coef: torch.Tensor,
    avg_coef: torch.Tensor,
    comp_th: torch.Tensor,
    exp_th: torch.Tensor,
    make_up: torch.Tensor,
    lookahead: torch.Tensor,
    sample_rate: int,
    pad_size=None,
    arange=None,
):
    """Compressor/Expander with gating functionality.
 
    Args:
        x (torch.Tensor): Input audio tensor with shape (bs, chs, seq_len)
        sample_rate (float): Audio sample rate.
        comp_threshold (torch.Tensor): Compressor threshold in dB.
        comp_ratio (torch.Tensor): Compressor ratio.
        exp_thresh (torch.Tensor): Expander threshold in dB.
        exp_ratio (torch.Tensor): Expander ratio.
        at (torch.Tensor): Attack time in ms.
        rt (torch.Tensor): Release time in ms.
        comp_gate (bool): If True, use compressor mode; if False, use expander mode.
 
    Returns:
        torch.Tensor: Processed audio signal.
    """
    bs, chs, seq_len = x.size()
 
    arange= arange.to(x.device)

    lookahead = lookahead
    lookahead_in_samples = lookahead * 0.001 *sample_rate
    sinc_filter = torch.sinc(arange - lookahead_in_samples).to(x.device)




    def lookahead_func_fn(gain, sinc_filter, pad_size):
        """
        Apply lookahead to the gain using batch-wise grouped convolution.
        
        gain: Tensor of shape (B, T)
        sinc_filter: Tensor of shape (B, K), one filter per batch item
        pad_size: Tuple (left, right) padding for conv1d
        """
        B, T = gain.shape
        K = sinc_filter.shape[1]
    
        # Reshape gain and filters for grouped conv1d
        gain_padded = F.pad(gain.unsqueeze(1), pad_size, mode="replicate")  # (B, 1, T+pad)
        gain_reshaped = gain_padded.view(1, B, -1)                          # (1, B, T+pad)
        
        filters = sinc_filter.view(B, 1, K)                                 # (B, 1, K)
        
        # Perform grouped convolution
        output = F.conv1d(gain_reshaped, filters, groups=B)                # (1, B, T)
        output = output.view(B, T)                                         # (B, T)
    
        return output


    lookahead_func = lambda gain: lookahead_func_fn(gain, sinc_filter=sinc_filter, pad_size=pad_size)

    #print("avg_coef", avg_coef.shape, x.shape)

    return compressor_expander_diffvox_func(
        x.reshape(-1, x.shape[-1]),
        avg_coef=avg_coef,
        cmp_th=comp_th.squeeze(-1),
        cmp_ratio=comp_ratio.squeeze(-1),
        exp_th=exp_th.squeeze(-1),
        exp_ratio=exp_ratio.squeeze(-1),
        at=at_coef.squeeze(-1),
        rt=rt_coef.squeeze(-1),
        make_up=make_up,
        lookahead_func=lookahead_func,
    ).view(*x.shape)

