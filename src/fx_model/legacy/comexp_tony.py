import torch
import torchcomp
from dasp_pytorch import Processor

EPS= 1e-8


class Compressor_Expander(Processor):
    def __init__(
        self,
        sample_rate: int,
        # comp min max
        min_comp_threshold_db: float = -30.0,
        max_comp_threshold_db: float =-5.0,
        min_comp_ratio: float = 1.5,
        max_comp_ratio: float = 6.0,
        # exp min max
        min_exp_threshold_db: float = -30.0,
        max_exp_threshold_db: float = -5.0,
        min_exp_ratio: float = 0.0+EPS,
        max_exp_ratio: float = 1.0-EPS,
        # comp at /rt min max
        min_attack_ms: float = 1.0,
        max_attack_ms: float = 20.0,
        min_release_ms: float = 20.0,
        max_release_ms: float = 500.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = compressor_expander
        self.param_ranges = {
            "comp_threshold": (min_comp_threshold_db, max_comp_threshold_db),
            "comp_ratio": (min_comp_ratio, max_comp_ratio),
            "exp_threshold": (min_exp_threshold_db, max_exp_threshold_db),
            "exp_ratio": (min_exp_ratio, max_exp_ratio),
            "at": (min_attack_ms, max_attack_ms),
            "rt": (min_release_ms, max_release_ms),
            "comp_gate": (0.0, 1.0), # gating
        }
        self.num_params = len(self.param_ranges)
 
 
def compressor_expander(
    x: torch.Tensor,
    sample_rate: float,
    comp_threshold: torch.Tensor,
    comp_ratio: torch.Tensor,
    exp_threshold: torch.Tensor,
    exp_ratio: torch.Tensor,
    at: torch.Tensor,
    rt: torch.Tensor,
    comp_gate: float,
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
 
    comp_threshold = torch.where(comp_gate > 0.5, comp_threshold, torch.tensor(0.0 + EPS).to(x.device))
    comp_ratio = torch.where(comp_gate > 0.5, comp_ratio, torch.tensor(1.0 + EPS).to(x.device))
    exp_threshold = torch.where(comp_gate > 0.5, torch.tensor(-80.0 + EPS).to(x.device), exp_threshold)
    exp_ratio = torch.where(comp_gate > 0.5, torch.tensor(1.0 - EPS).to(x.device), exp_ratio)
 
    # Apply compressor/expander gain
    x_out = x * torchcomp.compexp_gain(
        x.sum(axis=1).abs() + EPS, # for numerical stability
        comp_thresh=comp_threshold,
        comp_ratio=comp_ratio,
        exp_thresh=exp_threshold,
        exp_ratio=exp_ratio,
        at=torchcomp.ms2coef(at, sample_rate),
        rt=torchcomp.ms2coef(rt, sample_rate)
    ).unsqueeze(1)
 
    return x_out