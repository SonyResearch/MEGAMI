import torch

def compute_rms(x: torch.Tensor, **kwargs):
    """Compute root mean square energy.

    Args:
        x: (bs, 1, seq_len)

    Returns:
        rms: (bs, )
    """
    rms = torch.sqrt(torch.mean(x**2, dim=-1).clamp(min=1e-8))
    return rms

def compute_log_rms_gated_max(x: torch.Tensor, sample_rate=44100, **kwargs):
    """Compute gated log RMS energy.

    Frames the signal in 400 ms windows with 75% overlap, computes RMS,
    discards frames with RMS < -60 dBFS, and averages the log-RMS.

    If all frames in a given (batch, channel) are below -60 dBFS,
    returns -60 for that entry.

    Args:
        x: Tensor of shape (bs, c, seq_len)

    Returns:
        log_rms: Tensor of shape (bs, c)
    """
    seg_size = int(sample_rate * 0.4)
    hop_size = int(sample_rate * 0.1)

    # (bs, c, num_frames, seg_size)
    B, C, L = x.size()

    x_frames = x.unfold(2, seg_size, hop_size)  # (bs, c, num_frames, seg_size)

    # RMS over last dimension (seg_size)
    rms = torch.sqrt((x_frames ** 2).mean(dim=-1))  # (bs, c, num_frames)

    # dB conversion
    rms_db = 20 * torch.log10(rms.clamp(min=1e-8))  # (bs, c, num_frames)
    #print("rms db shape", rms_db.shape)

    #take the maximum RMS across all frames
    rms_max = rms_db.max(dim=2)[0]  # (bs, c)
    #print(f"RMS max shape: {rms_max.shape}")

    return rms_max


def compute_log_rms(x: torch.Tensor, **kwargs):
    """Compute root mean square energy.

    Args:
        x: (bs, 1, seq_len)

    Returns:
        rms: (bs, )
    """
    rms=compute_rms(x)
    return 20 * torch.log10(rms.clamp(min=1e-8))

def compute_crest_factor(x: torch.Tensor, **kwargs):
    """Compute crest factor as ratio of peak to rms energy in dB.

    Args:
        x: (bs, 2, seq_len)

    """
    num = torch.max(torch.abs(x), dim=-1)[0]
    den = compute_rms(x).clamp(min=1e-8)
    cf = 20 * torch.log10((num / den).clamp(min=1e-8))
    return cf

def compute_log_spread(x: torch.Tensor, **kwargs):
    """Compute log spread as the mean difference between log magnitude of samples and log RMS.
    
    Args:
        x: (bs, 1, seq_len)
        
    Returns:
        log_spread: (bs, )
    """
    # Compute log RMS
    log_rms = compute_log_rms(x).unsqueeze(-1)  # (bs, 1, 1)
    
    # Compute log magnitude of each sample
    log_magnitude = 20 * torch.log10(torch.abs(x).clamp(min=1e-8))  # (bs, 1, seq_len)
    
    # Compute the difference and take the mean
    log_spread = torch.mean(log_magnitude - log_rms, dim=-1).squeeze(1)  # (bs, )
    
    return log_spread


def compute_stereo_width(x: torch.Tensor, **kwargs):
    """Compute stereo width as ratio of energy in sum and difference signals.

    Args:
        x: (bs, 2, seq_len)

    """
    bs, chs, seq_len = x.size()

    assert chs == 2, "Input must be stereo"

    # compute sum and diff of stereo channels
    x_sum = x[:, 0, :] + x[:, 1, :]
    x_diff = x[:, 0, :] - x[:, 1, :]

    # compute power of sum and diff
    sum_energy = torch.mean(x_sum**2, dim=-1)
    diff_energy = torch.mean(x_diff**2, dim=-1)

    # compute stereo width as ratio
    stereo_width = diff_energy / sum_energy.clamp(min=1e-8)

    return stereo_width


def compute_stereo_imbalance(x: torch.Tensor, **kwargs):
    """Compute stereo imbalance as ratio of energy in left and right channels.

    Args:
        x: (bs, 2, seq_len)

    Returns:
        stereo_imbalance: (bs, )

    """
    left_energy = torch.mean(x[:, 0, :] ** 2, dim=-1)
    right_energy = torch.mean(x[:, 1, :] ** 2, dim=-1)

    stereo_imbalance = (right_energy - left_energy) / (
        right_energy + left_energy
    ).clamp(min=1e-8)

    return stereo_imbalance
