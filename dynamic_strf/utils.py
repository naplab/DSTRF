import numpy as np
import torch


def timestr(seconds):
    """
    Convert seconds into seconds, minutes or hours
    """
    
    if seconds < 60:
        return '{:.2f} seconds'.format(seconds)
    
    minutes = seconds / 60
    if minutes < 60:
        return '{:.2f} minutes'.format(minutes)
    
    hours = minutes / 60
    return '{:.2f} hours'.format(hours)


@torch.no_grad()
def corr(a, b, axis=None):
    """
    Compute Pearson's correlation of `a` and `b` along specified axis.
    """
    a_mean = a.mean(axis=axis, keepdims=True)
    b_mean = b.mean(axis=axis, keepdims=True)
    a, b = (a - a_mean), (b - b_mean)
    
    a_sum2 = (a ** 2).sum(axis=axis, keepdims=True)
    b_sum2 = (b ** 2).sum(axis=axis, keepdims=True)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        a, b = (a / np.sqrt(a_sum2)), (b / np.sqrt(b_sum2))
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a, b = (a / torch.sqrt(a_sum2)), (b / torch.sqrt(b_sum2))
    else:
        raise TypeError(f'Incompatible types: {type(a)} and {type(b)}')
    
    return (a * b).sum(axis=axis)


@torch.no_grad()
def smooth(x, span):
    """
    Moving average smoothing with a filter of size `span`.
    """
    x = torch.nn.functional.pad(x, (span//2, span//2))
    x = torch.stack([torch.roll(x, k, -1) for k in range(-span//2, span//2+1)]).sum(dim=0) / span
    return x


@torch.no_grad()
def find_shift(x, ref, max_shift):
    """
    Find the optimal shift value to align `x` with `ref`.
    
    Arguments:
        x: dSTRF tensor with shape [time * channel * lag * frequency] to be aligned to reference.
        ref: dSTRF with shape [channel * lag * frequency], used as alignment reference.
        max_shift: maximum shift amount allowed in each direction.
    
    Returns:
        shifts: optimal shift values for each channel and time point, with shape [time * channel].
    """
    const = x.std(dim=[2, 3]) == 0
    klist = torch.arange(-max_shift, max_shift+1)
    x_shifts = torch.stack([torch.roll(x, int(k), 2) for k in klist], dim=4)
    
    r = corr(x_shifts, ref.unsqueeze(0).unsqueeze(4), axis=[2, 3])
    r[torch.isnan(r)] = -np.inf
    
    shifts = klist[torch.argmax(r, dim=2)]
    shifts[const] = 0
    
    return shifts


@torch.no_grad()
def first_nonzero(x, axis=0):
    """
    Finds first nonzero element of `x` along specified axis.
    """
    return ((x != 0).cumsum(dim=axis) == 1).int().argmax(dim=axis)
