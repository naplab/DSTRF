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


def _slice_baskets(items, maxbaskets):
    '''
    Slice a list into a certain number of groups and minimizes the variance
    in group length. For example, a list of length 21 and maxbaskets=5 will
    get grouped into 1 group of 5 and 4 groups of 4, rather than 4 groups of 5
    and 1 group of 1.
    '''
    n_baskets = min(maxbaskets, len(items))
    return [items[i::n_baskets] for i in range(n_baskets)]

def leave_out_indices(num_trials, crossval=False, jackknife=False):
    if not crossval and not jackknife:
        instances = [[]]
    elif crossval and jackknife:
        if isinstance(jackknife, bool):
            instances = [[i, j] for i in range(num_trials) for j in range(i+1, num_trials)]
        elif isinstance(jackknife, int):
            instances = []
            for i in range(num_trials):
                other_trials = [x for x in range(num_trials)]
                for other_trial_subset in _slice_baskets(other_trials, jackknife):
                    subset = set([i] + other_trial_subset)
                    if subset not in instances:
                        instances.append(subset)
            instances = [list(x) for x in instances]
        else:
            raise TypeError(f'jackknife must be either a bool or an int specifying '
                            f'the number of jackknife groups, but got a {type(jackknife)}')
    else:
        if isinstance(jackknife, bool):
            instances = [[i] for i in range(num_trials)]
        elif isinstance(jackknife, int):
            lst = [i for i in range(num_trials)]
            n = jackknife
            instances = _slice_baskets(lst, n)
        else:
            raise TypeError(f'jackknife must be either a bool or an int specifying '
                            f'the number of jackknife groups, but got a {type(jackknife)}')
    
    return instances


def checkpoint_from_leave_out(leave_out_idx):
    if leave_out_idx:
        return f"model-{'_'.join(['%03d' % i for i in leave_out_idx])}.pt"
    else:
        return 'model-all.pt'


def leave_out_from_checkpoint(checkpoint):
    if checkpoint == 'model-all.pt':
        return []
    else:
        return [int(i) for i in checkpoint.split('-')[-1].split('.')[0].split('_')]
