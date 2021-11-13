import time
import math
import ipypb

import scipy.stats

import torch

import dynamic_strf.utils as utils


@torch.no_grad()
def dSTRF(model, x, chunk_size=100, verbose=False):
    """
    Estimate dynamic spectrotemporal receptive field (dSTRF) of a model for given input `x`.

    Arguments:
        mode: a pytorch model being analyzed. It should accept input of shape [time * in_channels]
            and return output of shape [time * out_channels].
        x: input of shape [time * in_channels].
        chunk_size: number of time samples to calculate dSTRF on simultaneously.
        verbose: a boolean, indicating whether to print out progress status.
    
    Returns:
        dstrfs: a dSTRF tensor of shape [time * out_channels * time_lag * in_channels].
    """
    # Pad input such that output has same shape and is half-precision
    context_size = model.receptive_field - 1
    x = torch.nn.functional.pad(x, (0, 0, context_size, 0)).half()
    def model_fx(x):
        with torch.cuda.amp.autocast():
            return model(x)[context_size:]
    
    dstrfs = []
    chunks = math.ceil((len(x) - context_size) / chunk_size)
    chunks = ipypb.irange(chunks) if verbose else range(chunks)
    for chunk_idx in chunks:
        chunk_start = chunk_idx * chunk_size
        chunk_length = min(chunk_size + context_size, len(x) - chunk_start)
        
        # Jacobian has shape [time * channel * lag * frequency]
        jacobian = torch.autograd.functional.jacobian(
            model_fx, x[chunk_start:chunk_start+chunk_length]
        ).cpu()
        
        dstrfs.append( # dSTRF is of shape [time * channel * lag * frequency]
            torch.stack([j[:, t:t+context_size+1] for t, j in enumerate(jacobian)], dim=0)
        )
        del jacobian
    
    return torch.cat(dstrfs, dim=0) # shape [time * channel * lag * frequency]

@torch.no_grad()
def complexity(dstrfs):
    """
    Measure general complexity of dSTRF function.

    Arguments:
        dstrfs: tensor of dSTRFs with shape [time * channel * lag * frequency]
    
    Returns:
        complexity: nonlinear function complexity, tensor of shape [channel]
    """
    tdim, cdim, ldim, fdim = dstrfs.shape
    dstrfs = dstrfs.transpose(0, 1).reshape(cdim, tdim, ldim*fdim)
    singular_vals = torch.linalg.svdvals(dstrfs.cpu())
    complexity = (singular_vals / singular_vals.max(dim=1, keepdims=True)[0]).sum(dim=1)
    return complexity

@torch.no_grad()
def gain_change(dstrfs):
    """
    Measure standard deviation of dSTRF gains.
    
    Arguments:
        dstrfs: tensor of dSTRFs with shape [time * channel * lag * frequency]
    
    Returns:
        gain_change: shape change parameter, tensor of shape [channel]
    """
    return dstrfs.norm(dim=[-2, -1]).std(dim=0).cpu()

@torch.no_grad()
def temporal_hold(dstrfs, lookahead=None, batch_size=1000):
    """
    Align dSTRFs to their adjacent neighbors by shifting to find how long an average dSTRF shifts along
    the time axis.
    
    Arguments:
        dstrfs: tensor of dSTRFs with shape [time * channel * lag * frequency]
        lookahead: the maximum amount of shift to consider
        batch_size: temporal batch size used for computing the shifts
    
    Returns:
        temporal_hold: temporal shift parameter, tensor of shape [channel]
    """
    tdim, cdim, ldim, _ = dstrfs.shape
    lookahead = max(ldim-10, min(ldim, 5)) if lookahead is None else lookahead
    dstrfs = torch.nn.functional.pad(dstrfs, (0, 0, 0, lookahead))

    # Shift all values, find best
    pval = torch.ones((lookahead+1, cdim), device=dstrfs.device)
    for lag in range(1, lookahead+1):
        dist0 = torch.zeros((tdim - lookahead, cdim))
        dist1 = torch.zeros((tdim - lookahead, cdim))
        
        for batch in range(math.ceil((tdim - lookahead)/batch_size)):
            batch_ind = slice(batch*batch_size, min((batch+1)*batch_size, tdim-lookahead))
            shift_ind = slice(batch*batch_size + lag, min((batch+1)*batch_size, tdim-lookahead) + lag)
            
            # Direct comparison
            dist0[batch_ind, :] = utils.corr(
                dstrfs[batch_ind],
                dstrfs[shift_ind],
                axis=[2, 3]
            )

            # Shift-corrected comparison
            dist1[batch_ind, :] = utils.corr(
                dstrfs[batch_ind],
                torch.nn.functional.pad(dstrfs[shift_ind, :, :-lag], (0, 0, lag, 0)),
                axis=[2, 3]
            )
        
        for c in range(cdim):
            pval[lag-1, c] = scipy.stats.wilcoxon(dist0[:, c], dist1[:, c], alternative='less').pvalue
    
    return utils.first_nonzero(pval >= 0.001, axis=0).cpu()

@torch.no_grad()
def shape_change(dstrfs, niter=100, span=None, batch_size=1000, return_shifts=False):
    """
    Align all dSTRFs to the global mean by shifting them along the lag axis.
    Compute shape change nonlinearity measure on the aligned dSTRFs.
    
    Arguments:
        dstrfs: tensor of dSTRFs with shape [time * channel * lag * frequency]
        niter: number of iterations
        span: the maximum shift allowed per iteration [-span, span]
        batch_size: temporal batch size used for computing the shifts
    
    Returns:
        shape_change: shape change parameter, tensor of shape [channel]
        dstrfs: shift-corrected and centered dSTRFs, returned if return_shifts is True
        shifts: Amount of shift used for each time point to achieve the final result,
            returned if return_shifts is True
    """
    tdim, cdim, ldim, fdim = dstrfs.shape
    span = max(ldim-10, min(ldim, 5)) if span is None else span
    dstrfs = torch.nn.functional.pad(dstrfs, (0, 0, span, span))
    
    max_shift = ldim//2 + span
    shifts = torch.zeros((tdim, cdim), dtype=int, device=dstrfs.device)
    
    t1 = time.time()
    for i in range(niter):
        dmean = dstrfs.mean(dim=0)
        shift = torch.zeros((tdim, cdim), dtype=int, device=dstrfs.device)
        for batch in range(math.ceil(tdim/batch_size)):
            batch_ind = slice(batch*batch_size, (batch+1)*batch_size)
            shift[batch_ind, :] = utils.find_shift(dstrfs[batch_ind], dmean, max_shift)
        
        for c in range(cdim):
            for k in range(-max_shift, max_shift+1):
                batch_ind = shift[:, c] == k
                dstrfs[batch_ind, c] = torch.roll(dstrfs[batch_ind, c], k, 1)
        
        power = dstrfs.abs().mean(dim=[0, 3])
        center = utils.smooth(power, 9).argmax(dim=1)
        
        for c in range(cdim):
            dstrfs[:, c] = torch.roll(dstrfs[:, c], int(max_shift-center[c]), 1)
            shifts[:, c] += shift[:, c] + max_shift - center[c]
        
        if (shift == 0).all():
            break
        else:
            print('Iteration #%d: %.4f average shift.' % (i+1, shift.float().mean().cpu()), flush=True)
    t2 = time.time()
    
    if not (shift == 0).all():
        print('Failed to converge to solution ({:s} elapsed).'.format(utils.timestr(t2-t1)))
    else:
        print('Converged in {:d} iterations ({:s} elapsed).'.format(i, utils.timestr(t2-t1)))
    
    power = dstrfs.abs().mean(dim=[0, 3])
    best_shift = torch.arange(-span, span+1)[
        torch.argmax(
            torch.stack([torch.roll(power, k, 1)[:, span:-span].sum(dim=1) for k in range(-span, span+1)]),
            dim=0
        )
    ]
    for c in range(cdim):
        dstrfs[:, c] = torch.roll(dstrfs[:, c], int(best_shift[c]), 1)
        shifts[:, c] += best_shift[c]
    dstrfs = dstrfs[:, :, span:-span]
    
    if return_shifts:
        return complexity(dstrfs), dstrfs.cpu(), shifts.cpu()
    else:
        return complexity(dstrfs)
