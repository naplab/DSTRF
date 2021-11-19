import os
import time
import ipypb
import string
import shutil
import ffmpeg
import numpy as np
import matplotlib.pyplot as plt

import torch


def _tempfile(loc='/tmp'):
    """
    Generate name for a temporary directory.
    """
    letters = ''.join(np.random.choice(list(string.ascii_letters), size=6))
    numbers = time.time_ns() % 1000_000
    path = os.path.join(loc, f'temp-{letters:s}-{numbers:06d}')

    return tempfile(loc) if os.path.exists(path) else path


def dSTRF(path, channels=slice(None), time_range=slice(None), figsize=(4, 4), aspect=2, cmap='bwr', vquant=0.90, output_prefix='./', verbose=0, **kwargs):
    """
    """
    # Load dSTRF from file and filter channels and time samples
    dstrf = torch.load(path)[time_range, channels, :, :]
    
    # If prefix leads to different directory, create if doesn't exist
    if '/' in output_prefix:
        parent_dir = '/'.join(output_prefix.split('/')[:-1])
        os.makedirs(parent_dir, exist_ok=True)
    
    # Loop over channels
    for c in (ipypb.irange if verbose >=1 else range)(dstrf.shape[1]):
        # Find color axis range
        vmax = dstrf[:,c].abs().max(dim=1)[0].max(dim=1)[0].float().quantile(vquant)
        
        # Make temporary directory to store frames
        tmp_dir = _tempfile('.')
        os.makedirs(tmp_dir, exist_ok=False)
        
        # Plot all frames for a channel
        for t in (ipypb.irange if verbose >=2 else range)(dstrf.shape[0]):
            plt.figure(
                figsize=figsize
            )
            
            plt.imshow(
                dstrf[t,c].float().T,
                aspect=aspect,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                origin='lower'
            )
            
            if 'xlabel' in kwargs:
                plt.xlabel(kwargs['xlabel'])
            
            if 'ylabel' in kwargs:
                plt.ylabel(kwargs['ylabel'])
            
            if 'xticks' in kwargs and 'xtick_labels' in kwargs:
                plt.xticks(kwargs['xticks'], kwargs['xtick_labels'])
            elif 'xticks' in kwargs:
                plt.xticks(kwargs['xticks'])
            
            if 'yticks' in kwargs and 'ytick_labels' in kwargs:
                plt.yticks(kwargs['yticks'], kwargs['ytick_labels'], fontsize=11)
            elif 'yticks' in kwargs:
                plt.yticks(kwargs['yticks'])
            
            plt.savefig(os.path.join(tmp_dir, f'channel-{c:04d}-frame-{t:04d}.png'))
            plt.close()
        
        # Make frames into video
        ffmpeg.input(
            os.path.join(tmp_dir, f'channel-{c:04d}-frame-%04d.png'), framerate=100
        ).output(
            f'{output_prefix}channel-{c:04d}.mp4', r=60, pix_fmt='yuv420p'
        ).run(
            overwrite_output=True
        )
        
        # Remove temporary directory
        shutil.rmtree(tmp_dir)
