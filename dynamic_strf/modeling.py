import os
import time
import math
import glob
import ipypb
from typing import Tuple

import numpy as np
import torch
import radam
import torchaudio
import pytorch_lightning as pl

import dynamic_strf.utils as utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None):
        """
        A simple dataset class, which takes list of inputs `x`, and outputs `y`. Each item in the
        list is considered a separate trial, such that the i-th element of `x` is the stimulus of
        the i-th trial, and the i-th element of `y` is the response of the i-th trial.

        Arguments:
            x: a list of inputs, each having shape [time * in_channels]
            y: a list of outputs, each  having shape [time * out_channels]
        """
        super().__init__()
        self._x = x
        self._y = y
    
    def __getitem__(self, idx):
        return self._x[idx], (self._y[idx] if self._y else None)
    
    def __len__(self):
        return len(self._x)
    
    def iterator(self, batch_size=64, num_workers=4):
        def collate_fn(xys):
            batch_size = len(xys)
            xs = np.ndarray(batch_size, dtype=object)
            ys = np.ndarray(batch_size, dtype=object)
            for i, (x, y) in enumerate(xys):
                xs[i] = x
                if self._y:
                    ys[i] = y
            return xs, ys
        
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers
        )


class SpectrogramParser(torch.nn.Sequential):
    def __init__(self, in_sr, out_sr, freqbins=64, f_min=20, f_max=8000, top_db=70, normalize=False):
        """
        A waveform to Mel-spectrogram parser.

        Arguments:
            in_sr: sampling rate of input waveform.
            out_sr: sampling rate of output spectrogram.
            freqbins: number of frequency bins of output spectrogram.
            f_min: minimum frequency of the spectrogram.
            f_max: maximum frequency of the spectrogram.
            top_db: the maximum decibel range between the highest and lowest spectrotemporal bins.
            normalize: whether to normalize the spectrogram power.
        """
        super().__init__(
            torchaudio.transforms.MelSpectrogram(
                in_sr, n_fft=1024, hop_length=int(in_sr/out_sr),
                f_min=f_min, f_max=f_max, n_mels=freqbins, power=2.0
            ),

            torchaudio.transforms.AmplitudeToDB(
                'power', top_db=top_db
            ),

            type("Normalize", (torch.nn.Module,), dict(
                forward=lambda self, x: (x - x.max()).squeeze(0).T.float() / top_db + 1
            ))() if normalize else type("Squeeze", (torch.nn.Module,), dict(
                forward=lambda self, x: x.squeeze(0).T.float()
            ))()
        )


class BaseEncoder(pl.LightningModule):
    def __init__(self, input_size, channels=1, optimizer_cfg={}, scheduler_cfg={}):
        """
        An abstract neural activity encoder for multiple electrodes.

        Arguments:
            input_size: input channels, i.e., frequency bins of input audio spectrogram.
            channels: number of output channels, i.e., electrodes being encoded.
            optimizer_cfg: a dictionary containing configuration for an RAdam optimizer.
            scheduler_cfg: a dictionary containing configuration for an exponential learning rate
                decay scheduler.
        """
        super().__init__()
        self.input_size = input_size
        self.channels = channels
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self._device = 'cpu'

        self.loss = torch.nn.MSELoss(reduction='mean')
    
    def to(self, device):
        super().to(device)
        self._device = device
        return self
    
    def forward(self, x):
        raise NotImplementedError()
    
    @property
    def device(self):
        return self._device
    
    def configure_optimizers(self):
        optimizer = radam.RAdam(self.parameters(), **{'lr': 0.003, 'weight_decay': 0.03, **self.optimizer_cfg})
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **{'gamma': 0.996, **self.scheduler_cfg})
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    
    def training_step(self, train_batch, batch_idx):
        loss = torch.mean(torch.stack([
            self.loss(
                self(x.to(self.device)),
                y.to(self.device)
            ) for x, y in zip(*train_batch)
        ]))
        return loss


class SharedEncoder(BaseEncoder):
    def __init__(self, input_size, hidden_size=128, channels=1, **kwargs):
        """
        A 1D-convolutional neural activity encoder for multiple electrodes which shares all hidden
        layers, except the final readout, between all electrodes.

        Arguments:
            input_size: input channels, i.e., frequency bins of input audio spectrogram.
            hidden_size: number of kernels in each layer of the network.
            channels: number of output channels, i.e., electrodes being encoded.
        """
        super().__init__(input_size, channels, **kwargs)
        self.hidden_size = hidden_size
        
        self.conv_shared = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, hidden_size, 5, dilation=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_size, hidden_size, 5, dilation=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_size, hidden_size, 5, dilation=2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_size, hidden_size, 5, dilation=4, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_size, hidden_size, 5, dilation=8, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_size, channels, 1, bias=True)
        )
    
    def forward(self, x):
        x = x.to(self.device)
        x = torch.nn.functional.pad(x, (0, 0, self.receptive_field-1, 0)).float()
        x = x.T.unsqueeze(dim=0)
        x = self.conv_shared(x)
        x = x.squeeze(dim=0).T
        return x
    
    @property
    def receptive_field(self):
        receptive_field = 1
        for m in self.conv_shared:
            if isinstance(m, torch.nn.Conv1d):
                receptive_field += (m.kernel_size[0] - 1) * m.dilation[0]
            elif not isinstance(m, torch.nn.ReLU):
                raise RuntimeError(f'Unsupported module {type(m)}')
        
        return receptive_field


def fit(model, data, trainer=None, leave_out_idx=[], batch_size=64, num_workers=4, gpus=1, precision=16, verbose=0):
    """
    """
    # Initialize training dataloader
    dataloader = Dataset(
        [x for i, x in enumerate(data[0]) if i not in leave_out_idx],
        [y for i, y in enumerate(data[1]) if i not in leave_out_idx],
    ).iterator(batch_size=batch_size, num_workers=num_workers)
    
    # Initialize trainer
    if not trainer:
        trainer = pl.Trainer(
            gpus=gpus,
            precision=precision,
            gradient_clip_val=10.0,
            max_epochs=1000,
            logger=False,
            detect_anomaly=True,
            enable_model_summary=(verbose >= 2),
            enable_progress_bar=(verbose >= 2),
            enable_checkpointing=False,
            callbacks=[]
        )
    
    # Fit model on train split
    trainer.fit(
        model,
        dataloader,
    )
    
    return model


def fit_multiple(builder, data, crossval=False, jackknife=False, trainer=None, save_dir=None, verbose=0, **kwargs):
    """
    """
    if save_dir is None:
        raise ValueError('Parameter `save_dir` cannot be empty.')
    
    if verbose >= 1 and os.path.exists(save_dir):
        print(f'Directory "{save_dir}" already exists.', flush=True)
    os.makedirs(save_dir, exist_ok=True)
    
    instances = utils.leave_out_indices(len(data[0]), crossval=crossval, jackknife=jackknife)
    
    checkpoints = []
    for leave_out_idx in instances:
        fpath = os.path.join(save_dir, utils.checkpoint_from_leave_out(leave_out_idx))
        checkpoints.append(fpath)
        
        if verbose >= 1:
            print(f"Fitting model for leave out: [{', '.join([str(i) for i in leave_out_idx])}]... ", flush=True, end='')
        
        if os.path.exists(fpath):
            # If trained model exists, skip
            print('Skip.', flush=True)
            continue
        elif verbose >= 2:
            print(flush=True)
        
        # Initialize model
        model = builder()
        
        # Fit model
        fit(
            model=model,
            data=data,
            trainer=trainer() if trainer else None,
            leave_out_idx=leave_out_idx,
            verbose=verbose,
            **kwargs
        )
        
        # Save model weights
        torch.save(model.state_dict(), fpath)
        
        if verbose >= 1:
            print('Done.', flush=True)
    
    return checkpoints


@torch.no_grad()
def test_jackknife(model, checkpoints, data, jackknife_mode='pred'):
    x = data[0].to(model.device)
    y = data[1].to(model.device)
    
    preds = []
    for checkpoint in checkpoints:
        model.load_state_dict(torch.load(checkpoint))
        preds.append(model(x))
    
    if jackknife_mode == 'pred':
        preds = torch.mean(torch.stack(preds, dim=0), dim=0)
        scores = utils.corr(preds, y, axis=0)
    elif jackknife_mode == 'score':
        scores = [utils.corr(pred, y, axis=0) for pred in preds]
        scores = torch.mean(torch.stack(scores, dim=0), dim=0)
    else:
        raise ValueError('Parameter `jackknife_mode` should be one of "pred" or "score".')
    
    return scores.cpu()


@torch.no_grad()
def test_multiple(model, checkpoints, data, crossval=False, jackknife_mode='pred', verbose=0):
    if os.path.isdir(checkpoints):
        checkpoints = sorted(glob.glob(os.path.join(checkpoints, 'model-*.pt')))
        if verbose >= 1:
            print(f'Found {len(checkpoints)} model checkpoints in specified directory.', flush=True)
    
    scores = []
    iterator = ipypb.irange(len(data[0])) if verbose >= 1 else range(len(data[0]))
    for i in iterator:
        if crossval:
            checkpoints_i = [ckpt for ckpt in checkpoints if i in utils.leave_out_from_checkpoint(ckpt)]
        else:
            checkpoints_i = checkpoints
        
        scores.append(
            test_jackknife(
                model=model,
                checkpoints=checkpoints_i,
                data=(data[0][i], data[1][i]),
                jackknife_mode=jackknife_mode,
            )
        )
    
    return torch.stack(scores, dim=0)


@torch.no_grad()
def infer_jackknife(model, checkpoints, data, jackknife_mode='pred'):
    x = data.to(model.device)
    
    preds = []
    for checkpoint in checkpoints:
        model.load_state_dict(torch.load(checkpoint))
        preds.append(model(x))
    
    if jackknife_mode == 'pred':
        preds = torch.mean(torch.stack(preds, dim=0), dim=0)
    else:
        raise ValueError('Parameter `jackknife_mode` should be one of "pred" or "score".')
    
    return preds.cpu()


@torch.no_grad()
def infer_multiple(model, checkpoints, data, crossval=False, jackknife_mode='pred', verbose=0):
    if os.path.isdir(checkpoints):
        checkpoints = sorted(glob.glob(os.path.join(checkpoints, 'model-*.pt')))
        if verbose >= 1:
            print(f'Found {len(checkpoints)} model checkpoints in specified directory.', flush=True)
    
    preds = []
    iterator = ipypb.irange(len(data)) if verbose >= 1 else range(len(data))
    for i in iterator:
        if crossval:
            checkpoints_i = [ckpt for ckpt in checkpoints if i in utils.leave_out_from_checkpoint(ckpt)]
        else:
            checkpoints_i = checkpoints
        
        preds.append(
            infer_jackknife(
                model=model,
                checkpoints=checkpoints_i,
                data=data[i],
                jackknife_mode=jackknife_mode,
            )
        )
    
    return preds
