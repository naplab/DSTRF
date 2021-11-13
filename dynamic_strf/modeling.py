import time
import math
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
    
    @torch.no_grad()
    def test_step(self, val_batch, batch_idx):
        corr = torch.mean(torch.tensor([
            utils.corr(
                self(x.to(self.device)),
                y.to(self.device),
                axis=0
            ).mean(dim=0) for x, y in zip(*val_batch)
        ]))
        self.log('test_corr', corr, batch_size=len(val_batch[0]))


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
        super().__init__(input_size, hidden_size, channels, **kwargs)
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


class BranchedEncoder(BaseEncoder):
    def __init__(self, input_size, hidden_size=128, channels=1, **kwargs):
        """
        A 1D-convolutional neural activity encoder for multiple electrodes which has a shared part
        and a unique part for each electrode.

        Arguments:
            input_size: input channels, i.e., frequency bins of input audio spectrogram.
            hidden_size: number of kernels in each layer of the network.
            channels: number of output channels, i.e., electrodes being encoded.
        """
        super().__init__(input_size, hidden_size, channels, **kwargs)
        self.hidden_size = hidden_size
        
        self.conv_shared = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, hidden_size, 5, dilation=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_size, hidden_size, 5, dilation=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_size, hidden_size, 5, dilation=2, bias=False),
            torch.nn.ReLU(),
        )
        
        self.conv_branch = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(hidden_size, hidden_size, 5, dilation=4, bias=False),
                torch.nn.ReLU(),
                torch.nn.Conv1d(hidden_size, hidden_size, 5, dilation=8, bias=False),
                torch.nn.ReLU(),
                torch.nn.Conv1d(hidden_size, 1, 1, bias=True)
            ) for _ in range(channels)
        ])
    
    def forward(self, x):
        x = x.to(self.device)
        x = torch.nn.functional.pad(x, (0, 0, self.receptive_field-1, 0)).float()
        x = x.T.unsqueeze(dim=0)
        x = self.conv_shared(x)
        x = [m(x).squeeze(dim=0).T for m in self.conv_branch]
        x = torch.cat(x, dim=-1)
        return x
    
    @property
    def receptive_field(self):
        receptive_field = 1
        for m in self.conv_shared:
            if isinstance(m, torch.nn.Conv1d):
                receptive_field += (m.kernel_size[0] - 1) * m.dilation[0]
            elif not isinstance(m, torch.nn.ReLU):
                raise RuntimeError(f'Unsupported module {type(m)}')
        
        for m in self.conv_branch[0]:
            if isinstance(m, torch.nn.Conv1d):
                receptive_field += (m.kernel_size[0] - 1) * m.dilation[0]
            elif not isinstance(m, torch.nn.ReLU):
                raise RuntimeError(f'Unsupported module {type(m)}')
        
        return receptive_field


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
