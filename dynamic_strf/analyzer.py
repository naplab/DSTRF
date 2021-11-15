import os
import time
import math
import ipypb
import string

import numpy as np
import scipy as sp
import scipy.stats

import torch
import pytorch_lightning as pl

import dynamic_strf.utils as utils
import dynamic_strf.estimate as estimate
import dynamic_strf.modeling as modeling
from dynamic_strf.modeling import Dataset


class Analyzer:
    def __init__(
            self, /,
            model = None,
            builder = None,
            trainer = None,
            data = None,
            validation = None,
            jackknife = False,
            store_loc = None,
            use_gpus = 0,
            precision = 16,
            device = 'cpu',
            verbose = 1,
            batch_size = 64,
            num_workers = 4
        ):
        """
        An all-in-one dynamic spectrotemporal receptive field (dSTRF) analyzer, with jackknifing support.

        model: target model to analyze.
        builder: a function that returns a newly initialized model when called.
        trainer: a function that returns a pytorch_lightning.Trainer object, used for training.
        data: data used to both train the model, and run the dSTRF analysis.
        validation: data to run dSTRF analysis on. None for cross-validating on training data.
        jackknife: whether to use data jackknifing for training the models.
        store_loc: path to directory where trained models and computed dSTRFs will be saved.
        use_gpus: number of GPUs to use for training.
        precision: numeral precision to use during training, one of 16 or 32.
        device: pytorch device to do all analysis on.
        verbose: whether to print out progress status.
        batch_size: batch size used during training.
        num_workers: number of workers used for data preparation.
        """
        self.model = model
        self.builder = builder
        self.trainer = trainer
        self.data = data
        self.validation = validation
        self.jackknife = jackknife
        self.use_gpus = use_gpus
        self.precision = precision
        self.device = device
        self.verbose = verbose
        self.batch_size = batch_size
        self.num_workers = num_workers

        if int(model is None) + int(builder is None) != 1:
            raise RuntimeError('Exactly one of model or builder should be specified.')
        if builder and not callable(builder):
            raise RuntimeError('Builder has to be a callable that returns an initialized model.')
        if jackknife and data is None:
            raise RuntimeError('Data jaccknifing can only be used with data.')
        
        if store_loc:
            self.store_loc = store_loc
            os.makedirs(self.store_loc, exist_ok=True)
        else:
            def tempdir():
                """Generate name for a temporary directory."""
                letters = ''.join(np.random.choice(list(string.ascii_letters), size=6))
                numbers = time.time_ns() % 1000_000
                path = f'{letters:s}-{numbers:06d}'

                return tempdir() if os.path.exists(path) else path
            
            self.store_loc = tempdir()
            os.makedirs(self.store_loc, exist_ok=False)
        
        if verbose >= 1:
            print(f'II: Storing results under ./{self.store_loc}/')
        
        self._nonlinearity = None
    
    def estimate_dstrfs(self, chunk_size=100):
        """
        Estimate dSTRFs for all validation data based on all trained models that had not seen that data
        during training.
        """
        checkpoints = modeling.fit_multiple(
            builder=self.builder,
            data=self.data,
            crossval=self.validation is None,
            jackknife=self.jackknife,
            save_dir=self.store_loc,
            trainer=self.trainer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            gpus=self.use_gpus,
            precision=self.precision,
            verbose=self.verbose
        )

        # modeling.evaluate(model, self.validation if self.validation else self.data)
        
        estimate.dSTRF_multiple(
            model=self.builder() if self.builder else self.model,
            checkpoints=checkpoints,
            data=self.validation[0] if self.validation else self.data[0],
            crossval=self.validation is None,
            jackknife=self.jackknife,
            save_dir=self.store_loc,
            chunk_size=chunk_size,
            verbose=self.verbose
        )
    
    @property
    def nonlinearity(self):
        """
        A dictionary of nonlinearity measure computed for the model being analyzed.
        """
        return self._nonlinearity.copy() if self._nonlinearity else None
    
    def compute_nonlinearities(self, per_trial=False):
        """
        Compute all introduced nonlinearities for the model being analyzed on the validation data.

        Arguments:
            per_trial: if set to True, returns separate nonlinearity values for each trial, otherwise
                returns the mean across trials.
        """
        nonliniearity = {
            'complexity': [],
            'gain change': [],
            'temporal hold': [],
            'shape change': []
        }

        for i, x in enumerate(self.validation[0] if self.validation else self.data[0]):
            if self.verbose >= 1:
                print(f"Estimating dSTRF nonlinearities for stimulus {i+1:02d}/{len(self.validation[0] if self.validation else self.data[0]):02d}... ", flush=True, end='')
            
            fpath = os.path.join(self.store_loc, f"dSTRF-{i:03d}.pt")
            dstrf = torch.load(fpath).to(self.device)
            
            nonliniearity['complexity'].append(estimate.complexity(dstrf))
            nonliniearity['gain change'].append(estimate.gain_change(dstrf))
            nonliniearity['temporal hold'].append(estimate.temporal_hold(dstrf))
            nonliniearity['shape change'].append(estimate.shape_change(dstrf))
            print('DONE', flush=True)
        
        for k in nonliniearity:
            nonliniearity[k] = torch.stack(nonliniearity[k], dim=1)
            if not per_trial:
                nonliniearity[k] = nonliniearity[k].mean(dim=1)
        
        self._nonlinearity = nonliniearity
        
        return self.nonliniearity
