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

import dynamic_strf.estimate as estimate
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
    
    def fit_on_subset(self, leave_out_idx):
        """
        Fit model on training data, after leaving out trials indicated by `leave_out_idx`.

        Arguments:
            leave_out_idx: list of indices of trials to leave out from training.
        """
        if self.builder is None:
            # If builder not provided, nothing to train
            return
        
        fname = f"model-{'_'.join([str(i) for i in leave_out_idx])}.pt"
        fpath = os.path.join(self.store_loc, fname)
        if os.path.exists(fpath):
            # If trained model exists, skip retraining
            return
        
        # Initialize model
        model = self.builder().to(self.device)

        # Initialize training dataloader
        train_dataloader = Dataset(
            [x for i, x in enumerate(self.data[0]) if i not in leave_out_idx],
            [y for i, y in enumerate(self.data[1]) if i not in leave_out_idx],
        ).iterator(batch_size=self.batch_size, num_workers=self.num_workers)
        
        # Initialize validation dataloader
        if self.validation:
            val_dataloader = Dataset(
                [x for i, x in enumerate(self.validation[0])],
                [y for i, y in enumerate(self.validation[1])]
            ).iterator(batch_size=self.batch_size, num_workers=self.num_workers)
        else:
            val_dataloader = Dataset(
                [x for i, x in enumerate(self.data[0]) if i in leave_out_idx],
                [y for i, y in enumerate(self.data[1]) if i in leave_out_idx]
            ).iterator(batch_size=self.batch_size, num_workers=self.num_workers)
        
        # Initialize trainer
        if self.trainer:
            trainer = self.trainer()
        else:
            trainer = pl.Trainer(
                gpus=self.use_gpus,
                precision=self.precision,
                gradient_clip_val=10.0,
                max_epochs=1000,
                logger=False,
                detect_anomaly=True,
                enable_model_summary=(self.verbose >= 2),
                enable_progress_bar=(self.verbose >= 2),
                enable_checkpointing=False,
                callbacks=[]
            )
        
        # Fit model on train split
        trainer.fit(
            model,
            train_dataloader,
        )
        
        # Test model on left-out split
        r = trainer.test(
            model,
            val_dataloader,
        )[0]['test_corr']
        print(f'r={r:0.4f} -- ', flush=True, end='')
        
        torch.save(model.state_dict(), fpath)
    
    def load_instance(self, leave_out_idx):
        """
        Load a model trained after leaving out trials indicated in `leave_out_idx`.

        Arguments:
            leave_out_idx: list of indices of trials that were left out from training.
        """
        if self.builder is None:
            model = self.model
        else:
            model = self.builder()
            fname = f"model-{'_'.join([str(i) for i in leave_out_idx])}.pt"
            fpath = os.path.join(self.store_loc, fname)
            model.load_state_dict(torch.load(fpath))
        
        return model.to(self.device).eval()
    
    def instance_dstrf(self, leave_out_idx, x, chunk_size=100):
        """
        Estimate dSTRFs on input `x`, from models that were trained by leaving out `leave_out_idx` trials.

        Arguments:
            leave_out_idx: list of indices of trials that were left out from training.
            x: input of shape [time * frequency] to calculate dSTRFs for.
            chunk_size: number of time samples to calculate dSTRF simultaneously on.
        """
        # Load trained model and move to `device`
        model = self.load_instance(leave_out_idx)
        
        # Compute dSTRF for input `x`
        return estimate.dstrf(model, x.to(self.device), chunk_size=chunk_size, verbose=self.verbose >= 2)
    
    def estimate_dstrfs(self):
        """
        Estimate dSTRFs for all validation data based on all trained models that had not seen that data
        during training.
        """
        if self.validation and not self.jackknife:
            instances = [[None]]
        elif self.validation is None and self.jackknife:
            instances = [[i, j] for i in range(len(self.data[0])) for j in range(i+1, len(self.data[0]))]
        else:
            instances = [[i] for i in range(len(self.data[0]))]
        
        for leave_out_idx in instances:
            if self.verbose >= 1:
                print(f"Fitting model for leave out: [{', '.join([str(i) for i in leave_out_idx])}]... ", flush=True, end='')
            self.fit_on_subset(leave_out_idx)
            print('DONE', flush=True)
        
        dstrfs = []
        for i, x in enumerate(self.validation[0] if self.validation else self.data[0]):
            if self.verbose >= 1:
                print(f"Computing dSTRFs for stimulus {i+1:02d}/{len(self.validation[0] if self.validation else self.data[0]):02d}... ", flush=True, end='')
            
            path = os.path.join(self.store_loc, f"dSTRF-{i:03d}.pt")
            if os.path.exists(path):
                dstrf = torch.load(path)
            elif self.validation:
                dstrf = torch.stack([
                    self.instance_dstrf(leave_out_idx, x) for leave_out_idx in instances
                ]).mean(dim=0)
            else:
                dstrf = torch.stack([
                    self.instance_dstrf(leave_out_idx, x) for leave_out_idx in instances if i in leave_out_idx
                ]).mean(dim=0)
            
            dstrfs.append(dstrf)
            if not os.path.exists(path):
                torch.save(dstrf, path)
            print('DONE', flush=True)
        
        return dstrfs
    
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
