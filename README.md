# DSTRF
A toolbox for dynamic spectrotemporal receptive field (dSTRF) analysis, as introduced in <a href="https://elifesciences.org/articles/53445">"Estimating and interpreting nonlinear receptive field of sensory neural responses with deep neural network models"</a>. In short, dSTRF is a method to analyze which parts of the input at any time point are being used by a feed-forward deep neural network to predict the response.

## Usage

There are two ways to use this toolbox:

1. Functional: This case entails using separate functions to train the model, estimate the dSTRFs from the model, and quantify the degree of nonlinearity of the estimated dSTRFs. All the main functions for the analysis are located in the <a href="https://github.com/naplab/DSTRF/blob/master/dynamic_strf/estimate.py">dynamic_strf.estimate</a> module.

2. Unified: In this case, we use an all-in-one <a href="https://github.com/naplab/DSTRF/blob/master/dynamic_strf/analyzer.py">Analyzer</a> class that takes a model and data, and trains the model on the data, estimates the dSTRFs, and computes the dSTRF nonlinearities. This class enables the following robustness funcionalities:

    1. Cross-validation: If separate validation data is not provided, the analyzer works in a cross-validation regime, where dSTRFs for each data trial are computed from a model trained on all the other trials. This means if the data consists of N trials, there will be N models trained.

    2. Data jackknifing: If jackknifing is enabled, the training set (after leaving out the validation if necessary), is jackknifed, leaving out one trial at a time and training the model on the remaining trials. This means if the data consists of N trials and jackknifing is enabled, (a) if separate validation data is provided, N models will be trained, (b) if cross-validating, N*(N-1)/2 models will be trained, i.e., leaving out two trials at a time (one as a validation trial, the other from the jackknifing procedure).

## Installation

To install this package through pip, run the following command:

`pip install git+https://github.com/naplab/DSTRF.git`

You can use the same command to update the package.

## To-do

- Add a jupyter notebook demonstrating the application of the toolbox on a toy example.
- Make jackknifing also accessible in a functional format.
