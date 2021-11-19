# DSTRF

A toolbox for dynamic spectrotemporal receptive field (dSTRF) analysis, as introduced in <a href="https://elifesciences.org/articles/53445">"Estimating and interpreting nonlinear receptive field of sensory neural responses with deep neural network models"</a>. In short, dSTRF is a method to analyze which parts of the input at any time point are being used by a feed-forward deep neural network to predict the response.

In mathematical terms, for a nonlinear model ***f(⋅)***, input at time ***t***, ***X<sub>t</sub>***, and output at time ***t***, ***Y<sub>t</sub> = f(X<sub>t</sub>)***, the dSTRF is the locally linear transformation ***W<sub>t</sub>***, such that: ***Y<sub>t</sub> = W<sub>t</sub> X<sub>t</sub>***. Now, if ***f(⋅)*** is a linear function, ***W<sub>t</sub>*** will be the same for all ***t***. If not, the linearized function applied to the input (***W<sub>t</sub>***) will depend on the given stimulus (***X<sub>t</sub>***), which is the case in a deep neural network.

A linear function mapping the auditory stimuli in the spectrotemporal domain (i.e., auditory spectrograms) to the neural activity recorded from the biological brain is called a spectrotemporal receptive field (STRF). Since we use a nonlinear mapping between the stimulus and response, and characterize it as a locally linear function at each time point, we call this a dynamic spectrotemporal receptive field (dSTRF).

Note that for the equality condition ***Y<sub>t</sub> = W<sub>t</sub> X<sub>t</sub>*** to hold for all ***t***, the intermediate layers should have no bias term, and ReLU or LeakyReLU activations. In this case, the ***Y<sub>t</sub> / X<sub>t</sub>*** is equivalent to the derivative of ***Y*** with respect to ***X***, at ***X<sub>t</sub>***, i.e., the Jacobian matrix of function ***f(⋅)***. If the above conditions are not satisfied, the Jacobian will only be approximating ***Y<sub>t</sub> / X<sub>t</sub>***.

## Usage

This toolbox has two main modules:

1. <a href="https://github.com/naplab/DSTRF/blob/master/dynamic_strf/modeling.py">**modeling**</a>: This module contains functions to fit and evaluate neural activity encoding models. The training supports cross-validation, and data jackknifing. It also contains two sample Encoder classes that map audio spectrogram to neural activity&mdash; a linear model, and a deep convolutional model.

2. <a href="https://github.com/naplab/DSTRF/blob/master/dynamic_strf/estimate.py">**estimate**</a>: This module contains all functions necessary to compute dSTRFs and analyze their nonlinear characteristics. The dSTRF estimation supports cross-validation and data jackknifing to be used with cross-validated and jackknifed training introduced in the above module.

All the main functionality of the toolbox, along with some examples is demonstrated in the <a href="https://nbviewer.org/github/naplab/DSTRF/blob/master/Examples/Tutorial.ipynb">Example</a> notebook.

## Installation

To install or update this package through pip, run the following command:

`pip install git+https://github.com/naplab/DSTRF.git`

## To-do

- Test for possible bugs
- Fix code documentation
