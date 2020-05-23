import os
import sys
import time
import ipypb
import logging
import warnings
from collections import OrderedDict
from multiprocessing import Pool

import pickle
import numpy as np
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from hdf5storage import savemat


def drop_nan(response, prediction):
    mask = tf.math.is_finite(response)
    return tf.boolean_mask(response, mask), tf.boolean_mask(prediction, mask)

def loss_se(response, prediction):
    """Squared error loss."""
    response, prediction = drop_nan(response, prediction)
    num = tf.reduce_mean(tf.square(response - prediction))
    den = tf.reduce_mean(tf.square(response - tf.reduce_mean(response)))
    return num / den

# Correlation metric
def metric_corr(resp, pred):
    resp, pred = drop_nan(resp, pred)
    resp, pred = tf.expand_dims(resp, 0), tf.expand_dims(pred, 0)
    return tfp.stats.correlation(resp, pred, 1, 0)

# Prepad in time
def pad(x, n):
    return np.pad(x, ((n-len(x), 0), (0, 0))) if len(x)<n else x


# path to data file

dbase_dir = "/tf/data/"
dbase_name = "sample_data.mat"

# load data

data = load_data() # insert data loading code

# prepare stimulus for computing DSTRF on
#     X has shape ( time x freqs )

X = data['stim']

# select model for single channel

channel = 0
print(f"Loading models for channel {channel:d}...")

# batch data if needed
#     new X has shape ( batch x time x freqs )

batch = False
if batch:
    # batch data into continuous chunks
    pass
else:
    # use full data as single batch
    X = X[np.newaxis]

# stimulus frequency channels
freqbins = est['stim'].shape[0]

# base path for loading model
model_dir = f"models/"
model_name = f"mdl-v1-{channel:03d}"
model_base = os.path.join(model_dir, model_name)

# compute DSTRF for trained models on jackknifed dataset
#     dstrf has shape ( njacks x time x lag x freqs )
#     - consider running on small chunks (500 samples) of data at a
#       time and concatenating results to improve calculation speed

dstrf = []
njacks = 20
for r in ipypb.irange(njacks):
    model = tf.keras.models.load_model(f"models/{model_base}-{r:d}.h5",
                                       custom_objects={"loss_se": loss_se, "metric_corr": metric_corr})
    
    x = tf.Variable(X)
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        z = model(x)
    dz_dx = g.jacobian(z, x).numpy()
    dz_dx = np.stack([pad(dz_dx[i, max(0, i-39):i+1, :], 40) for i in range(len(dz_dx))], 0)
    dstrf.append(dz_dx)

dstrf = np.stack(dstrf, 0)

# compute average and significance bounds on DSTRFs
#     dstrf_* have shape ( time x lag x freqs )

dstrf_mean = dstrf.mean(0)
dstrf_05, dstrf_95 = np.quantile(dstrf, [0.05, 0.95], 0)

# save results

savemat(f'dstrfs-{model_name}-jk{njacks}.mat',
        mdict={'channelid': channel, 'dstrf': dstrf_mean,
               'dstrf_05': dstrf_05, 'dstrf_95': dstrf_95})
