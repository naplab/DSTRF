import os
import sys
import time
import logging

import numpy as np
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


# drop nan elements for computing loss
def drop_nan(response, prediction):
    mask = tf.math.is_finite(response)
    return tf.boolean_mask(response, mask), tf.boolean_mask(prediction, mask)

# loss function
def loss_se(response, prediction):
    """Squared error loss."""
    response, prediction = drop_nan(response, prediction)
    num = tf.reduce_mean(tf.square(response - prediction))
    den = tf.reduce_mean(tf.square(response))
    # den = tf.reduce_mean(tf.square(response - tf.reduce_mean(response)))
    return num / den

# correlation metric
def metric_corr(response, prediction):
    response, prediction = drop_nan(response, prediction)
    response, prediction = tf.expand_dims(response, 0), tf.expand_dims(prediction, 0)
    return tfp.stats.correlation(response, prediction, 1, 0)

# regular correlation
def fn_corr(prediction):
    mask = np.isfinite(Y_te)
    return scipy.stats.pearsonr(Y_te[mask], prediction[mask])[0]

# noise-corrected correlation
def fn_ncorr(prediction):
    mask = np.isfinite(Y_te)
    r0 = scipy.stats.pearsonr(Y_r0[mask], prediction[mask])[0]
    r1 = scipy.stats.pearsonr(Y_r1[mask], prediction[mask])[0]
    rr = scipy.stats.pearsonr(Y_r0[mask], Y_r1[mask])[0]
    return (r0 + r1)/2 / np.sqrt(rr)


# path to data file

dbase_dir = "/tf/data/"
dbase_name = "sample_data.mat"

# load data

data = load_data() # insert data loading code

# prepare variables for training
#     X_* have shape ( time x freqs )
#     Y_* have shape ( time x electrode )
#     Y_te has shape ( 2 x time x electrode ) for even and odd repetitions of repeated task

X_tr, X_vl, X_te = data['stim_train'], data['stim_validation'], data['stim_test']
Y_tr, Y_vl, Y_te = data['resp_train'], data['resp_validation'], data['resp_test']
Y_r0, Y_r1, Y_te = Y_te[0], Y_te[1], Y_te.mean(0) # separate even/odd repetitions

# select response data for single channel
#     new Y_* have shape ( time )

channel = 0
print(f"Fitting models for channel {channel:d}...")

Y_tr = Y_tr[..., channel]
Y_vl = Y_vl[..., channel]
Y_te = Y_te[..., channel]
Y_r0 = Y_r0[..., channel]
Y_r1 = Y_r1[..., channel]

# batch data if needed
#     new X_* have shape ( batch x time x freqs )
#     new Y_* have shape ( batch x time )

batch = False
if batch:
    # batch data into continuous chunks
    pass
else:
    # use full data as single batch
    X_tr = X_tr[np.newaxis]
    Y_tr = Y_tr[np.newaxis]
    X_vl = X_vl[np.newaxis]
    Y_vl = Y_vl[np.newaxis]
    X_te = X_te[np.newaxis]
    Y_te = Y_te[np.newaxis]
    Y_r0 = Y_r0[np.newaxis]
    Y_r1 = Y_r1[np.newaxis]

# stimulus frequency channels
freqbins = est['stim'].shape[0]

# define model architecture
l2 = tf.keras.regularizers.l2(0.0001)
layers = (
    tf.keras.layers.InputLayer(input_shape=(None, freqbins)),
    tf.keras.layers.Reshape((-1, freqbins, 1)),
    
    tf.keras.layers.ZeroPadding2D(((2, 0), (1, 1))),
    tf.keras.layers.Conv2D(16, 3, activation='relu', use_bias=False, kernel_regularizer=l2),
    tf.keras.layers.ZeroPadding2D(((2, 0), (1, 1))),
    tf.keras.layers.Conv2D(16, 3, activation='relu', use_bias=False, kernel_regularizer=l2),
    tf.keras.layers.ZeroPadding2D(((2, 0), (1, 1))),
    tf.keras.layers.Conv2D(16, 3, activation='relu', use_bias=False, kernel_regularizer=l2),
    
    tf.keras.layers.ZeroPadding2D(((1, 0), (0, 0))),
    tf.keras.layers.MaxPool2D((2, 2), (1, 2)),
    
    tf.keras.layers.ZeroPadding2D(((39, 0), (0, 0))),
    tf.keras.layers.Conv2D(16, (40, 8), activation='relu', use_bias=False, kernel_regularizer=l2),
    
    tf.keras.layers.Dense(1, use_bias=True, kernel_regularizer=l2),
    tf.keras.layers.Flatten()
)

# make blueprint of model
blueprint = tf.keras.models.Sequential(layers)
# blueprint.summary()

# multiple random initializations
rand_inits = 5

# prepare directory for saving trained models
model_dir = f"models/"
os.makedirs(model_dir, exist_ok=True)

# base path for saving model
model_base = os.path.join(model_dir, f"mdl-v1-{channel:03d}")

# skip training if saved model exists
if os.path.exists(f"{model_base}.h5"):
    # Load saved model
    model = tf.keras.models.load_model(
        f"{model_base:s}.h5", custom_objects={"loss_se": loss_se, "metric_corr": metric_corr})

    # calculate performance on test set
    pred = model(X_te)
    print(f"(SKIP)\tsplit #{jk+1}\t{fn_ncorr(pred):.3f}")

    # skip training
    continue

# initialize multiple models and keep only best one
perfs = []
for r in range(rand_inits):
    # temporary path for this initialization of model
    model_path = f"{model_base:s}-{r:d}.h5"

    # initialize model from blueprint
    model = tf.keras.models.clone_model(blueprint)
    optim = tf.keras.optimizers.RMSprop(1e-3, momentum=0.9)
    model.compile(optimizer=optim, loss=loss_se, metrics=[metric_corr])

    # set callbacks
    callbk_early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_metric_corr', patience=500, mode='max')
    callbk_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor='val_metric_corr', save_best_only=True, mode='max')
    callbacks = [callbk_early_stop, callbk_checkpoint]

    # fit model to data
    history = model.fit(X_tr, Y_tr, validation_data=(X_vl, Y_vl),
                        epochs=5000, verbose=0, callbacks=callbacks)

    # store best validation performance
    perfs.append(max(history.history['val_metric_corr']))

# keep best model among multiple initializations and remove the rest
best_r = np.argmax(perfs)
os.rename(f"{model_base:s}-{best_r:d}.h5", f"{model_base:s}.h5")
for r in range(rand_inits):
    if r == best_r:
        continue
    os.remove(f"{model_base:s}-{r:d}.h5")

# calculate performance on test set
model.load_weights(f"{model_base:s}.h5")
pred = model(X_val)
print(f"\tsplit #{jk+1}\t{fn_ncorr(pred):.3f}")
