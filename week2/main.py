#!/usr/bin/env python
# coding: utf-8

from keras_uncertainty.losses import regression_gaussian_nll_loss, regression_gaussian_beta_nll_loss
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Input, Flatten

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model


import keras_uncertainty.backend as K
from keras_uncertainty.models import TwoHeadStochasticRegressor, StochasticRegressor
from keras_uncertainty.layers import DropConnectDense, StochasticDropout

tf.compat.v1.disable_eager_execution()

# Losses commonly used in uncertainty quantification and probabilistic forecasting

def regression_gaussian_nll_loss(variance_tensor, epsilon=1e-8, variance_logits=False):
    """
        Gaussian negative log-likelihood for regression, with variance estimated by the model.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.
    """
    def nll(y_true, y_pred):
        #if variance_logits:
        #    variance_tensor = K.exp(variance_tensor)

        return 0.5 * K.mean(K.log(variance_tensor + epsilon) + K.square(y_true - y_pred) / (variance_tensor + epsilon))

    return nll

def train_standard_model(x_train, y_train):
    """

    :param x_train:
    :param y_train:
    :return:
    """


    prob = 0.05
    inp = Input(shape=(1,))
    x = DropConnectDense(32, activation="relu", prob=prob)(inp)
    x = DropConnectDense(32, activation="relu", prob=prob)(x)
    mean = DropConnectDense(1, activation="linear")(x)
    var = DropConnectDense(1, activation="softplus")(x)

    train_model = Model(inp, mean)
    pred_model = Model(inp, [mean, var])

    opt = keras.optimizers.Adam(learning_rate=0.1)

    train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer=opt)
    train_model.fit(x_train, y_train, verbose=2, epochs=100)

    return train_model, pred_model

def evaluate_model(domain, pred_model):
    # model taken from given repository: returns the mean and std (two heads)
    model = TwoHeadStochasticRegressor(pred_model)

    # predict on the same model 20 times and disentangle the uncertainties
    # where aleatoric uncertainty = mean of the variances from each predicted model (previously trained)
    # and epistemic uncertainty = variance of the means from each predicted model (previously trained)
    mean_pred, epi_std, ale_std = model.predict(domain, 20, disentangle_uncertainty=True)

    return mean_pred, epi_std, ale_std

# generating big and small datasets
X = np.clip(np.random.normal(0.0, 1.0, 100).reshape(-1,1), -3, 3)

# let us generate a grid to check how models fit the data
x_grid = np.linspace(-5, 5, 100).reshape(-1,1)

# defining the function - noisy
# function adapted from : https://www.kaggle.com/code/gdmarmerola/risk-and-uncertainty-in-deep-learning/notebook
noise = lambda x: (x**2)/10
target_toy = lambda x: (x + 0.3*np.sin(2*np.pi*(x + noise(x)) +
                        0.3*np.sin(4*np.pi*(x + noise(x)) +
                        noise(x) - 0.5)))

# defining the function - no noise
target_toy_noiseless = lambda x: (x + 0.3*np.sin(2*np.pi*(x)) + 0.3*np.sin(4*np.pi*(x)) - 0.5)

# runnning the target
y = np.array([target_toy(e) for e in X])
y_noiseless = np.array([target_toy_noiseless(e) for e in x_grid])


# check the toy data
plt.figure(figsize=[12,6], dpi=200)

# return trained model and predictive model
train_model, pred_model =train_standard_model(X, y)

# predict the model on data and return epistemic and aleatoric noise
predicted_mean, epi, ale = evaluate_model(x_grid, pred_model)

# reshape the vectors
y_pred_mean = predicted_mean.reshape((-1,))
y_epi = epi.reshape((-1,))
y_ale =ale.reshape((-1,))

# predictive uncertainty is the sum of epistemic and aleatoric
y_pred_std = y_epi + y_ale

#  calculate lower and upper bounds
y_pred_up_1 = y_pred_mean + y_pred_std
y_pred_down_1 = y_pred_mean - y_pred_std


plt.figure(figsize=[12,6], dpi=200)

# plot aleatoric uncertainty with the predicted mean
plt.plot(X, y, 'kx', label='Toy data', alpha=0.5, markersize=5)
plt.plot(x_grid, predicted_mean, label='neural net fit', color='tomato', alpha=0.8)
plt.fill_between (x_grid.reshape(1,-1)[0], y_pred_mean-y_ale, y_pred_mean+y_ale, alpha=0.2, label="Aleatoric Uncertainty", color='orange')
plt.title('Neural network fit for median expected value with aleatoric uncertainty')
plt.xlabel('$x$'); plt.ylabel('$y$')
plt.xlim(-3.5,3.5); plt.ylim(-5, 3)
plt.legend()
plt.show()

# plot epistemic uncertainty with the predicted mean
plt.plot(X, y, 'kx', label='Toy data', alpha=0.5, markersize=5)
plt.plot(x_grid, predicted_mean, label='neural net fit', color='tomato', alpha=0.8)
plt.fill_between (x_grid.reshape(1,-1)[0], y_pred_mean-y_epi, y_pred_mean+y_epi, alpha=0.2, label="Epistemic Uncertainty", color='orange')
plt.title('Neural network fit for median expected value with epistemic uncertainty')
plt.xlabel('$x$'); plt.ylabel('$y$')
plt.xlim(-3.5,3.5); plt.ylim(-5, 3)
plt.legend()
plt.show()

# plot predictive uncertainty with the predicted mean
plt.plot(X, y, 'kx', label='Toy data', alpha=0.5, markersize=5)
plt.plot(x_grid, predicted_mean, label='neural net fit', color='tomato', alpha=0.8)
plt.fill_between (x_grid.reshape(1,-1)[0], y_pred_mean-y_pred_std , y_pred_mean+y_pred_std , alpha=0.2, label="Predictive Uncertainty", color='orange')
plt.title('Neural network fit for median expected value with predictive uncertainty')
plt.xlabel('$x$'); plt.ylabel('$y$')
plt.xlim(-3.5,3.5); plt.ylim(-5, 3)
plt.legend()
plt.show()
