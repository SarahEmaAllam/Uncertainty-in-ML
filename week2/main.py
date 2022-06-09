#!/usr/bin/env python
# coding: utf-8

from keras_uncertainty.losses import regression_gaussian_nll_loss, regression_gaussian_beta_nll_loss
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm, poisson
from scipy import stats
from scipy.stats import kurtosis, skew
import seaborn as sns
import math
from keras.layers import Dense, Input, Flatten

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

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


def train_standard_model(x_train, y_train, domain):
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

    model = TwoHeadStochasticRegressor(pred_model)

    mean_pred, epi_std, ale_std = model.predict(domain, 4, disentangle_uncertainty=True)

    return mean_pred, epi_std, ale_std

A = 5

num_samples = 100
test_samples = 140

sample = np.linspace(-5, 5, num=num_samples)
# range of 10 and 100 samples -> 10 samples per range unit (from -5 to -4 for example there are 10 samples)

print("Input array : \n", sample)

noise_sigma = 0.5
noise_fn = lambda x : abs (x+0.2)

x = A * np.sin(sample) + np.random.normal(loc = 0.0, scale = noise_fn (sample), size = num_samples)
# print("\nSine values : \n", x)

y = A * np.sin(sample)
y = y.reshape((-1, 1))
# X = np.random.normal(loc = 0.0, scale = 2.0, size = 1000)
print("shape of y", y.shape)
domain_samples = A * np.sin(np.linspace (-7, 7, num= test_samples))
domain = domain_samples + np.random.normal(loc = 0.0, scale = noise_fn (domain_samples), size = test_samples)
domain = domain.reshape((-1, 1))

print(f'μ={y.mean()}')
print(f'σ={y.std()}')

predicted_mean, epi, ale=train_standard_model(x, y, domain)

# print("pred mean", predicted_mean)
# print("pred std", predicted_std)

print("average ale", np.mean(ale))

y_pred_mean = predicted_mean.reshape((-1,))


y_epi = epi.reshape((-1,))
y_ale =ale.reshape((-1,))

y_pred_std = y_epi + y_ale


y_pred_up_1 = y_pred_mean + y_pred_std
y_pred_down_1 = y_pred_mean - y_pred_std


print (f'average standard deviation: {np.mean(y_pred_std)}')
plt.scatter(range (len (x )), x, label="Noisy Data Points", color='green')
plt.plot(y, label= "Sine", linewidth = 3)
plt.plot(y_pred_mean, label = "Predicted mean", linewidth = 3)
plt.fill_between (range (len(y_pred_mean)), y_pred_mean-y_ale, y_pred_mean+y_ale, alpha=0.2, label="Aleatoric Uncertainty", color='orange')
# plt.plot(y_pred_up_1, label = "one std up")
# plt.plot(y_pred_down_1, label = "one std below")
plt.legend()
plt.show ()
#
# plt.scatter (range (len (x)), x, label="Noisy Data Points", color='green')
# plt.plot(y, label= "Sine", linewidth = 3)
# plt.plot(y_pred_mean, label = "Predicted mean", linewidth = 3)
# plt.fill_between (range (num_samples), y_pred_mean-y_pred_std, y_pred_mean+y_pred_std, alpha=0.2, label="Standard Deviation", color='orange')
# # plt.plot(y_pred_up_1, label = "one std up")
# # plt.plot(y_pred_down_1, label = "one std below")
# plt.legend()
# plt.show ()
#
# plt.scatter (range (len (x)), x, label="Noisy Data Points", color='green')
# plt.plot(y, label= "Sine", linewidth = 3)
# plt.plot(y_pred_mean, label = "Predicted mean", linewidth = 3)
# plt.fill_between (range (num_samples), y_pred_mean-y_ale, y_pred_mean+y_ale, alpha=0.2, label="Aleatoric Uncertainty", color='orange')
# # plt.plot(y_pred_up_1, label = "one std up")
# # plt.plot(y_pred_down_1, label = "one std below")
# plt.legend()
# plt.show ()
#
#
# plt.scatter (range (len (x)), x, label="Noisy Data Points", color='green')
# plt.plot(y, label= "Sine", linewidth = 3)
# plt.plot(y_pred_mean, label = "Predicted mean", linewidth = 3)
# plt.fill_between (range (num_samples), y_pred_mean-y_epi, y_pred_mean+y_epi, alpha=0.2, label="Epistemic Uncertainty", color='orange')
# # plt.plot(y_pred_up_1, label = "one std up")
# # plt.plot(y_pred_down_1, label = "one std below")
# plt.legend()
# plt.show ()
