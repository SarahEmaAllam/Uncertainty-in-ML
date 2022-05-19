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
    inp = Input(shape=(1,))
    x = Dense(32, activation="relu")(inp)
    x = Dense(32, activation="relu")(x)
    mean = Dense(1, activation="linear")(x)
    var = Dense(1, activation="softplus")(x)

    train_model = Model(inp, mean)
    pred_model = Model(inp, [mean, var])

    opt = keras.optimizers.Adam (learning_rate=0.0001)

    train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer=opt)
    train_model.fit(x_train, y_train, verbose=2, epochs=300)

    mean_pred, var_pred = pred_model.predict(domain)
    std_pred = np.sqrt(var_pred)

    return mean_pred, std_pred


num_samples = 100

sample = np.linspace(-5, 5, num=num_samples)
print("Input array : \n", sample)

scales = np.linspace (0.1, 1.0, num=num_samples)

x = 2 * np.sin(sample) + np.random.normal(loc = 0.0, scale = scales, size = num_samples)
print("\nSine values : \n", x)

y = 2 * np.sin(sample)
y = y.reshape((-1, 1))
# X = np.random.normal(loc = 0.0, scale = 2.0, size = 1000)
print("shape of y", y.shape)

data_train,data_test,labels_train,labels_test = train_test_split(x,y, test_size = 0.20)

print(f'μ={y.mean()}')
print(f'σ={y.std()}')

predicted_mean, predicted_std=train_standard_model(data_train, labels_train, y)

# print("pred mean", predicted_mean)
# print("pred std", predicted_std)

y_pred_mean = predicted_mean.reshape((-1,))
y_pred_std = predicted_std.reshape((-1,))
y_pred_up_1 = y_pred_mean + y_pred_std
y_pred_down_1 = y_pred_mean - y_pred_std


plt.scatter (range (len (x)), x, label="Noisy Data Points", color='green')
plt.plot(y, label= "Sine", linewidth = 3)
plt.plot(y_pred_mean, label = "Predicted mean", linewidth = 3)
plt.fill_between (range (num_samples), y_pred_mean-y_pred_std, y_pred_mean+y_pred_std, alpha=0.2, label="Standard Deviation", color='orange')
# plt.plot(y_pred_up_1, label = "one std up")
# plt.plot(y_pred_down_1, label = "one std below")
plt.legend()
plt.show ()
