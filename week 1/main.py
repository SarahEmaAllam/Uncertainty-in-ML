# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm, poisson
from scipy import stats
from scipy.stats import kurtosis, skew
import seaborn as sns
import math

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers

import keras_uncertainty.backend as K

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

    return nl

def train_model(inputs, outputs, test_set):
    train_dataset = inputs
    test_dataset = test_set
    input = keras.layers.Input(shape=inputs.shape)
    x = keras.layers.Dense(32, activation="relu")(input)
    # output = keras.layers.Dense(1, activation="relu")(x)

    miu = keras.layers.Dense(1, activation="relu")(x)
    var = keras.layers.Dense(1, activation="softplus")(x)

    training_model = Model(input, miu)
    testing_model = Model(input, [miu, var])

    training_model.summary()

    training_model.compile(loss=out_loss, optimizer="adam")

    return training_model, testing_model

x = np.linspace(-5, 5, num=100)
print("Input array : \n", x)

x = np.sin(x)
print("\nSine values : \n", x)

y = np.random.choice(x, 100, replace=False)
y = x + np.random.normal(loc = 0.0, scale = 2.0)
# X = np.random.normal(loc = 0.0, scale = 2.0, size = 1000)

print(y)
print(f'μ={y.mean()}')
print(f'σ={y.std()}')

