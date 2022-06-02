#!/usr/bin/env python
# coding: utf-8

from keras_uncertainty.losses import regression_gaussian_nll_loss, regression_gaussian_beta_nll_loss
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

"""
Code provided by the assignment and retrieved from:
https://github.com/mvaldenegro/keras-uncertainty/blob/master/keras_uncertainty/losses.py
We use the Gaussian Negative Log-Likelihood loss: 
Loss commonly used in uncertainty quantification and probabilistic forecasting
"""
import keras_uncertainty.backend as K

tf.compat.v1.disable_eager_execution()

"""
Creates a model (one dense input layer, one hidden dense layer,
1 dense output layer for mean and another one for variance.
Returns the predicted mean and std (squared variance).
"""
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
    pred_model.compile (loss=regression_gaussian_nll_loss (var), optimizer=opt)
    train_model.fit(x_train, y_train, verbose=2, epochs=300)

    mean_pred, var_pred = pred_model.predict(domain)
    std_pred = np.sqrt(var_pred)

    return mean_pred, std_pred

# amplitude for noise
A = 3

# number of samples for training the model
num_samples = 10000

# generating an amount of num_samples between -5 and 5
sample = np.linspace(-5, 5, num=num_samples)
print("Input array : \n", sample)

noise_sigma = 0.5

# transform the data samples into sinusoid function and add noise
x = A * np.sin(sample) + np.random.normal(loc = 0.0, scale = noise_sigma, size = num_samples)
y = A * np.sin(sample)
y = y.reshape((-1, 1))

# split data samples and their labels into training and testing
data_train,data_test,labels_train,labels_test = train_test_split(x,y, test_size = 0.40)

print(f'μ={y.mean()}')
print(f'σ={y.std()}')

# train the model, return predicted mean and std
predicted_mean, predicted_std=train_standard_model(data_train, labels_train, y)

y_pred_mean = predicted_mean.reshape((-1,))
y_pred_std = predicted_std.reshape((-1,))

# add the upper and lower std to the mean
y_pred_up_1 = y_pred_mean + y_pred_std
y_pred_down_1 = y_pred_mean - y_pred_std


print (f'average standard deviation: {np.mean(y_pred_std)}')

# plot the noisy data points on which the model trained
plt.scatter (range (len (x)), x, label="Noisy Data Points", color='green')
plt.plot(y, label= "Sine", linewidth = 3)

# plot the predicted mean
plt.plot(y_pred_mean, label = "Predicted mean", linewidth = 3)
# plot std on the mean
plt.fill_between (range (num_samples), y_pred_mean-y_pred_std, y_pred_mean+y_pred_std, alpha=0.2, label="Standard Deviation", color='orange')
plt.legend()
plt.show ()
