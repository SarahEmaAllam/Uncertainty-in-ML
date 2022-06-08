#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
from keras_uncertainty.models import TwoHeadStochasticRegressor, StochasticRegressor, DeepEnsembleRegressor
from keras_uncertainty.layers import DropConnectDense, StochasticDropout

tf.compat.v1.disable_eager_execution()


def train_standard_model(x_train, y_train, domain):
    def model_fn():
        inp = Input(shape=(1,))
        x = Dense(32, activation="relu")(inp)
        x = Dense(32, activation="relu")(x)
        mean = Dense(1, activation="linear")(x)
        var = Dense(1, activation="softplus")(x)

        train_model = Model(inp, mean)
        pred_model = Model(inp, [mean, var])

        train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer="adam")

        return train_model, pred_model

    model = DeepEnsembleRegressor(model_fn, num_estimators=5)
    model.fit(x_train, y_train, verbose=2, epochs=100)

    mean_pred, epi_std, ale_std = model.predict(domain, 20, disentangle_uncertainty=True)


    #mean_preds = []
    #var_preds = []
    #std_preds = []

    #for i in range (10):
     #   mean, var = pred_model.predict(domain)
      #  mean_preds.append(mean)
       # std_preds.append(np.sqrt(var))
        #mean_pred[i], var_pred[i] = pred_model.predict(domain)
        #std_pred[i] = np.sqrt(var_pred)

    #mean_pred = np.mean (mean_preds, axis = 0)
    #std_pred = np.mean (std_preds, axis = 0)

    #print("pred mean", mean_pred)
    #print("pred std", std_pred)


    #epi_std = np.std(mean_pred, axis=0)
    #ale_std = np.mean(std_pred, axis=0)

    #print("epi", epi_std)
    #print("ale",ale_std)


    return mean_pred, epi_std, ale_std

A = 3

num_samples = 1000

sample = np.linspace(-np.pi, np.pi, num=num_samples//2)
print("Input array : \n", sample)

noise_sigma = 0.5
noise_fn = lambda x : abs (np.cos (x))

x = A * np.sin(sample) + np.random.normal(loc = 0.0, scale = noise_fn (sample), size =num_samples//2)
# print("\nSine values : \n", x)

y = A * np.sin(sample)
y = y.reshape((-1, 1))

domain = A * np.sin(np.linspace (-2*np.pi,2*np.pi, num= num_samples))
domain = domain.reshape((-1, 1))
# X = np.random.normal(loc = 0.0, scale = 2.0, size = 1000)
print("shape of y", y.shape)

# data_train,data_test,labels_train,labels_test = train_test_split(x,y, test_size = 0.40)

print(f'μ={y.mean()}')
print(f'σ={y.std()}')

predicted_mean, epi, ale=train_standard_model(x, y, domain)

# print("pred mean", predicted_mean)
# print("pred std", predicted_std)

#print("epi", epi)
#print("ale", ale)
print("average ale", np.mean(ale))

y_pred_mean = predicted_mean.reshape((-1,))


y_epi = epi.reshape((-1,))
y_ale =ale.reshape((-1,))

y_pred_std = y_epi + y_ale


y_pred_up_1 = y_pred_mean + y_pred_std
y_pred_down_1 = y_pred_mean - y_pred_std


print (f'average standard deviation: {np.mean(y_pred_std)}')

#fig, axs = plt.subplots(3)
#std_ax = axs[0]
#ale_ax = axs[1]
#epi_ax =axs[2]
plt.rcParams['figure.figsize'] = [10, 5]

# plt.scatter (range (len (x)), x, label="Noisy Data Points", color='green')
plt.plot(domain, label= "Sine", linewidth = 3)
plt.vlines (sample [0], -10, 10)
plt.vlines (sample [-1], -10, 10)
plt.plot(y_pred_mean, label = "Predicted mean", linewidth = 3)
plt.fill_between (range (len (y_epi)), y_pred_mean-y_epi, y_pred_mean+y_epi, alpha=0.2, label="Epi", color='orange')
plt.ylim(-10, 10)
plt.legend()
plt.show ()

# plt.scatter (range (len (x)), x, label="Noisy Data Points", color='green')
plt.plot(domain, label= "Sine", linewidth = 3)
plt.vlines (-np.pi, -10, 10)
plt.vlines (np.pi, -10, 10)
plt.plot(y_pred_mean, label = "Predicted mean", linewidth = 3)
plt.fill_between (range (len (y_ale)), y_pred_mean-y_ale, y_pred_mean+y_ale, alpha=0.2, label="Ale", color='orange')
plt.ylim(-10, 10)
plt.legend()
plt.show ()


# plt.scatter (range (len (x)), x, label="Noisy Data Points", color='green')
plt.plot(domain, label= "Sine", linewidth = 3)
plt.vlines (-np.pi, -10, 10)
plt.vlines (np.pi, -10, 10)
plt.plot(y_pred_mean, label = "Predicted mean", linewidth = 3)
plt.fill_between (range (len (y_pred_std)), y_pred_mean-y_pred_std, y_pred_mean+y_pred_std, alpha=0.2, label="std", color='orange')
plt.ylim(-10, 10)
plt.legend()
plt.show ()


# In[ ]:
