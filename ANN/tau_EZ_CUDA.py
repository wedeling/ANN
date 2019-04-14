import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import test_functions as tf
from NN_Cuda import *

plt.close('all')

###################################
#generate synthetic regression data
###################################

#number of data points
n_days = 50

#get the data
X, y, t = tf.get_tau_EZ_regres(n_days)

N = t.size

#standardize the features and the data
try:
    N_feat = X.shape[1]
    for i in range(N_feat):
        X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])

except IndexError:
    N_feat = 1
    X = (X - np.mean(X))/np.std(X) 
    
y = (y - np.mean(y))/np.std(y)

#split the data into a training and a validation data set, if required

#fraction of the data to be used for training
beta = 1.0
I = np.int(beta*y.size)

X_train = X[0:I,:]
y_train = y[0:I]

X_train_gpu = cp.asarray(X_train)
y_train_gpu = cp.asarray(y_train)

init_network(X = X_train, y = y_train, n_layers = 4, n_neurons = 128)
init_layers()

feed_forward(X_train_gpu[0].reshape([8,1]))

