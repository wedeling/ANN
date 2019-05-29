"""
TEST THE ANN CLASS ON A REAL REGRESSION PROBLEM
"""

import numpy as np
import matplotlib.pyplot as plt
from base import NN
import test_functions as tf
import time

plt.close('all')

###################################
#generate synthetic regression data
###################################

#number of data points
n_days = 365

#get the data
name = 'dZ'
X, y, t = tf.get_tau_EZ_regres(n_days, name)

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

#train on GPU or CPU
on_gpu = False

X_train = X[0:I,:]
y_train = y[0:I]

if on_gpu == True:
    import cupy as cp

    X_train = cp.asarray(X_train)
    y_train = cp.asarray(y_train)

ann = NN.ANN(X = X_train, y = y_train, alpha = 0.001, beta1 = 0.9, beta2=0.999, lamb_J = 5e-4, phi = 1e-3, decay_rate = 0.9, \
             decay_step=10**5, n_layers = 4, n_neurons=64, activation = 'hard_tanh', \
             neuron_based_compute=False, batch_size=128, param_specific_learn_rate=True, on_gpu=on_gpu, name=name)

ann.get_n_weights()

##############
#train the ANN
##############

t0 = time.time()
ann.train(5000, store_loss=True, check_derivative=False)
t1 = time.time()
print(t1-t0) 

#if len(ann.loss_vals) > 0:
#    fig_loss = plt.figure()
#    plt.yscale('log')
#    plt.plot(ann.loss_vals)

###############
##plot the data
###############
#
#fig = plt.figure()
#ax = fig.add_subplot(121, title=r'$\Delta E'+'\;\mathrm{data}$', xlabel=r'$t$')
#ax.plot(t[0:I], y[0:I], 'b+')
#ax.plot(t[I:], y[I:], 'r+')
##
#########################################
###plot the ANN regression after training
#########################################
#
#ax = fig.add_subplot(122, title='Neural net prediction', xlabel=r'$t$')
#
#y_hat = np.zeros(N)
#
##if standardize = True in ANN
#X_pred = (X - ann.X_mean)/ann.X_std
#
#for i in range(N):
#    y_hat[i] = ann.feed_forward(X_pred[i].reshape([1,N_feat]))
#    
#ax.plot(t[0:I], y_hat[0:I], 'b+')
#ax.plot(t[I:], y_hat[I:], 'r+')
#
#plt.tight_layout()
#
#plt.show()
