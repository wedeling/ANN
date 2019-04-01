"""
TEST THE ANN CLASS ON A REAL REGRESSION PROBLEM
"""

import numpy as np
import matplotlib.pyplot as plt
from base import NN
import test_functions as tf

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

##############
#plot the data
##############
fig = plt.figure()
ax = fig.add_subplot(131, title='data')
ax.plot(t[0:I], y[0:I], 'b+')
ax.plot(t[I:], y[I:], 'r+')

ann = NN.ANN(X_train, y_train, alpha = 0.01, beta = 0.9, decay_rate = 0.9,\
             decay_step=10**5, n_layers = 4, n_neurons=10, activation = 'tanh', \
             compute_h_at_neuron=False)

########################################
#plot the ANN regression before training
########################################

#ax = fig.add_subplot(132, title='before training')
#
#y_hat = np.zeros(N)
#
#for i in range(N):
#    y_hat[i] = ann.feed_forward(X[i])
#    
#ax.plot(t[0:I], y_hat[0:I], 'b+')
#ax.plot(t[I:], y_hat[I:], 'r+')

##############
#train the ANN
##############

import time
t0 = time.time()
ann.train(50000, store_loss=True, check_derivative=False)
t1 = time.time()
print(t1-t0) 

if len(ann.loss_vals) > 0:
    fig_loss = plt.figure()
    plt.yscale('log')
    plt.plot(ann.mean_loss_vals)

#######################################
#plot the ANN regression after training
#######################################

ax = fig.add_subplot(133, title='after training')

y_hat = np.zeros(N)

for i in range(N):
    y_hat[i] = ann.feed_forward(X[i])
    
ax.plot(t[0:I], y_hat[0:I], 'b+')
ax.plot(t[I:], y_hat[I:], 'r+')

plt.tight_layout()

plt.show()