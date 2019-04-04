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
n_days = 365

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
y_train = np.sign(y[0:I])

idx_1 = np.where(y_train == 1.0)[0]
idx_m1 = np.where(y_train == -1.0)[0]

#plot the training data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X_train[idx_1, 0], X_train[idx_1, 1], 'bo')
ax.plot(X_train[idx_m1, 0], X_train[idx_m1, 1], 'ro')
plt.tight_layout()

ann = NN.ANN(X_train, y_train, alpha = 0.001, beta = 0.9, loss = 'logistic', activation = 'relu',\
             decay_rate = 0.9, decay_step=10**5, n_layers = 4, n_neurons=10, batch_size=64)

##############
#train the ANN
##############

ann.compute_misclass()
ann.train(1000000, store_loss=True, check_derivative=False)
ann.compute_misclass()

if len(ann.loss_vals) > 0:
    fig_loss = plt.figure()
    plt.yscale('log')
    plt.plot(ann.mean_loss_vals)
    plt.tight_layout()

#######################
#plot the predictions #
#######################

y_hat = np.zeros(N)
for i in range(N):
    y_hat[i] = np.sign(ann.feed_forward(X[i])[0][0])
    
idx_1 = np.where(y_hat == 1.0)[0]
idx_m1 = np.where(y_hat == -1.0)[0]

#plot the training data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X_train[idx_1, 0], X_train[idx_1, 1], 'bo')
ax.plot(X_train[idx_m1, 0], X_train[idx_m1, 1], 'ro')
plt.tight_layout()

plt.show()