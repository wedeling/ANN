import numpy as np
import matplotlib.pyplot as plt
from base import NN
#import cupy as cp
import test_functions as tf
import time

plt.close('all')

###################################
#generate synthetic regression data
###################################

#number of data points
n_days = 365

#get the data
X, y, t = tf.get_tau_EZ_regres(n_days, 'dE')

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

X_train = X[0:I,:]
y_train = np.sign(y[0:I])

idx_1 = np.where(y_train == 1.0)[0]
idx_m1 = np.where(y_train == -1.0)[0]

if on_gpu == True:
    X_train = cp.asarray(X_train)
    y_train = cp.asarray(y_train)

#plot the training data
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(X_train[idx_1, 0], X_train[idx_1, 1], 'bo')
#ax.plot(X_train[idx_m1, 0], X_train[idx_m1, 1], 'ro')
#plt.tight_layout()

ann = NN.ANN(X_train, y_train, alpha = 0.001, beta1 = 0.9, beta2=0.999, loss = 'logistic', activation = 'relu',\
             decay_rate = 0.9, decay_step=10**5, n_layers = 4, n_neurons=1024, batch_size=128, \
             param_specific_learn_rate=True, on_gpu=on_gpu)

ann.get_n_weights()

##############
#train the ANN
##############

ann.compute_misclass()
t0 = time.time()
ann.train(1000, store_loss=True, check_derivative=False)
t1 = time.time()

print('===============================')
print('Training time =', t1-t0)
print('===============================')

ann.compute_misclass()

if len(ann.loss_vals) > 0:
    fig_loss = plt.figure()
    plt.yscale('log')
    plt.plot(ann.loss_vals)

#######################
#plot the predictions #
#######################

#y_hat = np.zeros(N)
#for i in range(N):
#    y_hat[i] = np.sign(ann.feed_forward(X[i])[0][0])
#    
#idx_1 = np.where(y_hat == 1.0)[0]
#idx_m1 = np.where(y_hat == -1.0)[0]
#
##plot the training data
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(X_train[idx_1, 0], X_train[idx_1, 1], 'bo')
#ax.plot(X_train[idx_m1, 0], X_train[idx_m1, 1], 'ro')
#plt.tight_layout()

plt.show()
