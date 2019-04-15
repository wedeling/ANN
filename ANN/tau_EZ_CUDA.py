import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import test_functions as tf
from NN_Cuda import *
import time

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
ax = fig.add_subplot(121, title='data')
ax.plot(t[0:I], y[0:I], 'b+')
ax.plot(t[I:], y[I:], 'r+')

X_train_gpu = cp.asarray(X_train.T)
y_train_gpu = cp.asarray(y_train)

batch_size = 32
init_network(X = X_train_gpu, y = y_train_gpu, n_layers = 2, n_neurons = 64, batch_size=batch_size)
init_layers()

N_train = X_train_gpu.shape[1]
t0 = time.time()

train(N_train)

t1 = time.time()
print('Training time =', t1 - t0)

#######################################
#plot the ANN regression after training
#######################################

ax = fig.add_subplot(122, title='trained prediction')

y_hat = cp.zeros(N_train)

for i in range(N_train):
    y_hat[i] = feed_forward(X_train_gpu[:,i].reshape([N_feat, 1]))[0][0]
    #print(feed_forward(X_train_gpu[:,i].reshape([N_feat, 1])))
   
y_hat = cp.asnumpy(y_hat)

ax.plot(t[0:I], y_hat[0:I], 'b+')
ax.plot(t[I:], y_hat[I:], 'r+')

plt.tight_layout()

plt.show()
