"""
TEST THE ANN CLASS ON A REAL REGRESSION PROBLEM
"""

import numpy as np
import matplotlib.pyplot as plt
from base import NN
import cupy as cp
import test_functions as tf
import time

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

#train on GPU or CPU
on_gpu = False

X_train = X[0:I,:]
y_train = y[0:I]

X_train = X[0:I,:]
y_train = np.sign(y[0:I])

X_train_gpu = cp.asarray(X_train)
y_train_gpu = cp.asarray(y_train)

idx_1 = np.where(y_train == 1.0)[0]
idx_m1 = np.where(y_train == -1.0)[0]

n_neurons = np.array([128, 256, 512, 1024, 2048])
gpu = np.array([False, True])

from itertools import product
settings = list(product(gpu, n_neurons))
print(settings)

results = {}
results['T_cpu'] = []
results['n_weights'] = []
results['T_gpu'] = []

for s in range(len(settings)):

    on_gpu = settings[s][0]
    n_neurons = settings[s][1]

    if on_gpu == False:
        X = eval('X_train')
        y = eval('y_train')
    else:
        X = eval('X_train_gpu')
        y = eval('y_train_gpu')
    
    ann = NN.ANN(X, y, alpha = 0.001, beta1 = 0.9, beta2=0.999, loss = 'logistic', activation = 'relu',\
                 decay_rate = 0.9, decay_step=10**5, n_layers = 4, n_neurons=n_neurons, batch_size=32, \
                 param_specific_learn_rate=True, on_gpu=on_gpu)
    
    n_weights = ann.get_n_weights()

    ##############
    #train the ANN
    ##############

    #ann.compute_misclass()
    t0 = time.time()
    ann.train(5000, store_loss=True, check_derivative=False)
    t1 = time.time()

    print('===============================')
    print('Training time =', t1-t0)
    print('===============================')

    if on_gpu == False:
        results['T_cpu'].append(t1-t0)
        results['n_weights'].append(n_weights)
    else:
        results['T_gpu'].append(t1-t0)

    #ann.compute_misclass()

#store results
import pickle
fname = './samples/scaling.pickle'
pickle.dump(results, open(fname, 'wb'))

#plot results
fig = plt.figure()
ax = fig.add_subplot(111, xscale='log', xlabel=r'number of weights', ylabel=r'training time')
ax.plot(results['n_weights'], results['T_cpu'], '-bo', label='CPU')
ax.plot(results['n_weights'], results['T_gpu'], '-rs', label='GPU')
leg = plt.legend()
plt.tight_layout()
plt.show()

