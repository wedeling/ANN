import numpy as np
import matplotlib.pyplot as plt
from base import NN
import test_functions as tf
from sklearn.model_selection import KFold
from itertools import product

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

on_gpu = False

if on_gpu == True:
    import cupy as cp

    X = cp.asarray(X)
    y = cp.asarray(y)
    
##############################
# define hyperparameter grid #
##############################
    
n_neurons = [32, 64, 128, 256]
n_layers = [2, 3, 4, 5, 6, 7, 8]
lamb = [0.0, 0.01, 0.1]

hyperparam = np.array(list(product(n_layers, n_neurons)))
errors = np.zeros(hyperparam.shape[0])

idx = 0

for hyper in hyperparam:

    n_layers = hyper[0]
    n_neurons = hyper[1]
    
    ###########################
    # k-fold cross validation #
    ###########################
        
    K = 3
    kfold = KFold(K, shuffle = True)
    
    error = 0.0
    
    # enumerate splits
    for train_idx, test_idx in kfold.split(X):
        print("TRAIN:", train_idx, "TEST:", test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
        #create the ANN
        ann = NN.ANN(X = X_train, y = y_train, alpha = 0.001, beta1 = 0.9, beta2=0.999, lamb = 0.0, decay_rate = 0.9, \
                     decay_step=10**5, n_layers = n_layers, n_neurons=n_neurons, activation = 'hard_tanh', \
                     neuron_based_compute=False, batch_size=128, param_specific_learn_rate=True, save = False, on_gpu=on_gpu)
    
        #train the ANN
        ann.train(4000, store_loss=True)
        
        #compute the error on the test set
        N_test = y_test.size
        y_hat = np.zeros(N_test)
    
        for i in range(N_test):
            y_hat[i] = ann.feed_forward(X_test[i])
        
        error_k = np.linalg.norm(y_test - y_hat)
        
        print('Error =', error_k)
        
        error += error_k
        
    errors[idx] = error/K
    idx += 1
    
    print('Total error =', error/K)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(errors)

plt.show()
