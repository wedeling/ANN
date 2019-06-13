import numpy as np
import matplotlib.pyplot as plt
from base import NN
import test_functions as tf
from sklearn.model_selection import TimeSeriesSplit
from itertools import product

plt.close('all')

#number of data points
n_days = 365

#get the data
name = 'dE'
n_bins = 10
X, y, bin_idx, bins, t = tf.get_tau_EZ_binned(n_days, name, n_bins)

N = t.size

##############################
# define hyperparameter grid #
##############################
    
n_neurons = [64]
n_layers = [2, 3, 4]
lamb = [0.0, 0.01, 0.1]

hyperparam = np.array(list(product(n_layers, n_neurons)))
errors = np.zeros(hyperparam.shape[0])

idx1 = 0

for hyper in hyperparam:

    n_layers = hyper[0]
    n_neurons = hyper[1]
    
    ##################################################################################
    # k-fold cross validation for time series: 
    # successive training sets are supersets of those that come before them
    # https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
    ##################################################################################
        
    K = 2
    ts_fold = TimeSeriesSplit(K)
    
    errors = np.zeros([hyperparam.shape[0], K])
    
    idx2 = 0
    
    # enumerate splits
    for train_idx, test_idx in ts_fold.split(X):
        print("TRAIN:", train_idx, "TEST:", test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        bin_idx_train, bin_idx_test = bin_idx[train_idx], bin_idx[test_idx]
    
        #create the ANN
        ann = NN.ANN(X = X_train, y = bin_idx_train, alpha = 0.001, decay_rate = 0.9, decay_step=10**4, n_out = n_bins, loss = 'cross_entropy', \
                     n_layers = n_layers, n_neurons=n_neurons, activation = 'hard_tanh', activation_out = 'linear', \
                     standardize_y = False, batch_size=512, name=name, save=False, aux_vars={'y':y[train_idx], 'bins':bins})
        
        #train the ANN
        ann.train(50000, store_loss=True)
        
        #compute the error on the test set
        error_k = ann.compute_misclass_softmax(X_test, bin_idx_test)
        
        print('Error =', error_k)
        
        errors[idx1, idx2] = error_k
        idx2 += 1
        
    idx1 += 1
    
print(errors)