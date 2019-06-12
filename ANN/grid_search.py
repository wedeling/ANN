import numpy as np
import matplotlib.pyplot as plt
from base import NN
import test_functions as tf
from sklearn.model_selection import KFold
from itertools import product

plt.close('all')

#number of data points
n_days = 365

#get the data
name = 'dE'
n_bins = 10
X, y, bin_idx, bins, t = tf.get_tau_EZ_binned(n_days, name, n_bins)

N = t.size

#standardize the features and the data
try:
    N_feat = X.shape[1]
except IndexError:
    N_feat = 1

##############################
# define hyperparameter grid #
##############################
    
n_neurons = [32, 64, 128, 256]
n_layers = [2, 3, 4]
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
        
    K = 2
    kfold = KFold(K, shuffle = False)
    
    error = 0.0
    
    # enumerate splits
    for train_idx, test_idx in kfold.split(X):
        print("TRAIN:", train_idx, "TEST:", test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        bin_idx_train, bin_idx_test = bin_idx[train_idx], bin_idx[test_idx]
    
        #create the ANN
        ann = NN.ANN(X = X_train, y = bin_idx_train, alpha = 0.001, decay_rate = 0.9, decay_step=10**4, n_out = n_bins, loss = 'cross_entropy', \
                     n_layers = n_layers, n_neurons=n_neurons, activation = 'hard_tanh', activation_out = 'relu', \
                     standardize_y = False, batch_size=512, name=name, save=False, aux_vars={'y':y, 'bins':bins})
        #train the ANN
        ann.train(10, store_loss=True)
        
        #compute the error on the test set
        
        error_k = ann.compute_misclass_softmax()
        
#        N_test = bin_idx_test.size
#        y_hat = np.zeros(N_test)
#    
#        for i in range(N_test):
#            y_hat[i] = ann.feed_forward(X_test[i])
#        
#        error_k = np.linalg.norm(y_test - y_hat)
#        

        print('Error =', error_k)
        
        error += error_k
        
    errors[idx] = error/K
    idx += 1
    
    print('Total error =', error/K)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(errors)

plt.show()
