import numpy as np
import matplotlib.pyplot as plt
from base import NN
import test_functions as tf

plt.close('all')

###################################
#generate synthetic regression data
###################################

#number of data points
n_days = 4*365 

#get the data
name = 'tau_EZ_T5_n_lags2'
n_softmax = 2
n_bins = 20
n_lags = 2
X_train, dE, bin_idx_dE, bins_dE, t = tf.get_tau_EZ_binned_lagged(n_days, 'dE', n_bins, n_lags)

#assuming the same features
_, dZ, bin_idx_dZ, bins_dZ, _ = tf.get_tau_EZ_binned_lagged(n_days, 'dZ', n_bins, n_lags)

#make one data vector (per sample) of size n_softmax*n_bins
bin_idx_train = np.concatenate([bin_idx_dE, bin_idx_dZ], axis = 1)

N = t.size

######################
#train or load the ANN
######################

train = True

if train == True:

    ann = NN.ANN(X = X_train, y = bin_idx_train, alpha = 0.001, decay_rate = 0.9, decay_step=10**4, n_out = n_bins*n_softmax, loss = 'cross_entropy', \
                 lamb = 0.0, n_layers = 3, n_neurons=256, activation = 'hard_tanh', activation_out = 'linear', n_softmax = n_softmax, \
                 standardize_y = False, batch_size=512, name=name, save=True, aux_vars={'dE':dE, 'dZ':dZ, 'bins_dE':bins_dE, 'bins_dZ':bins_dZ})
    ann.get_n_weights()

    ann.train(50000, store_loss=True)
    
    if len(ann.loss_vals) > 0:
        fig_loss = plt.figure()
        plt.yscale('log')
        plt.plot(ann.loss_vals)
else:
    ann = NN.ANN(X = X_train, y = bin_idx_train)
    ann.load_ANN(name)
    
########################################
#compute the number of misclassification
########################################

ann.compute_misclass_softmax()

#############
#plot results
#############

#predicted_bin = np.zeros(ann.n_train)
#
#for i in range(ann.n_train):
#    o_i, idx_max = ann.get_softmax(ann.X[i].reshape([1, ann.n_in]))
#    predicted_bin[i] = idx_max
#
#fig = plt.figure(figsize=[10,5])
#ax1 = fig.add_subplot(121, title=r'data classification', xlabel=r'$X_i$', ylabel=r'$X_j$')
#ax2 = fig.add_subplot(122, title=r'neural net prediction', xlabel=r'$X_i$', ylabel=r'$X_j$')
#
#for j in range(n_bins):
#    idx_pred = np.where(predicted_bin == j)[0]
#    idx_data = np.where(bin_idx_train[:, j] == 1.0)[0]
#    ax1.plot(ann.X[idx_data, 0], ann.X[idx_data, 1], 'o', label=r'$\mathrm{bin}\;'+str(j+1)+'$')
#    ax2.plot(ann.X[idx_pred, 0], ann.X[idx_pred, 1], 'o')
#
#ax1.legend()
#plt.tight_layout()

plt.show()