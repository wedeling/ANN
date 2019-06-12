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
name = 'dE'
n_bins = 10
X_train, y, bin_idx_train, bins, t = tf.get_tau_EZ_binned(n_days, name, n_bins)

N = t.size

######################
#train or load the ANN
######################

train = True

if train == True:

    ann = NN.ANN(X = X_train, y = bin_idx_train, alpha = 0.001, decay_rate = 0.9, decay_step=10**4, n_out = n_bins, loss = 'cross_entropy', \
                 n_layers = 3, n_neurons=32, activation = 'hard_tanh', activation_out = 'relu', \
                 standardize_y = False, batch_size=512, name=name, save=True, aux_vars={'y':y, 'bins':bins})

    ann.train(100000, store_loss=True)
    
    if len(ann.loss_vals) > 0:
        fig_loss = plt.figure()
        plt.yscale('log')
        plt.plot(ann.loss_vals)
else:
    ann = NN.ANN(X = X_train, y = bin_idx_train)
    ann.load_ANN(name)

ann.get_n_weights()
    
########################################
#compute the number of misclassification
########################################

#ann.compute_misclass_softmax()

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