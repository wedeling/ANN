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
X, y, t = tf.get_tau_EZ_binned(n_days, name, n_bins)

N = t.size

#standardize the features and the data
try:
    N_feat = X.shape[1]
except IndexError:
    N_feat = 1
    
#split the data into a training and a validation data set, if required

#fraction of the data to be used for training
beta = 1.0
I = np.int(beta*y.size)

#train on GPU or CPU
on_gpu = False

X_train = X[0:I,:]
y_train = y[0:I]

if on_gpu == True:
    import cupy as cp

    X_train = cp.asarray(X_train)
    y_train = cp.asarray(y_train)

ann = NN.ANN(X = X_train, y = y_train, alpha = 0.001, decay_rate = 0.9, n_out = n_bins, loss = 'cross_entropy', \
             n_layers = 3, n_neurons=32, activation = 'hard_tanh', activation_out = 'linear', \
             standardize_y = False, batch_size=512, name=name, save=True)

ann.get_n_weights()

##############
#train the ANN
##############

ann.train(500000, store_loss=True)

if len(ann.loss_vals) > 0:
    fig_loss = plt.figure()
    plt.yscale('log')
    plt.plot(ann.loss_vals)
    
########################################
#compute the number of misclassification
########################################

ann.compute_misclass_softmax()

#############
#plot results
#############


plt.show()