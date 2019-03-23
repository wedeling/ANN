"""
TEST THE ANN CLASS ON A NUMBER OF DIFFERENT SYNTHETIC (REGRESSION) PROBLEMS
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
n_days = 30

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

##############
#plot the data
##############
fig = plt.figure()
ax = fig.add_subplot(131, title='data')
ax.plot(t, y, 'b+')

ann = NN.ANN(X, y, alpha = 0.01)

########################################
#plot the ANN regression before training
########################################

ax = fig.add_subplot(132, title='before training')

for i in range(N):
    y_hat = np.sign(ann.feed_forward(X[i]))
    
    ax.plot(t[i], y_hat, 'b+')

##############
#train the ANN
##############

ann.train(50000, store_loss=True, check_derivative=False)

if len(ann.loss_vals) > 0:
    fig_loss = plt.figure()
    plt.yscale('log')
    plt.plot(ann.mean_loss_vals)

#######################################
#plot the ANN regression after training
#######################################

ax = fig.add_subplot(133, title='after training')

for i in range(N):
    y_hat = ann.feed_forward(X[i])
    
    ax.plot(t[i], y_hat, 'b+')

plt.tight_layout()

plt.show()