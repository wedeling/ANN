"""
TEST THE ANN CLASS ON A NUMBER OF DIFFERENT SYNTHETIC (CLASSIFICATION) PROBLEMS
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
N = 100

#get the data
X, y = tf.get_sin_regres(N)

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
ax = fig.add_subplot(221, title='data')
ax.plot(X, y, 'b+')

ann = NN.ANN(X, y, alpha = 0.01)

########################################
#plot the ANN regression before training
########################################

ax = fig.add_subplot(222, title='before training')

for i in range(N):
    y_hat = np.sign(ann.feed_forward(X[i]))
    
    ax.plot(X[i], y_hat, 'b+')

##############
#train the ANN
##############

ann.train(500000, store_loss=True, check_derivative=False)

if len(ann.loss_vals) > 0:
    fig_loss = plt.figure()
    plt.yscale('log')
    plt.plot(ann.mean_loss_vals)

#######################################
#plot the ANN regression after training
#######################################

ax = fig.add_subplot(223, title='after training')

for i in range(N):
    y_hat = ann.feed_forward(X[i])
    
    ax.plot(X[i], y_hat, 'b+')

##############################################
#plot the ANN regression on the validation set
##############################################

ax = fig.add_subplot(224, title='validation')

#get the labels
X_val, y_val = tf.get_sin_regres(N)

try:
    for i in range(N_feat):
        X_val[:,i] = (X_val[:,i] - np.mean(X_val[:,i]))/np.std(X_val[:,i])
except IndexError:
    X_val = (X_val - np.mean(X_val))/np.std(X_val) 

y_val = (y_val - np.mean(y_val))/np.std(y_val)

for i in range(N):
    y_hat_val = ann.feed_forward(X_val[i])
    
    ax.plot(X_val[i], y_hat_val, 'b+')

plt.tight_layout()

plt.show()