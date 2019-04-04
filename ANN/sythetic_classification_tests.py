"""
TEST THE ANN CLASS ON A NUMBER OF DIFFERENT SYNTHETIC (CLASSIFICATION) PROBLEMS
"""

import numpy as np
import matplotlib.pyplot as plt
from base import NN
import test_functions as tf

plt.close('all')

#######################################
#generate synthetic classification data
#######################################

#number of data points
N = 5000

#get the labels
X, y, idx1, idxm1 = tf.get_y_quadrant(N)

##############
#plot the data
##############
fig = plt.figure()
ax = fig.add_subplot(221, title='data')
ax.plot(X[idx1, 0], X[idx1, 1], 'b+')
ax.plot(X[idxm1, 0], X[idxm1, 1], 'r*')

ann = NN.ANN(X, y, alpha = 0.01, n_layers=4, n_neurons=10, activation='relu', loss='logistic', \
             beta=0.9, decay_rate=0.9, decay_step=10**4, batch_size=32)

#############################################
#plot the ANN classifications before training
#############################################

ax = fig.add_subplot(222, title='before training')

for i in range(N):
    y_hat = np.sign(ann.feed_forward(X[i]))
    
    if y_hat == 1.0:
        ax.plot(X[i, 0], X[i, 1], 'b+')
    else:
        ax.plot(X[i, 0], X[i, 1], 'r*')

##############
#train the ANN
##############

ann.compute_misclass()
ann.train(100000, store_loss=True, check_derivative=False)
ann.compute_misclass()

############################################
#plot the ANN classifications after training
############################################

ax = fig.add_subplot(223, title='after training')

for i in range(N):
    y_hat = ann.feed_forward(X[i])
    
    if np.sign(y_hat) == 1.0:
        ax.plot(X[i, 0], X[i, 1], 'b+')
    else:
        ax.plot(X[i, 0], X[i, 1], 'r*')

###################################################
#plot the ANN classifications on the validation set
###################################################

ax = fig.add_subplot(224, title='validation')

#get the labels
X_val, y_val, idx1, idxm1 = tf.get_y_quadrant(N)

for i in range(N):
    y_hat_val = ann.feed_forward(X_val[i])
    
    if np.sign(y_hat_val) == 1.0:
        ax.plot(X_val[i, 0], X_val[i, 1], 'b+')
    else:
        ax.plot(X_val[i, 0], X_val[i, 1], 'r*')

plt.tight_layout()

plt.show()