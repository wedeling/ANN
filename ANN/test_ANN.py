import numpy as np
import matplotlib.pyplot as plt
from base import NN
import test_functions as tf

plt.close('all')

##########################################################
#generate synthetic LINEARLY SEPERABLE classification data
##########################################################

#number of data points
N = 1000

#N draws from multivariate normal with mean mu and covariance matrix Sigma
mu = np.array([0, 0])
Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
X = np.random.multivariate_normal(mu, Sigma, size = N)

#get the labels
y, idx1, idxm1 = tf.get_y_quadrant(X, N)

#Test ANN

#plot the data
fig = plt.figure()
ax = fig.add_subplot(221, title='data')
ax.plot(X[idx1, 0], X[idx1, 1], 'b+')
ax.plot(X[idxm1, 0], X[idxm1, 1], 'r*')

ann = NN.ANN(X, y, alpha = 0.1)

ax = fig.add_subplot(222, title='before training')

for i in range(N):
    y_hat = np.sign(ann.feed_forward(X[i]))
    
    if y_hat == 1.0:
        ax.plot(X[i, 0], X[i, 1], 'b+')
    else:
        ax.plot(X[i, 0], X[i, 1], 'r*')

ann.compute_loss()
ann.train(500)
ann.compute_loss()

ax = fig.add_subplot(223, title='after training')

for i in range(N):
    y_hat = ann.feed_forward(X[i])
    
    if np.sign(y_hat) == 1.0:
        ax.plot(X[i, 0], X[i, 1], 'b+')
    else:
        ax.plot(X[i, 0], X[i, 1], 'r*')

ax = fig.add_subplot(224, title='validation')

X = np.random.multivariate_normal(mu, Sigma, size = N)

#get the labels
y, idx1, idxm1 = tf.get_y_quadrant(X, N)

for i in range(N):
    y_hat = ann.feed_forward(X[i])
    
    if np.sign(y_hat) == 1.0:
        ax.plot(X[i, 0], X[i, 1], 'b+')
    else:
        ax.plot(X[i, 0], X[i, 1], 'r*')

plt.tight_layout()

plt.show()