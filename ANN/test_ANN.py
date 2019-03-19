#create labels associated with X
def get_y(X):
    
    #create the classification labels
    y = np.zeros(N)
    
    #choose condition for label 1
    idx1 = np.where(X[:, 1] > -X[:,0])
    
    #condition for label -1 is just the complement of the label 1 set
    idxm1 = np.setdiff1d(np.arange(N), idx1)
    
    #set labels
    y[idx1] = 1.0
    y[idxm1] = -1.0

    return y, idx1, idxm1

import numpy as np
import matplotlib.pyplot as plt
from base import NN

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
y, idx1, idxm1 = get_y(X)

#Test ANN

#plot the data
fig = plt.figure()
ax = fig.add_subplot(131, title='data')
ax.plot(X[idx1, 0], X[idx1, 1], 'b+')
ax.plot(X[idxm1, 0], X[idxm1, 1], 'r*')

ann = NN.ANN(X, y, alpha = 0.1)

ax = fig.add_subplot(132, title='before training')

for i in range(N):
    y_hat = np.sign(ann.feed_forward(X[i]))
    
    if y_hat == 1.0:
        ax.plot(X[i, 0], X[i, 1], 'b+')
    else:
        ax.plot(X[i, 0], X[i, 1], 'r*')

ann.compute_loss()
ann.train(10000)
ann.compute_loss()

ax = fig.add_subplot(133, title='after training')

for i in range(N):
    y_hat = np.sign(ann.feed_forward(X[i]))
    
    if y_hat == 1.0:
        ax.plot(X[i, 0], X[i, 1], 'b+')
    else:
        ax.plot(X[i, 0], X[i, 1], 'r*')

plt.show()