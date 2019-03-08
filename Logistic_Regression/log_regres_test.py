"""
==================================
TEST THE LOGOSTIC REGERSSION CLASS 
==================================
"""

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
import Logistic_Regression as lr 

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

##########################################################
#generate synthetic LINEARLY SEPERABLE classification data
##########################################################

#number of data points
N = 1000

#N draws from multivariate normal with mean mu and covariance matrix Sigma
mu = np.array([0, 0])
Sigma = np.array([[1, 0.0], [0.0, 1.0]])
X = np.random.multivariate_normal(mu, Sigma, size = N)

#get the labels
y, idx1, idxm1 = get_y(X)

#plot the data
fig = plt.figure(figsize=[12,4])
ax = fig.add_subplot(131, title='data')
ax.plot(X[idx1, 0], X[idx1, 1], 'b+')
ax.plot(X[idxm1, 0], X[idxm1, 1], 'r*')

###################################
#test the Logistic Regression class
###################################

#create a Logistic Regression object
lregress = lr.Logistic_Regression(X, y, alpha=1.0)

#plot the predictions before training
ax = fig.add_subplot(132, title='before training')
y_hat = []
for i in range(N):
    y_hat.append(lregress.feed_forward(X[i]))

#contourmap of probability of 'success'
ax.tricontourf(X[:,0], X[:,1], y_hat)
#overlay data
ax.plot(X[idx1, 0], X[idx1, 1], 'b+')
ax.plot(X[idxm1, 0], X[idxm1, 1], 'r*')

#train the ANN
lregress.train(10000)

#plot the results after prediction
ax = fig.add_subplot(133, title='after training')
y_hat = []
for i in range(N):
    y_hat.append(lregress.feed_forward(X[i]))

#contourmap of probability of 'success'
ct = ax.tricontourf(X[:,0], X[:,1], y_hat)
#overlay data
ax.plot(X[idx1, 0], X[idx1, 1], 'b+')
ax.plot(X[idxm1, 0], X[idxm1, 1], 'r*')
plt.colorbar(ct)

plt.tight_layout()

plt.show()
