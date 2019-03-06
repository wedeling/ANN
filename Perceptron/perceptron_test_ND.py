"""
========================================================
TEST THE PERCEPTRON CLASSIFIER IN MORE THAN 2 DIMENSIONS
========================================================
"""

#create labels associated with X
def get_y(X):
    
    #create the classification labels
    y = np.zeros(N)
    
    #choose condition for label 1
    cond1 = X[:, 0]
    for i in range(1, N_dim):
        cond1 += X[:, i]
    
    idx1 = np.where(cond1 > 0)
    
    #condition for label -1 is just the complement of the label 1 set
    idxm1 = np.setdiff1d(np.arange(N), idx1)
    
    #set labels
    y[idx1] = 1.0
    y[idxm1] = -1.0

    return y, idx1, idxm1

import numpy as np
import matplotlib.pyplot as plt
import Perceptron as ptron

plt.close('all')

##########################################################
#generate synthetic LINEARLY SEPERABLE classification data
##########################################################

#number of data points
N = 10000

#number of dimensions (equals number of input nodes here)
N_dim = 50

print('Testing Perceptron using', N_dim, 'input nodes and', N, 'data points.')

#N draws from multivariate normal with mean mu and covariance matrix Sigma
mu = np.zeros(N_dim)
Sigma = np.eye(N_dim)
X = np.random.multivariate_normal(mu, Sigma, size = N)

#get the labels
y, idx1, idxm1 = get_y(X)

##########################
#test the Perceptron class
##########################

#create a Perceptron object
perceptron = ptron.Perceptron(X, y)

#train the perceptron
print('Initial loss =', perceptron.compute_loss())

N_epoch = 100000
print('Training model for', N_epoch, 'epochs...')
perceptron.train(N_epoch, store_loss = True)
print('done.')
print('Trained loss =', perceptron.compute_loss())

##########################################
#validate the trained model on unseen data
##########################################
N_val = 100
X_val = np.random.multivariate_normal(mu, Sigma, size = N_val)

#create the classification labels
y_val, _, _ = get_y(X_val)

loss_val = 0

for i in range(N_val):
    #trained prediction
    y_hat = perceptron.feed_forward(X_val[i])

    if y_hat != y_val[i]:
        loss_val += 1

print('Number of validation classification errors=', loss_val)

####################################################################
#compute loss function evolution if value was stored during training
####################################################################

if len(perceptron.loss) > 0:
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='epoch', ylabel='loss')
    ax.plot(perceptron.loss)
    plt.tight_layout()

plt.show()