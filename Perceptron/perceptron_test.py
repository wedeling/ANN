"""
==============================
TEST THE PERCEPTRON CLASSIFIER
==============================
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
import Perceptron as ptron

plt.close('all')

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
fig = plt.figure()
ax = fig.add_subplot(221, title='data')
ax.plot(X[idx1, 0], X[idx1, 1], 'b+')
ax.plot(X[idxm1, 0], X[idxm1, 1], 'r*')

##########################
#test the Perceptron class
##########################

#create a Perceptron object
perceptron = ptron.Perceptron(X, y)

#plot the results of the untrained perceptron
ax = fig.add_subplot(222, title = 'before training')

for i in range(N):
    #untrained prediction
    y_hat = perceptron.feed_forward(X[i])
    
    if y_hat == 1.0:
        ax.plot(X[i, 0], X[i, 1], 'b+')
    else:
        ax.plot(X[i, 0], X[i, 1], 'r*')

#train the perceptron
print(perceptron.compute_loss())
N_epoch = 10000
perceptron.train(N_epoch, store_loss = True)
print(perceptron.compute_loss())

#plot the results of the trained perceptron
ax = fig.add_subplot(223, title='after training')

for i in range(N):
    #trained prediction
    y_hat = perceptron.feed_forward(X[i])
    
    if y_hat == 1.0:
        ax.plot(X[i, 0], X[i, 1], 'b+')
    else:
        ax.plot(X[i, 0], X[i, 1], 'r*')

##########################################
#validate the trained model on unseen data
##########################################
N_val = 100
X_val = np.random.multivariate_normal(mu, Sigma, size = N_val)

#create the classification labels
y_val, _, _ = get_y(X_val)

#plot the results of the validation
ax = fig.add_subplot(224, title='validation')

loss_val = 0

for i in range(N_val):
    #trained prediction
    y_hat = perceptron.feed_forward(X_val[i])
    
    if y_hat == 1.0:
        ax.plot(X_val[i, 0], X_val[i, 1], 'b+')
    else:
        ax.plot(X_val[i, 0], X_val[i, 1], 'r*')
        
    if y_hat != y_val[i]:
        loss_val += 1

print('Number of validation classification errors=', loss_val)

plt.tight_layout()

####################################################################
#compute loss function evolution if value was stored during training
####################################################################

if len(perceptron.loss) > 0:
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='epoch', ylabel='loss')
    ax.plot(perceptron.loss)
    plt.tight_layout()

plt.show()