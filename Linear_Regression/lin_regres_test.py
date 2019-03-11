"""
================================
TEST THE LINEAR REGERSSION CLASS 
================================
"""

#create artificial data
def get_y():
    
    #create the classification labels
    X = np.linspace(0.0, 1.0, N)
    
    noise = np.random.randn(N)*1e-1
    y = X**2 + 2*X + noise

    return y, X

import numpy as np
import matplotlib.pyplot as plt
import Linear_Regression as lr 

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

###################################
#generate synthetic regression data
###################################

#number of data points
N = 100

#get feature and data
y, X = get_y()

#plot the data
fig = plt.figure(figsize=[12,4])
ax = fig.add_subplot(131, title='data')
ax.plot(X, y, 'b+')

###################################
#test the Logistic Regression class
###################################

#create a Logistic Regression object
lregress = lr.Linear_Regression(X, y, alpha=1.0)

#plot the predictions before training
ax = fig.add_subplot(132, title='before training')
y_hat = []
for i in range(N):
    y_hat.append(lregress.feed_forward(X[i]))
    
ax.plot(X, y, 'b+')
ax.plot(X, y_hat, 'r+')

#train the ANN
lregress.train(1000)

#plot the results after prediction
ax = fig.add_subplot(133, title='after training')
y_hat = []
for i in range(N):
    y_hat.append(lregress.feed_forward(X[i]))

ax.plot(X, y, 'b+')
ax.plot(X, y_hat, 'r+')

plt.tight_layout()

plt.show()
