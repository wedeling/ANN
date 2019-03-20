def get_y_lin(X, N):
    
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

def get_y_quad(X, N):

    #create the classification labels
    y = np.zeros(N)
    
    #choose condition for label 1
    idx1 = np.where(X[:, 1] > X[:,0]**2)
    
    #condition for label -1 is just the complement of the label 1 set
    idxm1 = np.setdiff1d(np.arange(N), idx1)
    
    #set labels
    y[idx1] = 1.0
    y[idxm1] = -1.0

    return y, idx1, idxm1

def get_y_quadrant(X, N):

    #create the classification labels
    y = np.zeros(N)
    
    #choose condition for label 1
    idx1 = np.where(np.sign(X[:, 1]) == np.sign(X[:,0]))
    
    #condition for label -1 is just the complement of the label 1 set
    idxm1 = np.setdiff1d(np.arange(N), idx1)
    
    #set labels
    y[idx1] = 1.0
    y[idxm1] = -1.0

    return y, idx1, idxm1
    
import numpy as np