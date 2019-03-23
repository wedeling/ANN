def get_y_lin(N):

    #N draws from multivariate normal with mean mu and covariance matrix Sigma
    mu = np.array([0, 0])
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    X = np.random.multivariate_normal(mu, Sigma, size = N)
    
    #create the classification labels
    y = np.zeros(N)
    
    #choose condition for label 1
    idx1 = np.where(X[:, 1] > -X[:,0])
    
    #condition for label -1 is just the complement of the label 1 set
    idxm1 = np.setdiff1d(np.arange(N), idx1)
    
    #set labels
    y[idx1] = 1.0
    y[idxm1] = -1.0

    return X, y, idx1, idxm1

def get_y_quad(N):
    
    #N draws from multivariate normal with mean mu and covariance matrix Sigma
    mu = np.array([0, 0])
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    X = np.random.multivariate_normal(mu, Sigma, size = N)

    #create the classification labels
    y = np.zeros(N)
    
    #choose condition for label 1
    idx1 = np.where(X[:, 1] > X[:,0]**2)
    
    #condition for label -1 is just the complement of the label 1 set
    idxm1 = np.setdiff1d(np.arange(N), idx1)
    
    #set labels
    y[idx1] = 1.0
    y[idxm1] = -1.0

    return X, y, idx1, idxm1

def get_y_quadrant(N):

    #N draws from multivariate normal with mean mu and covariance matrix Sigma
    mu = np.array([0, 0])
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    X = np.random.multivariate_normal(mu, Sigma, size = N)

    #create the classification labels
    y = np.zeros(N)
    
    #choose condition for label 1
    idx1 = np.where(np.sign(X[:, 1]) == np.sign(X[:,0]))
    
    #condition for label -1 is just the complement of the label 1 set
    idxm1 = np.setdiff1d(np.arange(N), idx1)
    
    #set labels
    y[idx1] = 1.0
    y[idxm1] = -1.0

    return X, y, idx1, idxm1

def get_lin_regres(N):
    
    a = -1.0; b = 1.0
    X = np.random.rand(N)*(b-a) + a
    noise = np.random.randn(N)*1e-2
    
    y = X + noise + 1.0
    
    return X, y

def get_quad_regres(N):
    
    a = -1.0; b = 1.0
    X = np.random.rand(N)*(b-a) + a
    noise = np.random.randn(N)*1e-2
    
    y = X**2 + noise
    
    return X, y

def get_sin_regres(N):

    a = 0.0; b = 3.0*np.pi
    X = np.random.rand(N)*(b-a) + a
    noise = np.random.randn(N)*1e-2
    y = np.sin(2*X) + noise
    
    return X, y

import numpy as np