"""
============================
SIMPLE PERCEPTRON CLASSIFIER
============================
"""

class Perceptron:

    #X = the features
    #y = the data labels (y \in \{-1, 1\})
    def __init__(self, X, y, alpha = 1.0):
        self.X = X
        self.y = y
        
        #number of training samples
        self.N_train = X.shape[0]
        
        #number of input nodes
        self.N_in = X.shape[1]
        
        #number of output nodes
        self.N_out = 1
        
        #training rate
        self.alpha = alpha
        
        #initialize weights
        self.w = np.random.randn(self.N_in)
        
        #list used to store loss values
        self.loss = []
        
    #predict
    def feed_forward(self, X_i):
        
        a = np.dot(self.w, X_i)
        
        return np.sign(a)

    #run one training cycle using gradient descent    
    def epoch(self, X_i, y_i):
        
        a = np.dot(self.w, X_i)
        
        #smooth surrogate objective function
        L_i = np.max([-y_i*a, 0.0])

        y_hat_i = self.feed_forward(X_i)

        #gradient of L_i wrt weights
        if L_i == 0.0:
            grad_L_i = 0.0
        elif y_hat_i == 1.0 and y_i == -1.0:
            grad_L_i = X_i
        elif y_hat_i == -1.0 and y_i == 1.0:
            grad_L_i = -X_i
            
        #gradient descent step
        self.w = self.w - self.alpha*grad_L_i
    
    #train the Perceptron for N_iter epochs
    def train(self, N_epoch, store_loss = False):
        
        for i in range(N_epoch):

            #select a random training instance (X, y)
            rand_idx = np.random.randint(0, self.N_train)

            self.epoch(self.X[rand_idx], self.y[rand_idx])
            
            if store_loss == True:
                self.loss.append(self.compute_loss())
            
    #compute the total 0/1 loss function, counts the number of misclassifications
    def compute_loss(self):
    
        a = np.dot(self.X, self.w)
        return np.sum(1.0 - np.sign(a)*self.y)
        
import numpy as np            