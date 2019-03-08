"""
=========================================
ANN IMPLEMENTATION OF LOGISTIC REGRESSION
=========================================
"""

class Logistic_Regression:

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
        
    #predict probability using the logistic function
    def feed_forward(self, X_i):
        
        a = np.dot(self.w, X_i)
        
        return 1.0/(1.0 + np.exp(-a))

    #run one training cycle using gradient descent    
    def epoch(self, X_i, y_i):
        
        a = np.dot(self.w, X_i)
        
        #smooth objective function: L_i = np.log(1.0 + np.exp(-y_i*a))

        #gradient of L_i wrt weights
        if y_i == 1.0:
            grad_L_i = -np.exp(-a)/(1.0 + np.exp(-a))*X_i
        elif y_i == -1.0:
            grad_L_i = np.exp(a)/(1.0 + np.exp(a))*X_i
            
        #gradient descent step
        self.w = self.w - self.alpha*grad_L_i
    
    #train the ANN for N_iter epochs
    def train(self, N_epoch, store_loss = False):
        
        for i in range(N_epoch):

            #select a random training instance (X, y)
            rand_idx = np.random.randint(0, self.N_train)

            self.epoch(self.X[rand_idx], self.y[rand_idx])
            
            if store_loss == True:
                self.loss.append(self.compute_loss())
        
import numpy as np            
