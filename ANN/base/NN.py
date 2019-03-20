import numpy as np
from .Layer import Layer

class ANN:

    def __init__(self, X, y, alpha = 1.0):

        #the features
        self.X = X
        
        #number of training data points
        self.n_train = X.shape[0]
        
        #the training outputs
        self.y = y
        
        #training rate
        self.alpha = alpha
        
        #number of input nodes
        try:
            self.n_in = X.shape[1]
        except IndexError:
            self.n_in = 1
        
        #number of layers (hidden + output)
        self.n_layers = 3

        #number of neurons in a hidden layer
        self.n_neurons_hid = 10

        #number of output neurons
        self.n_out = 1
        
        #loss function type
        self.loss = 'perceptron_crit'

        self.layers = []
        
        #add the input layer
        self.layers.append(Layer(self.n_in, 0, self.n_layers, 'linear', self.loss)) 
        
        #add the hidden layers
        for r in range(1, self.n_layers):
            self.layers.append(Layer(self.n_neurons_hid, r, self.n_layers, 'relu', self.loss))
        
        #add the output layer
        self.layers.append(Layer(self.n_out, self.n_layers, self.n_layers, 'linear', self.loss))
        
        self.connect_layers()
   
    #connect each layer in the NN with its previous and the next      
    def connect_layers(self):
        
        self.layers[0].meet_the_neighbors(None, self.layers[1])
        self.layers[-1].meet_the_neighbors(self.layers[-2], None)
        
        for i in range(1, self.n_layers):
            self.layers[i].meet_the_neighbors(self.layers[i-1], self.layers[i+1])
    
    #run the network forward
    def feed_forward(self, X):
        
        #set the features at the output of in the input layer
        self.layers[0].h = X
        
        for i in range(1, self.n_layers+1):
            self.layers[i].compute_output()
            
        return self.layers[-1].h
    
    def back_prop(self, y_i):

        #start back propagation over hidden layers, starting with layer before output layer
        for i in range(self.n_layers, 0, -1):
            self.layers[i].back_prop(y_i)
        
    #update step of the weights
    def epoch(self, X_i, y_i):
        
        self.feed_forward(X_i)
        self.back_prop(y_i)
        
        for i in range(1, self.n_layers+1):
            #gradient descent update step
            self.layers[i].W = self.layers[i].W - self.alpha*self.layers[i].L_grad_W
    
    #train the neural network        
    def train(self, n_epoch):
        
        for i in range(n_epoch):

            #select a random training instance (X, y)
            rand_idx = np.random.randint(0, self.n_train)
            
            self.epoch(self.X[rand_idx], self.y[rand_idx])
      
    #compute the number of misclassifications
    def compute_loss(self):
        
        n_misclass = 0.0
        
        for i in range(self.n_train):
            y_hat_i = np.sign(self.feed_forward(self.X[i]))
            
            if y_hat_i != self.y[i]:
                n_misclass += 1
                
        print('Number of misclassifications = ', n_misclass)
        
    #return the number of weights
    def get_n_weights(self):
        
        n_weights = 0
        
        for i in range(1, self.n_layers+1):
            n_weights += self.layers[i].W.size
            
        print('This neural network has', n_weights, 'weights.')