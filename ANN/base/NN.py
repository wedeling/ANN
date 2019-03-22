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
        self.n_layers = 2

        #number of neurons in a hidden layer
        self.n_neurons_hid = 10

        #number of output neurons
        self.n_out = 1
        
        #use bias neurons
        self.bias = True
        
        #loss function type
        #self.loss = 'perceptron_crit'
        #self.loss = 'hinge'
        #self.loss = 'logistic'
        self.loss = 'squared'

        self.loss_vals = []

        self.layers = []
        
        #add the input layer
        self.layers.append(Layer(self.n_in, 0, self.n_layers, 'linear', self.loss, self.bias)) 
        
        #add the hidden layers
        for r in range(1, self.n_layers):
            self.layers.append(Layer(self.n_neurons_hid, r, self.n_layers, 'tanh', self.loss))
        
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
    def train(self, n_epoch, store_loss = False, check_derivative = False):
        
        for i in range(n_epoch):

            #select a random training instance (X, y)
            rand_idx = np.random.randint(0, self.n_train)
            self.epoch(self.X[rand_idx], self.y[rand_idx])
            
            if check_derivative == True and np.mod(i, 1000) == 0:
                self.check_derivative(self.X[rand_idx], self.y[rand_idx], 10)
            
            if store_loss == True:
                l = 0.0
                for k in range(self.n_out):
                    l += self.layers[-1].neurons[k].L_i
                self.loss_vals.append(l)
                
                if np.mod(i, 1000) == 0:
                    print(np.mean(self.loss_vals))
                    
    #compare a random back propagation derivative with a finite-difference approximation
    def check_derivative(self, X_i, y_i, n_checks):
        
        eps = 1e-6
        print('==============================================')
        print('Performing derivative check of', n_checks, 'randomly selected neurons.')
        
        for i in range(n_checks):
            
            #'align' the netwrok with the newly computed gradient and compute the loss function
            self.feed_forward(X_i)
            self.layers[-1].neurons[0].compute_loss(y_i)            
            L_i_old = self.layers[-1].neurons[0].L_i

            #select a random neuron which has a nonzero gradient            
            L_grad_W_old = 0.0
            while L_grad_W_old == 0.0:

                #select a random neuron
                layer_idx = np.random.randint(1, self.n_layers+1)
                neuron_idx = np.random.randint(self.layers[layer_idx].n_neurons)
                weight_idx = np.random.randint(self.layers[layer_idx-1].n_neurons)
             
                #the unperturbed weight and gradient
                w_old = self.layers[layer_idx].W[weight_idx, neuron_idx]
                L_grad_W_old = self.layers[layer_idx].L_grad_W[weight_idx, neuron_idx]
            
            #perturb weight
            self.layers[layer_idx].W[weight_idx, neuron_idx] += eps
            
            #run the netwrok forward and compute loss
            self.feed_forward(X_i)
            self.layers[-1].neurons[0].compute_loss(y_i)            
            L_i_new = self.layers[-1].neurons[0].L_i
                        
            #simple FD approximation of the gradient
            L_grad_W_FD = (L_i_new - L_i_old)/eps
  
            print('Back-propogation gradient:', L_grad_W_old)
            print('FD approximation gradient:', L_grad_W_FD)
           
            #reset weights and network
            self.layers[layer_idx].W[weight_idx, neuron_idx] = w_old
            self.feed_forward(X_i)

        print('==============================================')
      
    #compute the number of misclassifications
    def compute_misclass(self):
        
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