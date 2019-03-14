from Layer import Layer

class ANN:

    def __init__(self, X, y, alpha = 1.0):

        #the features
        self.X = X
        
        #the training outputs
        self.y = y
        
        #number of input nodes
        try:
            self.n_in = X.shape[1]
        except IndexError:
            self.n_in = 1
        
        #number of layers (hidden + output)
        self.n_layers = 3

        #number of neurons in a hidden layer
        self.n_neurons_hid = 2

        #number of output neurons
        self.n_outputs = 1

        self.layers = []
        
        #add the input layer
        self.layers.append(Layer(self.n_in, 0)) 
        
        for r in range(1, self.n_layers-1):
            self.layers.append(Layer(self.n_neurons_hid, r))