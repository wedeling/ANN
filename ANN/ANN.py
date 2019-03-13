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
        self.n_layers = 2
        
        #number of neurons per hidden layer
        self.n_hidden = 2
        
        #number of output neurons
        self.n_outputs = 1

        #total number of neurons
        self.n_neurons = self.n_in + self.n_hidden*(self.n_layers-1) + self.n_outputs
        
