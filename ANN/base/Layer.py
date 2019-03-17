from .Neuron import Neuron

class Layer:
    
    def __init__(self, n_neurons, r, n_layers):
        
        self.n_neurons = n_neurons
        self.r = r
        self.n_layers = n_layers

    #connect this layer to its neighbors
    def meet_the_neighbors(self, layer_rm1, layer_rp1):
        #if this layer is an input layer
        if self.r == 0:
            self.layer_rm1 = None
            self.layer_rp1 = layer_rp1
        #if this layer is an output layer
        elif self.r == self.n_layers:
            self.layer_rm1 = layer_rm1
            self.layer_rp1 = None
        #if this layer is hidden
        else:
            self.layer_rm1 = layer_rm1
            self.layer_rp1 = layer_rp1
        
        if self.r != 0:
            self.seed_neurons()
            
    def seed_neurons(self):
        neurons = []
        
        for i in range(self.n_neurons):
            neurons.append(Neuron('test', self.layer_rm1, self, self.layer_rp1))
            
        self.neurons = neurons