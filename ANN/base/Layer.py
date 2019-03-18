from .Neuron import Neuron
import numpy as np

class Layer:
    
    def __init__(self, n_neurons, r, n_layers, activation):
        
        self.n_neurons = n_neurons
        self.r = r
        self.n_layers = n_layers
        self.activation = activation
        
        self.h = np.zeros(n_neurons)

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
            
    #initialize the neurons of this layer
    def seed_neurons(self):
        neurons = []
        
        for i in range(self.n_neurons):
            neurons.append(Neuron(self.activation, self.layer_rm1, self, self.layer_rp1))
            
        self.neurons = neurons
        
    #return the output of the current layer
    def compute_output(self):
        for i in range(self.n_neurons):
            self.h[i] = self.neurons[i].compute_h()