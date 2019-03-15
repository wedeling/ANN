from ANN import ANN
import numpy as np      
import Neuron as neuron

class Layer(ANN):
    
    def __init__(self, n_neurons, r):
        
        self.n_neurons = n_neurons
        self.r = r
        self.n_layers = super().n_layers

    def meet_the_neighbors(self):
        #if this layer is an input layer
        if self.r == 0:
            self.layer_rm1 = None
            self.layer_rp1 = super().get_layer(self.r+1)
        #if this layer is an output layer
        elif self.r == self.n_layer:
            self.layer_rp1 = None
            self.layer_rm1 = super().get_layer(self.r-1)
        #if this layer is hidden
        else:
            self.layer_rm1 = super.get_layer(self.r-1)        
            self.layer_rp1 = super.get_layer(self.r+1)        
       
    def set_h(self, idx):
        self.h[idx] = 1
        
