from .Neuron import Neuron
import numpy as np

class Layer:
    
    def __init__(self, n_neurons, r, n_layers, activation, loss):
        
        self.n_neurons = n_neurons
        self.r = r
        self.n_layers = n_layers
        self.activation = activation
        self.loss = loss
        
        self.h = np.zeros(n_neurons)
        self.delta_ho = np.zeros(n_neurons)
        self.grad_Phi = np.zeros(n_neurons)
        
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
        
        #fill the layer with neurons
        if self.r != 0:
            self.seed_neurons()
        
    #initialize the neurons of this layer
    def seed_neurons(self):

        #initialize the weight matrix
        self.W = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])
        self.L_grad_W = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])

        neurons = []
        
        for j in range(self.n_neurons):
            neurons.append(Neuron(self.activation, self.loss, self.layer_rm1, self, self.layer_rp1, j))
            
        self.neurons = neurons
        
    #return the output of the current layer
    def compute_output(self):
        for i in range(self.n_neurons):
            self.h[i] = self.neurons[i].compute_h()
    
    #perform the backpropogation operations of the current layer
    def back_prop(self, y_i):
 
        if self.r == self.n_layers:
            #initialize delta_oo
            for i in range(self.n_neurons):
                self.neurons[i].compute_delta_oo(y_i)
                self.neurons[i].compute_L_grad_W()
        else:
            for i in range(self.n_neurons):
                self.neurons[i].compute_delta_ho()
                self.neurons[i].compute_L_grad_W()