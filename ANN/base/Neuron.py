import numpy as np

class Neuron:
    
    def __init__(self, activation, layer_rm1, layer_r, layer_rp1):
        
        #activation function type
        self.activation = activation     

        #layer neighborhood
        self.layer_rm1 = layer_rm1
        self.layer_r = layer_r
        self.layer_rp1 = layer_rp1
        
        #initialize the weights
        self.w = np.random.randn(self.layer_rm1.n_neurons)
        
    #Compute the value of h, i.e. the post activation value of the neuron
    #Also update the value in layer_rp1 using its set_h subroutine
    def compute_h(self):
        
        #multiply output of previous layer with the weights of this layer
        a = np.dot(self.layer_rm1.h, self.w)
        
        #apply activation to a
        if self.activation == 'linear':
            return a
        elif self.activation == 'relu':
            return np.max([0, a])
        else:
            print('Unknown activation type')
            import sys; sys.exit()
   
    #compute $\Delta(h, 0):=\partial L/\partial h$
    #Requires: all delta_ho of layer_rp1
    #What else????
    def compute_delta_ho(self):
        print('compute delta_ho')
       
    def compute_grad_w(self):
        self.compute_delta_ho()