class Neuron:
    
    def __init__(self, typ, layer_rp1, layer_r, layer_rm1):
        
        self.typ = typ     
        
        self.layer_rp1 = layer_rp1
        self.layer_r = layer_r
        self.layer_rm1 = layer_rm1
        
    #Compute the value of h, i.e. the post activation value of the neuron
    #Also update the value in layer_rp1 using its set_h subroutine
    def compute_h(self):
        print('Compute h')
    
    #compute $\Delta(h, 0):=\partial L/\partial h$
    #Requires: all delta_ho of layer_rp1
    #What else????
    def compute_delta_ho(self):
        print('compute delta_ho')
        
    
    def compute_grad_w(self):
        self.compute_delta_ho()
        
import numpy as np