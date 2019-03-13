class Layer:
    
    def __init__(self, n_neurons, typ, r):
        
        self.n_neurons = n_neurons
        self.typ = typ
        self.h = np.zeros(10)
        self.r = r
        
    def set_h(self, idx):
        self.h[idx] = 1
        
import numpy as np      
        