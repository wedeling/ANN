class Layer:
    
    def __init__(self, n_neurons, r, n_layers):
        
        self.n_neurons = n_neurons
        self.r = r

    def meet_the_neighbors(self, layer_rm1, layer_rp1):
        #if this layer is an input layer
        if self.r == 0:
            self.layer_rm1 = None
            self.layer_rp1 = layer_rp1
        #if this layer is an output layer
        elif self.r == self.n_layer:
            self.layer_rm1 = layer_rm1
            self.layer_rp1 = None
        #if this layer is hidden
        else:
            self.layer_rm1 = layer_rm1
            self.layer_rp1 = layer_rp1