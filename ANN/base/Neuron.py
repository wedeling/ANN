import numpy as np

class Neuron:
    
    def __init__(self, activation, loss, layer_rm1, layer_r, layer_rp1, j):
        
        #activation function and loss function type
        self.activation = activation
        self.loss = loss
        
        #the neuron index, its place in the layer
        self.j = j

        #layer neighborhood
        self.layer_rm1 = layer_rm1
        self.layer_r = layer_r
        self.layer_rp1 = layer_rp1
        
        #initialize the weights
        self.layer_r.W[:, j] = np.random.randn(self.layer_rm1.n_neurons)*np.sqrt(1.0/layer_rm1.n_neurons)
        
        #pre-activation value
        self.a = 0.0        
        
        #post-activation value
        self.h = 0.0
        
    #Compute the value of h, i.e. the post activation value of the neuron
    #Also update the value in layer_rp1 using its set_h subroutine
    def compute_h(self):
        
        w = self.layer_r.W[:, self.j]
        
        #multiply output of previous layer with the weights of this layer
        a = np.dot(self.layer_rm1.h, w)
        
        #apply activation to a
        if self.activation == 'linear':
            self.h = a
        elif self.activation == 'relu':
            self.h = np.max([0, a])
        elif self.activation == 'tanh':
            self.h = np.tanh(a)
        else:
            print('Unknown activation type')
            import sys; sys.exit()
        
        self.a = a

        return self.h
    
    #compute the gradient in the activation function Phi wrt its input
    def compute_grad_Phi(self):
        
        if self.activation == 'linear':
            return 1.0
        elif self.activation == 'relu':
            if self.a >= 0.0:
                return 1.0
            else:
                return 0.0
        elif self.activation == 'tanh':
            return 1.0 - self.h**2
            
    #compute the value of the loss function
    def compute_loss(self, y_i):
        
        #only compute if in an output layer
        if self.layer_rp1 == None:
            if self.loss == 'perceptron_crit':
                self.L_i = np.max([-y_i*self.h, 0.0])
            elif self.loss == 'hinge':
                self.L_i = np.max([1.0 - y_i*self.h, 0.0])
            elif self.loss == 'logistic':
                self.L_i = np.log(1.0 + np.exp(-y_i*self.h))
            elif self.loss == 'squared':
                self.L_i = (y_i - self.h)**2
            else:
                print('Cannot compute loss: unknown loss and/or activation function')
                import sys; sys.exit()
    
    #initialize the value of delta_ho at the output layer
    def compute_delta_oo(self, y_i):
        #if the neuron is in the output layer, initialze delta_oo
        if self.layer_rp1 == None:
            
            #compute the loss function
            self.compute_loss(y_i)
            
            #in the case of the perceptron criterion loss
            if self.loss == 'perceptron_crit' and self.activation == 'linear':
                
                if self.L_i == 0.0:
                    self.delta_ho = 0.0
                elif y_i == 1.0:
                    self.delta_ho = -1.0
                else:
                    self.delta_ho = 1.0
                    
            elif self.loss == 'hinge' and self.activation == 'linear':
                
                if self.L_i == 0.0:
                    self.delta_ho = 0.0
                elif y_i == 1.0:
                    self.delta_ho = -1.0
                else:
                    self.delta_ho = 1.0
                    
            elif self.loss == 'logistic' and self.activation == 'linear':
                
                y_hat_i = self.h
                if y_i == 1.0: 
                    self.delta_ho = -np.exp(-y_hat_i)/(1.0 + np.exp(-y_hat_i))
                else:
                    self.delta_ho = np.exp(y_hat_i)/(1.0 + np.exp(y_hat_i))
                    
            elif self.loss == 'squared' and self.activation == 'linear':
                
                self.delta_ho = -2.0*(y_i - self.h)
            
            #store the value in the r-th layer object
            self.layer_r.delta_ho[self.j] = self.delta_ho
            
            #compute the gradient of the activation function, and store in layer
            self.grad_Phi = self.compute_grad_Phi()
            self.layer_r.grad_Phi[self.j] = self.grad_Phi
            
        else:
            print('Can only initialize delta_oo in output layer')
            import sys; sys.exit()
   
    def compute_delta_ho(self):
        #get the delta_ho values of the next layer (layer r+1)
        delta_h_rp1_o = self.layer_rp1.delta_ho
        
        #get the grad_Phi values of the next layer
        grad_Phi_rp1 = self.layer_rp1.grad_Phi
        
        #get the weights connecting this neuron to all neurons in the next layer
        w_rp1 = self.layer_rp1.W[self.j, :]
        
        #value of delta_ho for this neuron
        self.delta_ho = np.sum(grad_Phi_rp1*w_rp1*delta_h_rp1_o)
        
        #store the value
        self.layer_r.delta_ho[self.j] = self.delta_ho
        
        #compute the gradient of the activation function, 
        #and store in this layer, to be used in the next backprop iteration
        self.grad_Phi = self.compute_grad_Phi()
        self.layer_r.grad_Phi[self.j] = self.grad_Phi
    
    #compute the gradient of the loss function wrt the weights
    def compute_L_grad_W(self):
        
        #The gradient of L wrt all weights of the incoming edges of this neuron
        self.L_grad_W = self.delta_ho*self.grad_Phi*self.layer_rm1.h
        
        #store the result in the Layer_r object
        self.layer_r.L_grad_W[:, self.j] = self.L_grad_W