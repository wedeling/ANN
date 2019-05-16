from .Neuron import Neuron
#import numpy as np

class Layer:
    
    def __init__(self, n_neurons, r, n_layers, activation, loss, bias = False, \
                 neuron_based_compute = False, batch_size = 1, lamb = 0.0, on_gpu = False):
        
        self.n_neurons = n_neurons
        self.r = r
        self.n_layers = n_layers
        self.activation = activation
        self.loss = loss
        self.bias = bias
        self.neuron_based_compute = neuron_based_compute
        self.batch_size = batch_size
        self.lamb = lamb   
        
        #use either numpy or cupy via xp based on the on_gpu flag
        global xp
        if on_gpu == False:
            import numpy as xp
        else:
            import cupy as xp

        if self.bias == True:
            self.n_bias = 1
        else:
            self.n_bias = 0
       
        self.a = xp.zeros([n_neurons, batch_size])
        self.h = xp.zeros([n_neurons + self.n_bias, batch_size])
        self.delta_ho = xp.zeros([n_neurons, batch_size])
        self.grad_Phi = xp.zeros([n_neurons, batch_size])
        
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

        #initialize the weight, gradient and momentum matrix
#        self.W = xp.zeros([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])
        self.W = xp.random.randn(self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons)*xp.sqrt(1.0/self.layer_rm1.n_neurons)
        self.L_grad_W = xp.zeros([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])
        self.V = xp.zeros([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])
        self.A = xp.zeros([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])
        self.Lamb = xp.ones([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])*self.lamb
        
        #do not apply regularization to the bias terms
        if self.bias == True:
            self.Lamb[-1, :] = 0.0

        if self.neuron_based_compute == True:
            neurons = []
            
            for j in range(self.n_neurons):
                neurons.append(Neuron(self.activation, self.loss, self.layer_rm1, self, self.layer_rp1, j))
                
            for j in range(self.n_neurons, self.n_neurons + self.n_bias):
                neurons.append(Neuron('bias', self.loss, self.layer_rm1, self, self.layer_rp1, j))
                
            self.neurons = neurons
        
    #return the output of the current layer, computed locally at each neuron
    def compute_output_local(self):
        for i in range(self.n_neurons + self.n_bias):
            self.neurons[i].compute_h()
       
        #compute the gradient of the activation function, 
        self.compute_grad_Phi()
            
    #compute the output of the current layer in one shot using matrix - vector/matrix multiplication    
    def compute_output(self, batch_size):
        
        a = xp.dot(self.W.T, self.layer_rm1.h)
       
        #apply activation to a
        if self.activation == 'linear':
            self.h = a
        elif self.activation == 'relu':
            self.h = xp.maximum(xp.zeros([a.shape[0], a.shape[1]]), a)
        elif self.activation == 'tanh':
            self.h = xp.tanh(a)
        elif self.activation == 'hard_tanh':
            
            aa = xp.copy(a)
            idx_gt1 = xp.where(a >= 1.0)
            idx_ltm1 = xp.where(a <= -1.0)
            aa[idx_gt1[0], idx_gt1[1]] = 1.0
            aa[idx_ltm1[0], idx_ltm1[1]] = -1.0
            
            self.h = aa

        else:
            print('Unknown activation type')
            import sys; sys.exit()
            
        #add bias neuron output
        if self.bias == True:
            self.h = xp.vstack([self.h, xp.ones(batch_size)])
        self.a = a

        #compute the gradient of the activation function, 
        self.compute_grad_Phi()
                
    #compute the gradient in the activation function Phi wrt its input
    def compute_grad_Phi(self):
        
        if self.activation == 'linear':
            self.grad_Phi = xp.ones([self.n_neurons, self.batch_size])
        elif self.activation == 'relu':
            idx_lt0 = xp.where(self.a < 0.0)
            self.grad_Phi = xp.ones([self.n_neurons, self.batch_size])
            self.grad_Phi[idx_lt0[0], idx_lt0[1]] = 0.0
        elif self.activation == 'tanh':
            self.grad_Phi = 1.0 - self.h[0:self.n_neurons]**2
        elif self.activation == 'hard_tanh':
            idx = xp.where(xp.logical_and(self.a > -1.0, self.a < 1.0))
            self.grad_Phi = xp.zeros([self.n_neurons, self.batch_size])
            self.grad_Phi[idx[0], idx[1]] = 1.0

    #compute the value of the loss function
    def compute_loss(self, y_i):
        
        h = self.h
        
        #only compute if in an output layer
        if self.layer_rp1 == None:
            if self.loss == 'perceptron_crit':
                self.L_i = xp.max([-y_i*h, 0.0])
            elif self.loss == 'hinge':
                self.L_i = xp.max([1.0 - y_i*h, 0.0])
            elif self.loss == 'logistic':
                self.L_i = xp.log(1.0 + xp.exp(-y_i*h))
            elif self.loss == 'squared':
                self.L_i = (y_i - h)**2
            else:
                print('Cannot compute loss: unknown loss and/or activation function')
                import sys; sys.exit()
        
    #compute the gradient of the output wrt the activation functions of this layer
    def compute_delta_hy(self):

        #if this layer is the output layer        
        if self.layer_rp1 == None:
            self.delta_hy = xp.ones([self.n_neurons, self.batch_size])
        else:
            #get the delta_ho values of the next layer (layer r+1)
            delta_hy_rp1 = self.layer_rp1.delta_hy
            
            #get the grad_Phi values of the next layer
            grad_Phi_rp1 = self.layer_rp1.grad_Phi
            
            #the weight matrix of the next layer
            W_rp1 = self.layer_rp1.W
            
            self.delta_hy = xp.dot(W_rp1, delta_hy_rp1*grad_Phi_rp1)[0:self.n_neurons, :]
    
    #initialize the value of delta_ho at the output layer
    def compute_delta_oo(self, y_i):
        
        #if the neuron is in the output layer, initialze delta_oo
        if self.layer_rp1 == None:
            
            #compute the loss function
            self.compute_loss(y_i)
            
            h = self.h
            
            if self.loss == 'logistic' and self.activation == 'linear':

                self.delta_ho = -y_i*xp.exp(-y_i*h)/(1.0 + xp.exp(-y_i*h))

            elif self.loss == 'squared' and self.activation == 'linear':
                
                self.delta_ho = -2.0*(y_i - h)
        else:
            print('Can only initialize delta_oo in output layer')
            import sys; sys.exit()
            
    #compute the gradient of the loss function wrt the activation functions of this layer
    def compute_delta_ho(self):
        #get the delta_ho values of the next layer (layer r+1)
        delta_ho_rp1 = self.layer_rp1.delta_ho
        
        #get the grad_Phi values of the next layer
        grad_Phi_rp1 = self.layer_rp1.grad_Phi
        
        #the weight matrix of the next layer
        W_rp1 = self.layer_rp1.W
        
        self.delta_ho = xp.dot(W_rp1, delta_ho_rp1*grad_Phi_rp1)[0:self.n_neurons, :]

    #compute the gradient of the loss function wrt the weights of this layer
    def compute_L_grad_W(self):
        h_rm1 = self.layer_rm1.h
        
        delta_ho_grad_Phi = self.delta_ho*self.grad_Phi

        self.L_grad_W = xp.dot(h_rm1, delta_ho_grad_Phi.T)
    
    #perform the backpropogation operations of the current layer
    def back_prop(self, y_i):
        
        if self.neuron_based_compute == False:
            if self.r == self.n_layers:
                self.compute_delta_oo(y_i)
                self.compute_L_grad_W()
            else:
                self.compute_delta_ho()
                self.compute_L_grad_W()
        else:
            if self.r == self.n_layers:
                #initialize delta_oo
                for i in range(self.n_neurons):
                    self.neurons[i].compute_delta_oo(y_i)
                    self.neurons[i].compute_L_grad_W()
            else:
                for i in range(self.n_neurons):
                    self.neurons[i].compute_delta_ho()
                    self.neurons[i].compute_L_grad_W()            
