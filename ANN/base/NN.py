import numpy as np
import os, pickle
from .Layer import Layer

class ANN:

    def __init__(self, X, y, alpha = 0.001, decay_rate = 1.0, decay_step = 10**4, beta1 = 0.9, beta2 = 0.999, lamb = 0.0, \
                 phi = 0.0, lamb_J = 0.0, n_out = 1, \
                 param_specific_learn_rate = True, loss = 'squared', activation = 'tanh', activation_out = 'linear', \
                 n_layers = 2, n_neurons = 16, \
                 bias = True, neuron_based_compute = False, batch_size = 1, save = True, name='ANN', on_gpu = False, \
                 standardize_X = True, standardize_y = True):

        #the features
        self.X = X
        
        #number of training data points
        self.n_train = X.shape[0]
        
        #the training outputs
        self.y = y
                
        #number of input nodes
        try:
            self.n_in = X.shape[1]
        except IndexError:
            self.n_in = 1
        
        #use either numpy or cupy via xp based on the on_gpu flag
        global xp
        print('===============================')
        print('Neural net parameters')
        print('===============================')

        if on_gpu == False:
            print('Numpy-based computation')
            import numpy as xp
        else:
            print('Cupy-based computation')
            import cupy as xp
            
        print('Number of layers =', n_layers)
        print('Number of features =', self.n_in)
        print('Loss function =', loss)
        print('Number of neurons per hidden layer =', n_neurons)
        print('Number of output neurons =', n_out)
        print('Activation hidden layers =', activation)
        print('Activation output layer =', activation_out)
        print('Loss function =', loss)
        print('===============================')

        #standardize the training data
        if standardize_X == True:
            
            self.X_mean = xp.mean(X, axis = 0)
            self.X_std = xp.std(X, axis = 0)
            self.X = (X - self.X_mean)/self.X_std
        
        if standardize_y == True:
            self.y_mean = xp.mean(y, axis = 0)
            self.y_std = xp.std(y, axis = 0)
            self.y = (y - self.y_mean)/self.y_std
        
        #number of layers (hidden + output)
        self.n_layers = n_layers

        #number of neurons in a hidden layer
        self.n_neurons = n_neurons

        #number of output neurons
        self.n_out = n_out
        
        #use bias neurons
        self.bias = bias
        
        #loss function type
        self.loss = loss

        #training rate
        self.alpha = alpha

        #L2 regularization parameter
        self.lamb = lamb

        #Jacobian regularization and finite difference parameter (phi)
        self.lamb_J = lamb_J
        self.phi = phi
        self.test = []

        #the rate of decay and decay step for alpha
        self.decay_rate = decay_rate        
        self.decay_step = decay_step

        #momentum parameter
        self.beta1 = beta1
        
        #squared gradient parameter
        self.beta2 = beta2
        
        #use parameter specific learning rate
        self.param_specific_learn_rate = param_specific_learn_rate

        #activation function of the hidden layers
        self.activation = activation
        
        #activation function of the output layer
        self.activation_out = activation_out
        
        #save the neural network after training
        self.save = save
        self.name = name
        
        #determines where to compute the neuron outputs and gradients 
        #True: locally at the neuron, False: on the Layer level in one shot via linear algebra)
        self.neuron_based_compute = neuron_based_compute

        #size of the mini batch used in stochastic gradient descent
        self.batch_size = batch_size
        
        self.loss_vals = []

        self.layers = []
        
        #add the input layer
        self.layers.append(Layer(self.n_in, 0, self.n_layers, 'linear', \
                                 self.loss, self.bias, batch_size = batch_size, lamb = lamb, \
                                 neuron_based_compute=neuron_based_compute, on_gpu=on_gpu)) 
        
        #add the hidden layers
        for r in range(1, self.n_layers):
            self.layers.append(Layer(self.n_neurons, r, self.n_layers, self.activation, \
                                     self.loss, self.bias, batch_size=batch_size, lamb = lamb,\
                                     neuron_based_compute=neuron_based_compute, on_gpu=on_gpu))
        
        #add the output layer
        self.layers.append(Layer(self.n_out, self.n_layers, self.n_layers, self.activation_out, \
                                 self.loss, batch_size=batch_size, lamb = lamb,\
                                 neuron_based_compute = neuron_based_compute, on_gpu=on_gpu))
        
        self.connect_layers()
   
    #connect each layer in the NN with its previous and the next      
    def connect_layers(self):
        
        self.layers[0].meet_the_neighbors(None, self.layers[1])
        self.layers[-1].meet_the_neighbors(self.layers[-2], None)
        
        for i in range(1, self.n_layers):
            self.layers[i].meet_the_neighbors(self.layers[i-1], self.layers[i+1])
    
    #run the network forward
    #X_i needs to have shape [batch size, number of features]
    def feed_forward(self, X_i, batch_size = 1):
              
        #set the features at the output of in the input layer
        if self.bias == False:
            self.layers[0].h = X_i
        else:
            self.layers[0].h = xp.ones([self.n_in + 1, batch_size])
            self.layers[0].h[0:self.n_in, :] = X_i.T
                    
        for i in range(1, self.n_layers+1):
            if self.neuron_based_compute:
                #compute the output locally in each neuron
                self.layers[i].compute_output_local()
            else:
                #compute the output on the layer using matrix-maxtrix multiplication
                self.layers[i].compute_output(batch_size)
            
        return self.layers[-1].h
    
    #get the output of the softmax layer
    def get_softmax(self, X_i, batch_size = 1):
        
        h = self.feed_forward(X_i, batch_size = batch_size)
        return np.exp(h)/np.sum(np.exp(h), axis=0)
 
    #compute jacobian of the neural net via back propagation
    def jacobian(self, X_i, batch_size = 1, feed_forward = False):
        
        if feed_forward == True:
            self.feed_forward(X_i, batch_size=batch_size)
        
        for i in range(self.n_layers, -1, -1):
            self.layers[i].compute_delta_hy()
            #also compute the gradient of the output wrt the weights
            if i > 0:
                self.layers[i].compute_y_grad_W()
            
        #delta_hy of the input layer = the Jacobian of the neural net
        return self.layers[0].delta_hy
            
    #compute jacobian via finite differences
    def jacobian_FD(self, X_i, batch_size = 1):
        
        eps = 1e-8
        
        jac_FD = np.zeros([self.n_in, batch_size])
        
        h1 = self.feed_forward(X_i, batch_size = batch_size)
        
        for i in range(self.n_in):
            X2 = np.copy(X_i)
            
            #perturb a single feature
            X2[:, i] += eps
        
            h2 = self.feed_forward(X2, batch_size = batch_size)
        
            #FD approximation of gradient of NN output wrt i-th feature
            jac_FD[i] = (h2 - h1)/eps
            
        return jac_FD
    
    #back propagation algorithm    
    def back_prop(self, y_i):

        #start back propagation over hidden layers, starting with output layer
        for i in range(self.n_layers, 0, -1):
            self.layers[i].back_prop(y_i)

    #update step of the weights
    def batch(self, X_i, y_i, alpha=0.001, beta1=0.9, beta2=0.999, t=0):
        
        self.feed_forward(X_i, self.batch_size)
        self.back_prop(y_i)

        #if Jacobian regularization is used
        if self.phi > 0.0:
            self.jacobian(X_i, batch_size=self.batch_size)
            self.test.append(np.linalg.norm(self.layers[0].delta_hy)**2)
        
        for r in range(1, self.n_layers+1):

            layer_r = self.layers[r]
            
            #momentum 
            layer_r.V = beta1*layer_r.V + (1.0 - beta1)*layer_r.L_grad_W
            
            #moving average of squared gradient magnitude
            layer_r.A = beta2*layer_r.A + (1.0 - beta2)*layer_r.L_grad_W**2

            #select learning rate            
            if self.param_specific_learn_rate == False:
                #same alpha for all weights
                alpha_i = alpha
            #param specific learning rate
            else:
                #RMSProp
                alpha_i = alpha/(xp.sqrt(layer_r.A + 1e-8))
                
                #Adam
                #alpha_t = alpha*xp.sqrt(1.0 - beta2**t)/(1.0 - beta1**t)
                #alpha_i = alpha_t/(xp.sqrt(layer_r.A + 1e-8))

            #gradient descent update step
            if self.lamb > 0.0:
                #with L2 regularization
                layer_r.W = (1.0 - layer_r.Lamb*alpha_i)*layer_r.W - alpha_i*layer_r.V
            elif self.phi > 0.0:
                
                #dydx, the Jacobian of the output y
                dydX = self.layers[0].delta_hy
                
                #dydW
                dydW = layer_r.y_grad_W
                
                #regularize wrt squared Frobenius norm R ==> dRda = 2a, a = dydx
                #This is the 'adversarial direction', direc of max. change
                X_hat = X_i + self.phi*2.0*dydX.T
                
                #double back prop
                self.feed_forward(X_hat, batch_size=self.batch_size)
                self.jacobian(X_i, batch_size=self.batch_size)
                
                #output gradient wrt weights of the adversarial example
                #CHECK THIS TERM AS FUNCTION OF PHI
                dydW_hat = layer_r.y_grad_W
                
                #FD approximation of the mixed partial derivative of y wrt W and X
                #CHECK THIS TERM
                d2y_dWdx = (dydW_hat - dydW)/self.phi
                
                #weight update with Jacobian regularization (NO MOMENTUM TO MIXED TERM - OK??)
                #PLOT MIXED DERIVATIVE, SEE IF NOISY, AND IF IT COULD BENEFIT FROM MOMENTUM SMOOTHING
                layer_r.W = layer_r.W - alpha_i*layer_r.V - alpha_i*self.lamb_J*d2y_dWdx
                
            else:
                #without regularization
                layer_r.W = layer_r.W - alpha_i*layer_r.V
           
            #Nesterov momentum
            #layer_r[i].W += -alpha*beta1*layer_r.V
    
    #train the neural network        
    def train(self, n_epoch, store_loss = False, check_derivative = False):
        
        for i in range(n_epoch):

            #select a random training instance (X, y)
            rand_idx = np.random.randint(0, self.n_train, self.batch_size)
            
            #compute learning rate
            alpha = self.alpha*self.decay_rate**(np.int(i/self.decay_step))

            #run the batch
            self.batch(self.X[rand_idx], self.y[rand_idx].T, alpha=alpha, beta1=self.beta1, beta2=self.beta2, t=i+1)
            
            if check_derivative == True and np.mod(i, 1000) == 0:
                self.check_derivative(self.X[rand_idx], self.y[rand_idx], 10)
            
            #store the loss value 
            if store_loss == True:
                l = 0.0
                for k in range(self.n_out):
                    if self.neuron_based_compute:
                        l += self.layers[-1].neurons[k].L_i
                    else:
                        l += self.layers[-1].L_i
                
                if np.mod(i, 1000) == 0:
                    loss_i = xp.mean(l)
                    print('Batch', i, 'learning rate', alpha ,'loss:', loss_i)
                    #note: appending a cupy value to a list is inefficient - if done every iteration
                    #it will slow down executing significantly
                    self.loss_vals.append(loss_i)
                    
        if self.save == True:
            self.save_ANN()

    #save using pickle (maybe too slow for very large ANNs?)
    def save_ANN(self):
        
        #absolute path
        home = os.path.abspath(os.path.dirname(__file__))
        path = home + '/../saved_networks/'
        
        print('Saving ANN to', path + self.name + '.pickle')
        
        if os.path.exists(path) == False:
            os.makedirs(path)
        
        file = open(path + self.name + '.pickle', 'wb')
        pickle.dump(self.__dict__, file)
        file.close()

    #load using pickle
    def load_ANN(self, name):

        #absolute path
        home = os.path.abspath(os.path.dirname(__file__))
        path = home + '/../saved_networks/'

        print('Loading ANN from', path + name + '.pickle')

        file = open(path + name + '.pickle', 'rb')
        self.__dict__ = pickle.load(file)
        file.close()
        
    def set_batch_size(self, batch_size):
        
        self.batch_size = batch_size
        
        for i in range(self.n_layers+1):
            self.layers[i].batch_size = batch_size

    #compare a random back propagation derivative with a finite-difference approximation
    def check_derivative(self, X_i, y_i, n_checks):
        
        eps = 1e-6
        print('==============================================')
        print('Performing derivative check of', n_checks, 'randomly selected neurons.')
        
        for i in range(n_checks):
            
            #'align' the netwrok with the newly computed gradient and compute the loss function
            self.feed_forward(X_i)
            self.layers[-1].neurons[0].compute_loss(y_i)            
            L_i_old = self.layers[-1].neurons[0].L_i

            #select a random neuron which has a nonzero gradient            
            L_grad_W_old = 0.0
            while L_grad_W_old == 0.0:

                #select a random neuron
                layer_idx = np.random.randint(1, self.n_layers+1)
                neuron_idx = np.random.randint(self.layers[layer_idx].n_neurons)
                weight_idx = np.random.randint(self.layers[layer_idx-1].n_neurons)
             
                #the unperturbed weight and gradient
                w_old = self.layers[layer_idx].W[weight_idx, neuron_idx]
                L_grad_W_old = self.layers[layer_idx].L_grad_W[weight_idx, neuron_idx]
            
            #perturb weight
            self.layers[layer_idx].W[weight_idx, neuron_idx] += eps
            
            #run the netwrok forward and compute loss
            self.feed_forward(X_i)
            self.layers[-1].neurons[0].compute_loss(y_i)            
            L_i_new = self.layers[-1].neurons[0].L_i
                        
            #simple FD approximation of the gradient
            L_grad_W_FD = (L_i_new - L_i_old)/eps
  
            print('Back-propogation gradient:', L_grad_W_old)
            print('FD approximation gradient:', L_grad_W_FD)
           
            #reset weights and network
            self.layers[layer_idx].W[weight_idx, neuron_idx] = w_old
            self.feed_forward(X_i)

        print('==============================================')
      
    #compute the number of misclassifications
    def compute_misclass(self):
        
        n_misclass = 0.0
        
        for i in range(self.n_train):
            y_hat_i = xp.sign(self.feed_forward(self.X[i]))
            
            if y_hat_i != self.y[i]:
                n_misclass += 1
                
        print('Number of misclassifications = ', n_misclass)
        
    #compute the number of misclassifications for a sofmax layer
    def compute_misclass_softmax(self):
        
        n_misclass = 0.0
        
        for i in range(self.n_train):
            o_i = self.get_softmax(self.X[i].reshape([1, self.n_in])).flatten()
            
            idx1 = np.argmax(o_i)
            idx2 = np.where(self.y[i] == 1.0)[0][0]

            if idx1 != idx2:
                n_misclass += 1
                
        print('Number of misclassifications =', n_misclass)
        print('Misclassification percentage =', n_misclass/self.n_train*100, '%')
        
    #return the number of weights
    def get_n_weights(self):
        
        n_weights = 0
        
        for i in range(1, self.n_layers+1):
            n_weights += self.layers[i].W.size
            
        print('This neural network has', n_weights, 'weights.')

        return n_weights
