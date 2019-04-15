import numpy as np
import cupy as cp

#store all the parameters defining the ANN in the global P dict
def init_network(X = cp.zeros(1), y = cp.zeros(0), alpha = 0.001, decay_rate = 1.0, decay_step = 10**4, beta1 = 0.9, beta2 = 0.999, lamb = 0.0, \
                 param_specific_learn_rate = False, loss = 'squared', activation = 'tanh', n_layers = 2, n_neurons = 16, \
                 bias = True, neuron_based_compute = False, batch_size = 1, save = True, name='ANN'):
    
    all_vars = locals()

    for key in all_vars.keys():
        P[key] = all_vars[key]

    #number of training data points
    P['n_train'] = X.shape[1]
    
    #number of input nodes
    try:
        P['n_in'] = X.shape[0]
    except IndexError:
        P['n_in'] = 1

#initialize the parameters of the layers
def init_layers():

    n_layers = P['n_layers']
    batch_size = P['batch_size']

    #number of bias neurons
    if P['bias'] == True:
        n_bias = 1
    else:
        n_bias = 0

    #set hidden layer vars 
    for i in range(1, n_layers):
        layers[i] = {}
        layers[i]['n_neurons'] = P['n_neurons']
        layers[i]['activation'] = P['activation']
        layers[i]['n_bias'] = n_bias

    #set input layer vars
    layers[0] = {}
    layers[0]['n_neurons'] = P['n_in']
    layers[0]['activation'] = 'linear'
    layers[0]['n_bias'] = n_bias
    
    #set output layer vars
    layers[n_layers] = {}
    layers[n_layers]['n_neurons'] = 1 
    layers[n_layers]['activation'] = 'linear'
    layers[n_layers]['n_bias'] = 0

    for i in range(n_layers + 1):
        n_neurons = layers[i]['n_neurons']
        n_bias = layers[i]['n_bias']
        layers[i]['a'] = cp.zeros([n_neurons, batch_size])
        layers[i]['h'] = cp.zeros([n_neurons + n_bias, batch_size])
        layers[i]['delta_ho'] = cp.zeros([n_neurons, batch_size])
        layers[i]['grad_Phi'] = cp.zeros([n_neurons, batch_size])

        if i > 0:
            n_neurons_rm1 = layers[i-1]['n_neurons']
            n_bias_rm1 = layers[i-1]['n_bias']
            layers[i]['W'] = cp.random.randn(n_neurons_rm1 + n_bias_rm1, n_neurons)*cp.sqrt(1.0/n_neurons_rm1)
            layers[i]['L_grad_W'] = cp.zeros([n_neurons_rm1 + n_bias_rm1, n_neurons])
            layers[i]['V '] = cp.zeros([n_neurons_rm1 + n_bias_rm1, n_neurons])
            layers[i]['A'] = cp.zeros([n_neurons_rm1 + n_bias_rm1, n_neurons])

#run the network forward
def feed_forward(X_i, batch_size = 1):

    #set the output of the input layer to the features X_i
    if P['bias'] == False:
        layers[0]['h'] = X_i
    else:
        layers[0]['h'] = cp.zeros([layers[0]['n_neurons'] + layers[0]['n_bias'], batch_size])
        layers[0]['h'][0:-1, :] = X_i
        layers[0]['h'][-1, :] = 1.0
    
    for r in range(1, P['n_layers']+1):
        #compute input to all neurons in current layer
        a = cp.dot(layers[r]['W'].T, layers[r-1]['h'])

        #compute activations of all neurons in current layer
        if layers[r]['activation'] == 'tanh':
            layers[r]['h'] = cp.tanh(a)
        elif layers[r]['activation'] == 'linear':
            layers[r]['h'] = a

        #add ones to last rwo of h if this layers has a bias neuron
        if layers[r]['n_bias'] == 1:
            #NOTE: vstack is inefficient, use prev method, but reinit all h when changing batch size
            #layers[r]['h'][-1, :] = 1.0
            layers[r]['h'] = cp.vstack([layers[r]['h'], cp.ones(batch_size)])

        compute_grad_Phi(r)
    
    return layers[P['n_layers']]['h']

#compute the gradient of L wrt the activation of the output layer
def compute_delta_oo(y_i, r):

    h = layers[r]['h']

    #assumes a linear output layer
    if P['loss'] == 'squared':
        layers[r]['delta_ho'] = -2.0*(y_i - h)

#compute the gradient of L wrt the activation of the output layer
def compute_delta_ho(r):

    n_neurons = layers[r]['n_neurons']

    #get the delta_ho values of the next layer (layer r+1)
    delta_h_rp1_o = layers[r+1]['delta_ho']
    
    #get the grad_Phi values of the next layer
    grad_Phi_rp1 = layers[r+1]['grad_Phi']
    
    #the weight matrix of the next layer
    W_rp1 = layers[r+1]['W']
   
    #compute delta_ho := partial L / partial h
    layers[r]['delta_ho'] = cp.dot(W_rp1, delta_h_rp1_o*grad_Phi_rp1)[0:n_neurons, :]

#compute the gradient in the activation function Phi wrt its input
def compute_grad_Phi(r):

    activation = layers[r]['activation']
    h = layers[r]['h']
    n_neurons = layers[r]['n_neurons']
    batch_size = P['batch_size']

    if activation == 'linear':
        layers[r]['grad_Phi'] = cp.ones([n_neurons, batch_size])
    elif activation == 'tanh':
        layers[r]['grad_Phi'] = 1.0 - h[0:n_neurons, :]**2

def compute_L_grad_W(r, learn_rate):
    h_rm1 = layers[r-1]['h']
    delta_ho_grad_Phi = layers[r]['delta_ho']*layers[r]['grad_Phi']
    layers[r]['L_grad_W'] = cp.dot(h_rm1, delta_ho_grad_Phi.T)
    layers[r]['W'] -= learn_rate*layers[r]['L_grad_W']

#back propagation
def back_prop(y_i, n_layers, learn_rate):

    for r in range(n_layers, 0, -1):
        if r == n_layers:
            compute_delta_oo(y_i, r)
        else:
            compute_delta_ho(r)
        compute_L_grad_W(r, learn_rate)        

#train the neural network        
def train(n_epoch, store_loss = False, check_derivative = False):
   
    n_train = P['n_train']
    batch_size = P['batch_size']
    decay_step = P['decay_step']
    decay_rate = P['decay_rate']
    alpha = P['alpha']
    n_layers = P['n_layers']
    X = P['X']
    y = P['y']

    for i in range(n_epoch):

        #select a random training instance (X, y) -- use numpy, seems faster than cupy
        rnd_idx = np.random.randint(0, n_train, batch_size)
        
        #compute learning rate
        learn_rate = alpha*decay_rate**(np.int(i/decay_step))

        #run the batch
        feed_forward(X[:, rnd_idx], batch_size = batch_size)
        back_prop(y[rnd_idx], n_layers, learn_rate)

P = {}
layers = {}

