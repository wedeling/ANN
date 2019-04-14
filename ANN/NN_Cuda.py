import numpy as np
import cupy as cp

#store all the parameters defining the ANN in the global P dict
def init_network(X = np.zeros(1), y = np.zeros(0), alpha = 0.001, decay_rate = 1.0, decay_step = 10**4, beta1 = 0.9, beta2 = 0.999, lamb = 0.0, \
                 param_specific_learn_rate = False, loss = 'squared', activation = 'tanh', n_layers = 2, n_neurons = 16, \
                 bias = True, neuron_based_compute = False, batch_size = 1, save = True, name='ANN'):
    
    all_vars = locals()

    for key in all_vars.keys():
        P[key] = all_vars[key]

    #number of training data points
    P['n_train'] = X.shape[0]
    
    #number of input nodes
    try:
        P['n_in'] = X.shape[1]
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
        layers[0]['h'][0:-1, :] = X_i
        layers[0]['h'][-1, :] = 1.0
    
    for r in range(1, P['n_layers']+1):
        #compute input to all neurons in current layer
        a = cp.dot(layers[r]['W'].T, layers[r-1]['h'])

        #compute activations of all neurons in current layer
        if layers[r]['activation'] == 'tanh':
            layers[r]['h'][0:-1,:] = cp.tanh(a)
        elif layers[r]['activation'] == 'linear':
            layers[r]['h'] = a

        #add ones to last rwo of h if this layers has a bias neuron
        if layers[r]['n_bias'] == 1:
            layers[r]['h'][-1, :] = 1.0

P = {}
layers = {}

