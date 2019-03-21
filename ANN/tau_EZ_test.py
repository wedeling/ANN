import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from base import NN

HOME = os.path.abspath(os.path.dirname(__file__))

###########################
# load the reference data #
###########################
fname = HOME + '/samples/dE_dZ_training.hdf5'

h5f = h5py.File(fname, 'r')

QoI = list(h5f.keys())

print(QoI)

#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega
dt = 0.01
n_days = 365
N = np.int(n_days*day/dt)

sub = 100

for qoi in QoI:
    vars()[qoi] = h5f[qoi][0:N:sub]    

y = np.sign(h5f['e_n_HF'][0:N:sub] - h5f['e_n_LF'][0:N:sub])
N = y.size

N_feat = 8
X = np.zeros([N, N_feat])
X[:, 0] = z_n_LF
X[:, 1] = e_n_LF
X[:, 2] = u_n_LF
X[:, 3] = s_n_LF
X[:, 4] = v_n_LF
X[:, 5] = o_n_LF
X[:, 6] = sprime_n_LF
X[:, 7] = zprime_n_LF

idx1 = np.where(y == 1.0)[0]
idxm1 = np.where(y == -1.0)[0]

ann = NN.ANN(X, y, alpha = 0.05)

##############
#train the ANN
##############

ann.compute_loss()
ann.train(20000, store_loss = True)
ann.compute_loss()

if N_feat == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(X[idx1, 0], X[idx1, 1], 'b+')
    ax.plot(X[idxm1, 0], X[idxm1, 1], 'ro')


plt.show()