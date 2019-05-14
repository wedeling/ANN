def get_kde(X, N_points = 300):

    kernel = stats.gaussian_kde(X)
    x = np.linspace(np.min(X), np.max(X), N_points)
    pde = kernel.evaluate(x)
    return x, pde

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy import stats

plt.close('all')

HOME = os.path.abspath(os.path.dirname(__file__))
    
###########################
# load the reference data #
###########################
 
sim_ID = 'T2'
fname = HOME + '/samples/' + sim_ID + '_t_4495.1.hdf5'

h5f = h5py.File(fname, 'r')

QoI = list(h5f.keys())

print(QoI)

e_LF, pdf_e_LF = get_kde(h5f['e_n_LF'][0:-1], N_points = 200)
z_LF, pdf_z_LF = get_kde(h5f['z_n_LF'][0:-1], N_points = 200)
e_HF, pdf_e_HF = get_kde(h5f['e_n_HF'][0:-1], N_points = 200)
z_HF, pdf_z_HF = get_kde(h5f['z_n_HF'][0:-1], N_points = 200)

fig = plt.figure(figsize=[8, 4])
ax1 = fig.add_subplot(121, xlabel=r'energy', yticks = [])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax2 = fig.add_subplot(122, xlabel=r'enstropy', yticks = [])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

ax1.plot(e_HF, pdf_e_HF, '--k')
ax1.plot(e_LF, pdf_e_LF, 'r')

ax2.plot(z_HF, pdf_z_HF, '--k')
ax2.plot(z_LF, pdf_z_LF, 'r')

plt.show()
