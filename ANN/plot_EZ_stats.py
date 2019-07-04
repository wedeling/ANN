def get_pde(X, Npoints = 100):

    kernel = stats.gaussian_kde(X)
    x = np.linspace(np.min(X), np.max(X), Npoints)
    pde = kernel.evaluate(x)
    return x, pde

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from itertools import product, cycle, combinations
from scipy import stats
import sys
import json

HOME = os.path.abspath(os.path.dirname(__file__))
    
###########################
# load the reference data #
###########################
Omega = 7.292*10**-5
day = 24*60**2*Omega
sim_ID = 'tau_EZ_T5'
t_end = (250.0 + 4*365.)*day 
burn = 0#np.int(365*day)

fig = plt.figure(figsize=[8, 4])
ax1 = fig.add_subplot(121, xlabel=r'energy', yticks = [])
ax2 = fig.add_subplot(122, xlabel=r'enstropy', yticks = [])

#fpath = sys.argv[1]
#fp = open(fpath, 'r')
#N_surr = int(fp.readline())
#flags = json.loads(fp.readline())
#
#fname = HOME + '/samples/' + sim_ID + '_' + flags['input_file']  + '_t_' + str(np.around(t_end/day,1)) + '.hdf5'

fname = HOME + '/samples/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.hdf5'
fname_training = HOME + '/samples/tau_EZ_training_t_3170.0.hdf5'  

print('Loading samples ', fname)

try:
    h5f = h5py.File(fname, 'r')
    print(h5f.keys())

#    h5f = h5py.File(fname_training, 'r')
#    print(h5f.keys())
#
#    x_E_LF, pdf_E_LF = get_pde(h5f['e_n_LF'])
#    x_Z_LF, pdf_Z_LF = get_pde(h5f['z_n_LF'])
#
#    ax1.plot(x_E_LF, pdf_E_LF, label=r'$\mathrm{reduced}$')
#    ax2.plot(x_Z_LF, pdf_Z_LF, label=r'$\mathrm{reduced}$')
#   
#    x_E_HF, pdf_E_HF = get_pde(h5f['e_n_HF'])
#    x_Z_HF, pdf_Z_HF = get_pde(h5f['z_n_HF'])
#
#    #x_E_UP, pdf_E_UP = get_pde(h5f['e_n_UP'])
#    #x_Z_UP, pdf_Z_UP = get_pde(h5f['z_n_UP'])
#   
#    ax1.plot(x_E_HF, pdf_E_HF, '--k', label=r'$\mathrm{reference}$')
#    ax2.plot(x_Z_HF, pdf_Z_HF, '--k', label=r'$\mathrm{reference}$')
#
#    #ax1.plot(x_E_UP, pdf_E_UP, ':k', label=r'$\mathrm{unparam.}$')
#    #ax2.plot(x_Z_UP, pdf_Z_UP, ':k', label=r'$\mathrm{unparam.}$')

    ax1.hist([h5f['e_n_LF'][burn:], h5f['e_n_HF'][burn:]], 20, label=[r'$\mathrm{reduced}$', r'$\mathrm{reference}$'])
    ax2.hist([h5f['z_n_LF'][burn:], h5f['z_n_HF'][burn:]], 20, label=[r'$\mathrm{reduced}$', r'$\mathrm{reference}$'])

    ax1.legend(loc=0)
   
    plt.tight_layout()

    fig = plt.figure()
    plt.subplot(121, title=r'$\Delta E$', xlabel=r'$t$')
#    plt.plot(h5f['t'], h5f['r[0]'])
#    plt.plot(h5f['t'], h5f['dE_train'], linewidth=4)
    plt.plot(h5f['t'], h5f['e_n_LF'], 'r')
    plt.plot(h5f['t'], h5f['e_n_HF'], 'b')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.subplot(122, title=r'$\Delta Z$', xlabel=r'$t$')
#    plt.plot(h5f['t'], h5f['r[1]'])
#    plt.plot(h5f['t'], h5f['dZ_train'], linewidth=4)
    plt.plot(h5f['t'], h5f['z_n_LF'], 'r')
    plt.plot(h5f['t'], h5f['z_n_HF'], 'b')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()

except IOError:
    print('*****************************')
    print(fname, ' not found')
    print('*****************************')

leg = plt.legend(loc=0)
leg.set_draggable(True)

ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

fig.tight_layout()

plt.show()
