def get_pde(X, Npoints = 100):

    kernel = stats.gaussian_kde(X, bw_method='scott')
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

plt.close('all')

HOME = os.path.abspath(os.path.dirname(__file__))
    
###########################
# load the reference data #
###########################
Omega = 7.292*10**-5
day = 24*60**2*Omega
sim_ID = 'gen_tau_P_k_equal_nu_4'
t_end = (10.0*365)*day 
burn = np.int(250*day)

fig = plt.figure(figsize=[4, 4])
ax1 = fig.add_subplot(111, xlabel=r'$E$', yticks = [])
#ax2 = fig.add_subplot(122, xlabel=r'$Z$', yticks = [])
#ax3 = fig.add_subplot(133, xlabel=r'$Z_2$', yticks = [])

#fpath = sys.argv[1]
#fp = open(fpath, 'r')
#N_surr = int(fp.readline())
#flags = json.loads(fp.readline())
#
#fname = HOME + '/samples/' + sim_ID + '_' + flags['input_file']  + '_t_' + str(np.around(t_end/day,1)) + '.hdf5'

fname = HOME + '/samples/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.hdf5'
fname_unparam = HOME + '/samples/unparam_t_3900.0.hdf5'  

print('Loading samples ', fname)

h5f = h5py.File(fname, 'r')
print(h5f.keys())

h5f_unparam = h5py.File(fname_unparam, 'r')
print(h5f_unparam.keys())
x_E_UN, pdf_E_UN = get_pde(h5f_unparam['e_n_LF'][burn:])
x_Z_UN, pdf_Z_UN = get_pde(h5f_unparam['z_n_LF'][burn:])
#    x_W3_UN, pdf_W3_UN = get_pde(h5f_unparam['w3_n_LF'])

ax1.plot(x_E_UN, pdf_E_UN, ':', label=r'$\mathrm{eddy\;visc}$')
#ax2.plot(x_Z_UN, pdf_Z_UN, ':', label=r'$\mathrm{eddy\;visc}$')
#    ax3.plot(x_W3_UN, pdf_W3_UN, ':', label=r'$\mathrm{eddy\;visc}$')

x_E_LF, pdf_E_LF = get_pde(h5f['e_n_LF'][burn:])
x_Z_LF, pdf_Z_LF = get_pde(h5f['z_n_LF'][burn:])
#    x_W3_LF, pdf_W3_LF = get_pde(h5f['w3_n_LF'])

ax1.plot(x_E_LF, pdf_E_LF, label=r'$\mathrm{reduced}$')
#ax2.plot(x_Z_LF, pdf_Z_LF, label=r'$\mathrm{reduced}$')
#    ax3.plot(x_W3_LF, pdf_W3_LF, label=r'$\mathrm{reduced}$')

x_E_HF, pdf_E_HF = get_pde(h5f['e_n_HF'][burn:])
x_Z_HF, pdf_Z_HF = get_pde(h5f['z_n_HF'][burn:])
#    x_W3_HF, pdf_W3_HF = get_pde(h5f['w3_n_HF'])

#x_E_UP, pdf_E_UP = get_pde(h5f['e_n_UP'])
#x_Z_UP, pdf_Z_UP = get_pde(h5f['z_n_UP'])

ax1.plot(x_E_HF, pdf_E_HF, '--k', label=r'$\mathrm{reference}$')
#ax2.plot(x_Z_HF, pdf_Z_HF, '--k', label=r'$\mathrm{reference}$')
#    ax3.plot(x_W3_HF, pdf_W3_HF, '--k', label=r'$\mathrm{reference}$')

#ax1.plot(x_E_UP, pdf_E_UP, ':k', label=r'$\mathrm{unparam.}$')
#ax2.plot(x_Z_UP, pdf_Z_UP, ':k', label=r'$\mathrm{unparam.}$')

#    ax1.hist([h5f['e_n_LF'][burn:], h5f['e_n_HF'][burn:]], 20, label=[r'$\mathrm{reduced}$', r'$\mathrm{reference}$'])
#    ax2.hist([h5f['z_n_LF'][burn:], h5f['z_n_HF'][burn:]], 20, label=[r'$\mathrm{reduced}$', r'$\mathrm{reference}$'])

ax1.legend(loc=0)

plt.tight_layout()
#
#fig = plt.figure(figsize=[8,4])
#t = h5f_unparam['t'][:]
#plt.subplot(121, title=r'$\Delta E$', xlabel=r'$t\;\mathrm{[days]}$')
#plt.plot(t/day, h5f['dE'])
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.subplot(122, title=r'$\Delta Z$', xlabel=r'$t\;\mathrm{[days]}$')
#plt.plot(t/day, h5f['dZ'])
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.tight_layout()
#
ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
##ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#
#fig.tight_layout()
#
plt.show()
