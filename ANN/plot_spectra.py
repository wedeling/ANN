def draw():
    plt.subplot(121, xscale='log', yscale='log', title=r'energy', xlabel=r'$k$')
    plt.plot(bins+1e-8, mean_E_spec_HF, '--', label = r'$\mathrm{reference}$')
    plt.plot(bins+1e-8, mean_E_spec_LF, label=r'$\mathrm{reduced}$')
    plt.plot(bins+1e-8, mean_E_spec_UN, ':', label=r'$\mathrm{eddy\;visc.}$')
    plt.legend(loc=0)

    ax = plt.gca()
    axins = zoomed_inset_axes(ax, 3, loc=3)
    axins.plot(bins+1e-8, mean_E_spec_HF, '--')
    axins.plot(bins+1e-8, mean_E_spec_LF)
    axins.plot(bins+1e-8, mean_E_spec_UN, ':')

    ax.plot([Ncutoff_LF + 1, Ncutoff_LF + 1], [10, 0], 'lightgray')
    ax.plot([np.sqrt(2)*Ncutoff_LF + 1, np.sqrt(2)*Ncutoff_LF + 1], [10, 0], 'lightgray')

    # sub region of the original image
    # K <= k <= ceil(sqrt(2)*K)
    #x1, x2, y1, y2 = 20, 33, 10**-8, 2*10**-7
    # K - 5 <= k <= K
    x1, x2, y1, y2 = 15, 22, 5*10**-8, 5*10**-7
    axins.set_xscale('log')
    axins.set_yscale('log')
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
  
    axins.set_xticks([])
    axins.set_yticks([])
   
    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.plot([Ncutoff_LF + 1, Ncutoff_LF + 1], [10, 0], 'lightgray')
    plt.plot([np.sqrt(2)*Ncutoff_LF + 1, np.sqrt(2)*Ncutoff_LF + 1], [10, 0], 'lightgray')

    #########################3

    plt.subplot(122, xscale='log', yscale='log', title=r'enstrophy', xlabel=r'$k$')
    plt.plot(bins+1e-8, mean_Z_spec_HF, '--')
    plt.plot(bins+1e-8, mean_Z_spec_LF)
    plt.plot(bins+1e-8, mean_Z_spec_UN, ':')

    ax = plt.gca()
    axins = zoomed_inset_axes(ax, 3, loc=3)
    axins.plot(bins+1e-8, mean_Z_spec_HF, '--')
    axins.plot(bins+1e-8, mean_Z_spec_LF)
    axins.plot(bins+1e-8, mean_Z_spec_UN, ':')

    ax.plot([Ncutoff_LF + 1, Ncutoff_LF + 1], [10, 0], 'lightgray')
    ax.plot([np.sqrt(2)*Ncutoff_LF + 1, np.sqrt(2)*Ncutoff_LF + 1], [10, 0], 'lightgray')

    # sub region of the original image
    # K <= k <= ceil(sqrt(2)*K)
    #x1, x2, y1, y2 = 20, 33, 10**-8, 2*10**-7
    # K - 5 <= k <= K
    x1, x2, y1, y2 = 15, 22, 2*10**-5, 2*10**-4
    axins.set_xscale('log')
    axins.set_yscale('log')
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.set_xticks([])
    axins.set_yticks([])
   
    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.plot([Ncutoff_LF + 1, Ncutoff_LF + 1], [10, 0], 'lightgray')
    plt.plot([np.sqrt(2)*Ncutoff_LF + 1, np.sqrt(2)*Ncutoff_LF + 1], [10, 0], 'lightgray')

    plt.tight_layout()

#compute spectral filter
def get_P(cutoff):
    
    P = np.ones([N, int(N/2+1)])
    
    for i in range(N):
        for j in range(int(N/2+1)):
            
            if np.abs(kx[i, j]) > cutoff or np.abs(ky[i, j]) > cutoff:
                P[i, j] = 0.0
                
    return P

def get_P_k():
    
    k_min = Ncutoff_LF
    k_max = np.max(P_LF_full*binnumbers)
    
    P_k = np.zeros([N, N])    
    idx0, idx1 = np.where((binnumbers >= k_min) & (binnumbers <= k_max))
    
    P_k[idx0, idx1] = 1.0
    
    return P_k[0:N, 0:int(N/2+1)] 

#compute spectral filter
def get_P_full(cutoff):

    P = np.ones([N, N])

    for i in range(N):
        for j in range(N):

            if np.abs(kx_full[i, j]) > cutoff or np.abs(ky_full[i, j]) > cutoff:
                P[i, j] = 0.0

    return P

def freq_map():
    """
    Map 2D frequencies to a 1D bin (kx, ky) --> k
    where k = 0, 1, ..., sqrt(2)*Ncutoff
    """
   
    #edges of 1D wavenumber bins
    bins = np.arange(-0.5, np.ceil(2**0.5*Ncutoff)+1)
    #fmap = np.zeros([N,N]).astype('int')
    
    dist = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            #Euclidian distance of frequencies kx and ky
            dist[i, j] = np.sqrt(kx_full[i,j]**2 + ky_full[i,j]**2).imag
                
    #find 1D bin index of dist
    _, _, binnumbers = stats.binned_statistic(dist.flatten(), np.zeros(N**2), bins=bins)
    
    binnumbers -= 1
            
    return binnumbers.reshape([N, N]), bins

def spectrum(w_hat, P):

    #convert rfft2 coefficients to fft2 coefficients
    w_hat_full = np.zeros([N, N]) + 0.0j
    w_hat_full[0:N, 0:int(N/2+1)] = w_hat
    w_hat_full[map_I, map_J] = np.conjugate(w_hat[I, J])
    w_hat_full *= P
    
    psi_hat_full = w_hat_full/k_squared_no_zero_full
    psi_hat_full[0,0] = 0.0
    
    E_hat = -0.5*psi_hat_full*np.conjugate(w_hat_full)/N**4
    Z_hat = 0.5*w_hat_full*np.conjugate(w_hat_full)/N**4
    
    E_spec = np.zeros(N_bins)
    Z_spec = np.zeros(N_bins)
    
    for i in range(N):
        for j in range(N):
            bin_idx = binnumbers[i, j]
            E_spec[bin_idx] += E_hat[i, j].real
            Z_spec[bin_idx] += Z_hat[i, j].real
            
    return E_spec, Z_spec

#recursive formulas for the mean and variance
def recursive_mean(X_np1, mu_n, N):#, sigma2_n, N):

    mu_np1 = mu_n + (X_np1 - mu_n)/(N+1)

#    sigma2_np1 = sigma2_n + mu_n**2 - mu_np1**2 + (X_np1**2 - sigma2_n - mu_n**2)/(N+1)

    return mu_np1 #, sigma2_np1

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

plt.close('all')

HOME = os.path.abspath(os.path.dirname(__file__))

#number of gridpoints in 1D
I = 8
N = 2**I

#2D grid
h = 2*np.pi/N
axis = h*np.arange(1, N+1)
axis = np.linspace(0, 2.0*np.pi, N)
[x , y] = np.meshgrid(axis , axis)

#frequencies
k = np.fft.fftfreq(N)*N

kx = np.zeros([N, int(N/2+1)]) + 0.0j
ky = np.zeros([N, int(N/2+1)]) + 0.0j

for i in range(N):
    for j in range(int(N/2+1)):
        kx[i, j] = 1j*k[j]
        ky[i, j] = 1j*k[i]


k_squared = kx**2 + ky**2
k_squared_no_zero = np.copy(k_squared)
k_squared_no_zero[0,0] = 1.0

kx_full = np.zeros([N, N]) + 0.0j
ky_full = np.zeros([N, N]) + 0.0j

for i in range(N):
    for j in range(N):
        kx_full[i, j] = 1j*k[j]
        ky_full[i, j] = 1j*k[i]

k_squared_full = kx_full**2 + ky_full**2
k_squared_no_zero_full = np.copy(k_squared_full)
k_squared_no_zero_full[0,0] = 1.0

#cutoff in pseudospectral method
Ncutoff = np.int(N/3)
Ncutoff_LF = np.int(2**(I-2)/3)

#spectral filter
P = get_P(Ncutoff)
P_LF = get_P(Ncutoff_LF)
P_U = P - P_LF

#spectral filter for the full FFT2 (used in compute_E_Z)
P_full = get_P_full(Ncutoff)
P_LF_full = get_P_full(Ncutoff_LF)

binnumbers, bins = freq_map()
N_bins = bins.size
P_k = get_P_k()
P_k = P_LF

#map from the rfft2 coefficient indices to fft2 coefficient indices
#Use: see compute_E_Z subroutine
shift = np.zeros(N).astype('int')
for i in range(1,N):
    shift[i] = np.int(N-i)
I = range(N);J = range(np.int(N/2+1))
map_I, map_J = np.meshgrid(shift[I], shift[J])
I, J = np.meshgrid(I, J)

    
###########################
# load the reference data #
###########################

Omega = 7.292*10**-5
day = 24*60**2*Omega
sim_ID = 'gen_tau_P_k_equal_nu_3'
t_end = (250.0)*day 

fname = HOME + '/samples/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.hdf5'
fname_unparam = HOME + '/samples/unparam_spectrum_t_3900.0.hdf5'  

print('Loading samples ', fname)

fig = plt.figure(figsize=[8,4])

try:
    h5f = h5py.File(fname, 'r')
    print(h5f.keys())

    
    w_hat_n_HF = h5f['w_hat_n_HF']
    w_hat_n_LF = h5f['w_hat_n_LF']

    S = w_hat_n_HF.shape[0]
    
    mean_E_spec_HF = 0.0; mean_E_spec_LF = 0.0
    mean_Z_spec_HF = 0.0; mean_Z_spec_LF = 0.0
    
    for s in range(S):
        E_spec_HF, Z_spec_HF = spectrum(w_hat_n_HF[s], P_full)
        E_spec_LF, Z_spec_LF = spectrum(w_hat_n_LF[s], P_LF_full)
#        E_spec_UN, Z_spec_UN = spectrum(w_hat_n_UN[s], P_LF_full)

        mean_E_spec_HF = recursive_mean(E_spec_HF, mean_E_spec_HF, s)
        mean_E_spec_LF = recursive_mean(E_spec_LF, mean_E_spec_LF, s)
        mean_Z_spec_HF = recursive_mean(Z_spec_HF, mean_Z_spec_HF, s)
        mean_Z_spec_LF = recursive_mean(Z_spec_LF, mean_Z_spec_LF, s)
#        mean_E_spec_UN = recursive_mean(E_spec_UN, mean_E_spec_UN, s)
#        mean_Z_spec_UN = recursive_mean(Z_spec_UN, mean_Z_spec_UN, s)

except IOError:
    print('*****************************')
    print(fname, ' not found')
    print('*****************************')
    

try:

    h5f_unparam = h5py.File(fname_unparam, 'r')
    print(h5f_unparam.keys())
    
    w_hat_n_UN = h5f_unparam['w_hat_n_LF']
    S = w_hat_n_UN.shape[0]
    mean_E_spec_UN = 0.0; mean_Z_spec_UN = 0.0

    for s in range(S):
        print(s)
        E_spec_UN, Z_spec_UN = spectrum(w_hat_n_UN[s], P_LF_full)
        mean_E_spec_UN = recursive_mean(E_spec_UN, mean_E_spec_UN, s)
        mean_Z_spec_UN = recursive_mean(Z_spec_UN, mean_Z_spec_UN, s)

except IOError:
    print('*****************************')
    print(fname_unparam, ' not found')
    print('*****************************')

draw()
plt.show()
