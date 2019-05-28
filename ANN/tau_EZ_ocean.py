"""
*************************
* S U B R O U T I N E S *
*************************
"""

#pseudo-spectral technique to solve for Fourier coefs of Jacobian
def compute_VgradW_hat(w_hat_n, P):
    
    #compute streamfunction
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    
    #compute jacobian in physical space
    u_n = np.fft.irfft2(-ky*psi_hat_n)
    w_x_n = np.fft.irfft2(kx*w_hat_n)

    v_n = np.fft.irfft2(kx*psi_hat_n)
    w_y_n = np.fft.irfft2(ky*w_hat_n)
    
    VgradW_n = u_n*w_x_n + v_n*w_y_n
    
    #return to spectral space
    VgradW_hat_n = np.fft.rfft2(VgradW_n)
    
    VgradW_hat_n *= P
    
    return VgradW_hat_n

#get Fourier coefficient of the vorticity at next (n+1) time step
def get_w_hat_np1(w_hat_n, w_hat_nm1, VgradW_hat_nm1, P, norm_factor, sgs_hat = 0.0):
    
    #compute jacobian
    VgradW_hat_n = compute_VgradW_hat(w_hat_n, P)
    
    #solve for next time step according to AB/BDI2 scheme
    w_hat_np1 = norm_factor*P*(2.0/dt*w_hat_n - 1.0/(2.0*dt)*w_hat_nm1 - \
                               2.0*VgradW_hat_n + VgradW_hat_nm1 + mu*F_hat - sgs_hat)
    
    return w_hat_np1, VgradW_hat_n

#compute spectral filter
def get_P(cutoff):
    
    P = np.ones([N, int(N/2+1)])
    
    for i in range(N):
        for j in range(int(N/2+1)):
            
            if np.abs(kx[i, j]) > cutoff or np.abs(ky[i, j]) > cutoff:
                P[i, j] = 0.0
                
    return P

#compute spectral filter
def get_P_full(cutoff):

    P = np.ones([N, N])

    for i in range(N):
        for j in range(N):

            if np.abs(kx_full[i, j]) > cutoff or np.abs(ky_full[i, j]) > cutoff:
                P[i, j] = 0.0

    return P

#store samples in hierarchical data format, when sample size become very large
def store_samples_hdf5():
  
    fname = HOME + '/samples/' + store_ID + '_t_' + str(np.around(t_end/day, 1)) + '.hdf5'
    
    print('Storing samples in ', fname)
    
    if os.path.exists(HOME + '/samples') == False:
        os.makedirs(HOME + '/samples')
    
    #create HDF5 file
    h5f = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for q in QoI:
        h5f.create_dataset(q, data = samples[q])
        
    h5f.close()    

def draw_2w():
    plt.subplot(121, title=r'$Q_1\; ' + r't = '+ str(np.around(t/day, 2)) + '\;[days]$')
    #plt.contourf(x, y, w_np1_HF, 100)
    plt.plot(T, DE)
    plt.plot(T, DE_ANN)
    #plt.colorbar()
    plt.subplot(122,title=r'$Q_2$')
    #plt.contourf(x, y, w_np1_LF, 100)
#    plt.plot(T, DE_ANN)
    #plt.colorbar()
    plt.tight_layout()
    
def draw_stats():
    plt.subplot(121, xlabel=r't')
    plt.plot(T, energy_HF, label=r'$E^{HF}$')
    plt.plot(T, energy_LF, label=r'$E^{LF}$')
    plt.legend(loc=0)
    plt.subplot(122, xlabel=r't')
    plt.plot(T, enstrophy_HF, label=r'$Z^{HF}$')
    plt.plot(T, enstrophy_LF, label=r'$Z^{LF}$')
    plt.legend(loc=0)
    plt.tight_layout()
    
#compute the spatial correlation coeffient at a given time
def spatial_corr_coef(X, Y):
    return np.mean((X - np.mean(X))*(Y - np.mean(Y)))/(np.std(X)*np.std(Y))
   
#compute the energy and enstrophy at t_n
def compute_ZE(w_hat_n, verbose=True):

    #compute stats using Fourier coefficients - is faster
    #convert rfft2 coefficients to fft2 coefficients
    w_hat_full = np.zeros([N, N]) + 0.0j
    w_hat_full[0:N, 0:int(N/2+1)] = w_hat_n
    w_hat_full[map_I, map_J] = np.conjugate(w_hat_n[I, J])
    w_hat_full *= P_full
    
    #compute Fourier coefficients of stream function
    psi_hat_full = w_hat_full/k_squared_no_zero_full
    psi_hat_full[0,0] = 0.0

    #compute energy and enstrophy (density)
    Z = 0.5*np.sum(w_hat_full*np.conjugate(w_hat_full))/N**4
    E = -0.5*np.sum(psi_hat_full*np.conjugate(w_hat_full))/N**4

    if verbose:
        #print 'Energy = ', E, ', enstrophy = ', Z
        print('Z = ', Z.real, ', E = ', E.real)

    return Z.real, E.real

#compute all QoI at t_n
def compute_qoi(w_hat_n, verbose=True):

    #compute stats using Fourier coefficients - is faster
    #convert rfft2 coefficients to fft2 coefficients
    w_hat_full = np.zeros([N, N]) + 0.0j
    w_hat_full[0:N, 0:int(N/2+1)] = w_hat_n
    w_hat_full[map_I, map_J] = np.conjugate(w_hat_n[I, J])
    w_hat_full *= P_full
    
    #compute Fourier coefficients of stream function
    psi_hat_full = w_hat_full/k_squared_no_zero_full
    psi_hat_full[0,0] = 0.0

    #compute energy and enstrophy (density)
    Z = 0.5*np.sum(w_hat_full*np.conjugate(w_hat_full))/N**4
    E = -0.5*np.sum(psi_hat_full*np.conjugate(w_hat_full))/N**4
    U = 0.5*np.sum(psi_hat_full*np.conjugate(F_hat_full))/N**4
    S = 0.5*np.sum(psi_hat_full*np.conjugate(psi_hat_full))/N**4
    O = 0.5*np.sum(k_squared_full*w_hat_full*np.conjugate(w_hat_full))/N**4
    V = 0.5*np.sum(w_hat_full*np.conjugate(F_hat_full))/N**4

    Sprime = E**2/Z - S
    Zprime = Z - E**2/S

    if verbose:
        #print 'Energy = ', E, ', enstrophy = ', Z
        print('Z = ', Z.real, ', E = ', E.real)
        print('U = ', U.real, ', S = ', S.real)
        print('V = ', V.real, ', O = ', O.real)
        print('Sprime = ', Sprime.real, ', Zprime = ', Zprime.real)

    return Z.real, E.real, U.real, S.real, V.real, O.real, Sprime.real, Zprime.real

#######################
# ORTHOGONAL PATTERNS #
#######################

def get_psi_hat_prime(w_hat_n):

    #stream function
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0, 0] = 0.
    
    #compute stats using Fourier coefficients - is faster
    #convert rfft2 coefficients to fft2 coefficients
    w_hat_full = np.zeros([N, N]) + 0.0j
    w_hat_full[0:N, 0:int(N/2+1)] = w_hat_n
    w_hat_full[map_I, map_J] = np.conjugate(w_hat_n[I, J])
    w_hat_full *= P_full
    
    #compute Fourier coefficients of stream function
    psi_hat_full = w_hat_full/k_squared_no_zero_full
    psi_hat_full[0,0] = 0.0

#same, but integral compute using Simpson's rule
#    psi_n = np.fft.irfft2(psi_hat_n)
#    w_n = np.fft.irfft2(w_hat_n)
#
#    nom = simps(simps(w_n*psi_n, axis), axis)
#    denom = simps(simps(w_n*w_n, axis), axis)

    #compute integrals directly from Fourier coefficients
    nom = np.sum(w_hat_full*np.conjugate(psi_hat_full))/N**4
    denom = np.sum(w_hat_full*np.conjugate(w_hat_full))/N**4

    return psi_hat_n - nom/denom*w_hat_n

def get_w_hat_prime(w_hat_n):
    
    #stream function
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0, 0] = 0.

    #compute stats using Fourier coefficients - is faster
    #convert rfft2 coefficients to fft2 coefficients
    w_hat_full = np.zeros([N, N]) + 0.0j
    w_hat_full[0:N, 0:int(N/2+1)] = w_hat_n
    w_hat_full[map_I, map_J] = np.conjugate(w_hat_n[I, J])
    w_hat_full *= P_full
    
    #compute Fourier coefficients of stream function
    psi_hat_full = w_hat_full/k_squared_no_zero_full
    psi_hat_full[0,0] = 0.0

#same, but integral compute using Simpson's rule
#    psi_n = np.fft.irfft2(psi_hat_n)
#    w_n = np.fft.irfft2(w_hat_n)
#
#    nom = simps(simps(w_n*psi_n, axis), axis)
#    denom = simps(simps(psi_n*psi_n, axis), axis)

    nom = np.sum(w_hat_full*np.conjugate(psi_hat_full))/N**4
    denom = np.sum(psi_hat_full*np.conjugate(psi_hat_full))/N**4
    
    return w_hat_n - nom/denom*psi_hat_n

"""
***************************
* M A I N   P R O G R A M *
***************************
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.integrate import simps
from drawnow import drawnow
from base import NN

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

HOME = os.path.abspath(os.path.dirname(__file__))

#number of gridpoints in 1D
N = 2**7

#2D grid
h = 2*np.pi/N
#axis = h*np.arange(1, N+1)
axis = np.linspace(0.0, 2.0*np.pi, N)
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
Ncutoff = N/3
Ncutoff_LF = 2**6/3 

#spectral filter for the real FFT2
P = get_P(Ncutoff)
P_LF = get_P(Ncutoff_LF)
P_U = P - P_LF

#spectral filter for the full FFT2 (used in compute_E_Z)
P_full = get_P_full(Ncutoff_LF)

#map from the rfft2 coefficient indices to fft2 coefficient indices
#Use: see compute_E_Z subroutine
shift = np.zeros(N).astype('int')
for i in range(1,N):
    shift[i] = np.int(N-i)
I = range(N);J = range(np.int(N/2+1))
map_I, map_J = np.meshgrid(shift[I], shift[J])
I, J = np.meshgrid(I, J)

#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

#viscosities
decay_time_nu = 5.0
decay_time_mu = 90.0
nu = 1.0/(day*Ncutoff**2*decay_time_nu)
#nu_LF = 1.0/(day*Ncutoff_LF**2*decay_time_nu)
nu_LF = 1.0/(day*Ncutoff**2*decay_time_nu)
mu = 1.0/(day*decay_time_mu)

#start, end time (in days) + time step
t = 250.0*day
t_end = (t + 8.0*365)*day
#t_end = 251.0*day

#time step
dt = 0.01
n_steps = np.ceil((t_end-t)/dt).astype('int')

#############
# USER KEYS #
#############

sim_ID = 'ANN'
store_ID = 'T2'
plot_frame_rate = np.floor(1.0*day/dt).astype('int')
#store_frame_rate = np.floor(0.5*day/dt).astype('int')
store_frame_rate = 1
S = np.floor(n_steps/store_frame_rate).astype('int')

state_store = False
restart = True
store = True
plot = True
on_the_fly = True
eddy_forcing_type = 'ann_tau_ortho'

####################
# STORE PARAMETERS #
####################

#QoI to store, First letter in caps implies an NxN field, otherwise a scalar 
QoI = ['z_n_HF', 'e_n_HF', \
       'z_n_LF', 'e_n_LF', 'u_n_LF', 's_n_LF', 'v_n_LF', 'o_n_LF', \
       'sprime_n_LF', 'zprime_n_LF', \
       'tau_E', 'tau_Z', 't']

Q = len(QoI)

#allocate memory
samples = {}

if store == True:
    samples['S'] = S
    samples['N'] = N
    
    for q in range(Q):
        
        #a field
        if QoI[q][0].isupper():
            #samples[QoI[q]] = np.zeros([S, N, N/2+1]) + 0.0j
            samples[QoI[q]] = np.zeros([S, N, N])
        #a scalar
        else:
            samples[QoI[q]] = np.zeros(S)

##################       
# ANN PARAMETERS #
##################

if eddy_forcing_type == 'ann_tau_ortho':

    #create empty ANN object
    dE_ann = NN.ANN(X = np.zeros(10), y = np.zeros(1), standardize = False)
    #load trained ann
    dE_ann.load_ANN(name='dE')

    #create empty ANN object
    dZ_ann = NN.ANN(X = np.zeros(10), y = np.zeros(1), standardize = False)
    #load trained ann
    dZ_ann.load_ANN(name='dZ')
    
    #NOTE: making the assumption here that both ANNs use the same features
    X_mean = dE_ann.X_mean
    X_std = dE_ann.X_std
    
    #number of featues
    N_feat = dE_ann.N_in

    dE_mean = dE_ann.y_mean
    dE_std = dE_ann.y_std
    dZ_mean = dZ_ann.y_mean
    dZ_std = dZ_ann.y_std
    
    batch_size = 32
    X_on_the_fly = np.zeros([batch_size, N_feat])
    dE_on_the_fly = np.zeros(batch_size)
    dZ_on_the_fly = np.zeros(batch_size)

##################

#forcing term
F = 2**1.5*np.cos(5*x)*np.cos(5*y)
F_hat = np.fft.rfft2(F)
F_hat_full = np.fft.fft2(F)

if restart == True:
    
    fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t/day,1)) + '.hdf5'
    
    #create HDF5 file
    h5f = h5py.File(fname, 'r')
    
    for key in h5f.keys():
        print(key)
        vars()[key] = h5f[key][:]
        
    h5f.close()
   
else:
    
    #initial condition
    w = np.sin(4.0*x)*np.sin(4.0*y) + 0.4*np.cos(3.0*x)*np.cos(3.0*y) + \
        0.3*np.cos(5.0*x)*np.cos(5.0*y) + 0.02*np.sin(x) + 0.02*np.cos(y)

    #initial Fourier coefficients at time n and n-1
    w_hat_n_HF = P*np.fft.rfft2(w)
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
    
    w_hat_n_LF = P_LF*np.fft.rfft2(w)
    w_hat_nm1_LF = np.copy(w_hat_n_LF)
    
    #initial Fourier coefficients of the jacobian at time n and n-1
    VgradW_hat_n_HF = compute_VgradW_hat(w_hat_n_HF, P)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)
    
    VgradW_hat_n_LF = compute_VgradW_hat(w_hat_n_LF, P_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)
    
#constant factor that appears in AB/BDI2 time stepping scheme   
norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared + mu)
norm_factor_LF = 1.0/(3.0/(2.0*dt) - nu_LF*k_squared + mu)

j = 0; j2 = 0; idx = 0

if plot == True:
    plt.figure()
    energy_HF = []; energy_LF = []; enstrophy_HF = []; enstrophy_LF = []; T = []
    #TEST: REMOVE LATER
    DE = []; DE_ANN = []

#time loop
for n in range(n_steps):    
    
    #solve for next time step
    w_hat_np1_HF, VgradW_hat_n_HF = get_w_hat_np1(w_hat_n_HF, w_hat_nm1_HF, VgradW_hat_nm1_HF, P, norm_factor)
  
    #exact eddy forcing
    EF_hat_nm1_exact = P_LF*VgradW_hat_nm1_HF - VgradW_hat_nm1_LF

    #EXACT eddy forcing (for reference)
    if eddy_forcing_type == 'exact':
        EF_hat = EF_hat_nm1_exact
        
    #exact orthogonal pattern forcing
    elif eddy_forcing_type == 'tau_ortho':
        
        z_n_HF, e_n_HF = compute_ZE(P_LF*w_hat_n_HF, verbose=False)
        z_n_LF, e_n_LF, u_n_LF, s_n_LF, v_n_LF, o_n_LF, sprime_n_LF, zprime_n_LF = compute_qoi(w_hat_n_LF, verbose=False)
    
        src_E = e_n_LF**2/z_n_LF - s_n_LF
        src_Z = -e_n_LF**2/s_n_LF + z_n_LF
    
        dE = e_n_HF - e_n_LF
        dZ = z_n_HF - z_n_LF

        #inverse unclosed time scales
        tau_E = np.tanh(dE/e_n_LF)*np.sign(src_E)
        tau_Z = np.tanh(dZ/z_n_LF)*np.sign(src_Z)
        
        #orthogonal patterns
        psi_hat_n_prime = get_psi_hat_prime(w_hat_n_LF)
        w_hat_n_prime = get_w_hat_prime(w_hat_n_LF)

        #reduced model-error source term
        EF_hat = -tau_E*psi_hat_n_prime - tau_Z*w_hat_n_prime 
        
    #ANN surrogate of the orthogonal pattern forcing
    elif eddy_forcing_type == 'ann_tau_ortho':

        #compute features
        z_n_LF, e_n_LF, u_n_LF, s_n_LF, v_n_LF, o_n_LF, sprime_n_LF, zprime_n_LF = compute_qoi(w_hat_n_LF, verbose=False)
    
        #source terms of the E and Z ODEs
        src_E = e_n_LF**2/z_n_LF - s_n_LF
        src_Z = -e_n_LF**2/s_n_LF + z_n_LF

        #EXACT dE and dZ, leave uncommented for one-way coupled simulations
        z_n_HF, e_n_HF = compute_ZE(P_LF*w_hat_n_HF, verbose=False)
#        dE_tilde = e_n_HF - e_n_LF
#        dZ_tilde = z_n_HF - z_n_LF

        #features
        X_feat = np.array([z_n_LF, e_n_LF, u_n_LF, s_n_LF, v_n_LF, o_n_LF, sprime_n_LF, zprime_n_LF])
        
        #standardize by data mean and std if standardize flag was set to True during ann training
        X_feat = (X_feat - X_mean)/X_std
        
        #feed forward of the neural net
        dE_tilde = dE_ann.feed_forward(X_feat.reshape([1, 8]))[0][0]
        dZ_tilde = dZ_ann.feed_forward(X_feat.reshape([1, 8]))[0][0]
        
        #if standardize flag was True during ANN training
        dE_tilde = dE_tilde*dE_std + dE_mean
        dZ_tilde = dZ_tilde*dZ_std + dZ_mean

        #inverse unclosed time scales
        tau_E = np.tanh(dE_tilde/e_n_LF)*np.sign(src_E)
        tau_Z = np.tanh(dZ_tilde/z_n_LF)*np.sign(src_Z)
        
        #orthogonal patterns
        psi_hat_n_prime = get_psi_hat_prime(w_hat_n_LF)
        w_hat_n_prime = get_w_hat_prime(w_hat_n_LF)

        #reduced model-error source term
        EF_hat = -tau_E*psi_hat_n_prime - tau_Z*w_hat_n_prime 
        
    #NO eddy forcing
    elif eddy_forcing_type == 'unparam':
        EF_hat = np.zeros([N, int(N/2+1)])

    #resolved model run
    w_hat_np1_LF, VgradW_hat_n_LF = get_w_hat_np1(w_hat_n_LF, w_hat_nm1_LF, VgradW_hat_nm1_LF, P_LF, norm_factor_LF, EF_hat)
   
    #plot results to screen during iteration
    if j == plot_frame_rate and plot == True:
        j = 0
        
        #HF and LF vorticities
        w_np1_HF = np.fft.irfft2(P_LF*w_hat_np1_HF)
        w_np1_LF = np.fft.irfft2(w_hat_np1_LF)

        #compute stats
        Z_HF, E_HF = compute_ZE(P_LF*w_hat_np1_HF)
        Z_LF, E_LF, U_LF, S_LF, V_LF, O_LF, Sprime_LF, Zprime_LF = compute_qoi(w_hat_np1_LF)
        print('------------------')
        
        energy_HF.append(E_HF); energy_LF.append(E_LF)
        enstrophy_HF.append(Z_HF); enstrophy_LF.append(Z_LF)
        T.append(t)

        drawnow(draw_stats)
        #drawnow(draw_2w)
        
    #store samples to dict
    if j2 == store_frame_rate and store == True:
        j2 = 0
                
        if np.mod(n, np.round(day/dt)) == 0:
            print('n =', n, 'of', n_steps)
            
        for qoi in QoI:
            samples[qoi][idx] = eval(qoi)        
        
        samples['t'][idx] = t
        
        idx += 1  
        
    #update variables
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
    w_hat_n_HF = np.copy(w_hat_np1_HF)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)

    w_hat_nm1_LF = np.copy(w_hat_n_LF)
    w_hat_n_LF = np.copy(w_hat_np1_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)
    
    t += dt
    j += 1
    j2 += 1
    
#store the state of the system to allow for a simulation restart at t > 0
if state_store == True:
    
    keys = ['w_hat_nm1_HF', 'w_hat_n_HF', 'VgradW_hat_nm1_HF', \
            'w_hat_nm1_LF', 'w_hat_n_LF', 'VgradW_hat_nm1_LF']
    
    if os.path.exists(HOME + '/restart') == False:
        os.makedirs(HOME + '/restart')
    
    #cPickle.dump(state, open(HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.pickle', 'w'))
    
    fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.hdf5'
    
    #create HDF5 file
    h5f = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for key in keys:
        qoi = eval(key)
        h5f.create_dataset(key, data = qoi)
        
    h5f.close()   

#store the samples
if store == True:
    store_samples_hdf5() 

plt.show()