
##########################
# S U B R O U T I N E S  #
##########################

#pseudo-spectral technique to solve for Fourier coefs of Jacobian
def compute_VgradW_hat(w_hat_n, P):
    
    #compute streamfunction
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    
    #compute jacobian in physical space
    u_n = np.fft.ifft2(-ky*psi_hat_n)
    w_x_n = np.fft.ifft2(kx*w_hat_n)

    v_n = np.fft.ifft2(kx*psi_hat_n)
    w_y_n = np.fft.ifft2(ky*w_hat_n)
    
    VgradW_n = u_n*w_x_n + v_n*w_y_n
    
    #return to spectral space
    VgradW_hat_n = np.fft.fft2(VgradW_n)
    
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
    
    P = np.ones([N, N])
    
    for i in range(N):
        for j in range(N):
            
            if np.abs(kx[i, j]) > cutoff or np.abs(ky[i, j]) > cutoff:
                P[i, j] = 0.0
                
    return P

def get_P_k(k_min, k_max):

    P_k = np.zeros([N, N])    
    idx0, idx1 = np.where((binnumbers >= k_min) & (binnumbers <= k_max))
    
    P_k[idx0, idx1] = 1.0
    
    return P_k

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

def draw():
#    plt.subplot(121, xlabel=r'$x$', ylabel=r'$y$', title=r'$\omega^{\mathcal{R}}$')
#    plt.contourf(x, y, w_n_HF, 100)
#    plt.colorbar()
#    plt.subplot(122, xlabel=r'$x$', ylabel=r'$y$', title=r'$\mathrm{eddy\;forcing}\;\bar{r}$')
#    plt.contourf(x, y, EF, 100)
#    plt.colorbar()
#    plt.tight_layout()
    plt.subplot(121, xscale='log', yscale='log')
    plt.plot(bins+1.5, E_spec_HF, '--')
    plt.plot(bins+1.5, E_spec_LF, '--')

    plt.plot([k_min1 + 1.5, k_min1 + 1.5], [10, 0], 'lightgray')
    plt.plot([k_min2 + 1.5, k_min2 + 1.5], [10, 0], 'lightgray')
    plt.plot([k_max1 + 1.5, k_max1 + 1.5], [10, 0], 'lightgray')
    plt.plot([k_max2 + 1.5, k_max2 + 1.5], [10, 0], 'lightgray')

    plt.plot([np.sqrt(2)*Ncutoff_LF + 1.5, np.sqrt(2)*Ncutoff_LF + 1.5], [10, 0], 'lightgray')
    plt.subplot(122, xscale='log', yscale='log')
    plt.plot(bins+1.5, Z_spec_HF, '--')
    plt.plot(bins+1.5, Z_spec_LF, '--')

    plt.plot([k_min1 + 1.5, k_min1 + 1.5], [10, 0], 'lightgray')
    plt.plot([k_min2 + 1.5, k_min2 + 1.5], [10, 0], 'lightgray')
    plt.plot([k_max1 + 1.5, k_max1 + 1.5], [10, 0], 'lightgray')
    plt.plot([k_max2 + 1.5, k_max2 + 1.5], [10, 0], 'lightgray')
#    
#    plt.plot(np.array(T)/day, Z_LF)
#    plt.subplot(221, title=r'$E$', xlabel=r'$t\;[day]$')
#    plt.plot(np.array(T)/day, E_HF, 'o')
#    plt.plot(np.array(T)/day, E_LF)
#    plt.subplot(222, title=r'$Z_0$', xlabel=r'$t\;[day]$')
#    plt.plot(np.array(T)/day, W1_HF, 'o')
#    plt.plot(np.array(T)/day, W1_LF)
#    plt.subplot(223, title=r'$Z_1$', xlabel=r'$t\;[day]$')
#    plt.plot(np.array(T)/day, Z_HF, 'o')
#    plt.plot(np.array(T)/day, Z_LF)
#    plt.subplot(224, title=r'$Z_2$', xlabel=r'$t\;[day]$')
#    plt.plot(np.array(T)/day, W3_HF, 'o')
#    plt.plot(np.array(T)/day, W3_LF)
#    
#    plt.subplot(133, title=r'$\tau$', xlabel=r'$t\;[day]$')
#    plt.plot(np.array(T)/day, TAU1)
#    plt.plot(np.array(T)/day, TAU2)
#    plt.plot(np.array(T)/day, TAU3)
#    plt.subplot(133, title=r'$\bar{r}$', xlabel=r'$t\;[day]$')
#    plt.contourf(x, y, EF, 100)
#    plt.colorbar()
#    plt.subplot(133, xlabel=r'$t\;[day]$')
#    plt.plot(np.array(T)/day, TEST)
#    
#    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    plt.tight_layout()

#return the fourier coefs of the stream function
def get_psi_hat(w_hat_n):

    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0

    return psi_hat_n

###############################
# DATA-DRIVEN TAU SUBROUTINES #
###############################

def compute_cij(T_hat, V_hat):

    n_scalar = T_hat.shape[0]
    c_ij = np.zeros([n_scalar, n_scalar-1])

    for i in range(n_scalar):
        A = np.zeros([n_scalar-1, n_scalar-1])
        b = np.zeros(n_scalar-1)

        k = np.delete(np.arange(n_scalar), i)

        for j1 in range(n_scalar-1):
            for j2 in range(n_scalar-1):
                integral = compute_int(V_hat[k[j1]], T_hat[i, j2+1])
                A[j1, j2] = integral

        for j1 in range(n_scalar-1):
            integral = compute_int(V_hat[k[j1]], T_hat[i, 0])
            b[j1] = integral

        if n_scalar == 2:
            c_ij[i,:] = b/A
        else:
            c_ij[i,:] = np.linalg.solve(A, b)
            
    return c_ij

#compute all QoI at t_n
def compute_qoi(w_hat, P, verbose=False):

    #compute Fourier coefficients of stream function
    psi_hat = w_hat/k_squared_no_zero
    psi_hat[0,0] = 0.0
    
    w_hat_conj = np.conjugate(w_hat)

    w_hat_squared = P*np.fft.fft2(np.fft.ifft2(w_hat)**2)

    #compute energy and enstrophy (density)
    E = -0.5*np.sum(P*psi_hat*w_hat_conj)/N4
    W1 = np.sum(P*V_hat_w1*w_hat_conj)/N4
    Z = 0.5*np.sum(P*w_hat*w_hat_conj)/N4
    W3 = 1.0/3.0*np.sum(P*w_hat_squared*w_hat_conj)/N4
    
    if verbose:
        #print 'Energy = ', E, ', enstrophy = ', Z
        print('E = ', E.real, 'W1 = ', W1.real)
        print('Z = ', Z.real, 'W3 = ', W3.real)
        
    return E.real, W1.real, Z.real, W3.real

def compute_int(X1_hat, X2_hat):
    
    integral = np.sum(X1_hat*np.conjugate(X2_hat))/N4
    return integral.real
    
#    X1 = np.fft.ifft2(X1_hat)
#    X2 = np.fft.ifft2(X2_hat)
#    
#    return simps(simps(X1*X2, axis), axis)/(2*np.pi)**2
   
#######################
# ORTHOGONAL PATTERNS #
#######################

def get_psi_hat_prime(w_hat_n):
    
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0, 0] = 0.

    psi_n = np.fft.ifft2(psi_hat_n)
    w_n = np.fft.ifft2(w_hat_n)

    nom = simps(simps(w_n*psi_n, axis), axis)
    denom = simps(simps(w_n*w_n, axis), axis)

    return psi_hat_n - nom/denom*w_hat_n

def get_w_hat_prime(w_hat_n):
    
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0, 0] = 0.

    psi_n = np.fft.ifft2(psi_hat_n)
    w_n = np.fft.ifft2(w_hat_n)

    nom = simps(simps(w_n*psi_n, axis), axis)
    denom = simps(simps(psi_n*psi_n, axis), axis)

    return w_hat_n - nom/denom*psi_hat_n

#compute the (temporal) correlation coeffient 
def corr_coef(X, Y):
    return np.mean((X - np.mean(X))*(Y - np.mean(Y)))/(np.std(X)*np.std(Y))

#recursive formulas for the mean and variance
def recursive_moments(X_np1, mu_n, sigma2_n, N):

    mu_np1 = mu_n + (X_np1 - mu_n)/(N+1)

    sigma2_np1 = sigma2_n + mu_n**2 - mu_np1**2 + (X_np1**2 - sigma2_n - mu_n**2)/(N+1)

    return mu_np1, sigma2_np1

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
            dist[i, j] = np.sqrt(kx[i,j]**2 + ky[i,j]**2).imag
                
    #find 1D bin index of dist
    _, _, binnumbers = stats.binned_statistic(dist.flatten(), np.zeros(N**2), bins=bins)
    
    binnumbers -= 1
            
    return binnumbers.reshape([N, N]), bins

def spectrum(w_hat, P):
    
    psi_hat = w_hat/k_squared_no_zero
    psi_hat[0,0] = 0.0
    
    E_hat = -0.5*psi_hat*np.conjugate(w_hat)/N4
    Z_hat = 0.5*w_hat*np.conjugate(w_hat)/N4
    
    E_spec = np.zeros(N_bins)
    Z_spec = np.zeros(N_bins)
    
    for i in range(N):
        for j in range(N):
            bin_idx = binnumbers[i, j]
            E_spec[bin_idx] += E_hat[i, j].real
            Z_spec[bin_idx] += Z_hat[i, j].real
            
    return E_spec, Z_spec

###########################
# M A I N   P R O G R A M #
###########################

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.integrate import simps
import sys
from drawnow import drawnow
from scipy import stats

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

HOME = os.path.abspath(os.path.dirname(__file__))

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

#number of gridpoints in 1D
I = 8
N = 2**I

N4 = N**4

#2D grid
h = 2*np.pi/N
axis = h*np.arange(1, N+1)
axis = np.linspace(0, 2.0*np.pi, N)
[x , y] = np.meshgrid(axis , axis)

#frequencies
k = np.fft.fftfreq(N)*N

kx = np.zeros([N, N]) + 0.0j
ky = np.zeros([N, N]) + 0.0j

for i in range(N):
    for j in range(N):
        kx[i, j] = 1j*k[j]
        ky[i, j] = 1j*k[i]

k_squared = kx**2 + ky**2
k_squared_no_zero = np.copy(k_squared)
k_squared_no_zero[0,0] = 1.0

#cutoff in pseudospectral method
Ncutoff = np.int(N/3)
Ncutoff_LF = np.int(2**(I-2)/3)

#spectral filter
P = get_P(Ncutoff)
P_LF = get_P(Ncutoff_LF)
P_U = P - P_LF

binnumbers, bins = freq_map()
N_bins = bins.size    

k_min1 = Ncutoff_LF
k_max1 = np.max(P_LF*binnumbers)
P_k1 = get_P_k(k_min1, k_max1)

k_min2 = 0
k_max2 = 5
P_k2 = get_P_k(k_min2, k_max2)

#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

#viscosities
decay_time_nu = 5.0
decay_time_mu = 90.0
nu = 1.0/(day*Ncutoff**2*decay_time_nu)
nu_LF = 1.0/(day*Ncutoff_LF**2*decay_time_nu)
mu = 1.0/(day*decay_time_mu)

#start, end time, end time of data (training period), time step
dt = 0.005
t = 0.0*day
t_end = t + 250.0*day
n_steps = np.int(np.round((t_end-t)/dt))

#############
# USER KEYS #
#############

#simulation name
sim_ID = 'gen_tau_2P_k'
#framerate of storing data, plotting results, computing correlations (1 = every integration time step)
#store_frame_rate = np.floor(5.0*day/dt).astype('int')
store_frame_rate = 1
plot_frame_rate = np.floor(1.0*day/dt).astype('int')
#length of data array
S = np.floor(n_steps/store_frame_rate).astype('int')

#Manual specification of flags 
state_store = True     #store the state at the end
restart = False          #restart from prev state
store = False            #store data
plot = True            #plot results while running, requires drawnow package
compute_ref = True     #compute the reference solution as well, keep at True, will automatically turn off in surrogate mode
load_ref = False

eddy_forcing_type = 'tau_ortho'  

store_ID = sim_ID
    
###############################
# SPECIFY WHICH DATA TO STORE #
###############################

#TRAINING DATA SET
QoI = ['z_n_HF', 'e_n_HF', #'w3_n_HF', 
       'z_n_LF', 'e_n_LF', #'w3_n_LF', 
       'src1', 'src2', #'src3',
       'c_12', 'c_22', #'c_13', 'c_23', 'c_32', 'c_33',
       'tau_1', 'tau_2',# 'tau_3',
       'dE', 'dZ']#, 'dW3']

#QoI = ['w_hat_n_LF', 'w_hat_n_HF']

#PREDICTION DATA SET
#QoI = ['z_n_LF', 'e_n_LF', 'w3_n_LF', 't']

Q = len(QoI)

#allocate memory
samples = {}

if store == True:
    samples['S'] = S
    samples['N'] = N
    
    for q in range(Q):
        
        #assume a field contains the string '_hat_'
        if '_hat_' in QoI[q]:
            samples[QoI[q]] = np.zeros([S, N, N]) + 0.0j
        #a scalar
        else:
            samples[QoI[q]] = np.zeros(S)

#forcing term
F = 2**1.5*np.cos(5*x)*np.cos(5*y);
F_hat = np.fft.fft2(F);
F_hat_full = np.fft.fft2(F)

V_hat_w1 = P_LF*np.fft.fft2(np.ones([N,N]))

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
    w_hat_n_HF = P*np.fft.fft2(w)
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
    
    w_hat_n_LF = P_LF*np.fft.fft2(w)
    w_hat_nm1_LF = np.copy(w_hat_n_LF)
    
    #initial Fourier coefficients of the jacobian at time n and n-1
    VgradW_hat_n_HF = compute_VgradW_hat(w_hat_n_HF, P)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)
    
    VgradW_hat_n_LF = compute_VgradW_hat(w_hat_n_LF, P_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)

#constant factor that appears in AB/BDI2 time stepping scheme   
norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared + mu)        #for reference solution
norm_factor_LF = 1.0/(3.0/(2.0*dt) - nu_LF*k_squared + mu)  #for Low-Fidelity (LF) or resolved solution

#some counters
j = 0; j2 = 0; idx = 0;

T = []; E_LF = []; Z_LF = []; E_HF = []; Z_HF = []; W3_HF = []; W3_LF = []
W1_HF = []; W1_LF = []
TAU1 = []; TAU2 = []; TAU3 = []
TEST = []

if eddy_forcing_type == 'tau_ortho_mean':
    fname = HOME + '/samples/gen_tau_t_3900.0.hdf5'
    h5f = h5py.File(fname, 'r')
    dE_mean = np.mean(h5f['dE'][:])
    dZ_mean = np.mean(h5f['dZ'][:])
    
if load_ref == True:
    fname = HOME + '/samples/gen_tau_t_3900.0.hdf5'
    h5f = h5py.File(fname, 'r')  

fig = plt.figure(figsize=[8, 8])

#time loop
for n in range(n_steps):

    if compute_ref == True:
        
        #solve for next time step
        w_hat_np1_HF, VgradW_hat_n_HF = get_w_hat_np1(w_hat_n_HF, w_hat_nm1_HF, VgradW_hat_nm1_HF, P, norm_factor)
        
        #exact eddy forcing
        EF_hat_nm1_exact = P_LF*VgradW_hat_nm1_HF - VgradW_hat_nm1_LF 
    
        #reference energy and enstrophy
        #e_np1_HF, z_np1_HF, _ = get_EZS(P_LF*w_hat_np1_HF)
        e_n_HF, w1_n_HF, z_n_HF, w3_n_HF = compute_qoi(w_hat_n_HF, P_k1)
        e_n_HF2, w1_n_HF2, z_n_HF2, w3_n_HF2 = compute_qoi(w_hat_n_HF, P_k2)
        
    if load_ref == True:
        e_n_HF = h5f['e_n_HF'][n]
        z_n_HF = h5f['z_n_HF'][n]
        w1_n_HF = h5f['w1_n_HF'][n]
        w3_n_LF = h5f['w3_n_HF'][n]
        
    #######################################
    # covariates (conditioning variables) #
    #######################################
    e_n_LF, w1_n_LF, z_n_LF, w3_n_LF = compute_qoi(w_hat_n_LF, P_k1)
    e_n_LF2, w1_n_LF2, z_n_LF2, w3_n_LF2 = compute_qoi(w_hat_n_LF, P_k2)
   
    #exact orthogonal pattern surrogate
    if eddy_forcing_type == 'tau_ortho' or eddy_forcing_type == 'tau_ortho_mean':
        ###############################
        # generalized scalar tracking #
        ###############################
        
        psi_hat_n_LF = get_psi_hat(w_hat_n_LF)
#        
#        V_hat_1 = -psi_hat_n_LF
#        V_hat_2 = V_hat_w1
#        V_hat_3 = w_hat_n_LF
#        V_hat_4 = w_hat_n_LF_squared

        V_hat_1 = -P_k1*psi_hat_n_LF
        V_hat_2 = P_k1*w_hat_n_LF
        V_hat_3 = -P_k2*psi_hat_n_LF
        V_hat_4 = P_k2*w_hat_n_LF

        T_hat_11 = V_hat_1
        T_hat_12 = V_hat_2
        T_hat_13 = V_hat_3
        T_hat_14 = V_hat_4

        T_hat_21 = V_hat_2
        T_hat_22 = V_hat_1
        T_hat_23 = V_hat_3
        T_hat_24 = V_hat_4

        T_hat_31 = V_hat_3
        T_hat_32 = V_hat_1
        T_hat_33 = V_hat_2
        T_hat_34 = V_hat_4

        T_hat_41 = V_hat_4
        T_hat_42 = V_hat_1
        T_hat_43 = V_hat_2
        T_hat_44 = V_hat_3
   
        ##############################
        
        T_hat = np.zeros([4,4,N,N]) + 0.0j
        T_hat[0,0] = T_hat_11
        T_hat[0,1] = T_hat_12
        T_hat[0,2] = T_hat_13
        T_hat[0,3] = T_hat_14

        T_hat[1,0] = T_hat_21
        T_hat[1,1] = T_hat_22
        T_hat[1,2] = T_hat_23
        T_hat[1,3] = T_hat_24

        T_hat[2,0] = T_hat_31
        T_hat[2,1] = T_hat_32
        T_hat[2,2] = T_hat_33
        T_hat[2,3] = T_hat_34

        T_hat[3,0] = T_hat_41
        T_hat[3,1] = T_hat_42
        T_hat[3,2] = T_hat_43
        T_hat[3,3] = T_hat_44
        
        V_hat = np.zeros([4, N, N]) + 0.0j
        V_hat[0] = V_hat_1
        V_hat[1] = V_hat_2
        V_hat[2] = V_hat_3
        V_hat[3] = V_hat_4
        
        c_ij = compute_cij(T_hat, V_hat)
        c_12 = c_ij[0,0]; c_13 = c_ij[0,1]; c_14 = c_ij[0,2]
        c_22 = c_ij[1,0]; c_23 = c_ij[1,1]; c_24 = c_ij[1,2]
        c_32 = c_ij[2,0]; c_33 = c_ij[2,1]; c_34 = c_ij[2,2]
        c_42 = c_ij[3,0]; c_43 = c_ij[3,1]; c_44 = c_ij[3,2]
        
        P_hat_1 = T_hat_11 - c_12*T_hat_12 - c_13*T_hat_13 - c_14*T_hat_14
        P_hat_2 = T_hat_21 - c_22*T_hat_22 - c_23*T_hat_23 - c_24*T_hat_24
        P_hat_3 = T_hat_31 - c_32*T_hat_32 - c_33*T_hat_33 - c_34*T_hat_34
        P_hat_4 = T_hat_41 - c_42*T_hat_42 - c_43*T_hat_43 - c_44*T_hat_44
    
    #    src_E = compute_int(V_hat_1, P_hat_1)
    #    src_Z = compute_int(V_hat_2, P_hat_2)
    #    src_W3 = compute_int(V_hat_3, P_hat_3)

        if eddy_forcing_type == 'tau_ortho_mean':
            dE = dE_mean
            dZ = dZ_mean            
        else:
            dQ1 = e_n_HF - e_n_LF
            dQ2 = z_n_HF - z_n_LF
            dQ3 = e_n_HF2 - e_n_LF2
            dQ4 = z_n_HF2 - z_n_LF2
        
    #    tau_1 = np.tanh(dE/e_n_LF)*np.sign(src_E)
    #    tau_2 = np.tanh(dZ/z_n_LF)*np.sign(src_Z)
    #    tau_3 = 10**-4*np.tanh(dW3/w3_n_LF)*np.sign(src_W3)
        
        src1 = compute_int(V_hat_1, P_hat_1)
        src2 = compute_int(V_hat_2, P_hat_2)
        src3 = compute_int(V_hat_3, P_hat_3)
        src4 = compute_int(V_hat_4, P_hat_4)
        
        tau_1 = dQ1/src1
        tau_2 = dQ2/src2
        tau_3 = dQ3/src3
        tau_4 = dQ4/src4

        EF_hat = -tau_1*P_hat_1 - tau_2*P_hat_2 - tau_3*P_hat_3 - tau_4*P_hat_4

    #unparameterized solution
    elif eddy_forcing_type == 'unparam':
        EF_hat = np.zeros([N, N])
    #exact, full-field eddy forcing
    elif eddy_forcing_type == 'exact':
        EF_hat = EF_hat_nm1_exact
    else:
        print('No valid eddy_forcing_type selected')
        sys.exit()
   
    #########################
    #LF solve
    w_hat_np1_LF, VgradW_hat_n_LF = get_w_hat_np1(w_hat_n_LF, w_hat_nm1_LF, VgradW_hat_nm1_LF, P_LF, norm_factor_LF, EF_hat)

    t += dt
    j += 1
    j2 += 1    
    
    #plot solution every plot_frame_rate. Requires drawnow() package
    if j == plot_frame_rate and plot == True:
        j = 0

#        EF = np.fft.ifft2(EF_hat)
#        w_n_LF = np.fft.ifft2(w_hat_n_LF)
#        w_n_HF = np.fft.ifft2(P_LF*w_hat_n_HF)
#        
        T.append(t)
        E_LF.append(e_n_LF); Z_LF.append(z_n_LF)
        E_HF.append(e_n_HF); Z_HF.append(z_n_HF)
        W3_HF.append(w3_n_HF); W3_LF.append(w3_n_LF)
        W1_HF.append(w1_n_HF); W1_LF.append(w1_n_LF)
#
#        TAU1.append(tau_1); TAU2.append(tau_2); #TAU3.append(tau_3)
#        
#        print('e_n_HF: %.4e' % e_n_HF, 'w1_n_HF: %.4e' % w1_n_HF, 'z_n_HF: %.4e' % z_n_HF, 'w3_n_HF: %.4e' % w3_n_HF)
#        print('e_n_LF: %.4e' % e_n_LF, 'w1_n_LF: %.4e' % w1_n_LF, 'z_n_LF: %.4e' % z_n_LF, 'w3_n_LF: %.4e' % w3_n_LF)
#
        print('e_n_HF: %.4e' % e_n_HF, 'z_n_HF: %.4e' % z_n_HF)
        print('e_n_LF: %.4e' % e_n_LF, 'z_n_LF: %.4e' % z_n_LF)
        print('e_n_HF2: %.4e' % e_n_HF2, 'z_n_HF: %.4e' % z_n_HF2)
        print('e_n_LF2: %.4e' % e_n_LF2, 'z_n_LF: %.4e' % z_n_LF2)
        
#        
        E_spec_HF, Z_spec_HF = spectrum(w_hat_n_HF, P)
        E_spec_LF, Z_spec_LF = spectrum(w_hat_n_LF, P_LF)
#        
#        #compute_qoi(w_hat_n_LF)
#        
        drawnow(draw)
        
    #store samples to dict
    if j2 == store_frame_rate and store == True:
        j2 = 0
        
        if np.mod(n, np.round(day/dt)) == 0:
            print('n = ', n, ' of ', n_steps)

        for qoi in QoI:
            samples[qoi][idx] = eval(qoi)

        idx += 1  

    #update variables
    if compute_ref == True: 
        w_hat_nm1_HF = np.copy(w_hat_n_HF)
        w_hat_n_HF = np.copy(w_hat_np1_HF)
        VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)

    w_hat_nm1_LF = np.copy(w_hat_n_LF)
    w_hat_n_LF = np.copy(w_hat_np1_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)
    
####################################

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

####################################

#store the samples
if store == True:
    store_samples_hdf5() 

plt.show()