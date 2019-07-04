
##########################
# S U B R O U T I N E S  #
##########################

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

def draw_w():
    plt.subplot(111)
    plt.plot(T, R_DE)
    plt.plot(T, r_dE[0:n+1])
    plt.tight_layout()

def draw_2w():
    plt.subplot(121, title=r'$\Delta E$', xlabel=r'$t\;[day]$')
    plt.plot(np.array(T)/day, R_DE)
    plt.plot(np.array(T)/day, r_dE[0:n+1], linewidth=4)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.subplot(122, title=r'$\Delta Z$', xlabel=r'$t\;[day]$')
    plt.plot(np.array(T)/day, R_DZ)
    plt.plot(np.array(T)/day, r_dZ[0:n+1], linewidth=4)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()

#return the fourier coefs of the stream function
def get_psi_hat(w_hat_n):

    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0

    return psi_hat_n

###############################
# DATA-DRIVEN TAU SUBROUTINES #
###############################

def get_data_driven_tau_src_EZ(w_hat_n_LF, w_hat_n_HF, P, tau_max_E, tau_max_Z):
    
    E_LF, Z_LF, S_LF = get_EZS(w_hat_n_LF)

    src_E = E_LF**2/Z_LF - S_LF
    src_Z = -E_LF**2/S_LF + Z_LF

    E_HF = compute_E(P*w_hat_n_HF)
    Z_HF = compute_Z(P*w_hat_n_HF)

    dE = (E_HF - E_LF)#/E_LF
    dZ = (Z_HF - Z_LF)#/Z_LF

    tau_E = tau_max_E*np.tanh(dE/E_LF)*np.sign(src_E)
    tau_Z = tau_max_Z*np.tanh(dZ/Z_LF)*np.sign(src_Z)
    
    return tau_E, tau_Z, dE, dZ

def get_surrogate_tau_src_EZ(w_hat_n_LF, r, tau_max_E, tau_max_Z):
    
    E_LF, Z_LF, S_LF = get_EZS(w_hat_n_LF)

    src_E = E_LF**2/Z_LF - S_LF
    src_Z = -E_LF**2/S_LF + Z_LF

    dE = r[0] 
    dZ = r[1]

    tau_E = tau_max_E*np.tanh(dE/E_LF)*np.sign(src_E)
    tau_Z = tau_max_Z*np.tanh(dZ/Z_LF)*np.sign(src_Z)
    
    return tau_E, tau_Z

def get_EZS(w_hat_n):

    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    psi_n = np.fft.irfft2(psi_hat_n)
    w_n = np.fft.irfft2(w_hat_n)
    
    e_n = -0.5*psi_n*w_n
    z_n = 0.5*w_n**2
    s_n = 0.5*psi_n**2 

    E = simps(simps(e_n, axis), axis)/(2*np.pi)**2
    Z = simps(simps(z_n, axis), axis)/(2*np.pi)**2
    S = simps(simps(s_n, axis), axis)/(2*np.pi)**2

    return E, Z, S

#compute the energy at t_n
def compute_E(w_hat_n):
    
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    psi_n = np.fft.irfft2(psi_hat_n)
    w_n = np.fft.irfft2(w_hat_n)
    
    e_n = -0.5*psi_n*w_n

    E = simps(simps(e_n, axis), axis)/(2*np.pi)**2
    
    return E

#compute the enstrophy at t_n
def compute_Z(w_hat_n):
    
    w_n = np.fft.irfft2(w_hat_n)
    
    z_n = 0.5*w_n**2

    Z = simps(simps(z_n, axis), axis)/(2*np.pi)**2
    
    return Z

#######################
# ORTHOGONAL PATTERNS #
#######################

def get_psi_hat_prime(w_hat_n):
    
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0, 0] = 0.

    psi_n = np.fft.irfft2(psi_hat_n)
    w_n = np.fft.irfft2(w_hat_n)

    nom = simps(simps(w_n*psi_n, axis), axis)
    denom = simps(simps(w_n*w_n, axis), axis)

    return psi_hat_n - nom/denom*w_hat_n

def get_w_hat_prime(w_hat_n):
    
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0, 0] = 0.

    psi_n = np.fft.irfft2(psi_hat_n)
    w_n = np.fft.irfft2(w_hat_n)

    nom = simps(simps(w_n*psi_n, axis), axis)
    denom = simps(simps(psi_n*psi_n, axis), axis)

    return w_hat_n - nom/denom*psi_hat_n

#compute the (temporal) correlation coeffient 
def corr_coef(X, Y):
    return np.mean((X - np.mean(X))*(Y - np.mean(Y)))/(np.std(X)*np.std(Y))


###########################
# M A I N   P R O G R A M #
###########################

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.integrate import simps
from itertools import combinations, chain
import sys
import json
from base import NN
import SimpleBin as binning
from drawnow import drawnow

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

HOME = os.path.abspath(os.path.dirname(__file__))

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

#number of gridpoints in 1D
I = 7
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

#cutoff in pseudospectral method
Ncutoff = N/3
Ncutoff_LF = 2**(I-1)/3 

#spectral filter
P = get_P(Ncutoff)
P_LF = get_P(Ncutoff_LF)
P_U = P - P_LF

#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

#viscosities
decay_time_nu = 5.0
decay_time_mu = 90.0
nu = 1.0/(day*Ncutoff**2*decay_time_nu)
nu_LF = 1.0/(day*Ncutoff**2*decay_time_nu)
mu = 1.0/(day*decay_time_mu)

#start, end time, end time of data (training period), time step
t = 250.0*day
t_end = t + 8*365*day
dt = 0.01
n_steps = np.ceil((t_end-t)/dt).astype('int')

#############
# USER KEYS #
#############

#simulation name
sim_ID = 'tau_EZ'
#framerate of storing data, plotting results, computing correlations (1 = every integration time step)
store_frame_rate = 1
plot_frame_rate = np.floor(1.0*day/dt).astype('int')
corr_frame_rate = np.floor(0.25*day/dt).astype('int')
#length of data array
S = np.floor(n_steps/store_frame_rate).astype('int')

#user-specified parameter of tau_E and tau_Z terms
tau_E_max = 1.0
tau_Z_max = 1.0

##read flags from input file
#fpath = sys.argv[1]
#fp = open(fpath, 'r')
#N_surr = int(fp.readline())
#inputs = []
#
#flags = json.loads(fp.readline())
#print('*********************')
#print('Simulation flags')
#print('*********************')
#
#for key in flags.keys():
#    vars()[key] = flags[key]
#    print(key, '=', flags[key])
#
#print('*********************')

#Manual specification of flags 
state_store = False      #store the state at the end
restart = True           #restart from prev state
store = True             #store data
plot = False              #plot results while running, requires drawnow package
compute_ref = True       #compute the reference solution as well, keep at True, will automatically turn off in surrogate mode

eddy_forcing_type = 'tau_ortho_ann'  #which eddy forcing to use (tau_ortho, tau_ortho_ann, exact, unparam)
input_file = 'T2'

store_ID = sim_ID + '_' + input_file 

##################       
# ANN PARAMETERS #
##################

if eddy_forcing_type == 'tau_ortho_ann':

#    #create empty ANN object
#    dE_ann = NN.ANN(X = np.zeros(10), y = np.zeros(1))
#    #load trained ann
#    dE_ann.load_ANN(name='dE_T2')
#
#    #create empty ANN object
#    dZ_ann = NN.ANN(X = np.zeros(10), y = np.zeros(1))
#    #load trained ann
#    dZ_ann.load_ANN(name='dZ_T2')
#    
#    #NOTE: making the assumption here that both ANNs use the same features
#    X_mean = dE_ann.X_mean
#    X_std = dE_ann.X_std
#    
#    #number of featues
#    n_feat = dE_ann.n_in
#    
#    #reference data for resampling
#    r_dE = dE_ann.aux_vars['y']
#    bins_dE = dE_ann.aux_vars['bins']
#    r_dZ = dZ_ann.aux_vars['y']
#    bins_dZ = dZ_ann.aux_vars['bins']

    #create empty ANN object
    dE_dZ_ann = NN.ANN(X = np.zeros(10), y = np.zeros(1))
    #load trained ann
    dE_dZ_ann.load_ANN(name='dE_dZ')

    #reference data for resampling
    r_dE = dE_dZ_ann.aux_vars['dE']
    bins_dE = dE_dZ_ann.aux_vars['bins_dE']
    r_dZ = dE_dZ_ann.aux_vars['dZ']
    bins_dZ = dE_dZ_ann.aux_vars['bins_dZ']

    #NOTE: making the assumption here that both ANNs use the same features
    X_mean = dE_dZ_ann.X_mean
    X_std = dE_dZ_ann.X_std

    #number of featues
    n_feat = dE_dZ_ann.n_in
        
    #bin samplers
    dE_sampler = binning.SimpleBin(r_dE, bins_dE)
    dZ_sampler = binning.SimpleBin(r_dZ, bins_dZ)

###############################
# SPECIFY WHICH DATA TO STORE #
###############################

#QoI to store, First letter in caps implies an NxN field, otherwise a scalar 

#TRAINING DATA SET
QoI = ['z_n_HF', 'e_n_HF', \
       'z_n_LF', 'e_n_LF', 'u_n_LF', 's_n_LF', 'v_n_LF', 'o_n_LF', \
       'sprime_n_LF', 'zprime_n_LF', \
       'tau_E', 'tau_Z', 't']

#PREDICTION DATA SET
#QoI = ['z_n_LF', 'e_n_LF', 't']

Q = len(QoI)

#allocate memory
samples = {}

if store == True:
    samples['S'] = S
    samples['N'] = N
    
    for q in range(Q):
        
        #a field
        if QoI[q][0].isupper():
            samples[QoI[q]] = np.zeros([S, N, int(N/2+1)]) + 0.0j
        #a scalar
        else:
            samples[QoI[q]] = np.zeros(S)

#forcing term
F = 2**1.5*np.cos(5*x)*np.cos(5*y);
F_hat = np.fft.rfft2(F);

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
norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared + mu)        #for reference solution
norm_factor_LF = 1.0/(3.0/(2.0*dt) - nu_LF*k_squared + mu)  #for Low-Fidelity (LF) or resolved solution

#some counters
j = 0; j2 = 0; idx = 0;

T = []; R_DE = []; R_DZ = []

fig = plt.figure(figsize=[10, 5])

#time loop
for n in range(n_steps):
    
    #orthogonal patterns
    psi_hat_n_prime = get_psi_hat_prime(w_hat_n_LF)
    w_hat_n_prime = get_w_hat_prime(w_hat_n_LF)

    if compute_ref == True:
        
        #solve for next time step
        w_hat_np1_HF, VgradW_hat_n_HF = get_w_hat_np1(w_hat_n_HF, w_hat_nm1_HF, VgradW_hat_nm1_HF, P, norm_factor)
        
        #exact eddy forcing
        EF_hat_nm1_exact = P_LF*VgradW_hat_nm1_HF - VgradW_hat_nm1_LF 
 
        #exact tau_E and tau_Z
        tau_E, tau_Z, dE, dZ = get_data_driven_tau_src_EZ(w_hat_n_LF, w_hat_n_HF, P_LF, tau_E_max, tau_Z_max)
    
        #E & Z tracking eddy forcing
        EF_hat_n_ortho = -tau_E*psi_hat_n_prime - tau_Z*w_hat_n_prime 

        #reference energy and enstrophy
        e_n_HF, z_n_HF, _ = get_EZS(P_LF*w_hat_n_HF)

    #######################################
    # covariates (conditioning variables) #
    #######################################
    e_n_LF, z_n_LF, s_n_LF = get_EZS(w_hat_n_LF)
    psi_n_LF = np.fft.irfft2(get_psi_hat(w_hat_n_LF))
    u_n_LF = 0.5*simps(simps(psi_n_LF*F, axis), axis)/(2.0*np.pi)**2
    w_n_LF = np.fft.irfft2(w_hat_n_LF)
    v_n_LF = 0.5*simps(simps(w_n_LF*F, axis), axis)/(2.0*np.pi)**2
    nabla2_w_n_LF = np.fft.irfft2(k_squared*w_hat_n_LF)
    o_n_LF = 0.5*simps(simps(nabla2_w_n_LF*w_n_LF, axis), axis)/(2.0*np.pi)**2

    #compute S' and Z'
    sprime_n_LF = e_n_LF**2/z_n_LF - s_n_LF
    zprime_n_LF = z_n_LF - e_n_LF**2/s_n_LF

    ##############

    #exact orthogonal pattern surrogate
    if eddy_forcing_type == 'tau_ortho':
        EF_hat = -tau_E*psi_hat_n_prime - tau_Z*w_hat_n_prime
    elif eddy_forcing_type == 'tau_ortho_ann':
        
        #EF_hat = -tau_E*psi_hat_n_prime - tau_Z*w_hat_n_prime
        
        #features
        X_feat = np.array([z_n_LF, e_n_LF, u_n_LF, s_n_LF, v_n_LF, o_n_LF, sprime_n_LF, zprime_n_LF])

        #standardize by data mean and std if standardize flag was set to True during ann training
        X_feat = (X_feat - X_mean)/X_std

#        _, idx_max_dE = dE_ann.get_softmax(X_feat.reshape([1, n_feat]))
#        _, idx_max_dZ = dZ_ann.get_softmax(X_feat.reshape([1, n_feat]))
        
        _, idx_max = dE_dZ_ann.get_softmax(X_feat.reshape([1, n_feat]))
        
        r = [dE_sampler.draw(idx_max[0]), dZ_sampler.draw(idx_max[1])]
        
        r_tau_E, r_tau_Z = get_surrogate_tau_src_EZ(w_hat_n_LF, r, 1.0, 1.0)
        
        EF_hat = -r_tau_E*psi_hat_n_prime - r_tau_Z*w_hat_n_prime
        
        R_DE.append(r[0])
        R_DZ.append(r[1])
        
        T.append(t)
        
    #unparameterized solution
    elif eddy_forcing_type == 'unparam':
        EF_hat = np.zeros([N, int(N/2+1)])
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

        w_np1_LF = np.fft.irfft2(w_hat_np1_LF)
        EF = np.fft.irfft2(EF_hat)
        
        e_n_HF, z_n_HF, s_n_HF = get_EZS(P_LF*w_hat_n_HF)
        e_n_LF, z_n_LF, s_n_LF = get_EZS(w_hat_n_LF)
        
        print('e_n_HF: %.4f' % e_n_HF, 'z_n_HF: %.4f' % z_n_HF, 's_n_HF: %.4f' % s_n_HF)
        print('e_n_LF: %.4f' % e_n_LF, 'z_n_LF: %.4f' % z_n_LF, 's_n_LF: %.4f' % s_n_LF)

        drawnow(draw_2w)
        
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
