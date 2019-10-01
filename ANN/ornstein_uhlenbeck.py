class Ornstein_Uhlenbeck:
    
    def __init__(self, mu, sigma, dt):
        
        self.mu = mu
        self.sigma = sigma
        self.tau = 10.0
        self.sigma_bis = sigma * np.sqrt(2. / self.tau)
        self.x_i = 0.0
        self.dt = dt
        self.sqrtdt = np.sqrt(dt)
        
    def step(self):
        self.x_i = self.x_i + dt * (-(self.x_i - self.mu) / self.tau) + \
                  self.sigma_bis * self.sqrtdt * np.random.randn()
        
    def get_x(self):
        return self.x_i
    
    def set_x(self, x_i):
        self.x_i = x_i
        
import numpy as np

if __name__ == '__main__':
    
    import os
    import h5py 
    import matplotlib.pyplot as plt
    
    HOME = os.path.abspath(os.path.dirname(__file__))
    fname = HOME + '/samples/gen_tau_t_3900.0.hdf5'
    h5f = h5py.File(fname, 'r')  
    
    mu = np.mean(h5f['dE'][:])
    sigma = np.std(h5f['dE'][:])
    dt = 0.01
    
    plt.plot()
    
    ou_proc = Ornstein_Uhlenbeck(mu, sigma, dt)
    
    X = []
    N = h5f['dE'][:].size
    T = []
    
    for i in range(N):
        ou_proc.step()
        X.append(ou_proc.get_x())
        T.append((i+1)*dt)

    plt.plot(X)
    #plt.plot([T[0], T[-1]], [mu, mu], '--r')
    plt.plot(h5f['dE'][:], 'ro')
    
    plt.show()    