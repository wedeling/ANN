import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pickle.load(open('./samples/scaling.pickle', 'rb'))

print(data)


#plt.close('all')
#
#N = 1000
#
#x = np.linspace(0, 2*np.pi, N)
#noise = np.random.randn(N)*0.1
#y = np.cos(2*x) + noise
#
#plt.plot(x, y, 'bo', alpha=0.2)
#
#beta = 0.9
#alpha = 0.001
#
#V = []
#V.append(0.0)
#
#for i in range(1, N):
#    #beta_i = beta*(1.0 - beta**i)
#    beta_i = beta
#    V_i = beta_i*V[i-1] + (1-beta_i)*y[i]
#    V.append(V_i)
#    
#plt.plot(x[1:], V[1:], 'r')
