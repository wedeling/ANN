import numpy as np
import matplotlib.pyplot as plt

N = 500
x = np.linspace(-2.0*np.pi, 2.0*np.pi, N)
y = np.cos(x) + np.random.randn(N)*0.2

plt.plot(x, y, 'bo', alpha = 0.2)
plt.plot(x, np.cos(x), 'r')

beta = 0.999

V = []
V.append(y[0])

for i in range(1, N):
    V.append(beta*V[i-1] + (1-beta)*y[i])
    
plt.plot(x, V, 'g')