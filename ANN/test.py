# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold
# data sample
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# prepare cross validation
kfold = KFold(3, True, 1)
# enumerate splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (data[train], data[test]))


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
