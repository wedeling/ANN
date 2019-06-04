def get_kde(X, N_points = 100):

    kernel = stats.gaussian_kde(X)
    x = np.linspace(np.min(X), np.max(X), N_points)
    pde = kernel.evaluate(x)
    return x, pde

import numpy as np
import matplotlib.pyplot as plt
import test_functions as tf
from scipy import stats

plt.close('all')

#number of data points
n_days = 8*365

#get the data
name = 'dE'
X, y, t = tf.get_tau_EZ_regres(n_days, name)

N = t.size

N_bins = 10
bins = np.linspace(np.min(y), np.max(y), N_bins)
count, _, binnumbers = stats.binned_statistic(y, np.zeros(y.size), statistic='count', bins=bins)

unique_binnumbers = np.unique(binnumbers)

fig = plt.figure()
ax = fig.add_subplot(111)

for i in unique_binnumbers:
    idx = np.where(binnumbers == i)[0]
    
    ax.plot(X[idx, 5], X[idx, 0], 'o', label=i)
    
leg = plt.legend()
leg.draggable(True)

plt.show()