import numpy as np
from scipy import stats

class SimpleBin:
    
    def __init__(self, y, bins, n_feat = 1, max_lag = 1):
        
        self.count, _, self.binnumbers = \
        stats.binned_statistic(y, np.zeros(y.size), statistic='count', bins=bins)
        
        self.n_bins = bins.size - 1
        
        self.unique_binnumbers = np.unique(self.binnumbers)

        self.n_feat = n_feat
        self.max_lag = max_lag
        self.feat = {}
        for i in range(n_feat):
            self.feat[i] = []
        
        self.r_ip1 = {}
        for i in self.unique_binnumbers:
            idx = np.where(self.binnumbers == i)
            self.r_ip1[i-1] = y[idx]
            
        self.mapping = np.zeros(self.n_bins + 2).astype('int')
        self.mapping[1:-1] = range(self.n_bins)
        self.mapping[-1] = self.n_bins - 1
            
    def draw(self, bin_idx):
        return np.random.choice(self.r_ip1[bin_idx])
    
    #append the features supplied to the binning object during simulation to self.feat
    #Note: use list to dynamically append, array is very slow
    #Note 2: X_i must be of size [n_in, 1], where n_in is 
    #the number of input features at a SINGLE time instance
    def append_covar(self, X_i):
       
        for i in range(self.n_feat):
            self.feat[i].append(X_i[i])
            
            #if max number of lagged features is reached, remove first item
            if len(self.covar[i]) > self.max_lag:
                self.covar[i].pop(0)

#    #return lagged covariates, assumes constant lag
#    #Note: typecast to array if spatially varying lag is required
#    def get_covar(self, lags):
#        
#        c_i = np.zeros([self.N**2, self.N_c])
#
#        for i in range(self.N_c):
#            
#            if lags[i] <= self.max_lag:
#                c_i[:, i] = self.covar[i][-lags[i]]
#            else:
#                print 'Warning, max lag exceeded'
#                import sys; sys.exit()
#            
#        return c_i       
