import numpy as np
from scipy import stats

class SimpleBin:
    
    def __init__(self, y, bins):
        
        self.count, _, self.binnumbers = \
        stats.binned_statistic(y, np.zeros(y.size), statistic='count', bins=bins)
        
        self.n_bins = bins.size - 1
        
        self.unique_binnumbers = np.unique(self.binnumbers)
        
        self.r_ip1 = {}
        for i in self.unique_binnumbers:
            idx = np.where(self.binnumbers == i)
            self.r_ip1[i-1] = y[idx]
            
        self.mapping = np.zeros(self.n_bins + 2).astype('int')
        self.mapping[1:-1] = range(self.n_bins)
        self.mapping[-1] = self.n_bins - 1
            
    def draw(self, bin_idx):
        return np.random.choice(self.r_ip1[bin_idx])        