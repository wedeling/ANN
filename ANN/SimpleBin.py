import numpy as np
from scipy import stats

class SimpleBin:
    
    def __init__(self, y, bins):
        
        self.count, _, self.binnumbers = \
        stats.binned_statistic(y, np.zeros(y.size), statistic='count', bins=bins)
        
        self.unique_binnumbers = np.unique(self.binnumbers)
        
        self.r_ip1 = {}
        for i in self.unique_binnumbers:
            idx = np.where(self.binnumbers == i)
            self.r_ip1[i-1] = y[idx]
            
    def draw(self, bin_idx):
        return np.random.choice(self.r_ip1[bin_idx])        