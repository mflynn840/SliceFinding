from SliceFinder import SliceFinder
import numpy as np
from utils import getLogger
import logging
from utils import unpickleDataset
import math
import pandas as pd



class GASliceFinder(SliceFinder):
    def __init__(self, X, e:list, k=1, sigma=1, alpha=0.85, L=2, logger=None, auto=True):
        super().__init__(X, e, k, sigma, alpha, L, logger, auto)
    
class SliceLine(SliceFinder):
    def __init__(self, X, e:list, k=1, sigma=1, alpha=0.85, L=2, logger=None, auto=True):
        super().__init__(X, e, k, sigma, alpha, L, logger, auto)
        
    def score(self, slice_sizes, slice_errors, total_samples):
        """
        Summary:
            Apply the SliceLines scoring function. If division by zero occurs, return `-inf`.
        
        :param sliceSizes: 
            Vector of sizes for each slice.
        
        :param sliceErrors: 
            Vector of errors for each slice.
        
        :param nrows: 
            Number of samples in the entire dataset.
        
        :return: 
            The computed scores for each slice, with `-inf` for cases where division by zero occurs.
        """
        sc = self.alpha * ((slice_errors/slice_sizes) / self.avg_error - 1) - (1-self.alpha) * (total_samples/slice_sizes - 1)
        return np.nan_to_num(sc, nan=-np.inf)
    

    def upperbound_score(self, ss, se, sm):

        """
        Summary:
            Compute an upper bound for the SliceLines scoring function.
        
        :param ss: 
            Vector containing the size of each slice.
        
        :param se: 
            Vector containing the error for each slice.
        
        :param sm: 
            Vector containing the maximum error for each slice.
        

        :return: 
            The computed upper bound for the SliceLines scoring function.
        """
        # probe interesting points of sc in the interval [minSup, ss],
        # and compute the maximum to serve as the upper bound 
        
        #s is the critical points we have to check
        s = np.column_stack((np.full((ss.shape[0]), self.sigma), np.maximum(se/sm, self.sigma), ss))
        ex = np.ones((3))
        smex = s*np.column_stack((sm, sm, sm))
        seex = np.column_stack((se, se, se))
        pmin = np.minimum(smex, seex)

        sc = np.max(self.alpha * ((pmin/s) / self.avg_error-1) - (1-self.alpha) * (1/s*self.m2 - 1), axis=1)
        sc = np.nan_to_num(sc, nan=-np.inf)
        
        
        return sc
    

def testBed():
    """

        Summary:
            Run sliceline on a toy example and print the results


    """
    
    #dummy data has 3 samples of 4 features where sample i has all features set to value i
    #errors are 1,2,3
    
    logger = getLogger(__name__, "test.log")
    n_rows = 3
    n_cols = 4
    X_train = np.arange(1, n_rows+1).reshape(-1, 1) * np.ones((1, n_cols), dtype=int)
    errors = np.arange(1, 4).reshape(-1,1)
    
    sf = SliceFinder(X_train, errors, k=4, sigma=1, alpha=0.95, L=1, logger=logger, auto=False)
    sf.run()
    
    SliceFinder.pretty_print_results(sf.result, ["feature 1", "feature 2", "feature 3", "feature 4"])
    #print("foo result")
    #print(foo.result)
    


def testAdult():
    """
        Summary:
            Run sliceLine on adult dataset with dummy errors

    """
    logger = getLogger(__name__, "test.log")
    logger.setLevel(logging.INFO)
    X_train, _, _, _ = unpickleDataset("./Data/Adult/train.pkl", "./Data/Adult/test.pkl")
    np.random.seed(1)
    errors = np.random.random(size=X_train.shape[0]).reshape(-1,1)
    
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv("./Data/Adult/dummyErrors.csv", header=True)
    
    sigma = max(math.ceil(0.01 * X_train.shape[0]), 8)
    sf = SliceLine(X_train, errors, k=4, sigma=5, alpha=0.95, L=2, logger=logger, auto=False)
    sf.run()
    print("done")
    print(sf.result)
    SliceLine.pretty_print_results(sf.result, ["feature" + str(i) for i in range(86)])
    