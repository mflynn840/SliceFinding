
from Dataset import Dataset
import numpy as np
from utils import prepAdult


'''
    An implementation of SliceLine Fast Linear Alebra Based Slice Finding
    
    Algorithm Credit goes to: https://dl.acm.org/doi/10.1145/3448016.3457323


    Parameters:
        K (int)
        X (matrix of size of size n x m)
        e  (vector of size containing error for each point) 
        σ (int, size threshold)
        ∝ (score function parameter)
        L (how many level of the lattice to prune)
    

    Returns: 
        TS (a K x m matrix)
            Each row is a slice that was picked as topk
            Each feature is in {0, 1} where a 0 means a feature is not considered in the slice (free)
        TR (a K x 3 matrix of statistics for each slice)
            Score
            Error
            Size
'''



class Slice:
    def __init__(self, idxs, error):
        self.idxs = idxs
        self.error = error
        self.size = len(idxs)


#------------------------------
#TODO X0 has integer encoded featrues (categorical) our dataset does not !!!!
#------------------------------  

class SliceFinder:

    def __init__(self, X, e:list, k=1, sigma=10, alpha=1.0, L=5):
        
        #Return the top k slices from D with largest score
        #e: list of errors for each datapoint
        #sigma: size threshold for smallest allowable slice
        #alpha: score function balancing parameter
        #L : how many levels of the lattice to prune
        
        
        self.X = X
        self.k = k
        self.topK = [Slice([-1],-1) for i in range(k)]
        
        print(self.topk)
        
        self.errors = e
        self.sigma = sigma
        self.alpha = alpha
        self.L_max = L
        self.n, self.m = self.X_train.shape[0], self.X_train.shape[1]
        self.avgError = self.computeAverageError()
        
        

        print("samples: " + str(self.n))
        print("features: " + str(self.m))
        print("avg error: " + str(self.avgError))
        
        
        self.prepareTheData()
        self.findAndScoreBasicSlices()
        self.mainLoop()
        
        
    def findAndScoreBasicSlices(self):
        
        #slice sizes are the sum of columns since matrix is 1-hot
        ss0 = self.X.sum(axis=0).T
        
        #slice errors
        se0 = np.dot(self.e.T).T 
        
        sc = self.score()
        
        #current valid slice indexes
        self.CI = np.select([">="],[])
        
        #current slice errors
        self.SE = np.dot(self.errors.T, self.X).T
        
        #basic slice scores
        sc = self.score()
        
        
        
        #
        

    def prune(slice):
        
        
        #select features satsiying ss0 >= sigma and se0 > 0
        return np.where(self.ss0 >= self.sigma and SE > )
        
    def mainLoop(self):
        L = 1
        self.X = self.prune(CI) 
        
        
        #while there are still valid slices left and 
        while(S.shape[1] > 0 and L < self.L_max):
            L = L+1
            self.S, self.R = self.recursiveStep()
            S = self.pair_candidates(S)
            
            for i in rangeeach row of R
                Ri <- evalSlices(X, e ebar, this rounds s2)
            
            

        '''
        X = X[,cI] //select features satsiying ss0 >= sigma and se0 > 0
        
        while nrow(S) > 0 and L < ceiling(L) do 
            L <- L+1
            s <- getPairCandidates(S, R, TS, TR, K, L, avgerr, sigma, alpha, fb, fe)
            R <- matrix(0, nrow(S), 4)
            CI <- pruneCandidates(s)
            S2 <- S[, CI]
            
            
            for i in nrows(S) do 
                Ri <- evalSlices(X, e, ebar, S2i.T, L, alpha)
                TS, TR = maintainTopK(S,R, TS, TR, K, sigma)
                
        '''

        
                

    
    '''
        Find domains for each feature
        Compute start and end offset of each feature
        Get matrix X which has one hot encoding of each feature in each domain for each datapoint
        
        fdom <- colMaxs(X0) //fdom is a 1 x m matrix
        fb <- cumsum(fdom) - fdom
        fe <- cumsum(fdom)
        X <- onehot(x0 + fb) //X is an nx1 matrix
        
        
        
    '''
    def prepareTheData(self):
        
        # X0 is an n x m matrix. n samples of m features
        # Assume that the values for each feature fi is in [0-di]
        # we want to find the largest value in each columnn, this is di
        
        
        #if X0 has 10 datapoints of 2 features then the output should be a 2x1 matrix  
        #find the maximum value along ecah column of the dataset
        fdom = np.max(self.X_train, axis=0)
        print("fdom: " + str(np.max(self.X_train, axis=0)))
        
        #fb is the feature offsets
        fb = np.cumsum(fdom) - fdom
        print("fb: " + str(fb))
        
        fe = np.cumsum(fdom)
        

        
 
       
    '''
    
        The average dataset error is used in the scoring function
    ''' 
    
    def computeAverageError(self):
        return np.sum(self.errors)/self.n
        
    '''
    
        compute the scoring function
    
    '''
    def score(self, slice:Slice):
        avg_slice_error = sum(slice.error) / slice.size
        return self.alpha * ((avg_slice_error / self.avgError) - 1) - ((1 - self.alpha) * self.n / slice.size)
        

    def recursiveStep(self):
        bar=1
        
    
    
    
    #Find the topk slices for a given dataset D and error vector E
    
    

def testBed():
    
    X_train, _, _, _, _, _ = prepAdult()
    

    errors = np.random.randint(0, 10, size=(X_train.shape[0]))
    
    
    print("Xtrain: ")
    print(X_train)
    print("errors: ")
    print(errors)
    
    

    
    foo = SliceFinder(X_train, errors)
    
    
testBed()
    
