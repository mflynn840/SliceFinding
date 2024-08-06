
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

    def __init__(self, X, e:list, k=1, sigma=1, alpha=1.0, L=5):
        
        #Return the top k slices from D with largest score
        #e: list of errors for each datapoint
        #sigma: size threshold for smallest allowable slice
        #alpha: score function balancing parameter
        #L : how many levels of the lattice to prune
        
        
        self.X0 = X
        self.k = k
        self.topK = [Slice([-1],-1) for i in range(k)]
        
        print(self.topK)
        
        self.errors = e
        self.sigma = sigma
        self.alpha = alpha
        self.L_max = L
        self.n, self.m = self.X0.shape[0], self.X0.shape[1]
        self.avgError = self.computeAverageError()
        
        

        print("samples: " + str(self.n))
        print("features: " + str(self.m))
        print("avg error: " + str(self.avgError))
        
        
        self.fdom, self.fb, self.fe, self.X = self.prepareTheData()
        self.S, self.R, self.CI = self.findAndScoreBasicSlices()
        self.TS, self.TR = self.maintainTopK(self.S, self.R, 0, 0, self.k, self.sigma)
        self.mainLoop()
        
    def maintainTopK(self, S, R, TS, TR, k, sigma):
        
        #TS is top k slices
        #TR are the top k scores, errors and sizes
        
        #R["sc"] is a vector of scores for each slice
        if len(R["sc"]) < k:
            # If fewer scores, find all available indices and fill the rest with -1
            top_k_indices = np.concatenate((np.argsort(R["sc"])[::-1], -np.ones(k - len(R["sc"]), dtype=int)))
        else:
            # Otherwise, get the indices of the top k largest scores
            top_k_indices = np.argsort(R["sc"])[-k:][::-1]

        print("top k indicies: "  + str(top_k_indices))
        #find the slice indexes with the top k 
        TS =None
        TR = None

        return TS, TR
        
                  

    def onehotColumn(self, col):
        unique_values = np.unique(col)
        one_hot = np.zeros((col.size, unique_values.size))
        for i, value in enumerate(col):
            one_hot[i, np.where(unique_values == value)[0]] = 1
            
        return one_hot
        
    
    def oneHotMatrix(self, X):
        
        cols = []
        for i in range(X.shape[1]):
            column = X[:, i]
            encoded_column = self.onehotColumn(column)
            cols.append(encoded_column)
        
        encoded_matrix = np.hstack(cols)
        return encoded_matrix
        
            
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
        fdom = np.max(self.X0, axis=0)
        print("fdom: " + str(np.max(self.X0, axis=0)))
        
        #fb is the feature offsets
        fb = np.cumsum(fdom) - fdom
        print("fb: " + str(fb))
        
        fe = np.cumsum(fdom)
        print("fe: " + str(fe))
        
        print("X0 + fb" + str(self.X0+fb))
        X = self.oneHotMatrix(self.X0 + fb)
        print(X.shape)
        
        
        return fdom, fb, fe, X
    
    '''
        Returns
        S basic slices
        R slice errors
        CI 
    
    '''
    def findAndScoreBasicSlices(self):
        
        #slice sizes for each basic slice
        ss0 = self.X.sum(axis=0).T
        print("ss0: "  + str(ss0))
        
        #error for each slice is se0
        self.errors = self.errors[np.newaxis, :]
        se0 = np.sum(self.errors.T * self.X, axis=0).T
        print("se0" + str(se0))
        
        #pruning indicator
        CI = np.where((ss0 >= self.sigma) & (se0 > 0))
        print("CI: " + str(CI))
        
        #prune basic slices
        ss = ss0[CI]
        se = se0[CI]
        
        #basic slice scores
        sc = self.alpha * ((se/ss) / self.avgError - 1) - (1-self.alpha) * (self.n/ss - 1)
        print("sc: " + str(sc))
        
        #statistics vector
        R = {
            "sc" : sc,
            "se" : se,
            "ss" : ss            
        }
        

        #slices
        # Initialize a dictionary to store indices for each slice
        slices = {i: [] for i in range(self.X.shape[1])}

        # Iterate over each feature (column)
        for feature_index in range(self.X.shape[1]):
            # Find indices of rows where the current column (feature) has a 1
            indices = np.where(self.X[:, feature_index] == 1)[0]
            # Store the indices in the dictionary
            slices[feature_index] = indices.tolist()
            
        print("slices: " + str(slices))
        return slices, R, CI
        
        
        
        

    def prune(slice):
        
        return slice
        #select features satsiying ss0 >= sigma and se0 > 0
        #return np.where(self.ss0 >= sigma(
        #) self.sigma and SE > )
        
    def mainLoop(self):
        L = 1
        #self.X = self.prune(CI) 
        
        
        #while there are still valid slices left and 
        while(self.S["sc"].shape[0] > 0 and L < self.L_max):
            L = L+1
            self.S = self.getPairCandidates(self.S, self.R, self.TS, self.TR, self.k, L, self.avgError, self.sigma, self.alpha, self.fb, self.fe)
            R = np.zeros(self.S.shape[0], 4)
            S2 = self.S[self.CI]
            
            for i in range(self.S.shape[0]):
                R[i] = self.evalSlices(self.X, self.e, self.avgError, S2[i].T, L, self.alpha)
            
            TS, TR = self.maintainTopK(S, R, TS, TR, self.k, self.sigma)
            
        return self.decodeTopK(self.TS, self.fb, self.fe), TR
            
            #for i in rangeeach row of R
            #    Ri <- evalSlices(X, e ebar, this rounds s2)
            
            

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
    
    n_rows = 2
    n_cols = 3
    #X_train, _, _, _, _, _ = prepAdult()
    X_train = np.arange(n_rows).reshape(-1, 1) * np.ones((1, n_cols), dtype=int)
    errors = np.random.randint(0, 10, size=(X_train.shape[0]))

    
    print("Xtrain: ")
    print(X_train)
    print("errors: ")
    print(errors)
    
    

    
    foo = SliceFinder(X_train, errors)
    
    
testBed()
    
