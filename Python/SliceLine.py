import numpy as np
import pandas as pd
from utils import prepAdult, getLogger
from sklearn.preprocessing import OneHotEncoder


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

    def __init__(self, X, e:list, k=1, sigma=1, alpha=1.0, L=5, logger=None):
        
        #Return the top k slices from D with largest score
        #e: list of errors for each datapoint
        #sigma: size threshold for smallest allowable slice
        #alpha: score function balancing parameter
        #L : how many levels of the lattice to prune
        
        
        #Initilize variables
        self.logger = logger
        self.X0 = X
        self.k = k
        self.topK = [Slice([-1],-1) for i in range(k)]
        self.logger.debug("top k: " + str(self.topK) )
        self.errors = e
        self.sigma = sigma
        self.alpha = alpha
        self.L_max = L
        self.n, self.m = self.X0.shape[0], self.X0.shape[1]
        self.avgError = self.computeAverageError()
        self.logger.debug("samples: " + str(self.n))
        self.logger.debug("features: " + str(self.m))
        self.logger.debug("avg error: " + str(self.avgError))
        
        #do the sliceline algorithm
        self.fdom, self.fb, self.fe, self.X = self.data_prep()
        self.S, self.R, self.CI = self.find_score_basic_slices(self.X)
        self.TS, self.TR = self.maintainTopK(self.S, self.R, 0, 0, self.k, self.sigma)
        self.mainLoop()

        
            
    '''
        Find domains for each feature
        Compute start and end offset of each feature
        Get matrix X which has one hot encoding of each feature in each domain for each datapoint
        
        fdom <- colMaxs(X0) //fdom is a 1 x m matrix
        fb <- cumsum(fdom) - fdom
        fe <- cumsum(fdom)
        X <- onehot(x0 + fb) //X is an nx1 matrix  
    '''
    
    
    def data_prep(self):
        
        # X0 is an n x m matrix. n samples of m features
        # Assume that the values for each feature fi is in [0-di]
        # we want to find the largest value in each columnn, this is di
        
        
        #find the maximum value along ecah column of the dataset (domain sizes)
        fdom = np.max(self.X0, axis=0)
        self.logger.debug("fdom: " + str(np.max(self.X0, axis=0)))
        
        #fb is the feature offsets
        fb = np.cumsum(fdom) - fdom
        self.logger.debug("fb: " + str(fb))
        
        #fe are feature ending points
        fe = np.cumsum(fdom)
        self.logger.debug("fe: " + str(fe))
        
        encoder = OneHotEncoder()
        X = encoder.fit_transform(self.X0).toarray()
        print("X: \n" + str(X)) 
        return fdom, fb, fe, X
        

    
    
    '''
        Returns
        S basic slices
        R slice errors
        CI 
    
    '''
    def find_score_basic_slices(self, X):
        
        n,m = X.shape
        #ss0 = sizes of each basic slice are column sums
        ss0 = X.sum(axis=0)
        print("ss0: "  + str(ss0))
        
        #se0 = error of each basic slice
        se0 = (self.errors * X).sum(axis=0)
        print("se0" + str(se0))
        
        sm0 = (self.errors * X).max(axis=0)
        print("sm0: \n" + str(sm0))
        
        #pruning indicator
        CI = (ss0 >= self.sigma) & (se0 > 0)
        print("CI: \n" + str(CI))

        #prune basic slice scores and errors
        ss = ss0[CI]
        se = se0[CI]
        sm = sm[CI]
        
        print("ss: \n"+ str(ss))
        print("se: \n" + str(se))
        
        #select unpruned one hot slice representations of slices
        slices = X[:,CI]
        
        #score slices
        sc = self.score(ss, se, self.n)
        print("sc: " + str(sc))
        
        #statistics vector
        R = {
            "sc" : sc,
            "se" : se,
            "ss" : ss,   
            "sm" : sm         
        }
        

        #slices
        # Initialize a dictionary to store indices for each slice
        
        return slices, R, CI
    
    def score(self, sliceSizes, sliceErrors, nrows):
        sc = self.alpha * ((sliceErrors/sliceSizes) / self.avgError - 1) - (1-self.alpha) * (nrows/sliceSizes - 1)
        return np.nan_to_num(sc, nan=-np.inf)
        
    '''
    
        S: a set of slices from last iteration
        R: stastics where R[:, 4] is the sizes of slices and R[:, 2] are slice sizes
    
        Our implementation has 4 steps
        1. prune invalid input slices using thresholds for size (sigma) and error (non-negative)
            a) S = removeEmpty(S * (R[:, 4]>= sigma and R[:, 2] > 0))
            Reduces the input size of the pair generation
            Does not jeopardize overall pruning because we handle missing parents
            
        2. join compatable slices via a self join on S
            a) I = upper.tri((S hadmard S.T) = (L-2), values=True)
            -We are comparing the matmul output with L-2 to ensure compatability
            -eg) L-2=1 matches for level L=3 which checks that ab and ac have 1 item overlap to form abc
            -get upper traingle because S hadmard S.T is symetric
            
        3. Create combined slices by converting I to row-column index pairs and get extraction matricies P1 and P2
            - rix = matrix(I * seq(1, nr), nr*nc, 1)
            - rix = removeEmpty(target=rix, margin="rows")
            - P1 = table(seq(1, nrow(rix)), rix, nrow(rix), nrow(S)) 
            
            -Merge the combined slices via P = ((P1 hadmard S) + P2 hadmard S)) != 0
            -extract combined sizes ss, total errors se, maximinum errors sm as the minimum of parent slices with
            ss = min(p1 hadmard R[:, 4], P2 hadmard R[:, 4])
            
        4. Discard invalid slices with multiple assignments per feature using feature offsets fb and fe
        
            -with fb and fe, scan over P and check that I = I AND (rowSums(P[:,:]) <=1)
            for each original feature and retain only rows in P where no feature assignment is violated
            
        5. We now have valid slices for level L but there are duplicates.  Multiple parents create good pruning but exponentially increasing redudancy
            -use deduplication via slice ids
            -interpret one hot vectors as binary integers makes for overflow
            -Instead the id for slices is determined using dom=fe-fb+1 and compute ids like an ND-array index
            -scan over P and compute the sum of feature contributions by ID = ID + scale * rowIndexMax(P[:,:] * rowMaxs(P[:][:])) where scale is the feature entry from cumprod(dom)
            
            -duplicate slices now map to the same id and can be eliminated
            -domain can be large so we tranforms ids via frame recoding to consecutive integers
            -for pruning and deduplication we matrialize hte mapping as
                -M = table(ID, seq(1, nrow(P)))
                -deduplicate using P=M hadmard P
            
        6. Candidate pruning
            -before the final deduplication, apply all pruning techniques from section 3.2 with respect to all parents of a slice
            -compute the upper bound slice sizes, errors and sm (minimum of all parents), and number of parents np as
            -ss.bound = 1/rowMaxs(M * (1/ss.T))
            -np = rowSms((m hadmard (P1 + P2)) != 0)
            
            -minimize by maximizing the recirical (replacing infinity with 0)
            -accounting only existing parents while avoiding large dense intermediates
            
            Equation 3 computes the upper bound scores and all pruning becomes a simple filter over M
            
            M = M * (ss.bound > sigma and sc.bound > sck and sc.bound >=0 and np=L)
            
            -discard empty rows in M to get M'
            -deduplicate slices with S = M' hadmard P
                -S = P[,rowIndexMax(M')]
            return S as the new slice cnadidates
    
    '''
    def getPairCandidates(self, S, R, TS, TR, K, L, e, alpha, fb, fe):
        
        #1. prune invalid input slices using thresholds for size (sigma) and error (non-negative)
        CI = R["ss"] >= self.sigma & R["se"] > 0
        S = S[:, CI]
        
        #(S ⊙ S.T)
        SST = S * S.T
        valid_SST = SST == L-2
        I = np.triu(valid_SST)
        
        print(I)
        
        #create combined slices by converting I to row column index pairs
        nr, nc = I.shape
        rows, cols = np.nonzero(I)

        seq = np.arange(1, nr+1)
        rix = np.array([seq[row] for row in rows])
        
        
        #get P1 and P2 (parents 1 and parents 2)
        P1 = np.zeros((len(rows), nr))
        P2 = np.zeros(len(cols), nc)
        
        for i, row in enumerate(rows):
            P1[i, row] = 1
            
        for i, col in enumerate(cols):
            P2[i, col] = 1
        
        print("P1: \n" + str(P1))
        print("P2: \n" + str(P2))

        #form combined slices by combining parents
        P = (((P1 * S) + (P2 * S)) != 0)

        #extract combined slices and errors as minimum of parents
        ss = np.minimum(P1 * R["ss"], P2 * R["ss"])
        se = np.minimum(P1 * R["se"], P2 * R["se"])
        sm = np.minimum(P1 * R["sm"], P2 * R["sm"])
         

        #step 4: discard invalid slices with multiple assignments per feature
            #with each pair of fb and fe we scan over P and check if
                #I = I and (rowSums(P:,beg:end)<=1) for each feature
                #retain only rows in P where no feature assignment is violated
                
                

        #step 5: duduplication
        dom = fe-fb+1
        scale = np.cumprod(dom)
        ids = np.sum(combined_slices * scale, axis=1)
        
        unique_ids, indices = np.unique(ids, return_index=True)
        deduplicated_slices =combined_slices[indices]
        
        #step 6: candidate pruning
        ss_bound = np.max(deduplicated_slices @ (1 / ss.T), axis=0)
        np_count = np.sum(deduplicated_slices @ (P1 + P2) != 0, axis=1)
        
        sc_bound = np.maximum(ss_bound, 0)
        valid_candidates = (ss_bound > σ) & (sc_bound >= 0) & (np_count == L)
        
        final_slices = deduplicated_slices[valid_candidates]
        return final_slices
    
        
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
        
       
    '''
    
        The average dataset error is used in the scoring function
    ''' 
    
    def computeAverageError(self):
        return np.sum(self.errors)/self.n
        
    '''
    
        compute the scoring function
    
    '''


    def recursiveStep(self):
        bar=1
        
    
    
    
    #Find the topk slices for a given dataset D and error vector E
    
    

def testBed():
    

    logger = getLogger(__name__, "test.log")
    n_rows = 2
    n_cols = 3
    #X_train, _, _, _, _, _ = prepAdult()
    X_train = np.arange(1, n_rows+1).reshape(-1, 1) * np.ones((1, n_cols), dtype=int)
    errors = np.random.randint(0, 10, size=(X_train.shape[0]))[:,np.newaxis]

    
    logger.debug("Xtrain: ")
    logger.debug(X_train)
    logger.debug("errors: ")
    logger.debug(errors)
    
    

    
    foo = SliceFinder(X_train, errors, logger=logger)
    
    
testBed()




'''
        #row and column index
        rix = pd.Series((np.arange(1, self.m+1).reshape(-1,1) @ np.ones((1,self.n))).T.flatten())
        print(rix.shape)
        #rix = pd.Series(np.reshape(np.arange(1, self.m+1).reshape(-1,1) @ np.ones((1,self.n)), (self.m*self.n, 1)))
        cix = pd.Series((self.X0 + fb).flatten())
        
        self.logger.debug("rix: \n" + str(rix))
        self.logger.debug("cix: \n" + str(cix))

        #contingency table
        X = pd.crosstab( rix, cix).to_numpy()
        print(X.shape)
        self.logger.debug("X: \n" + str(X))
        
        return fdom, fb, fe, X

'''
    
