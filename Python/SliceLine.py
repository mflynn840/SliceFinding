import numpy as np
import pandas as pd
from utils import prepAdult, getLogger
from sklearn.preprocessing import OneHotEncoder
import time

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
        self.n, self.m = self.X0.shape[0], self.X0.shape[1]
        self.L_max = min(L, self.m)
        self.avgError = SliceFinder.avg_error(self.errors, self.n)
        
        #report results
        self.logger.info("samples: " + str(self.n))
        self.logger.info("features: " + str(self.m))
        self.logger.info("avg error: " + str(self.avgError))
        
        #do the sliceline algorithm
        fdom, fb, fe, X = self.data_prep()
        S, R, CI = self.find_score_basic_slices(X)
        TS, TR = self.maintainTopK(S, R, 0, 0)
        
        self.mainLoop(X, S, R, CI, TS, TR, fb, fe)

        
            

         
    def mainLoop(self, X, S, R, CI, TS, TR, fb, fe):
        
        t0 = time.time()
        
        self.logger.info("Entering the main loop")
        L = 1
        
        self.logger.info("Pruning current slices")
        X = X[:, CI]
        
        #while there are still valid slices left and 
        while(S.shape[0] > 0 and L < self.L_max):
            L = L+1
            self.logger.info("Forming new slices for level " + str(L))
            S = self.get_pair_candidates(S, R, TS, TR, L, fb, fe)
            
            self.logger.info("Evaluating new slices")
            R = np.zeros(self.S.shape[0], 4)
            S2 = S[CI]
            for i in range(S.shape[0]):
                R[i] = self.eval_slices(X, self.e, S2[i].T, L)
            
            self.logger.info("Maintain topk")
            TS, TR = self.maintainTopK(S, R, TS, TR)
            
        self.logger.debug("finished in " + str(t0-time.time()))
        self.logger.debug("decoding top-k slices")
        return self.decodeTopK(self.TS, self.fb, self.fe), TR
            

    '''
        Data Prep (SliceLine):
            1. Find domains for each feature
            2. Compute start and end offset of each feature
            3. Get matrix X which has one hot encoding over all features and domains
    '''
    
    def data_prep(self):
        
        #fdom (feature domain sizes) are the maximum value along ecah column of the dataset for continuous integer encoding
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
      
      
    def getCounts(self, vi, vmax):
        if type(vi) is np.int64:
            vec = np.zeros((vmax))
            vec[vi] = 1
            return vec
        elif type(vi) is np.ndarray:
            vec = np.zeros((vmax))
            for vij in vi:
                vec[vij] += 1
        else:
            self.logger.error("unssupported use of getCounts")
            exit()
                
                
                
        

    '''
    

    
    
    '''
    def expand(self, v, m=None):
        
        
        if m == None:
            m = np.max(v)
            
        R = np.zeros( (v.shape[0], m), dtype=int)
        #count number of occourances for each value in each row
        for i in range(v.shape[0]):

            R[i,:] = self.getCounts(v[i], m)

        return R
    
    
    '''
        Returns
        S basic slices
        R slice errors
        CI pruning indicator
    '''
    def find_score_basic_slices(self, X):
        
        n,m = X.shape
        #ss0 = sizes of each basic slice are column sums
        ss0 = X.sum(axis=0)
        self.logger.debug("ss0: \n"  + str(ss0))
        
        #se0 = error of each basic slice
        se0 = (self.errors * X).sum(axis=0)
        self.logger.debug("se0: \n" + str(se0))
        
        sm0 = (self.errors * X).max(axis=0)
        self.logger.debug("sm0: \n" + str(sm0))
        
        #pruning indicator
        CI = (ss0 >= self.sigma) & (se0 > 0)
        self.logger.debug("CI: \n" + str(CI))

        #prune basic slice scores and errors
        ss = ss0[CI]
        se = se0[CI]
        sm = sm0[CI]
        
        self.logger.debug("ss: \n"+ str(ss))
        self.logger.debug("se: \n" + str(se))
        self.logger.debug("sm: \n" + str(sm))
        
        
        #a set of pruned slices
        slices = self.expand(np.arange(0, m)[CI], m=m)
        print("slices" + str(slices))
        
        
        #slice scores
        sc = self.score(ss, se, self.n)
        self.logger.debug("sc: \n" + str(sc))
        
        #statistics vector
        R = {
            "sc" : sc,
            "se" : se,
            "ss" : ss,   
            "sm" : sm         
        }
        
        
        return slices, R, CI
    
    
    
    '''
    
        Apply scoring function to the current sizes errors,sizes and n
        if divide by 0 occours, turn into -inf
    
    '''
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
    def get_pair_candidates(self, S, R, TK, TKC, L, fb, fe):
        
        #1. prune invalid input slices using thresholds for size (sigma) and error (non-negative)
        CI = (R["ss"] >= self.sigma) & (R["se"] > 0)
        print("CI: " + str(CI))
        S = S[:, CI]
        

        #(S ⊙ S.T) creates every possible combination of slices
        SST = S @ S.T
        print("SST: \n" + str(SST))
        
        #valid slices should have L-2 overlap
        #at level L=3, L-2=1 checks that ab and ac have 1 item overlap to generate abc
        valid_SST = (SST == (L-2))
        print("valid SST: \n" + str(valid_SST))
        
        #since the matrix is symetric we only need top half
        I = np.triu(valid_SST) * valid_SST
        
        #I is a logical matrix indicating where the upper triangular part of SST is true
        print("I: \n" + str(I))
        
        #create combined slices by converting I to row column index pairs
        n_rows, n_cols = I.shape
        
        #form row index matrix
        rix = np.tile(np.arange(1, n_rows+1)[:, np.newaxis], (1,n_cols))
        print("rix1: \n" + str(rix))
        rix = I * rix
        print("rix2: \n" + str(rix))
        #flatten by going down columns instead of rowss        
        rix = rix.flatten(order="F")
        print("rix3: \n" + str(rix))
        rix = rix[rix!=0]
        print("rix4: \n" + str(rix))
        rix = rix-1
        
        
        cix = np.tile(np.arange(1,n_cols+1), (n_rows, 1))
        print("cix1: \n" + str(cix))
        cix = I * cix
        print("cix2: \n" + str(cix))
        cix = cix.flatten(order="F")
        print("cix3: \n" + str(cix))
        cix = cix[cix != 0]
        print("cix4: \n" + str(cix))
        cix = cix-1
        
        print("rix shape: \n" + str(rix.shape))
        print("cix shape: \n" + str(cix.shape))
        
        if np.sum(rix) != 0:
            #get parents from row and column indicies
            P1 = self.expand(rix, S.shape[0])
            P2 = self.expand(cix, S.shape[0])
            
            
            print("P1: \n" + str(P1))
            print("P2: \n" + str(P2))
            
            newSlices = P1 + P2
            P = (((P1 @ S) + (P2 @ S)) != 0) * 1
            print("P: \n" + str(P))
            print(P.shape)
                 
        else:
            print("Error, ran out of valid slices")
            exit()

        
        #extract combined slice sizes and errors as minimum of parents
        ss = np.minimum(P1 @ R["ss"], P2 @ R["ss"])
        se = np.minimum(P1 @ R["se"], P2 @ R["se"])
        sm = np.minimum(P1 @ R["sm"], P2 @ R["sm"])
        
        print("ss: \n" + str(ss))
        print("se: \n" + str(se))
        print("sm: \n" + str(sm))

        #step 4: discard invalid slices with multiple assignments per feature
            #with each pair of fb and fe we scan over P and check if
                #I = I and (rowSums(P:,beg:end)<=1) for each feature
                #retain only rows in P where no feature assignment is violated
        I = np.ones((P.shape[0]), dtype=bool)
        #print("I1: \n" + str(I))
        #print("I shape" + str(I.shape))
        
        #find cases where a slice has more than one value assigned to the same feature
        for j in range(fb.shape[0]):
            beg = fb[j]
            end = fe[j]
            
            #print("beg: \n" + str(beg))
            #print("end: \n" + str(end))
            
            rowsums = np.sum(P[:, beg:end], axis=1)
            rowsums = rowsums <= 1
            #print("rowsums: \n" + str(rowsums))
            #print("rowsums shape" + str(rowsums.shape))

            I = I & rowsums

        
        print("I2: \n" + str(I))
        
        #prune out invalid slices (those with more than 1 assignment per feature)
        newSlices = newSlices[I]
        print("old p shape" + str(P.shape))
        P = P[I]

        print("new p shape" + str(P.shape))
        ss = ss[I]
        se = se[I]
        sm = sm[I] 
        

                 

        #step 5: duduplication (right now we have slices for level L but there are duplicates)
        dom = fe-fb+1
        ids = np.zeros(P.shape[0])
        
        '''
        We scan over P, and compute the sum of feature contributions by 
        ID = ID + scale · rowIndexMax(P:,beg:end) · rowMaxs(P:,beg:end)
        where scale is the feature entry from cumprod(dom)

        '''
        
        #get a unique id for each slice
        ID = np.zeros((P.shape[0]))
        print(ID.shape)
        for j in range(dom.shape[0]):
            beg = fb[j]
            end = fe[j]
            
            max_col = np.argmax(P[:, beg:end], axis=1) + 1
            rowsums = np.sum(P[:, beg:end], axis=1)
            
            #print("P: \n" + str(P[:,beg:end ]))
            print("max_col2: \n" + str(max_col))
            print("rowsums2: \n " + str(rowsums))
            
            I = max_col * rowsums
            #print("I")
            #print(I)
            
            scale = 1
            if j < dom.shape[0]:
                scale = np.prod(dom[j+1:])
                
            ID = ID + I * scale
            
        print("ID: \n" + str(ID))
        print("ID shape: " + str(ID.shape))
        #TODO from here on

        #size pruning with rowMin-rowMax transform
        map = pd.crosstab(pd.Series(ID), pd.Series(np.arange(1, P.shape[0]+1))).to_numpy()
        ex = np.arange(1, map.shape[0]+1)
        
        ubSizes = 1/np.max(map * (1/ex @ ss.T), axis=0)
        ubSizes[ubSizes == np.inf] = 0
        fSizes = ubSizes >= self.sigma
        
        #error pruning (using upperbound)
        ubError = 1/np.max(map * (1/ex @ se.T))
        ubError[ubError==np.inf] = 0
        
        ubMError = 1/np.max(map * (1/ ex @ sm.T))
        ubMError[ubMError == np.inf] = 0
        
        ubScores = self.upperbound_score(ubSizes, ubError, ubMError)
        TMP3 = self.analyzeTopK(TKC)
        minsc = TMP3[["minsc"]]
        fScores = (ubScores > minsc) & (ubScores > 0)
        
        
        #missing parents pruning
        numParents = np.sum((map @ newSlices) != 0)
        fParents = (numParents == L)
        
        
        #apply all pruning
        map = map * (fSizes & fScores & fParents) @ np.ones(map.shape[1])
        print("map2")
        print(map)
        
        #deduplication of join outputs
        Dedup = map[np.max(map) != 0,:] != 0
        P = (Dedup @ P) != 0
        
        return P


    '''
        Compute TS (topk slice representations) and TR (top k slice stastics)
        by selecting the topk scoring slices from the current set of slices S
    
    '''
    def maintainTopK(self, S, R, TS=None, TR=None):
        top_k_indicies = None
        
        #R["sc"] is a vector of scores for each slice
        if len(R["sc"]) < self.k:
            # If fewer than k scores, find all available indices and fill the rest with -1
            top_k_indices = np.concatenate((np.argsort(R["sc"])[::-1], -np.ones(self.k - len(R["sc"]), dtype=int)))
        else:
            # Otherwise, get the indices of the top k largest scores
            top_k_indices = np.argsort(R["sc"])[-self.k:][::-1]

        print("top k indicies: "  + str(top_k_indices))
        
        #find the slice indexes with the top k 
        TS = S[top_k_indicies]
        
        TR = {}
        for i in ["sc", "sm", "ss", "se"]:
            TR[i] = R[i][top_k_indicies]

        return TS, TR
        
       
    '''
    
        The average dataset error is used in the scoring function
    ''' 
    @staticmethod
    def avg_error(errors, n):
        return np.sum(errors)/n
        
    '''
    
        compute the scoring function
    
    '''


    def recursiveStep(self):
        bar=1
        
    
    
    
    #Find the topk slices for a given dataset D and error vector E
    
    

def testBed():
    

    logger = getLogger(__name__, "test.log")
    n_rows = 3
    n_cols = 4
    #X_train, _, _, _, _, _ = prepAdult()
    X_train = np.arange(1, n_rows+1).reshape(-1, 1) * np.ones((1, n_cols), dtype=int)
    errors = np.arange(1, 4).reshape(-1,1)
    print(errors)

    
    logger.debug("Xtrain: ")
    logger.debug(X_train)
    logger.debug("errors: ")
    logger.debug(errors)
    
    

    
    foo = SliceFinder(X_train, errors, k=1, sigma=0, alpha=0.95, L=2, logger=logger)
    
    
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
    
