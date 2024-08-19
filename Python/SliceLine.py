import numpy as np
import pandas as pd
from utils import prepAdult, getLogger, unpickleDataset
from sklearn.preprocessing import OneHotEncoder
import time
import logging
import math
from scipy.sparse import csr_matrix

'''
    An implementation of SliceLine Fast Linear Alebra Based Slice Finding
    
    Algorithm Credit: https://dl.acm.org/doi/10.1145/3448016.3457323


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


#------------------------------
#TODO X0 has integer encoded featrues (categorical) 1-|fi| our dataset does not !!!!
#------------------------------  


class SliceFinder:

    def __init__(self, X, e:list, k=1, sigma=1, alpha=1.0, L=5, logger=None, auto=True):
        #e: list of errors for each datapoint
        #sigma: size threshold for smallest allowable slice
        #alpha: score function balancing parameter
        #L : how many levels of the lattice to prune
        
        
        #Initilize variables
        self.logger = logger
        self.X0 = X
        self.k = k
        self.errors = e
        self.sigma = sigma
        self.alpha = alpha
        self.n, self.m = self.X0.shape[0], self.X0.shape[1]
        self.L_max = min(L, self.m)
        self.avgError = np.sum(self.errors)/self.n
        self.logger.info("samples: " + str(self.n))
        self.logger.info("features: " + str(self.m))
        self.logger.info("avg error: " + str(self.avgError))
        
        
        
        if auto:
            self.run()

    
    
    def run(self):
        #one hot encode dataset and get new feature begginings (fb) and ends (fe)
        fdom, fb, fe, X = self.data_prep()
        
        #new number of features
        self.n2 = X.shape[1]
        
        #get 1-predicate slices, stats, pruning indicator
        S, R, CI = self.find_score_basic_slices(X)
        
        #Top slices (TS), top statistics (TR)
        TS, TR = self.maintainTopK(S, R, np.zeros((0, self.n2)), np.zeros((0,4)))
        
        #Run recursive step
        self.result = self.mainLoop(X, S, R, CI, TS, TR, fb, fe)
    
    
    
    def mainLoop(self, X, S, R, CI, TK, TKC, fb, fe):
        
        t0 = time.time()
        
        self.logger.info("Entering the main loop")
        L = 1
        
        self.logger.info("Pruning current slices")
        X = X[:, CI]
        
        #while there are still valid slices left and 
        while(S.shape[0] > 0 and L < self.L_max):
            L = L+1
            self.logger.info(" Forming new slices for level " + str(L))
            S = self.get_pair_candidates(S, R, TK, TKC, L, fb, fe)
            
            self.logger.info("  Evaluating new slices")
            R = np.zeros((S.shape[0], 4))

            for i in range(S.shape[0]):
                R[i] = self.eval_slice(X, self.errors, S[i], L)
            
            self.logger.info("  Maintaing topk")
            TK, TKC = self.maintainTopK(S, R, TK, TKC)
            
        self.logger.debug("finished in " + str(time.time() - t0 ))
        self.logger.debug("decoding top-k slices")
        TK = self.decode_top_k(TK, fb, fe)
        return {"TK" : TK, "TKC" : TKC}
            
    
    '''
        un-one hot the matrix to get slice representations from original dataset
    '''
    def decode_top_k(self, TK, fb, fe):
        
        R = np.ones((TK.shape[0], fb.shape[0]))

        #for each feature beg-end
        if TK.shape[0] > 0:
            for j in range(0, len(fb)):
                beg = fb[j]
                end = fe[j]

                
                #find the index in the expanded fature with a 1, that poisition is feature number
                sub_TK = TK[:, beg:end]  
                mask = np.sum(sub_TK, axis=1) > 0 
                maxcol = np.argmax(sub_TK, axis=1)
                maxcol[mask] += 1
                I = np.sum(sub_TK, axis=1) * maxcol

                R[:, j] = I
  
        return R
                 

    def eval_slice(self, X, e, tS, l, alpha):
        I = (X @ tS) == 1
        ss = np.sum(I, axis=1)
        se = (e.T @ I).T
        sm = np.sum(I * (e @ np.ones((1, I.shape[1]))))
        
        sc = self.score(ss, se, X.shape[0])
        R = np.vstack((sc, se, sm, ss)).T
        return R
    
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
        #print("X: \n" + str(X)) 
        return fdom, fb, fe, X
      
    
    '''
        compute count bins for an int or vector
        
        for ints it returns a set of bins with that ints bin set to value 1
        for vectors it returns a set of bins where bins[i] is the number of occourances of i in the vector
    '''
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
            self.logger.error("unsupported use of getCounts")
            exit()
                
                
                
        

    '''
        one hot encode a vector v into m bins
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
        #print("slices" + str(slices))
        
        
        #slice scores
        sc = self.score(ss, se, self.n)
        self.logger.debug("sc: \n" + str(sc))
        
        #statistics vector
        R = np.vstack((sc, se, sm, ss)).T



        return slices, R, CI
        


    '''
        Form a set of valid L-predicate slices using the given L-1 predicate slices S
    '''
    def get_pair_candidates(self, S, R, TK, TKC, L, fb, fe):
        
        #1. prune invalid input slices using thresholds for size (sigma) and error (non-negative)
        CI = (R[:, 3] >= self.sigma) & (R[:, 1] > 0)
        S = S[CI]

        #(S ⊙ S.T) creates every possible combination of slices
        SST = S @ S.T

        #valid slices should have L-2 overlap
        valid_SST = (SST == (L-2))
        #print("valid SST: \n" + str(valid_SST))
        
        
        #since the matrix is symetric we only need top half
        #I is a mask for new slices with L-2 overlap
        I = np.triu(valid_SST) * valid_SST
        

        #create combined slices by converting I to row, column index pairs (rix and cix)
        n_rows, n_cols = I.shape
        
        #form row index matrix
        rix = np.tile(np.arange(1, n_rows+1)[:, np.newaxis], (1,n_cols))
        rix = I * rix
        rix = rix.flatten(order="F")
        rix = rix[ rix!=0 ]
        rix = rix-1
        
        #column index matrix
        cix = np.tile(np.arange(1,n_cols+1), (n_rows, 1))
        cix = I * cix
        cix = cix.flatten(order="F")
        cix = cix[cix != 0]
        cix = cix-1
        
        #if there are new slices
        if np.sum(rix) != 0:
            
            #get parents from row and column indicies
            P1 = self.expand(rix, S.shape[0])
            P2 = self.expand(cix, S.shape[0])
            #print("P1: \n" + str(P1))
            #print("P2: \n" + str(P2))
            
            #form new slices
            newSlices = P1 + P2
            P = (((P1 @ S) + (P2 @ S)) != 0) * 1
                 
        else:
            print("Error, ran out of valid slices")
            exit()

        
        #extract combined slice sizes and errors as minimum of parents (with parent handling)
        se = np.minimum(P1 @ R[:, 1], P2 @ R[:, 1])
        sm = np.minimum(P1 @ R[:, 2], P2 @ R[:, 2])
        ss = np.minimum(P1 @ R[:, 3], P2 @ R[:, 3])
        #print("ss: \n" + str(ss))
        #print("se: \n" + str(se))
        #print("sm: \n" + str(sm))



        #step 4: discard invalid slices with multiple assignments per feature
        
        #indicator for valid new slices      
        I = np.ones((P.shape[0]), dtype=bool)

        #with each expanded feature (from beg-end)
        for j in range(fb.shape[0]):
            beg = fb[j]
            end = fe[j]
            
            #make sure it only has 1 domain value assigned to it
            rowsums = np.sum(P[:, beg:end], axis=1)
            rowsums = rowsums <= 1

            #update mask to exclude features with more than one value
            I = I & rowsums


        #prune slices with more than one assignment per feature
        newSlices = newSlices[I]
        P = P[I]
        ss = ss[I].reshape(-1,1)
        se = se[I].reshape(-1,1)
        sm = sm[I].reshape(-1,1)



        #step 5: duduplication 
        dom = fe-fb+1

        
        
        self.logger.info("getting ids")
        #unique ids for each slice are the sum of weighted feature contributions
        ID = np.zeros((P.shape[0]), dtype=object)
        for j in range(dom.shape[0]):
            beg = fb[j]
            end = fe[j]
            
            max_col = np.argmax(P[:, beg:end], axis=1) + 1
            rowsums = np.sum(P[:, beg:end], axis=1)
            I = max_col * rowsums

            scale = 1
            if j < dom.shape[0]:
                scale = np.prod(dom[j+1:])
            
            I = I.astype(object)
            scale = int(scale)

            ID = ID + I * scale
            
        self.logger.info("got ids")

        #size pruning by upper bounding sizes
        
        #print("getting map")
        map = pd.crosstab(pd.Series(ID), pd.Series(np.arange(0, P.shape[0]))).to_numpy()

        #print("got map")
        #print("map shape: " + str(map.shape))
        ex = np.ones((map.shape[0])).reshape(-1,1)
        ubSizes = 1/np.max(map * (1/ex @ ss.T), axis=0)
        ubSizes[ubSizes == np.inf] = 0
        fSizes = ubSizes >= self.sigma
        #print("size pruning done")
        
        #error pruning mask
        ubError = 1/(np.max(map * (1/(ex @ se.T)), axis=0))
        ubError[ubError == np.inf] = 0
        ubMError = 1/np.max(map * (1/ (ex @ sm.T)), axis=0)
        ubMError[ubMError == np.inf] = 0.0
        #print("error pruning done")

        
        #score pruning mask
        ubScores = self.upperbound_score(ubSizes, ubError, ubMError, self.sigma, self.alpha, self.avgError, self.n2)
        TMP3 = self.analyze_topK(TKC)
        minsc = TMP3["minScore"]
        fScores = (ubScores > minsc) & (ubScores > 0)
        #print("score pruning done")
        
        
        #missing parents pruning mask
        sparse_map = csr_matrix(map)
        sparse_slices = csr_matrix(newSlices)
        result_sparse = sparse_map.dot(sparse_slices)
        numParents = result_sparse.nnz
        fParents = (numParents == L)
        #print("parent pruning done")
        
        #print("applying pruning masks")
        #combine all masks
        map = map * (fSizes & fScores & fParents) @ np.ones(map.shape[1])


        #apply masks and deduplication
        dedup = map[np.max(map, axis=0) != 0,:] != 0
        P = (dedup @ P) != 0
        #print("done applying pruning masks")
        #return new slices
        return P
    
    
    '''
        find minimum and maximum score in the current topk set TKC
        
        returns {"maxScore" : float, "minScore": float}
    '''
    def analyze_topK(self, TKC):
        maxScore = -np.inf
        minScore = -np.inf
        
        if TKC.shape[0] > 0:
            maxScore = TKC[0,0]
            minScore = TKC[0, TKC.shape[1]-1]
            

        return {"maxScore" : maxScore, "minScore" : minScore}



    
    '''
        Apply scoring function to the current sizes errors,sizes and n
        if divide by 0 occours, turn into -inf
    
    '''
    def score(self, sliceSizes, sliceErrors, nrows):
        sc = self.alpha * ((sliceErrors/sliceSizes) / self.avgError - 1) - (1-self.alpha) * (nrows/sliceSizes - 1)
        return np.nan_to_num(sc, nan=-np.inf)
    
    

    '''
        upper bound slicelines scoring function by computing it at several critical points and taking the maximum
    
    
    '''

    def upperbound_score(self, ss, se, sm, minSup, alpha, avg_error, n):
        # probe interesting points of sc in the interval [minSup, ss],
        # and compute the maximum to serve as the upper bound 
        
        #s is the critical points we have to check
        s = np.column_stack((np.full((ss.shape[0]), minSup), np.maximum(se/sm, minSup), ss))
        ex = np.ones((3))
        smex = s*np.column_stack((sm, sm, sm))
        seex = np.column_stack((se, se, se))
        pmin = np.minimum(smex, seex)

        sc = np.max(alpha * ((pmin/s) / avg_error-1) - (1-alpha) * (1/s*n - 1), axis=1)
        sc = np.nan_to_num(sc, nan=-np.inf)
        
        
        return sc
            
    '''
        Compute TS (topk slice representations) and TR (top k slice stastics)
        by selecting the topk scoring slices from the current set of slices S
    
    '''
    def maintainTopK(self, S, R, TK, TKC):
        
        self.logger.debug("Maintain topk:")
        top_k_indicies = None
        

        #prune on size and score
        I = (R[:,0] > 0) & (R[:,3] >= self.sigma)
        
        #if there are still valid slices left
        if np.sum(I) != 0:
            S = S[I]
            R = R[I]
            
            #???
            if S.shape[1] != TK.shape[1] and S.shape[1] == 1:
                S = S.T 
                R = R.T
                
            self.logger.debug(" S \n" + str(S))
            self.logger.debug(" R \n" + str(R))

            
            slices = np.vstack((TK, S))
            scores = np.vstack((TKC, R))
            self.logger.debug("slices \n" + str(slices))
            self.logger.debug("scores \n" + str(scores))
            
            
            #find new top-k slice indicies
            sorted_idxs = np.argsort(scores[:, 0])
            top_idxs = sorted_idxs[:min(self.k, len(sorted_idxs))]
            self.logger.debug("sorted idxs" + str(sorted_idxs))
            self.logger.debug("topk idxs" + str(top_idxs))
            
            #update top slices and top statistics with new topk values
            TK = slices[top_idxs, :]
            TKC = scores[top_idxs, :]
            
        
        #print("TK org")
        #print(TK)
        #print("TKC org")
        #print(TKC)
        return TK, TKC
        
    
    @staticmethod
    def pretty_print_results(result, featureNames, domainValues=None):
        TK = result["TK"]
        TKC = result["TKC"]
        
        #print("TK")
        #print(TK)
        
        #print("TKC")
        #print(TKC)
        
        count = 0
        
        
        for slice in TK:
            print("Slice " + str(count))
            
            print("   ----------------------------")
            #print features
            for j in range(len(slice)):
                if slice[j] != 0:
                    
                    if domainValues == None:
                        print("   " + str(featureNames[j] + " = " + str(slice[j])))
                    else:
                        print("   " + str(featureNames[j] + " = " + str(domainValues[j][slice[j]-1])))
                        
                    
            #print scores
            print("   ----------------------------")
            print("   score: " + str(TKC[count][0]))
            print("   avg. error: " + str(TKC[count][1]))
            print("   max error: " + str(TKC[count][2]))
            print("   size: " + str(TKC[count][3]))
            
            
            count += 1
            print("--------------------------------------------------")
        
        
      
       

def testBed():
    
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
    logger = getLogger(__name__, "test.log")
    logger.setLevel(logging.INFO)
    X_train, _, _, _ = unpickleDataset("./Data/Adult/train.pkl", "./Data/Adult/test.pkl")
    np.random.seed(1)
    errors = np.random.random(size=X_train.shape[0]).reshape(-1,1)
    
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv("./Data/Adult/dummyErrors.csv", header=True)
    
    sigma = max(math.ceil(0.01 * X_train.shape[0]), 8)
    sf = SliceFinder(X_train, errors, k=4, sigma=5, alpha=0.95, L=2, logger=logger, auto=False)
    sf.run()
    SliceFinder.pretty_print_results(sf.result, ["feature" + str(i) for i in range(86)])
    
#testBed()
testAdult()

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

