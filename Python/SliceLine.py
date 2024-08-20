import numpy as np
import pandas as pd
from utils import prepAdult, getLogger, unpickleDataset
from sklearn.preprocessing import OneHotEncoder
import time
import logging
import math
from scipy.sparse import csr_matrix


"""_summary_

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
"""


#------------------------------
#TODO X0 has integer encoded featrues (categorical) 1-|fi| our dataset does not !!!!
#------------------------------  


class SliceFinder:

    def __init__(self, X, e:list, k=1, sigma=1, alpha=0.85, L=2, logger=None, auto=True):
        #e: list of errors for each datapoint
        #sigma: size threshold for smallest allowable slice
        #alpha: score function balancing parameter
        #L : how many levels of the lattice to prune
        
        
        #Initilize variables
        if logger == None:
            self.logger = getLogger(__name__, "dummyLog.log")
        else:
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
    
    
    
    
    """
        Summary:
            Executes the recursive part of sliceLine
            
        Parameters:
            X - a one hot encoded dataset matrix
            S - a set of 1-predicate slices
            R - statistics for the 1-predicate slices (R[0] = 
            CI - pruning indicator for basic slices
            TK - top-k slice representations
            TKC - top-k slice statistics 
            fb - feature begginings
            fe - feature endings
            
        Returns:
            {"TK" : TK, "TKC" : TKC}
            TK is the final top-k slices
            TKC is the final top-k statistics
    """
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
            S = self.get_pair_candidates(S, R, TKC, L, fb, fe)
            
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
            
    
    """
        Summary:
            convert slice representations back to original form (before X was one hot encoded)
        
        Paramaters:
            TK - matrix of the top-k slices
            fb - list of feature begginings
            fe - list of feature endings
            

        Returns:

    
    
    """
    def decode_top_k(self, TK, fb, fe):
        
        TK_decoded = np.ones((TK.shape[0], fb.shape[0]))

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

                TK_decoded[:, j] = I
        return TK_decoded


    """
        Summary:
            get statistics for slice S 
        
        Parameters:
            X: one hot encoded dataset
            e: error vector for entire dataset
            S: a slice that needs to be evaluated
        
        Returns:
            stats: vector where stats[0]=score, stats[1] = error, stats[2] = max error, stats[3] = size
    """
    def eval_slice(self, X, e, S):
        
        #indicator for which columns of error vector this slice needs
        s_mask = (X @ S) == 1
        
        #slice size, total error, max error, score
        ss = np.sum(s_mask, axis=1)
        se = (e.T @ s_mask).T
        sm = np.sum(s_mask * (e @ np.ones((1, s_mask.shape[1]))))
        sc = self.score(ss, se, X.shape[0])
        
        #stack them to form a row in 
        stats = np.vstack((sc, se, sm, ss)).T
        return stats
    
    """
        Summary: 
            1. Find domains for each feature
            2. Compute start and end offset of each feature
            3. get one hot encoding of entire dataset
        
        Returns:
            fdom - domains sizes for each feature
            fb - feature begginings in one hot matrix
            fe - feature endings in one hot matrix
            X - one hot encoded dataset
    """
    
    def data_prep(self):
        
        #fdom (feature domain sizes) are the maximum value along ecah column of the dataset for continuous integer encoding
        fdom = np.max(self.X0, axis=0)
        self.logger.debug("fdom: " + str(np.max(self.X0, axis=0)))
        
        #fb is where each feature in new matrix starts
        fb = np.cumsum(fdom) - fdom
        self.logger.debug("fb: " + str(fb))
        
        #fe is where each feature in new matrix ends
        fe = np.cumsum(fdom)
        self.logger.debug("fe: " + str(fe))
        
        
        #one hot encode the entire dataset
        encoder = OneHotEncoder()
        X = encoder.fit_transform(self.X0).toarray()


        return fdom, fb, fe, X
    
    
    """
        Summary:
            -count bins for an int or vector
            -for ints it returns a set of bins with that ints bin set to value 1
            -for vectors it returns a set of bins where bins[i] is the number of occourances of i in the vector
        
        Parameters:
            v - vector or integer
            vmax - the number of bins to include, must be >= the largest element in v
        
    """
    def getCounts(self, v, bins):
        
        if type(v) is np.int64:
            vec = np.zeros((bins))
            vec[v] = 1
            return vec
        
        elif type(v) is np.ndarray:
            vec = np.zeros((bins))
            for vij in vi:
                vec[vij] += 1
        else:
            self.logger.error("unsupported use of getCounts")
            exit()
                
                
                
        

    """
        Summary:
            gets a dense matrix representation of a sparse matrix v
        
    """
    def expand(self, v, m=None):
        
        
        if m == None:
            m = np.max(v)
            
        R = np.zeros( (v.shape[0], m), dtype=int)
        #count number of occourances for each value in each row
        for i in range(v.shape[0]):

            R[i,:] = self.getCounts(v[i], m)

        return R
    
    
    """
        Summary:
            Find the 1-predicate slices and their statistics
        Parameters:
            X - one hot encoded dataset
            
        Returns:
            S - basic slices 
            R - basic slice statistics
            CI - pruning indicator on size and score
    """
    def find_score_basic_slices(self, X):
        
        n,m = X.shape
        #ss0 = sizes of each basic slice are column sums
        ss0 = X.sum(axis=0)
        self.logger.debug("ss0: \n"  + str(ss0))
        
        #se0 = error of each basic slice
        se0 = (self.errors * X).sum(axis=0)
        
        #sm0 = the maximum error among elements in each slice
        sm0 = (self.errors * X).max(axis=0)

        #pruning indicator
        CI = (ss0 >= self.sigma) & (se0 > 0)

        #prune basic slice scores and errors
        ss = ss0[CI]
        se = se0[CI]
        sm = sm0[CI]
        
        self.logger.debug("ss: \n"+ str(ss))
        self.logger.debug("se: \n" + str(se))
        self.logger.debug("sm: \n" + str(sm))
        
        
        #a set of pruned slices (very sparse matrix representation)
        slices = self.expand(np.arange(0, m)[CI], m=m)

        #slice scores
        sc = self.score(ss, se, self.n)

        
        #form statistics matrix
        R = np.vstack((sc, se, sm, ss)).T

        return slices, R, CI
        


    """
        Summary:
            Form a set of valid L-predicate slices using the given L-1 predicate slices S
            
            1. form every combination of slices from level L-1
            2. prune those without L-2 overlap
            3. form row and column index pairs of valid combinations
            4. get parents using row,column index pairs
            5. form new slices
            6. prune slices that assign more than one value to a feature
            7. extract size and error for new slices as the minimum of its parents
            8. deduplication using unique ids
            9. upper bound the scoring function and prune new slices
            
        Parameters:
            S - slices from level L-1
            R - statistics for those slices
            TKC - top-k slice statistics
            L - current level
            fb - feature begginings
            fe - feature endings
            
        Returns:
            P - a set of valid pair candidates 
    """
    def get_pair_candidates(self, S, R, TKC, L, fb, fe):
        
        #prune invalid input slices using thresholds for size (sigma) and error (non-negative)
        CI = (R[:, 3] >= self.sigma) & (R[:, 1] > 0)
        S = S[CI]

        #(S ⊙ S.T) creates every possible combination of slices
        SST = S @ S.T

        #valid slices should have L-2 overlap
        valid_SST = (SST == (L-2))

        
        #since the matrix is symetric we only need top half
        #I is a mask for new slices with L-2 overlap
        I = np.triu(valid_SST) * valid_SST
        

        #create combined slices parents by converting I to row, column index pairs (rix and cix)
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
            
            #get parents
            P1 = self.expand(rix, S.shape[0])
            P2 = self.expand(cix, S.shape[0])
            
            #form new slices
            newSlices = P1 + P2
            P = (((P1 @ S) + (P2 @ S)) != 0) * 1
                 
        else:
            print("Error, ran out of valid slices")
            exit()

        
        #extract new slice sizes and errors as minimum of parents (with parent handling)
        se = np.minimum(P1 @ R[:, 1], P2 @ R[:, 1])
        sm = np.minimum(P1 @ R[:, 2], P2 @ R[:, 2])
        ss = np.minimum(P1 @ R[:, 3], P2 @ R[:, 3])



        #discard invalid slices with multiple assignments per feature using indicator I
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



        #deduplication 
        dom = fe-fb+1
        #unique ids for each slice are the sum of weighted feature contributions.  Makes it so slices map to same ID if they have same features
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
    

        #map cotains deduplication pruning mask
        map = pd.crosstab(pd.Series(ID), pd.Series(np.arange(0, P.shape[0]))).to_numpy()
        ex = np.ones((map.shape[0])).reshape(-1,1)
        
        #size pruning by upper bounding sizes
        ub_sizes = 1/np.max(map * (1/ex @ ss.T), axis=0)
        ub_sizes[ub_sizes == np.inf] = 0
        f_sizes = ub_sizes >= self.sigma
        
        
        #error pruning mask
        ub_error = 1/(np.max(map * (1/(ex @ se.T)), axis=0))
        ub_error[ub_error == np.inf] = 0
        ub_max_error = 1/np.max(map * (1/ (ex @ sm.T)), axis=0)
        ub_max_error[ub_max_error == np.inf] = 0.0


        #score pruning mask
        ub_scores = self.upperbound_score(ub_sizes, ub_error, ub_max_error, self.n2)
        TMP3 = self.analyze_topK(TKC)
        minsc = TMP3["minScore"]
        fScores = (ub_scores > minsc) & (ub_scores > 0)
        
        
        #missing parents pruning mask
        sparse_map = csr_matrix(map)
        sparse_slices = csr_matrix(newSlices)
        result_sparse = sparse_map.dot(sparse_slices)
        num_parents = result_sparse.nnz
        f_parents = (num_parents == L)

        #combine all pruning masks
        map = map * (f_sizes & fScores & f_parents) @ np.ones(map.shape[1])


        #apply masks
        dedup = map[np.max(map, axis=0) != 0,:] != 0
        P = (dedup @ P) != 0

        return P
    
    
    """
        Summary:
            find minimum and maximum score in the current topk set TKC
        
        Paramters:
            TKC: a k x 4 matrix of statistics for current slices

        Returns:
            {"maxScore" : float, "minScore": float}
    """
    def analyze_topK(self, TKC):
        maxScore = -np.inf
        minScore = -np.inf
        
        if TKC.shape[0] > 0:
            maxScore = TKC[0,0]
            minScore = TKC[0, TKC.shape[1]-1]
            

        return {"maxScore" : maxScore, "minScore" : minScore}



    
    """
        Summary:
            Apply SliceLines scoring function if divide by 0 occours, turn into -inf
            
        Parameters:
            sliceSizes: vector of sizes for each slice
            sliceErrors: vector of errors for each slice
            nrows: number of samples in the entire dataset
    
    """
    def score(self, slice_sizes, slice_errors, total_samples):
        sc = self.alpha * ((slice_errors/slice_sizes) / self.avgError - 1) - (1-self.alpha) * (total_samples/slice_sizes - 1)
        return np.nan_to_num(sc, nan=-np.inf)
    
    

    """
        Summary:
            Compute an upper bound for slicelines scoring function
        
        Parameters:
            ss: size of each slice
            se: error for each slice
            sm: max error for each slice
            min

    
    
    """

    def upperbound_score(self, ss, se, sm, n):
        # probe interesting points of sc in the interval [minSup, ss],
        # and compute the maximum to serve as the upper bound 
        
        #s is the critical points we have to check
        s = np.column_stack((np.full((ss.shape[0]), self.sigma), np.maximum(se/sm, self.sigma), ss))
        ex = np.ones((3))
        smex = s*np.column_stack((sm, sm, sm))
        seex = np.column_stack((se, se, se))
        pmin = np.minimum(smex, seex)

        sc = np.max(self.alpha * ((pmin/s) / self.avg_error-1) - (1-self.alpha) * (1/s*n - 1), axis=1)
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
#testAdult()

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

