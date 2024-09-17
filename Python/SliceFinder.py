import numpy as np
import pandas as pd
from utils import prepAdult, getLogger, unpickleDataset
from sklearn.preprocessing import OneHotEncoder
import time
import logging
import math
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod



#------------------------------
#TODO X0 has integer encoded featrues (categorical) 1-|fi| our dataset does not !!!!
#------------------------------  


class SliceFinder(ABC):
    
    """

        An implementation of SliceLine Fast Linear Alebra Based Slice Finding
        
         Credit: https://dl.acm.org/doi/10.1145/3448016.3457323


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
    

    

    def __init__(self, X, e:list, k=1, sigma=1, alpha=0.85, L=2, logger=None, auto=True):
        #e: list of errors for each datapoint
        #sigma: size threshold for smallest allowable slice
        #alpha: score function balancing parameter
        #L : how many levels of the lattice to prune
        
        
        #Initilize variables
        if logger == None:
            self.logger = getLogger(__name__, "dummyLog.log")
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
            
        self.X0 = X
        self.k = k
        self.errors = e
        self.sigma = sigma
        self.alpha = alpha
        self.n, self.m = self.X0.shape[0], self.X0.shape[1]
        self.L_max = min(L, self.m)
        self.avg_error = np.sum(self.errors)/self.n
        self.logger.info("samples: " + str(self.n))
        self.logger.info("features: " + str(self.m))
        self.logger.info("avg error: " + str(self.avg_error))
        
        print("k: " + str(self.k))
        
        
        
        if auto:
            self.run()

    @abstractmethod
    def score(self, slice_sizes, slice_errors, total_samples):
        pass
    
    @abstractmethod
    def upperbound_score(self, ss, se, sm):
        pass
    
    def run(self):
        #one hot encode dataset and get new feature begginings (fb) and ends (fe)
        domains, f_beg, f_end, X = self.data_prep()
        
        #new number of features
        self.features = X.shape[1]
        
        #get 1-predicate slices, stats, pruning indicator
        S, R, CI = self.find_score_basic_slices(X)
        
        #Top slices (TS), top statistics (TR)
        TS, TR = self.maintainTopK(S, R, np.zeros((0, self.features)), np.zeros((0,4)))
        
        #Run recursive step
        self.result = self.mainLoop(X, S, R, CI, TS, TR, f_beg, f_end)
    
    
    
    

    def mainLoop(self, X, S, R, CI, TK, TKC, fb, fe):
        """
            Executes the recursive part of sliceLine
            
            :param X: 
                a one hot encoded dataset matrix
                
            :param S: 
                a set of 1-predicate slices
                
            :param R: 
                Statistics for the 1-predicate slices. (R[0] = ...)
            
            :param CI: 
                Pruning indicator for basic slices.
            
            :param TK: 
                Top-k slice representations.
            
            :param TKC: 
                Top-k slice statistics.
            
            :param fb: 
                Feature beginnings.
            
            :param fe: 
                Feature endings.

            :return: 
                Top slices (TK) and statistics (TKC) in a dict
            :rtype: dict
                

    """
        
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
            
    

    def decode_top_k(self, TK, fb, fe):
        """
        Summary:
            Convert slice representations back to their original form 
            (before `X` was one-hot encoded).
        
        :param TK: 
            Matrix of the top-k slices.
        
        :param fb: 
            List of feature beginnings.
        
        :param fe: 
            List of feature endings.
        
        :return: 
            Top-k slice representations from the original dataset `X`.
        """
        
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



    def eval_slice(self, X, e, S):

        """
        Summary:
            Get statistics for slice `S`.
        
        :param X: 
            One-hot encoded dataset.
        
        :param e: 
            Error vector for the entire dataset.
        
        :param S: 
            A slice that needs to be evaluated.
        
        :return: 
            Vector where `stats[0]` is the score, `stats[1]` is the error, 
            `stats[2]` is the maximum error, and `stats[3]` is the size.
        """
        
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
    
    
    def data_prep(self):
        
        """
        Summary: 
            1. Find domains for each feature.
            2. Compute start and end offsets of each feature.
            3. Get one-hot encoding of the entire dataset.
        
        :return: 
            - fdom: Domains sizes for each feature.
            - fb: Feature beginnings in the one-hot matrix.
            - fe: Feature endings in the one-hot matrix.
            - X: One-hot encoded dataset.
        """
        
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
    

    def getCounts(self, v, bins):
        # print(type(v))
        """
        Summary:
            - Count bins for an integer or vector.
            - For integers, it returns a set of bins with that integer's bin set to value 1.
            - For vectors, it returns a set of bins where `bins[i]` is the number of occurrences of `i` in the vector.
        
        :param v: 
            Vector or integer to be counted.
        
        :param vmax: 
            The number of bins to include. Must be >= the largest element in `v`.
        
        :return: 
            Bins for the given input. For integers, it returns a set with the corresponding bin set to 1. For vectors, it returns a set where each bin contains the count of occurrences of the index in the vector.
        """            
        
        
        if type(v) is np.int64 or type(v) is np.int32:
            vec = np.zeros((bins))
            vec[v] = 1
            return vec
        
        elif type(v) is np.ndarray:
            vec = np.zeros((bins))
            for vij in v:
                vec[vij] += 1
        else:
            self.logger.error("unsupported use of getCounts")
            exit()
                
                
                
        


    def expand(self, v, m=None):
        """
            Summary:
                gets a dense matrix representation of a sparse matrix v
            
        """
        
        
        if m == None:
            m = np.max(v)
            
        R = np.zeros( (v.shape[0], m), dtype=int)
        #count number of occourances for each value in each row
        for i in range(v.shape[0]):

            R[i,:] = self.getCounts(v[i], m)

        return R
    
    

    def find_score_basic_slices(self, X):
        """
        Summary:
            Find the 1-predicate slices and their statistics.
        
        :param X: 
            One-hot encoded dataset.
        
        :return: 
            - S: Basic slices.
            - R: Basic slice statistics.
            - CI: Pruning indicator based on size and score.
        """
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
        print("scores: " + str(sc))

        
        #form statistics matrix
        R = np.vstack((sc, se, sm, ss)).T

        return slices, R, CI
        



    def get_pair_candidates(self, S, R, TKC, L, fb, fe):
        """
        Summary:
            Form a set of valid L-predicate slices using the given L-1 predicate slices `S`.

            1. Form every combination of slices from level L-1.
            2. Prune those without L-2 overlap.
            3. Form row and column index pairs of valid combinations.
            4. Get parents using row, column index pairs.
            5. Form new slices.
            6. Prune slices that assign more than one value to a feature.
            7. Extract size and error for new slices as the minimum of their parents.
            8. Perform deduplication using unique IDs.
            9. Upper bound the scoring function and prune new slices.
        
        :param S: 
            Slices from level L-1.
        
        :param R: 
            Statistics for those slices.
        
        :param TKC: 
            Top-k slice statistics.
        
        :param L: 
            Current level of slices.
        
        :param fb: 
            Feature beginnings.
        
        :param fe: 
            Feature endings.
        
        :return: 
            - P: A set of valid pair candidates.
        """
        
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
        ub_scores = self.upperbound_score(ub_sizes, ub_error, ub_max_error)
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
    
    

    def analyze_topK(self, topK):
        """
        Summary:
            Find the minimum and maximum score in the current top-k set `TKC`.
        
        :param TKC: 
            A k x 4 matrix of statistics for the current slices.
        
        :return: 
            A dictionary containing:
            - `"maxScore"`: The maximum score (float).
            - `"minScore"`: The minimum score (float).
        """
        if topK.shape[0] > 0:
            maxScore = topK[0,0]
            minScore = topK[0, topK.shape[1]-1]
            

        return {"maxScore" : maxScore, "minScore" : minScore}



            

    def maintainTopK(self, S, R, TK, TKC):
        """
        Summary:
            Update the top-k slices and their statistics given the new slices `S` and statistics `R`.
        
        :param S: 
            New slices to be considered for updating the top-k.
        
        :param R: 
            New statistics corresponding to the new slices `S`.
        
        :param TK: 
            Current top-k slices that will be updated.
        
        :param TKC: 
            Current top-k statistics that will be updated.
        
        :return: 
            - `TK`: Updated top-k slices.
            - `TKC`: Updated top-k statistics.
        """

        

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
            

        return TK, TKC
        
    
    

    @staticmethod
    def pretty_print_results(result, featureNames, domainValues=None):
        """
        Summary:
            Print the top-k selected slices and their statistics. 
            Fill in the numeric column names and domain values with their actual values.
        
        :param result: 
            A dictionary containing:
            - `"TK"`: Top-k slices.
            - `"TKC"`: Top-k statistics.
        
        :param featureNames: 
            A list of names for each feature, 1-indexed. 
            `featureNames[0]` should be an empty string.
        
        :param domainValues: 
            A matrix where `domainValues[i][k]` is the name of the feature `i` taking on the `k`th domain value.
        
        :return: 
            None. Prints the output to the screen.
        """
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
                        print("   " + str(featureNames[j-1] + " = " + str(slice[j])))
                    else:
                        print("   " + str(featureNames[j-1] + " = " + str(domainValues[featureNames[j-1]][slice[j]])))
                        
                    
            #print scores
            print("   ----------------------------")
            print("   score: " + str(TKC[count][0]))
            print("   toal error: " + str(TKC[count][1]))
            print("   max error: " + str(TKC[count][2]))
            print("   size: " + str(TKC[count][3]))
            
            
            count += 1
            print("--------------------------------------------------")
        
        
